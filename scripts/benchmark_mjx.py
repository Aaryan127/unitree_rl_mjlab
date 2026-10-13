"""Benchmark script for MJX training performance.

This script measures:
- Steps per second (throughput)
- GPU utilization
- Compile time
- Memory usage at different batch sizes
"""

import os
import time
import argparse
from typing import Dict, List

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx


def create_simple_env(model_path: str, num_envs: int):
    """Create a simple MJX environment."""
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(mj_model)

    default_data = mjx.make_data(mjx_model)
    mjx.forward(mjx_model, default_data)

    batch_data = default_data.replace(
        qpos=jp.tile(default_data.qpos[None, :], (num_envs, 1)),
        qvel=jp.tile(default_data.qvel[None, :], (num_envs, 1)),
        ctrl=jp.zeros((num_envs, mj_model.nu)),
    )

    step_fn = jax.jit(mjx.step)
    forward_fn = jax.jit(mjx.forward)

    return {
        "mj_model": mj_model,
        "mjx_model": mjx_model,
        "batch_data": batch_data,
        "step_fn": step_fn,
        "forward_fn": forward_fn,
        "num_envs": num_envs,
        "nu": mj_model.nu,
    }


def benchmark_step(env: Dict, num_steps: int = 100, warmup: int = 10) -> float:
    """Benchmark simulation step performance.

    Returns steps per second.
    """
    data = env["batch_data"]
    step_fn = env["step_fn"]
    forward_fn = env["forward_fn"]

    for _ in range(warmup):
        data = step_fn(env["mjx_model"], data)
        data = forward_fn(env["mjx_model"], data)

    jax.block_until_ready(data)

    start = time.time()
    for _ in range(num_steps):
        data = step_fn(env["mjx_model"], data)
        data = forward_fn(env["mjx_model"], data)

    jax.block_until_ready(data)
    elapsed = time.time() - start

    steps_per_sec = num_steps * env["num_envs"] / elapsed

    return steps_per_sec


def benchmark_batch_sizes(
    model_path: str, batch_sizes: List[int], num_steps: int = 100
):
    """Benchmark performance across different batch sizes."""
    results = []

    print(f"{'Batch Size':<12} {'Steps/sec':<15} {'Compile Time (s)':<18}")
    print("-" * 50)

    for batch_size in batch_sizes:
        env = create_simple_env(model_path, batch_size)

        compile_start = time.time()
        step_fn = jax.jit(mjx.step)
        _ = step_fn(env["mjx_model"], env["batch_data"])
        jax.block_until_ready(_)
        compile_time = time.time() - compile_start

        steps_per_sec = benchmark_step(env, num_steps)

        results.append(
            {
                "batch_size": batch_size,
                "steps_per_sec": steps_per_sec,
                "compile_time": compile_time,
            }
        )

        print(f"{batch_size:<12} {steps_per_sec:<15,.0f} {compile_time:<18.2f}")

    return results


def get_memory_usage():
    """Get current GPU memory usage."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0


def benchmark_memory(model_path: str, batch_sizes: List[int]):
    """Benchmark memory usage across batch sizes."""
    print(f"\n{'Batch Size':<12} {'Memory (MB)':<15}")
    print("-" * 30)

    for batch_size in batch_sizes:
        env = create_simple_env(model_path, batch_size)

        _ = env["step_fn"](env["mjx_model"], env["batch_data"])
        jax.block_until_ready(env["batch_data"])

        memory_mb = get_memory_usage()
        print(f"{batch_size:<12} {memory_mb:<15}")

    print("\nNote: Memory usage shown is total GPU memory")
    print("Actual per-environment memory may vary")


def profile_jit_overhead(env: Dict, num_iterations: int = 1000):
    """Profile JIT compilation overhead."""
    print(f"\nJIT Overhead Profile ({num_iterations} iterations):")
    print("-" * 40)

    data = env["batch_data"]
    mjx_model = env["mjx_model"]
    step_fn = env["step_fn"]
    forward_fn = env["forward_fn"]

    jax.block_until_ready(data)

    times = []
    for _ in range(num_iterations):
        start = time.time()
        data = step_fn(mjx_model, data)
        data = forward_fn(mjx_model, data)
        jax.block_until_ready(data)
        times.append(time.time() - start)

    import numpy as np

    times = np.array(times) * 1000

    print(f"Mean time per step: {times.mean():.2f} ms")
    print(f"Std dev: {times.std():.2f} ms")
    print(f"Min: {times.min():.2f} ms")
    print(f"Max: {times.max():.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="MJX Benchmark Script")
    parser.add_argument(
        "--model", type=str, default="robot.xml", help="Path to MuJoCo XML model"
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512, 1024, 2048, 4096],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of steps per benchmark"
    )
    args = parser.parse_args()

    os.environ["MUJOCO_GL"] = "egl"

    print("=" * 60)
    print("MJX Performance Benchmark")
    print("=" * 60)
    print(f"Device: {jax.default_backend()}")
    print(f"Batch sizes: {args.batch_sizes}")
    print()

    print("Throughput Benchmark:")
    benchmark_batch_sizes(args.model, args.batch_sizes, args.steps)

    print("\nMemory Usage Benchmark:")
    benchmark_memory(args.model, args.batch_sizes)

    env = create_simple_env(args.model, args.batch_sizes[-1])
    profile_jit_overhead(env)

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
