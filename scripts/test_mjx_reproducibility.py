"""Reproducibility test script for MJX training.

This script verifies:
1. Reset reproducibility with same seed
2. Reward scale similarity
3. Determinism across runs
4. No exploding gradients
"""

import os
import numpy as np
from typing import Dict, List, Tuple

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx


def create_env(model_path: str, num_envs: int, seed: int):
    """Create MJX environment with seed."""
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(mj_model)

    default_data = mjx.make_data(mjx_model)
    mjx.forward(mjx_model, default_data)

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    qpos_noise = jax.random.normal(subkey, (num_envs, mj_model.nq)) * 0.01
    qvel_noise = jax.random.normal(subkey, (num_envs, mj_model.nv)) * 0.01

    batch_data = default_data.replace(
        qpos=default_data.qpos[None, :] + qpos_noise,
        qvel=default_data.qvel[None, :] + qvel_noise,
        ctrl=jp.zeros((num_envs, mj_model.nu)),
    )

    return {
        "mj_model": mj_model,
        "mjx_model": mjx_model,
        "batch_data": batch_data,
        "key": key,
    }


def run_simulation(env: Dict, num_steps: int) -> List[jp.ndarray]:
    """Run simulation and return trajectory."""
    data = env["batch_data"]
    step_fn = jax.jit(mjx.step)
    forward_fn = jax.jit(mjx.forward)

    trajectory = []

    for _ in range(num_steps):
        data = step_fn(env["mjx_model"], data)
        data = forward_fn(env["mjx_model"], data)

        trajectory.append(data.qpos.copy())

    return trajectory


def test_reset_determinism(model_path: str, num_envs: int = 32, num_steps: int = 100):
    """Test that reset with same seed produces same trajectory."""
    print("\n" + "=" * 60)
    print("Test 1: Reset Reproducibility")
    print("=" * 60)

    seed = 42

    env1 = create_env(model_path, num_envs, seed)
    traj1 = run_simulation(env1, num_steps)

    env2 = create_env(model_path, num_envs, seed)
    traj2 = run_simulation(env2, num_steps)

    max_diff = 0.0
    for t1, t2 in zip(traj1, traj2):
        diff = np.abs(np.array(t1 - t2)).max()
        max_diff = max(max_diff, diff)

    print(f"Max difference across {num_steps} steps: {max_diff}")
    print(
        f"Test PASSED: {max_diff < 1e-6}"
        if max_diff < 1e-6
        else f"Test FAILED: {max_diff >= 1e-6}"
    )

    return max_diff < 1e-6


def test_different_seeds_diverge(
    model_path: str, num_envs: int = 32, num_steps: int = 100
):
    """Test that different seeds produce different trajectories."""
    print("\n" + "=" * 60)
    print("Test 2: Different Seeds Diverge")
    print("=" * 60)

    env1 = create_env(model_path, num_envs, seed=42)
    traj1 = run_simulation(env1, num_steps)

    env2 = create_env(model_path, num_envs, seed=123)
    traj2 = run_simulation(env2, num_steps)

    diverged = False
    for i, (t1, t2) in enumerate(zip(traj1, traj2)):
        diff = np.abs(np.array(t1 - t2)).max()
        if diff > 1e-4:
            diverged = True
            print(f"Trajectories diverged at step {i}, diff: {diff}")
            break

    print(
        f"Test PASSED: Trajectories diverged"
        if diverged
        else "Test FAILED: Trajectories did not diverge"
    )

    return diverged


def test_reward_scale(model_path: str, num_envs: int = 32, num_steps: int = 100):
    """Test that reward scales are reasonable."""
    print("\n" + "=" * 60)
    print("Test 3: Reward Scale")
    print("=" * 60)

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

    rewards = []
    for _ in range(num_steps):
        ctrl = jax.random.normal(jax.random.PRNGKey(_), (num_envs, mj_model.nu)) * 10
        batch_data = batch_data.replace(ctrl=ctrl)
        batch_data = step_fn(mjx_model, batch_data)
        batch_data = forward_fn(mjx_model, batch_data)

        energy = jp.sum(jp.abs(batch_data.ctrl), axis=-1)
        rewards.append(energy.mean())

    reward_array = np.array(rewards)

    print(f"Mean reward: {reward_array.mean():.4f}")
    print(f"Std reward: {reward_array.std():.4f}")
    print(f"Max reward: {reward_array.max():.4f}")
    print(f"Min reward: {reward_array.min():.4f}")

    scale_reasonable = reward_array.std() < 100 and not np.any(np.isnan(reward_array))
    print(
        f"Test PASSED: Reward scale reasonable"
        if scale_reasonable
        else "Test FAILED: Reward scale unreasonable"
    )

    return scale_reasonable


def test_gradient_stability(model_path: str, num_envs: int = 32):
    """Test that gradients don't explode."""
    print("\n" + "=" * 60)
    print("Test 4: Gradient Stability")
    print("=" * 60)

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(mj_model)

    default_data = mjx.make_data(mjx_model)
    mjx.forward(mjx_model, default_data)

    batch_data = default_data.replace(
        qpos=jp.tile(default_data.qpos[None, :], (num_envs, 1)),
        qvel=jp.tile(default_data.qvel[None, :], (num_envs, 1)),
        ctrl=jp.zeros((num_envs, mj_model.nu)),
    )

    def loss_fn(ctrl):
        data = batch_data.replace(ctrl=ctrl)
        step_fn = jax.jit(mjx.step)
        forward_fn = jax.jit(mjx.forward)
        data = step_fn(mjx_model, data)
        data = forward_fn(mjx_model, data)

        qvel_norm = jp.sum(data.qvel**2)
        return qvel_norm

    ctrl = jax.random.normal(jax.random.PRNGKey(0), (num_envs, mj_model.nu)) * 10

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(ctrl)

    grad_norm = np.sqrt(np.sum(np.array(grads) ** 2))

    print(f"Gradient norm: {grad_norm:.4f}")

    stable = grad_norm < 1e6 and not np.isnan(grad_norm)
    print(
        f"Test PASSED: Gradients stable"
        if stable
        else "Test FAILED: Gradients exploded"
    )

    return stable


def test_batching_correctness(
    model_path: str, batch_size: int = 128, num_steps: int = 50
):
    """Test that vmap batched simulation produces correct results."""
    print("\n" + "=" * 60)
    print("Test 5: Batching Correctness")
    print("=" * 60)

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(mj_model)

    default_data = mjx.make_data(mjx_model)
    mjx.forward(mjx_model, default_data)

    def run_single_env(key):
        qpos_noise = jax.random.normal(key, (mj_model.nq,)) * 0.01
        qvel_noise = jax.random.normal(jax.random.split(key)[0], (mj_model.nv,)) * 0.01

        data = default_data.replace(
            qpos=default_data.qpos + qpos_noise,
            qvel=default_data.qvel + qvel_noise,
            ctrl=jp.zeros(mj_model.nu),
        )

        step_fn = jax.jit(mjx.step)
        forward_fn = jax.jit(mjx.forward)

        for _ in range(num_steps):
            data = step_fn(mjx_model, data)
            data = forward_fn(mjx_model, data)

        return data.qpos

    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    single_results = jp.array([run_single_env(k) for k in keys])

    batched_data = default_data.replace(
        qpos=jp.tile(default_data.qpos[None, :], (batch_size, 1))
        + jax.random.normal(jax.random.PRNGKey(0), (batch_size, mj_model.nq)) * 0.01,
        qvel=jp.tile(default_data.qvel[None, :], (batch_size, 1))
        + jax.random.normal(jax.random.PRNGKey(0), (batch_size, mj_model.nv)) * 0.01,
        ctrl=jp.zeros((batch_size, mj_model.nu)),
    )

    step_fn = jax.jit(mjx.step)
    forward_fn = jax.jit(mjx.forward)

    for _ in range(num_steps):
        batched_data = step_fn(mjx_model, batched_data)
        batched_data = forward_fn(mjx_model, batched_data)

    batched_results = batched_data.qpos

    diff = np.abs(np.array(single_results - batched_results)).max()

    print(f"Max difference between single and batched: {diff}")
    print(
        f"Test PASSED: {diff < 1e-5}" if diff < 1e-5 else f"Test FAILED: {diff >= 1e-5}"
    )

    return diff < 1e-5


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MJX Reproducibility Tests")
    parser.add_argument(
        "--model", type=str, default="robot.xml", help="Path to MuJoCo XML model"
    )
    parser.add_argument(
        "--num_envs", type=int, default=32, help="Number of environments for tests"
    )
    args = parser.parse_args()

    os.environ["MUJOCO_GL"] = "egl"

    print("=" * 60)
    print("MJX Reproducibility Test Suite")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Num envs: {args.num_envs}")
    print(f"Device: {jax.default_backend()}")

    results = []

    results.append(
        ("Reset Determinism", test_reset_determinism(args.model, args.num_envs))
    )
    results.append(
        (
            "Different Seeds Diverge",
            test_different_seeds_diverge(args.model, args.num_envs),
        )
    )
    results.append(("Reward Scale", test_reward_scale(args.model, args.num_envs)))
    results.append(
        ("Gradient Stability", test_gradient_stability(args.model, args.num_envs))
    )
    results.append(
        ("Batching Correctness", test_batching_correctness(args.model, args.num_envs))
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")

    return all_passed


if __name__ == "__main__":
    main()
