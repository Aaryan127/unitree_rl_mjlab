"""Debug a task without training.

Builds the environment with a few robots, steps it, and prints every observation
term, reward term, command and sensor (shape, min/max/mean, NaNs). Use it after every
change to a config, observation or reward, before starting a long training run.

  python scripts/debug_env.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2
  python scripts/debug_env.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2 --agent random --steps 200
  python scripts/debug_env.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2 --breakpoint

With --breakpoint you land in pdb after stepping, with these variables ready:
  env, obs, robot = env.scene["robot"], and every manager (env.reward_manager, ...).
Useful pdb commands: p obs["height"].shape, p robot.data.joint_pos[0], c (continue), q.
To stop inside one term instead, put `breakpoint()` as the first line of that
observation/reward function and run this script.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Debug on CPU.

import numpy as np
import torch
import tyro

import mjlab.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg


@dataclass
class DebugConfig:
  num_envs: int = 2
  steps: int = 50
  agent: Literal["zero", "random"] = "zero"
  small_terrain: bool = True
  """Use a 2 x 5 tile terrain grid (instead of 10 x 20): much faster on CPU."""
  play: bool = False
  """Use the play config (no noise, no pushes)."""
  env_id: int = 0
  """Which robot's values to print."""
  breakpoint: bool = False


def _stats(x: torch.Tensor) -> str:
  x = x.float()
  nan = int(torch.isnan(x).sum())
  flag = f"  <-- {nan} NaN!" if nan else ""
  return f"min {x.min():+8.3f}  max {x.max():+8.3f}  mean {x.mean():+8.3f}{flag}"


def print_observations(env: ManagerBasedRlEnv, obs: dict, env_id: int) -> None:
  manager = env.observation_manager
  print("\n=== Observations (term ranges over all robots) ===")
  for group, terms in manager.active_terms.items():
    data = obs[group]
    print(f"[{group}] shape {tuple(data.shape)}")
    start = 0
    for name, dims in zip(terms, manager.group_obs_term_dim[group], strict=True):
      size = int(np.prod(dims))
      term = data[:, start : start + size]
      start += size
      sample = term[env_id, : min(size, 6)].tolist()
      print(f"   {name:24s} {size:4d}  {_stats(term)}   env{env_id}: {np.round(sample, 3).tolist()}")


def print_rewards(env: ManagerBasedRlEnv, step_rewards: torch.Tensor, steps: int) -> None:
  manager = env.reward_manager
  print(f"\n=== Rewards: weighted value per step, averaged over {steps} steps and robots ===")
  print("   (positive = encouraged, negative = penalty; mjlab also multiplies by dt)")
  means = step_rewards.mean(dim=(0, 1))
  for name, cfg, mean in zip(manager._term_names, manager._term_cfgs, means, strict=True):
    print(f"   {name:24s} weight {cfg.weight:+10.2e}   {mean:+9.4f}")


def print_commands_and_sensors(env: ManagerBasedRlEnv, env_id: int) -> None:
  print(f"\n=== Commands (env {env_id}) ===")
  for name in env.command_manager.active_terms:
    value = env.command_manager.get_command(name)[env_id]
    print(f"   {name:24s} {np.round(value.tolist(), 3).tolist()}")
  print("\n=== Sensors ===")
  names = list(env.scene.sensors.keys())
  for name, sensor in env.scene.sensors.items():
    if any(name.startswith(other + "_") for other in names if other != name):
      continue  # Internal per-geom channel of a contact sensor listed above.
    data = sensor.data
    if isinstance(data, torch.Tensor):
      print(f"   {name:24s} {tuple(data.shape)}  {_stats(data)}")
      continue
    fields = {
      field: value
      for field, value in vars(data).items()
      if isinstance(value, torch.Tensor)
    } if hasattr(data, "__dict__") else {}
    print(f"   {name:24s} {type(sensor).__name__}")
    for field, value in fields.items():
      print(f"      .{field:20s} {tuple(value.shape)}  {_stats(value)}")


def main() -> None:
  task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
  )
  cfg = tyro.cli(DebugConfig, args=remaining, prog=f"{sys.argv[0]} {task}")

  env_cfg = load_env_cfg(task, play=cfg.play)
  env_cfg.scene.num_envs = cfg.num_envs
  terrain = env_cfg.scene.terrain
  if cfg.small_terrain and terrain is not None and terrain.terrain_generator is not None:
    terrain.terrain_generator.num_rows = 2
    terrain.terrain_generator.num_cols = 5
    terrain.max_init_terrain_level = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  robot = env.scene["robot"]  # noqa: F841  (for --breakpoint)
  print(f"\nTask {task}: {cfg.num_envs} robots, joints {list(robot.joint_names)}")

  obs, _ = env.reset()
  num_actions = env.action_manager.total_action_dim
  step_rewards = []
  resets = 0
  for _ in range(cfg.steps):
    if cfg.agent == "zero":
      action = torch.zeros(cfg.num_envs, num_actions)
    else:
      action = 2.0 * torch.rand(cfg.num_envs, num_actions) - 1.0
    obs, _, terminated, truncated, _ = env.step(action)
    step_rewards.append(env.reward_manager._step_reward.clone())
    resets += int((terminated | truncated).sum())

  print_observations(env, obs, cfg.env_id)
  print_rewards(env, torch.stack(step_rewards), cfg.steps)
  print_commands_and_sensors(env, cfg.env_id)
  print(f"\nEpisode resets during {cfg.steps} steps: {resets}")

  if cfg.breakpoint:
    breakpoint()
  env.close()


if __name__ == "__main__":
  main()
