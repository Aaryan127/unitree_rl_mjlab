"""Behaviour metrics that expose reward hacking.

Reward only says how much the policy scored, not how. These numbers measure what the
robot physically does, so a policy that scores by lying down, shuffling or ignoring the
command shows up even when the reward curve looks fine:

  base_height        base above the terrain under it [m] (Go2 walks at ~0.27)
  low_base_frac      fraction of robots with the base below `low_height`
  body_ground_frac   fraction of robots with trunk/hips on the ground (if sensor exists)
  leg_ground_frac    fraction of robots with a thigh/calf on the ground
  speed_ratio        speed achieved along the commanded direction / commanded speed,
                     over robots commanded to move (1 = tracks, ~0 = ignores command)
  feet_down_moving   fraction of moving robots with all four feet on the ground
                     (shuffling / sliding instead of stepping)
  standing_drift     speed of robots commanded to stand [m/s]
  action_jitter      mean |a_t - a_(t-1)| over the joints (tanh-bounded actions: a full-range
                     3 Hz sine gives 0.24, slamming the joints back and forth ~2)
  action_saturation  fraction of tanh-bounded actions beyond +-0.95 (bang-bang control)
  action_tremor      mean |a_t - 2a_(t-1) + a_(t-2)|: fast changes on top of the gait (any
                     motion of at most 3 Hz within the action range stays below 0.09)

`hacking_monitor` logs them every step as Metrics/hack/* (it is a reward term that always
returns 0, so it never changes training); scripts/check_reward_hacking.py calls
`behaviour_metrics` directly on any checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.rewards import base_height_above_terrain

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def behaviour_metrics(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  height_sensor: str = "height_scanner",
  feet_sensor: str = "feet_ground_contact",
  leg_sensor: str = "leg_ground_contact",
  body_sensor: str = "body_ground_contact",
  low_height: float = 0.15,
  moving_speed: float = 0.2,
) -> dict[str, torch.Tensor]:
  """Per-robot behaviour metrics, each of shape [num_envs] (NaN where not applicable)."""
  robot = env.scene["robot"]
  sensors = env.scene.sensors
  out: dict[str, torch.Tensor] = {}

  base_height = base_height_above_terrain(env, height_sensor)
  out["base_height"] = base_height
  out["low_base_frac"] = (base_height < low_height).float()

  if body_sensor in sensors:
    out["body_ground_frac"] = env.scene[body_sensor].data.found.flatten(1).any(-1).float()
  out["leg_ground_frac"] = env.scene[leg_sensor].data.found.flatten(1).any(-1).float()

  command = env.command_manager.get_command(command_name)
  cmd_xy = command[:, :2]
  cmd_speed = cmd_xy.norm(dim=-1)
  vel_xy = robot.data.root_link_lin_vel_b[:, :2]
  moving = cmd_speed > moving_speed
  along = (vel_xy * cmd_xy).sum(-1) / cmd_speed.clamp(min=1e-6)
  nan = torch.full_like(cmd_speed, torch.nan)
  out["speed_ratio"] = torch.where(moving, along / cmd_speed.clamp(min=1e-6), nan)
  feet_down = env.scene[feet_sensor].data.found.flatten(1).bool().all(-1).float()
  out["feet_down_moving"] = torch.where(moving, feet_down, nan)
  standing = (cmd_speed < 0.05) & (command[:, 2].abs() < 0.05)
  out["standing_drift"] = torch.where(standing, vel_xy.norm(dim=-1), nan)
  actions = env.action_manager
  out["action_jitter"] = (actions.action - actions.prev_action).abs().mean(-1)
  out["action_saturation"] = (actions.action.abs() > 0.95).float().mean(-1)
  out["action_tremor"] = (
    actions.action - 2 * actions.prev_action + actions.prev_prev_action
  ).abs().mean(-1)
  return out


def hacking_monitor(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  """Reward term that logs behaviour_metrics as Metrics/hack/* and always returns 0."""
  log = env.extras.setdefault("log", {})
  for name, value in behaviour_metrics(env, **params).items():
    valid = value[~torch.isnan(value)]
    if valid.numel():
      log[f"Metrics/hack/{name}"] = valid.mean()
  return torch.zeros(env.num_envs, device=env.device)
