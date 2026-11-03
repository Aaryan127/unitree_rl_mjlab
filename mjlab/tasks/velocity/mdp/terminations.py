from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.any(sensor.data.found, dim=-1)


def physics_exploded(
  env: ManagerBasedRlEnv,
  max_joint_vel: float = 100.0,
  max_root_vel: float = 20.0,
  asset_name: str = "robot",
) -> torch.Tensor:
  """True for robots whose simulation has blown up: a joint faster than `max_joint_vel`
  (rad/s; the Go2's motors top out near 30) or the base faster than `max_root_vel` (m/s).

  Such a robot stays finite, so nan_detection misses it, but it keeps feeding absurd
  values into the batch (rung 7: hip angles of ~1e6 rad, default_pose of -2e13 per
  episode) until it times out. Ending its episode resets its whole physics state.
  """
  data = env.scene[asset_name].data
  fast_joint = data.joint_vel.abs().amax(dim=-1) > max_joint_vel
  fast_base = data.root_link_lin_vel_w.norm(dim=-1) > max_root_vel
  return fast_joint | fast_base
