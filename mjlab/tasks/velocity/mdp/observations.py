from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


##
# PGTT observations.
##


def height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  base_height: float = 0.3,
  clip: float = 1.0,
) -> torch.Tensor:
  """Terrain height map around the base, relative to the robot.

  value = (base_z - terrain_z) - base_height, one value per ray, clipped to [-clip, clip].

  On flat ground with the base at its nominal height every value is ~0, so 0 means
  "ground where expected"; positive values are terrain lower than expected (holes,
  stairs down), negative values are terrain higher than expected (steps up). Missed rays
  also return 0. The real robot falls back to all-zeros when no height map is available,
  which is the same as "flat ground".

  Ray order follows GridPatternCfg: y outer loop, x inner loop, yaw-aligned with the base.
  Shape: [num_envs, num_rays].
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  data = sensor.data
  heights = data.pos_w[:, 2:3] - data.hit_pos_w[..., 2] - base_height
  heights = torch.where(data.distances >= 0, heights, torch.zeros_like(heights))
  return torch.clamp(heights, -clip, clip)


def gait_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """[cos(phi_i) x legs, sin(phi_i) x legs, f] from the gait phase command."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return command


def geom_friction(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Sliding friction coefficient of the selected geoms (e.g. feet). Privileged.

  Reads the per-environment value set by the friction randomization event.
  Shape: [num_envs, num_geoms].
  """
  asset: Entity = env.scene[asset_cfg.name]
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]
  return env.sim.model.geom_friction[:, geom_ids, 0]


def body_external_wrench(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """External force and torque applied to the selected bodies (world frame). Privileged.

  Shape: [num_envs, num_bodies * 6] as [fx, fy, fz, tx, ty, tz] per body.
  """
  asset: Entity = env.scene[asset_cfg.name]
  wrench = asset.data.body_external_wrench[:, asset_cfg.body_ids, :]
  return wrench.reshape(env.num_envs, -1)


def contact_normals(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Contact normals (world frame) of a contact sensor, zero when not in contact.

  Shape: [num_envs, num_primaries * 3]. Privileged.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.normal is not None, f"'{sensor_name}' needs the 'normal' field."
  return sensor.data.normal.flatten(start_dim=1)
