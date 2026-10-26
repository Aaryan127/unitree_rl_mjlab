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


##
# LiDAR-built height map (sim version of scripts/lidar_height_map.py).
##


def grid_height_map_from_points(
  points_w: torch.Tensor,
  valid: torch.Tensor,
  base_pos_w: torch.Tensor,
  base_yaw: torch.Tensor,
  size_x: float,
  size_y: float,
  resolution: float,
  base_height: float,
  min_z: float,
  max_z: float,
  body_x: float,
  body_y: float,
  body_min_z: float,
  cell_radius: float,
  min_points: int,
  quantile: float,
  min_coverage: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Point cloud -> height map, exactly as the robot's LiDAR bridge builds it.

  points_w [B, N, 3] world points, valid [B, N]. Each point goes to its nearest grid cell
  (grid yaw-aligned with the base, same order as GridPatternCfg: y outer, x inner);
  a cell's ground is the `quantile` lowest point (lower index); cells with fewer than
  `min_points` points are 0; if fewer than `min_coverage` of the cells are observed the
  whole map is 0 ("no map", what the controller sees when the bridge sends nothing).
  Returns (values [B, M], coverage [B]).
  """
  nx = int(round(size_x / resolution)) + 1
  ny = int(round(size_y / resolution)) + 1
  m = nx * ny
  b, n, _ = points_w.shape
  rel = points_w - base_pos_w[:, None, :]
  c, s = torch.cos(base_yaw)[:, None], torch.sin(base_yaw)[:, None]
  lx = c * rel[..., 0] + s * rel[..., 1]
  ly = -s * rel[..., 0] + c * rel[..., 1]
  z = rel[..., 2]
  ix = torch.round((lx + size_x / 2) / resolution)
  iy = torch.round((ly + size_y / 2) / resolution)
  d2 = (lx - (ix * resolution - size_x / 2)) ** 2 + (ly - (iy * resolution - size_y / 2)) ** 2
  on_body = (lx.abs() <= body_x) & (ly.abs() <= body_y) & (z >= body_min_z)
  ok = (
    valid
    & (z >= min_z) & (z <= max_z)
    & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    & (d2 <= cell_radius**2)
    & ~on_body
  )
  cell = torch.where(ok, (iy * nx + ix).long(), torch.full_like(ix, m).long())  # m = dropped

  # Sort points by (cell, height): key = cell + normalized height in [0, 1).
  zn = torch.clamp((z - min_z) / (max_z - min_z), 0.0, 0.9999)
  order = torch.argsort(cell.double() + zn.double(), dim=1)
  z_sorted = torch.gather(z, 1, order)
  counts = torch.zeros(b, m + 1, device=z.device).scatter_add_(1, cell, torch.ones_like(z))[:, :m]
  starts = torch.cumsum(counts, dim=1) - counts
  k = torch.floor(quantile * torch.clamp(counts - 1, min=0))
  idx = torch.clamp(starts + k, max=n - 1).long()
  ground = torch.gather(z_sorted, 1, idx)

  observed = counts >= min_points
  values = torch.where(observed, torch.clamp(-ground - base_height, -1.0, 1.0), torch.zeros_like(ground))
  coverage = observed.float().mean(dim=1)
  values = torch.where((coverage >= min_coverage)[:, None], values, torch.zeros_like(values))
  return values, coverage


def lidar_height_map(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  size_x: float = 1.0,
  size_y: float = 0.8,
  resolution: float = 0.1,
  base_height: float = 0.25,
  min_z: float = -0.9,
  max_z: float = 0.45,
  body_x: float = 0.30,
  body_y: float = 0.15,
  body_min_z: float = -0.10,
  cell_radius: float = 0.075,
  min_points: int = 2,
  quantile: float = 0.2,
  min_coverage: float = 0.5,
  point_noise_std: float = 0.01,
) -> torch.Tensor:
  """Height map built from a simulated LiDAR, with the robot bridge's exact gridding.

  Same layout and values as `height_scan` ((base_z - ground_z) - base_height), but the
  ground comes from LiDAR returns (beams from the head mount: blind spots, ring gaps,
  sparser far away) instead of 99 perfect vertical rays. The settings are exported to
  deploy.yaml so scripts/lidar_height_map.py uses the same ones on the robot.
  Shape: [num_envs, (size_x/res + 1) * (size_y/res + 1)].
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  data = sensor.data
  points = data.hit_pos_w
  if point_noise_std > 0:
    points = points + torch.randn_like(points) * point_noise_std
  robot: Entity = env.scene["robot"]
  values, coverage = grid_height_map_from_points(
    points,
    data.distances >= 0,
    robot.data.root_link_pos_w,
    robot.data.heading_w,
    size_x, size_y, resolution, base_height, min_z, max_z,
    body_x, body_y, body_min_z, cell_radius, min_points, quantile, min_coverage,
  )
  env.extras.setdefault("log", {})["Metrics/lidar_map_coverage"] = coverage.mean()
  return values
