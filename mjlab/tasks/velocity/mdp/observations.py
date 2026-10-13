from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import pdb

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor, BuiltinSensor
from scipy.spatial.transform import Rotation as R
import numpy as np

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

def height_map(env: ManagerBasedRlEnv, sensor_name: str)-> torch.Tensor:
  # REFACTOR: 1.2 - convert height_map to per-foot, multi-radius, foot-relative samples
  sensor: RayCastSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  robot = env.scene.entities["robot"]
  hit_pos = sensor_data.hit_pos_w
  foot_ids = [
    robot.site_names.index("FL"),
    robot.site_names.index("FR"),
    robot.site_names.index("RL"),
    robot.site_names.index("RR"),
  ]
  # pdb.set_trace()
  foot_pos = robot.data.site_pos_w[:, foot_ids, :]
  # Sample radii around each foot (per-paper): 5 concentric radii
  radii = [0.05, 0.10, 0.15, 0.20, 0.25]

  # hit_pos: (E, S, 3) where S is number of samples (e.g. 121)
  # foot_pos: (E, 4, 3)
  # For each foot and each radius, compute samples that fall within that radius
  # and return the height values relative to the foot z-position.

  # precompute 2D positions
  hit_xy = hit_pos[..., :2]  # (E, S, 2)
  foot_xy = foot_pos[..., :2]  # (E, 4, 2)
  foot_z = foot_pos[..., 2]  # (E, 4)

  # diff: (E, 4, S, 2)
  diff = hit_xy[:, None, :, :] - foot_xy[:, :, None, :]
  dist = torch.norm(diff, dim=-1)  # (E, 4, S)

  heights = hit_pos[..., 2]  # (E, S)

  per_foot_rad_heights = []
  for r in radii:
    mask = dist <= r  # (E, 4, S)
    masked_heights = torch.where(mask, heights[:, None, :], torch.full_like(heights[:, None, :], float("nan")))
    # mean over sample dim (S); nanmean yields NaN if no True samples
    mean_heights = torch.nanmean(masked_heights, dim=-1)  # (E,4)
    # Convert to height relative to foot z
    rel = mean_heights - foot_z
    per_foot_rad_heights.append(rel)

  # Stack radii features: list of (E,4) -> (E,4,R)
  features = torch.stack(per_foot_rad_heights, dim=-1)
  # Flatten per-foot per-radius to shape (E, 4*R)
  E, F, R = features.shape
  out = features.reshape(E, F * R)
  # Replace NaNs (from no samples) with 0.0
  out = torch.nan_to_num(out, nan=0.0)
  # Log-compress to keep numeric range small (match other obs formatting)
  return torch.sign(out) * torch.log1p(torch.abs(out))

def external_forces(env: ManagerBasedRlEnv) -> torch.Tensor:
  # import pdb; pdb.set_trace()
  robot = env.scene.entities["robot"]
  assert robot.data.body_external_force is not None
  assert robot.data.body_external_torque is not None
  f = robot.data.body_external_force.flatten(start_dim=-1)
  t = robot.data.body_external_torque.flatten(start_dim=-1)
  force = torch.sign(f) * torch.log1p(torch.abs(f))
  torque = torch.sign(t)* torch.log1p(torch.abs(t))
  return torch.cat((force,torque),dim=-1)

def orientation(env: ManagerBasedRlEnv):
  robot = env.scene.entities["robot"]
  assert robot.data.root_link_quat_w is not None
  quat = robot.data.root_link_quat_w
  return quat
  # return torch.sign(quat)* torch.log1p(torch.abs(quat)) #*Check required - is this needed?

def shank_thigh_contact(env: ManagerBasedRlEnv, sensor_name: str):
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float() #check again

def contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  return sensor_data.force

def contact_normals(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  #We will rely on the fact that the force attribute is always populated
  B,N,D = sensor_data.force.shape
  if sensor_data.normal is not None:
    return sensor_data.normal
  else:
    # If normals are not provided, we can't compute them from forces without knowing the friction model.
    # As a fallback, we return a default normal vector pointing upwards in the contact frame.
    return torch.tensor([0.0, 0.0, 1.0], device=sensor_data.force.device).expand(B, N, 3)


def friction(env: ManagerBasedRlEnv):
  """Estimate per-contact friction coefficient from a contact sensor.

  This computes mu ~= ||tangential_force|| / |normal_force| per contact slot.
  It prefers using the contact-normal if provided by the sensor (global frame);
  otherwise it assumes the contact frame uses index 2 as the normal axis.

  Args:
    env: The environment.
    sensor_name: (optional) name of ContactSensor to use; default 'feet_ground_contact'
  Returns:
    Tensor of shape [B, N] with estimated friction coefficients (clamped).
  """
  # Default sensor commonly used in velocity tasks.
  sensor_name = "feet_ground_contact"
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data

  if data.force is None:
    raise RuntimeError(f"Contact sensor '{sensor_name}' has no force field available")

  f = data.force  # [B, N, 3]

  # If sensor provides contact normals, use them to project forces.
  if data.normal is not None:
    n = data.normal
    n_norm = torch.norm(n, dim=-1, keepdim=True).clamp_min(1e-8)
    n_unit = n / n_norm
    # normal force = projection of force onto normal (signed)
    normal_force = (f * n_unit).sum(dim=-1)
    tangential = f - normal_force.unsqueeze(-1) * n_unit
    tangential_mag = torch.norm(tangential, dim=-1)
  else:
    # Assume contact-frame ordering where index 2 is the normal component.
    normal_force = f[..., 2]
    tangential_mag = torch.norm(f[..., :2], dim=-1)

  mu = tangential_mag / (torch.abs(normal_force) + 1e-6)
  # Clamp to avoid infinities from tiny normals and keep values reasonable.
  return torch.clamp(mu, max=50.0)


def terrain_geom_friction(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
  """Read geom friction entries for an entity (e.g., the terrain).

  Returns a flattened tensor per environment containing the geom friction axes
  for all geoms belonging to the given entity.
  """
  asset = env.scene[asset_cfg.name]
  # geom_ids is a tensor of global geom indices for this entity
  geom_ids = asset.indexing.geom_ids
  # env.sim.model.geom_friction has shape [nworld, n_geoms, axes]
  model_field = env.sim.model.geom_friction
  # Select only this entity's geoms and flatten last two dims
  vals = model_field[:, geom_ids, :]
  return vals.flatten(start_dim=1)

def body_velocity(env: ManagerBasedRlEnv):
  robot = env.scene.entities["robot"]
  assert robot.data.root_link_vel_w is not None
  return robot.data.root_link_vel_w


def quat_to_rpy_scipy(quat: torch.Tensor):
    """
    quat: (N, 4) tensor in (w, x, y, z)
    returns: (N, 3) tensor (roll, pitch, yaw) in radians
    """

    q_np = quat.detach().cpu().numpy()

    q_scipy = np.column_stack((q_np[:, 1], q_np[:, 2], q_np[:, 3], q_np[:, 0]))

    r = R.from_quat(q_scipy)

    rpy = r.as_euler('xyz', degrees=False)

    return torch.from_numpy(rpy).to(quat.device)

def body_orientation(env: ManagerBasedRlEnv):
  robot = env.scene.entities["robot"]
  assert robot.data.root_link_pos_w is not None
  return robot.data.root_link_pos_w