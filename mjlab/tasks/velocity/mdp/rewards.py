from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

#NOTE: root_link_lin_vel_b is for vel in base frame, (commands are in this frame)
#NOTE: root_link_lin_vel_w is for vel in world frame 
def track_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """
  1. r = exp(-|v|2), if |vdes| = 0
  2. r = 1.0, else if vdes · v > |vdes|
  3. exp(-(vdes · v - |vdes|)2), otherwise
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  actual = asset.data.root_link_lin_vel_b
  v_des = torch.norm(command[:, :2], dim=1)  # (batch,)
  v = torch.norm(actual[:, :2], dim=1)       # (batch,)
  dot = torch.sum(command[:, :2] * actual[:, :2], dim=1)
  proj = dot / (v_des + 1e-6)
  reward = torch.zeros_like(v_des)
  mask_zero = v_des < 1e-6
  reward[mask_zero] = torch.exp(-v[mask_zero] ** 2)
  mask_fast = (~mask_zero) & (proj > v_des)
  reward[mask_fast] = 1.0
  mask_else = (~mask_zero) & (~mask_fast)
  reward[mask_else] = torch.exp(
      -(proj[mask_else] - v_des[mask_else]) ** 2
  )
  return reward


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  w_des = command[:, 2]   # (batch,)
  w = actual[:, 2]        # (batch,)
  reward = torch.zeros_like(w_des)
  mask_zero = torch.abs(w_des) < 1e-6
  reward[mask_zero] = torch.exp(-w[mask_zero] ** 2)
  mask_fast = (~mask_zero) & (w * torch.sign(w_des) > torch.abs(w_des))
  reward[mask_fast] = 1.0
  mask_else = (~mask_zero) & (~mask_fast)
  reward[mask_else] = torch.exp(
      -(w[mask_else] - w_des[mask_else]) ** 2
  )
  return reward

def linear_orthogonal(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
)-> torch.Tensor:
  """
  Penalizes the velocity orthogonal to the target vel
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  actual = asset.data.root_link_lin_vel_b
  # Use only XY plane
  v = actual[:, :2]         # (batch, 2)
  v_des = command[:, :2]    # (batch, 2)
  v_des_norm = torch.norm(v_des, dim=1, keepdim=True)
  v_des_unit = v_des / (v_des_norm + 1e-6)
  proj_scalar = torch.sum(v * v_des_unit, dim=1, keepdim=True)
  v_proj = proj_scalar * v_des_unit
  v_orth = v - v_proj
  x = torch.sum(v_orth ** 2, dim=1)
  return torch.exp(-3.0 * x)

def body_motion_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
)-> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  ang_actual = asset.data.root_com_ang_vel_b
  wx = ang_actual[:,0]
  wy = ang_actual[:,1]
  vz = asset.data.root_com_lin_vel_b[:,2]
  return torch.Tensor(-1.25*(torch.square(vz)) -0.4*(torch.abs(wx))-0.4*(torch.abs(wy)))
  

def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_squared / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions.

  Returns the number of self-collisions detected by the specified contact sensor.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.3,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward

def feet_terrain_clearence(env: ManagerBasedRlEnv
    ) -> torch.Tensor:
    robot = env.scene.entities["robot"]
    hit_pos = env.scene.sensors["height_scanner"].data.hit_pos_w

    foot_ids = [
    robot.site_names.index("FL"),
    robot.site_names.index("FR"),
    robot.site_names.index("RL"),
    robot.site_names.index("RR"),
  ]
    # pdb.set_trace()
    foot_pos = robot.data.site_pos_w[:, foot_ids, :]
    foot_z = foot_pos[:,:,2]
    radius = 0.2
    diff = hit_pos[:, None, :, :2] - foot_pos[:, :, None, :2]
    dist = torch.norm(diff, dim=-1)
    mask = dist < radius
    heights = hit_pos[..., 2]  # (E, 121)
    heights = heights.unsqueeze(1).expand(-1,4,-1)
    masked_heights = torch.where(mask,heights, torch.full_like(heights, -1e6))
    max_height = masked_heights.max(dim=2).values 
    # print(max_height.shape)
    reward_mask = max_height - foot_z < -0.2
    reward = reward_mask * (-1)
    average_reward = reward.sum(dim=1)
    # print(average_reward)
    return average_reward

def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.mean(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


# def feet_slip(
#   env: ManagerBasedRlEnv,
#   sensor_name: str,
#   command_name: str,
#   command_threshold: float = 0.01,
#   asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
# ) -> torch.Tensor:
#   """Penalize foot sliding (xy velocity while in contact)."""
#   asset: Entity = env.scene[asset_cfg.name]
#   contact_sensor: ContactSensor = env.scene[sensor_name]
#   command = env.command_manager.get_command(command_name)
#   assert command is not None
#   linear_norm = torch.norm(command[:, :2], dim=1)
#   angular_norm = torch.abs(command[:, 2])
#   total_command = linear_norm + angular_norm
#   active = (total_command > command_threshold).float()
#   assert contact_sensor.data.found is not None
#   in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
#   foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
#   vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
#   vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
#   cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
#   num_in_contact = torch.sum(in_contact)
#   mean_slip_vel = torch.sum(torch.sqrt(vel_xy_norm) * in_contact) / torch.clamp(
#     num_in_contact, min=1
#   )
#   env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
#   return cost

"""Rewards having Curriculum Factor"""

def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  
  """Penalize foot sliding (xy velocity while in contact)."""

  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]

  vel_sq = torch.sum(foot_vel_xy ** 2, dim=-1)
  slip_sum = torch.sum(vel_sq * in_contact, dim=1)

  return slip_sum

def joint_torque_l2(
    env:ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  
  """Torque penalty"""
  asset: Entity = env.scene[asset_cfg.name]
  torques = asset.data.joint_torques

  torque_sq = torch.square(torques)
  return torch.sum(torque_sq, dim=1)

def target_smoothness_l2(
   env:ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:  
  
  """Penalize first and second finite differences of desired joint targets"""
  q_t = env.action_manager.action  # [B, D]

  q_t1 = env.action_manager.last_action  # [B, D]

  if not hasattr(env, "_prev_prev_action"):
    env._prev_prev_action = torch.zeros_like(q_t1)

  q_t2 = env._prev_prev_action  # [B, D]

  first_diff = q_t - q_t1
  second_diff = q_t - 2 * q_t1 + q_t2

  env._prev_prev_action = q_t1.clone()

  return torch.sum(first_diff**2 + second_diff**2, dim=1)

def joint_constraint_l2(
    env: ManagerBasedRlEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    
    """Soft joint constraint"""

    asset: Entity = env.scene[asset_cfg.name]

    q = asset.data.joint_pos[:, asset_cfg.joint_ids]

    violation = torch.clamp(q - threshold, min=0.0)

    cost = torch.sum(violation ** 2, dim=1)

    return cost

def shank_knee_collision(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize collision of shank or knee with terrain"""

    contact_sensor: ContactSensor = env.scene[sensor_name]
    in_contact = contact_sensor.data.found > 0

    collision = torch.any(in_contact, dim=1).float()

    return collision

def joint_motion_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """ Penalize joint velocity and acceleration"""

    asset: Entity = env.scene[asset_cfg.name]

    q_dot = asset.data.joint_vel  # [B, D]
    q_ddot = asset.data.joint_acc  # [B, D]

    cost = torch.sum(
        0.01*q_dot**2+q_ddot**2,
        dim=1,
    )
    return cost

def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward

