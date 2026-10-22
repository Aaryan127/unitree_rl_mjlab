from __future__ import annotations

import math
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


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
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
  z_error = torch.square(command[:, 2] - actual[:, 2])
  return torch.exp(-z_error / std**2)


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


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
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



##
# PGTT rewards (Ntagkas et al. 2025, Table I).
##


def track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-|v_cmd_xy - v_xy|^2 / sigma), base frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-error / sigma)


def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-(w_cmd_z - w_z)^2 / sigma), base frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / sigma)


def lin_vel_z_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """v_z^2 (base frame): penalize bouncing."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """|w_xy|^2 (base frame): penalize roll and pitch rates."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def joint_power(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """sum |tau| * |q_dot| (mechanical power). Actuators are in joint order on the Go2."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.abs(asset.data.actuator_force) * torch.abs(asset.data.joint_vel), dim=1
  )


class weighted_default_pose:
  """sum_i w_i * (q_i - q_default_i)^2, with per-joint weights given by name regex."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, weights = resolve_matching_names_values(
      data=cfg.params["weights"], list_of_strings=joint_names
    )
    self.weights = torch.tensor(weights, device=env.device, dtype=torch.float32)

  def __call__(
    self, env: ManagerBasedRlEnv, weights, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    del weights  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    error = (
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(self.weights * torch.square(error), dim=1)


def swing_foot_contact(
  env: ManagerBasedRlEnv, sensor_name: str, command_name: str
) -> torch.Tensor:
  """Number of feet touching the ground while their gait phase says swing.

  The contact sensor's feet must be listed in the same order as the gait command legs.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  gait = env.command_manager.get_term(command_name)
  in_contact = sensor.data.found > 0
  return torch.sum((in_contact & gait.swing_mask).float(), dim=1)


def _smoothstep(s: torch.Tensor) -> torch.Tensor:
  """Cubic Hermite with zero end slopes: 0 -> 1 on s in [0, 1]."""
  s = torch.clamp(s, 0.0, 1.0)
  return s * s * (3.0 - 2.0 * s)


def desired_foot_height(
  phase: torch.Tensor,
  terrain_delta: torch.Tensor,
  stance_height: float,
  swing_height: float,
) -> torch.Tensor:
  """PGTT swing trajectory (stance ratio 0.5).

    phi in [0, pi)        stance:     d_b
    phi in [pi, 1.5 pi)   swing up:   d_b -> d_s + dH
    phi in [1.5 pi, 2 pi) swing down: d_s + dH -> d_b

  dH is the terrain height variation around the foot, so the swing apex rises on
  uneven ground (stairs) and stays low on flat ground.
  """
  apex = swing_height + terrain_delta
  up = stance_height + (apex - stance_height) * _smoothstep(
    (phase - math.pi) / (0.5 * math.pi)
  )
  down = apex + (stance_height - apex) * _smoothstep(
    (phase - 1.5 * math.pi) / (0.5 * math.pi)
  )
  target = torch.where(phase < 1.5 * math.pi, up, down)
  return torch.where(phase < math.pi, torch.full_like(phase, stance_height), target)


def _foot_terrain(
  env: ManagerBasedRlEnv,
  site_ids: list[int],
  sensor_name: str,
  radius: float,
  asset_name: str = "robot",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Terrain around each foot, measured with the height-scan rays.

  Returns (all [B, L], L = number of feet):
    foot_z    foot site height (world)
    ground_z  height of the nearest valid ray hit = ground directly under the foot
    z_max     highest terrain within `radius` of the foot (ground_z if no ray is near)
    z_min     lowest terrain within `radius` of the foot (ground_z if no ray is near)
  """
  foot_pos = env.scene[asset_name].data.site_pos_w[:, site_ids]  # [B, L, 3]
  scan = env.scene[sensor_name].data
  hit_z = scan.hit_pos_w[..., 2]  # [B, N]
  valid = (scan.distances >= 0).unsqueeze(1)  # [B, 1, N]
  dist = torch.norm(
    scan.hit_pos_w[:, None, :, :2] - foot_pos[:, :, None, :2], dim=-1
  )  # [B, L, N]

  nearest = torch.argmin(torch.where(valid, dist, torch.inf), dim=-1)  # [B, L]
  ground_z = torch.gather(hit_z, 1, nearest)  # [B, L]

  near = valid & (dist < radius)
  any_near = near.any(dim=-1)
  hit_z_l = hit_z.unsqueeze(1).expand_as(dist)
  z_max = torch.where(any_near, torch.where(near, hit_z_l, -torch.inf).amax(dim=-1), ground_z)
  z_min = torch.where(any_near, torch.where(near, hit_z_l, torch.inf).amin(dim=-1), ground_z)
  return foot_pos[..., 2], ground_z, z_max, z_min


class _GaitFeetTerm:
  """Base for class terms that need the foot sites in gait-leg order."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    gait_cfg = env.command_manager.get_term_cfg(cfg.params["command_name"])
    self.site_ids, _ = asset.find_sites(gait_cfg.leg_names, preserve_order=True)


class foot_phase_tracking(_GaitFeetTerm):
  """PGTT foot-phase reward: sum_i exp(-(p_des_i - p_i)^2 / sigma).

  p_i is the foot height above the ground directly under the foot, and p_des_i comes
  from `desired_foot_height` with dH_i = (max - min) terrain height within
  `terrain_radius` of the foot. Both are measured with the height-scan rays, so the
  reward follows the terrain.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    stance_height: float,
    swing_height: float,
    sigma: float,
    terrain_radius: float,
    max_terrain_delta: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    gait = env.command_manager.get_term(command_name)
    foot_z, ground_z, z_max, z_min = _foot_terrain(
      env, self.site_ids, sensor_name, terrain_radius, asset_cfg.name
    )
    delta = torch.clamp(z_max - z_min, 0.0, max_terrain_delta)

    foot_height = foot_z - ground_z
    target = desired_foot_height(gait.phase, delta, stance_height, swing_height)
    error = torch.square(target - foot_height)

    log = env.extras["log"]
    log["Metrics/foot_height_error"] = torch.mean(torch.sqrt(error))
    log["Metrics/terrain_delta_mean"] = torch.mean(delta)
    return torch.sum(torch.exp(-error / sigma), dim=1)


##
# Optional terrain-aware and regularization terms (from the cpu branch and Miki et al.
# 2022, with the cpu-branch bugs fixed). They are registered with weight 0 in
# make_pgtt_env_cfg (a weight of 0 skips the computation entirely); enable one from
# the command line, e.g.  --env.rewards.swing-clearance.weight -0.5
##


class swing_terrain_clearance(_GaitFeetTerm):
  """Miki foot clearance: number of swing feet lower than the terrain around them.

  A swing foot whose height is below (highest terrain within `radius`) + `margin` is
  about to hit a step edge. (The cpu branch had this inverted: it penalized feet that
  were HIGH above the terrain.)
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    radius: float,
    margin: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    gait = env.command_manager.get_term(command_name)
    foot_z, _, z_max, _ = _foot_terrain(env, self.site_ids, sensor_name, radius, asset_cfg.name)
    too_low = foot_z < z_max + margin
    return torch.sum((too_low & gait.swing_mask).float(), dim=1)


class stance_on_edge(_GaitFeetTerm):
  """Metres of terrain unevenness under stance feet: penalize planting a foot on an edge."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    radius: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    gait = env.command_manager.get_term(command_name)
    _, _, z_max, z_min = _foot_terrain(env, self.site_ids, sensor_name, radius, asset_cfg.name)
    return torch.sum((z_max - z_min) * (~gait.swing_mask).float(), dim=1)


def leg_ground_contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Number of thigh/shank geoms touching the terrain (Miki r_shank_contact).

  Uses a leg-vs-terrain sensor (the cpu branch used a thigh-vs-shank self-contact
  sensor, which never sees the ground).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.sum((sensor.data.found > 0).float(), dim=1)


def lin_vel_orthogonal_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """|velocity component perpendicular to the commanded direction|^2 (Miki).

  Zero when the command is (almost) zero, where the direction is undefined.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  v = asset.data.root_link_lin_vel_b[:, :2]
  c = command[:, :2]
  c_norm = torch.norm(c, dim=1, keepdim=True)
  direction = c / torch.clamp(c_norm, min=1e-6)
  v_orth = v - torch.sum(v * direction, dim=1, keepdim=True) * direction
  active = (c_norm.squeeze(1) > command_threshold).float()
  return torch.sum(torch.square(v_orth), dim=1) * active
