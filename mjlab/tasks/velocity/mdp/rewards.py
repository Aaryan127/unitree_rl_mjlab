from __future__ import annotations

# Reward functions for the velocity tasks.
#
# Every function (or class with __call__) here is a reward TERM: the env config pairs it
# with a weight (RewardTermCfg(func=..., weight=...)), and reward_manager.compute() calls
# it once per policy step as func(env, **params). It returns one raw value per robot,
# a tensor of shape [B] (B = number of parallel robots). The manager multiplies it by
# weight x dt; positive weight = reward, negative = penalty.
#
# Top part: the stock mjlab terms (used by the stock Go2/G1 tasks). From "PGTT rewards"
# down: the terms written for this branch, which the PGTT tasks and the ladder use.
# Shapes in comments: [B] per robot, [B, L] per robot and leg (L = 4), [B, N] per robot
# and height-scan ray (N = 99).

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
# Used by the repro/rung tasks (env_cfgs.unitree_go2_pgtt_repro_env_cfg) and the old
# teacher task (velocity_env_cfg.make_pgtt_env_cfg). Typical sizes per second, after the
# weight, are in doc/reward_hacking_runs_1-3.md, "How the reward is computed".


def track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-|v_cmd_xy - v_xy|^2 / sigma), base frame."""
  # The main task reward: 1 when the trunk moves exactly as commanded, falling smoothly
  # towards 0 with the error. With sigma 0.25, a 0.5 m/s error still pays exp(-1) = 0.37,
  # which is why standing still under a slow command keeps a large part of it.
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)  # [B, 3]: vx, vy, yaw rate
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.sum(  # squared xy speed error, [B]
    # root_link_lin_vel_b: trunk velocity in the trunk's own frame (x forward, y left).
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-error / sigma)  # [B], in (0, 1]


def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-(w_cmd_z - w_z)^2 / sigma), base frame."""
  # The same for turning: 1 when the yaw rate (rad/s) matches the command.
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])  # [B]
  return torch.exp(-error / sigma)


def lin_vel_z_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """v_z^2 (base frame): penalize bouncing."""
  # Up/down speed of the trunk, squared. Weight -2: a trunk bobbing at 0.1 m/s costs 0.02/s;
  # tumbling (rung 5: -0.54/s) costs a lot.
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])  # [B]


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """|w_xy|^2 (base frame): penalize roll and pitch rates."""
  # How fast the trunk rocks sideways (roll) and forwards (pitch), squared. Not the tilt
  # itself: a robot that leans still costs nothing here (that would be `orientation`).
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)  # [B]


def joint_power(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """sum |tau| * |q_dot| (mechanical power). Actuators are in joint order on the Go2."""
  # Old teacher only (weight -2e-5). Motor torque x joint speed = watts spent.
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.abs(asset.data.actuator_force) * torch.abs(asset.data.joint_vel), dim=1
  )


class weighted_default_pose:
  """sum_i w_i * (q_i - q_default_i)^2, with per-joint weights given by name regex."""
  # A class (not a function) so the regex -> per-joint weight lookup happens once, at
  # build time, instead of every step. The Go2 configs give hips weight 1, thighs and
  # calves 0: only sideways leg splay is penalized (PGTT).

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    # Called once when the reward manager builds its terms.
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)  # all 12 joints
    # {r".*_hip_joint": 1.0, ...} -> one weight per joint, in joint order.
    _, _, weights = resolve_matching_names_values(
      data=cfg.params["weights"], list_of_strings=joint_names
    )
    self.weights = torch.tensor(weights, device=env.device, dtype=torch.float32)  # [12]

  def __call__(
    self, env: ManagerBasedRlEnv, weights, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    del weights  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    error = (  # each joint's angle minus its standing angle (rad), [B, 12]
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(self.weights * torch.square(error), dim=1)  # [B]


def swing_foot_contact(
  env: ManagerBasedRlEnv, sensor_name: str, command_name: str
) -> torch.Tensor:
  """Number of feet touching the ground while their gait phase says swing.

  The contact sensor's feet must be listed in the same order as the gait command legs.
  """
  # PGTT's "contact" term (weight -0.25): enforces the gait's TIMING. foot_phase only
  # scores heights loosely; this charges every foot that stays down when it should lift.
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  gait = env.command_manager.get_term(command_name)  # the phase clock (gait_command.py)
  in_contact = sensor.data.found > 0  # [B, 4] True = touching the ground
  # swing_mask: [B, 4] True where the leg's phase is in [pi, 2 pi) (its swing half).
  return torch.sum((in_contact & gait.swing_mask).float(), dim=1)  # [B], 0-4


def _smoothstep(s: torch.Tensor) -> torch.Tensor:
  """Cubic Hermite with zero end slopes: 0 -> 1 on s in [0, 1]."""
  # 3s^2 - 2s^3: starts and ends flat, so the foot target lifts off and lands gently.
  s = torch.clamp(s, 0.0, 1.0)  # outside [0, 1] it holds at 0 or 1
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
  # All tensors [B, L]. Heights are in whatever frame the caller uses: above the ground
  # (foot_phase_tracking) or relative to the IMU (foot_phase_tracking_body, negative).
  apex = swing_height + terrain_delta  # highest point of this swing
  # Lift: s goes 0 -> 1 as the phase goes pi -> 1.5 pi.
  up = stance_height + (apex - stance_height) * _smoothstep(
    (phase - math.pi) / (0.5 * math.pi)
  )
  # Lower: s goes 0 -> 1 as the phase goes 1.5 pi -> 2 pi.
  down = apex + (stance_height - apex) * _smoothstep(
    (phase - 1.5 * math.pi) / (0.5 * math.pi)
  )
  target = torch.where(phase < 1.5 * math.pi, up, down)  # first or second half of the swing
  # First half of the cycle (phase < pi): stance, the foot stays at stance_height.
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
  # The 99 rays hit the ground on a 10 cm grid around the trunk. For each foot, this
  # looks at the rays' hit points near it.
  foot_pos = env.scene[asset_name].data.site_pos_w[:, site_ids]  # [B, L, 3]
  scan = env.scene[sensor_name].data
  hit_z = scan.hit_pos_w[..., 2]  # [B, N]  height of each ray's hit point
  valid = (scan.distances >= 0).unsqueeze(1)  # [B, 1, N]  -1 = the ray hit nothing
  # Horizontal distance from every foot to every hit point.
  dist = torch.norm(
    scan.hit_pos_w[:, None, :, :2] - foot_pos[:, :, None, :2], dim=-1
  )  # [B, L, N]

  # The ground directly under the foot = the nearest valid hit point.
  nearest = torch.argmin(torch.where(valid, dist, torch.inf), dim=-1)  # [B, L]
  ground_z = torch.gather(hit_z, 1, nearest)  # [B, L]

  # Hit points within `radius` of the foot: their highest and lowest heights. On flat
  # ground z_max = z_min; next to a 10 cm step, z_max - z_min = 0.10.
  near = valid & (dist < radius)
  any_near = near.any(dim=-1)  # [B, L] at least one ray within the radius
  hit_z_l = hit_z.unsqueeze(1).expand_as(dist)  # the heights, repeated per leg
  z_max = torch.where(any_near, torch.where(near, hit_z_l, -torch.inf).amax(dim=-1), ground_z)
  z_min = torch.where(any_near, torch.where(near, hit_z_l, torch.inf).amin(dim=-1), ground_z)
  return foot_pos[..., 2], ground_z, z_max, z_min


class _GaitFeetTerm:
  """Base for class terms that need the foot sites in gait-leg order."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    gait_cfg = env.command_manager.get_term_cfg(cfg.params["command_name"])
    # Foot sites named like the gait's legs ("FL", "FR", "RL", "RR"), in that order, so
    # foot i here is leg i of the phase clock.
    self.site_ids, _ = asset.find_sites(gait_cfg.leg_names, preserve_order=True)


class foot_phase_tracking(_GaitFeetTerm):
  """PGTT foot-phase reward: exp(-sum_i (p_des_i - p_i)^2 / sigma), in [0, 1].

  As in the official PGTT code: one exponential of the error summed over the feet, so
  the reward is high only when all feet follow their phase, and sigma is in m^2 (0.05).
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
    # Old teacher task's version (heights above the ground). The ladder uses
    # foot_phase_tracking_body below.
    gait = env.command_manager.get_term(command_name)
    foot_z, ground_z, z_max, z_min = _foot_terrain(
      env, self.site_ids, sensor_name, terrain_radius, asset_cfg.name
    )
    delta = torch.clamp(z_max - z_min, 0.0, max_terrain_delta)  # dH, capped (e.g. 0.15 m)

    foot_height = foot_z - ground_z  # [B, L] foot above the ground under it
    target = desired_foot_height(gait.phase, delta, stance_height, swing_height)
    error = torch.square(target - foot_height)  # [B, L] in m^2

    log = env.extras["log"]  # extra TensorBoard metrics (not part of the reward)
    log["Metrics/foot_height_error"] = torch.mean(torch.sqrt(error))
    log["Metrics/terrain_delta_mean"] = torch.mean(delta)
    # One exponential of the error SUMMED over the 4 feet (max 1). The earlier per-foot
    # sum of exponentials paid planted feet most of a 4x larger reward (run v1's hack).
    return torch.exp(-torch.sum(error, dim=1) / sigma)


class foot_phase_tracking_body(_GaitFeetTerm):
  """PGTT foot-phase reward as the official code computes it.

  Same exp(-sum_i (p_des_i - p_i)^2 / sigma) as `foot_phase_tracking`, but p_i is the
  foot height relative to `ref_site` (the IMU) in the body frame, as PGTT's framepos
  sensors measure it. Stance then means "feet `stance_height` below the body" (-0.30 m
  in PGTT), so planting the feet only scores when the body is at walking height, and
  lying or crouching low loses the reward.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)  # finds the 4 foot sites (self.site_ids)
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    ref_ids, _ = asset.find_sites(cfg.params["ref_site"])  # the "imu" site on the trunk
    self.ref_id = ref_ids[0]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    ref_site: str,
    stance_height: float,
    swing_height: float,
    sigma: float,
    terrain_radius: float,
    max_terrain_delta: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    # The ladder's foot_phase (weight +1, sigma 0.05, stance -0.30, swing -0.20).
    del ref_site  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    gait = env.command_manager.get_term(command_name)
    sites = asset.data.site_pos_w  # world positions of all sites
    rel_w = sites[:, self.site_ids] - sites[:, self.ref_id].unsqueeze(1)  # foot - IMU, [B, L, 3]
    # Rotate that vector into the trunk's frame, so "z" is along the trunk's up axis.
    quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, rel_w.shape[1], -1)
    foot_height = quat_apply_inverse(quat, rel_w)[..., 2]  # [B, L], about -0.30 standing

    # Terrain unevenness near each foot, used only to raise the swing apex. Because the
    # rest is body-relative, a slight trunk tilt shifts the rear feet against the front
    # ones by only ~1.7 cm per 5 degrees: cheap under sigma 0.05 (a backward lean barely
    # costs anything here).
    _, _, z_max, z_min = _foot_terrain(
      env, self.site_ids, sensor_name, terrain_radius, asset_cfg.name
    )
    delta = torch.clamp(z_max - z_min, 0.0, max_terrain_delta)
    target = desired_foot_height(gait.phase, delta, stance_height, swing_height)
    error = torch.square(target - foot_height)  # [B, L]

    log = env.extras["log"]
    log["Metrics/foot_height_error"] = torch.mean(torch.sqrt(error))
    # Max 1 when all four feet are where the clock wants them. Note: it is paid even if
    # the robot doesn't move (stepping in place earns it fully), which is why the
    # tracking terms and swing_contact are needed next to it.
    return torch.exp(-torch.sum(error, dim=1) / sigma)


def torques_pgtt(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """PGTT torque cost: |tau|_2 + |tau|_1."""
  # Motor effort (weight -2e-6, tiny): the length of the 12-torque vector plus the sum
  # of their sizes, in Nm.
  tau = env.scene[asset_cfg.name].data.actuator_force  # [B, 12] Nm
  return torch.norm(tau, dim=1) + torch.sum(torch.abs(tau), dim=1)


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
    too_low = foot_z < z_max + margin  # [B, L] below the highest nearby terrain + margin
    return torch.sum((too_low & gait.swing_mask).float(), dim=1)  # only swinging feet count


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
    # Unevenness around each stance foot (0 on flat ground, the step height on an edge).
    return torch.sum((z_max - z_min) * (~gait.swing_mask).float(), dim=1)


def base_height_above_terrain(
  env: ManagerBasedRlEnv, sensor_name: str, radius: float = 0.2
) -> torch.Tensor:
  """Base height above the ground under it: base z minus the median height-scan hit
  within `radius` (horizontal) of the base, so a step ahead does not count."""
  scan = env.scene[sensor_name].data  # the height scanner, attached to the trunk
  # Horizontal distance of every ray's hit point from the trunk, [B, N].
  offset = torch.norm(scan.hit_pos_w[..., :2] - scan.pos_w[:, None, :2], dim=-1)
  # Heights of the hits within `radius` (20 cm) of the trunk; all others become NaN.
  ground_z = torch.where(
    (scan.distances >= 0) & (offset < radius), scan.hit_pos_w[..., 2], torch.nan
  )
  # Trunk height minus the median of those (the median ignores a few rays on a step).
  return scan.pos_w[:, 2] - torch.nanmedian(ground_z, dim=-1).values  # [B] metres


def base_height_below(
  env: ManagerBasedRlEnv, sensor_name: str, target: float, radius: float = 0.2
) -> torch.Tensor:
  """(target - h)^2 when the base is lower than `target` above the terrain, else 0.

  One-sided, so it stops crouching / lying low without paying the robot to stand
  taller. 0 if no ray hit the ground near the base.
  """
  # Rung 5+ (weight -50, target 0.25 m): 0 at the walking height (~0.28 m);
  # -50 x (0.25 - 0.20)^2 = -0.125/s at 0.20 m, -0.8/s at 0.12 m.
  h = base_height_above_terrain(env, sensor_name, radius)  # [B]
  # clamp(min=0): only being too LOW counts. nan_to_num: no ray near the base -> 0.
  return torch.nan_to_num(torch.square(torch.clamp(target - h, min=0.0)), nan=0.0)


def leg_ground_contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Number of thigh/shank geoms touching the terrain (Miki r_shank_contact).

  Uses a leg-vs-terrain sensor (the cpu branch used a thigh-vs-shank self-contact
  sensor, which never sees the ground).
  """
  # Rung 5+ (weight -0.25): each thigh or calf shape on the ground costs 0.25/s, so
  # kneeling on both hind calves (2-4 shapes) costs ~0.5-1/s. Up to 12 shapes.
  sensor: ContactSensor = env.scene[sensor_name]  # leg_ground_contact (env_cfgs.py)
  assert sensor.data.found is not None
  return torch.sum((sensor.data.found > 0).float(), dim=1)  # [B], 0-12


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
  v = asset.data.root_link_lin_vel_b[:, :2]  # actual xy velocity, [B, 2]
  c = command[:, :2]  # commanded xy velocity, [B, 2]
  c_norm = torch.norm(c, dim=1, keepdim=True)
  direction = c / torch.clamp(c_norm, min=1e-6)  # unit vector of the command
  # v minus its part along the command = the sideways drift.
  v_orth = v - torch.sum(v * direction, dim=1, keepdim=True) * direction
  active = (c_norm.squeeze(1) > command_threshold).float()
  return torch.sum(torch.square(v_orth), dim=1) * active
