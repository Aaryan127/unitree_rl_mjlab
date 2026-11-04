# What this file is for, and how to change it
# --------------------------------------------
# Observation functions: each returns one tensor per robot ([B, ...]). Which ones the policy,
# critic and student actually get is set in velocity_env_cfg.py (make_pgtt_env_cfg,
# `observations = {`). height_scan is the teacher's clean 11 x 9 map, lidar_height_map the
# student's (not computed in the ladder), the rest are privileged sim-only values.
# - A new observation: write the function here, add an ObservationTermCfg to a group there.
#   Adding to `policy` (or `proprio_history`, `height`, `privileged`, `critic`) changes a
#   network's input size, so old checkpoints no longer load or resume. Only `policy` has to
#   come from the real robot: after that change, re-export (scripts/export_deploy.py) and
#   update the robot side. For the ladder only, add the term in config/go2/env_cfgs.py
#   (cfg.observations["<group>"].terms["<name>"] = ObservationTermCfg(...)).

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.tasks.velocity.mdp.rewards import _foot_terrain, base_height_above_terrain
from mjlab.utils.lab_api.math import quat_apply_inverse

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


DYNAMICS_PARTS = (
  "trunk_mass", "trunk_com", "leg_mass", "torque_limit", "kp", "kd", "armature", "damping",
  "joint_friction", "encoder_bias", "delay",
)
# Rung 8: the teacher's actor gets the parts with a real effect that the student has some
# chance to infer; the critic gets the per-joint rest (small effect, not inferable from
# the student's sensors: a probe found R^2 ~ 0), see doc/experiments.md, rung 8.
DYNAMICS_ACTOR_PARTS = ("trunk_mass", "trunk_com", "leg_mass", "kp", "kd", "delay")
DYNAMICS_CRITIC_PARTS = ("torque_limit", "armature", "damping", "joint_friction", "encoder_bias")


def robot_dynamics(
  env: ManagerBasedRlEnv,
  trunk_body: str = "base_link",
  asset_name: str = "robot",
  parts: tuple[str, ...] = DYNAMICS_PARTS,
) -> torch.Tensor:
  """The randomized dynamics parameters of each robot, as the sim set it. Privileged.

  The teacher sees what makes this robot different (rung 8+), so it can learn the best
  response to each; the student has to infer it from its history. All parts, 90 values:
    trunk_mass added [kg] (1), trunk_com shift [m] (3), leg_mass x nominal (1), and per
    joint (12 each): torque_limit, kp, kd, armature and damping, each x nominal;
    joint_friction [Nm], encoder_bias [rad]; then the motor delay [physics steps] (1).
  `parts` picks which, in DYNAMICS_PARTS order. A field that is not randomized in this
  task reads as its nominal value. Shape: [num_envs, sum of the parts' sizes].
  """
  sim = env.sim
  asset: Entity = env.scene[asset_name]
  n = env.num_envs

  def field(name: str) -> torch.Tensor:  # [n, ...]; unrandomized fields are shared by all
    value = getattr(sim.model, name)
    return value if value.dim() > 1 and value.shape[0] == n else value.expand(n, *value.shape)

  def ratio(name: str, ids, col: int | None = None) -> torch.Tensor:
    now, default = field(name)[:, ids], sim.get_default_field(name)[ids]
    if col is not None:
      now, default = now[..., col], default[..., col]
    return now / torch.where(default.abs() > 1e-9, default, torch.ones_like(default))

  trunk = asset.indexing.body_ids[asset.find_bodies(trunk_body)[0][0]]
  legs = asset.indexing.body_ids[asset.find_bodies(r".*_(hip|thigh|calf)")[0]]
  dofs = asset.indexing.joint_v_adr  # the 12 leg joints' velocity addresses
  ctrl = asset.indexing.ctrl_ids  # the 12 motors
  mass = field("body_mass")

  def lag() -> torch.Tensor:
    value = torch.zeros(n, 1, device=env.device)
    for actuator in asset.actuators:
      for buffer in getattr(actuator, "_delay_buffers", {}).values():
        value = buffer.current_lags.float().unsqueeze(-1)  # one lag per robot, shared by joints
    return value

  compute = {
    "trunk_mass": lambda: (mass[:, trunk] - sim.get_default_field("body_mass")[trunk]).unsqueeze(-1),
    "trunk_com": lambda: field("body_ipos")[:, trunk] - sim.get_default_field("body_ipos")[trunk],
    "leg_mass": lambda: (
      mass[:, legs].sum(-1) / sim.get_default_field("body_mass")[legs].sum()
    ).unsqueeze(-1),
    "torque_limit": lambda: ratio("actuator_forcerange", ctrl, 1),
    "kp": lambda: ratio("actuator_gainprm", ctrl, 0),
    "kd": lambda: ratio("actuator_biasprm", ctrl, 2),
    "armature": lambda: ratio("dof_armature", dofs),
    "damping": lambda: ratio("dof_damping", dofs),
    "joint_friction": lambda: field("dof_frictionloss")[:, dofs],
    "encoder_bias": lambda: asset.data.encoder_bias,
    "delay": lag,
  }
  unknown = set(parts) - set(compute)
  if unknown:
    raise ValueError(f"Unknown dynamics parts {sorted(unknown)}; known: {DYNAMICS_PARTS}")
  return torch.cat([compute[p]().float() for p in DYNAMICS_PARTS if p in parts], dim=-1)


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
# Full simulator state for the teacher's critic (rung 8+, group privileged_full):
# everything the sim knows that the privileged terms do not already give. Never on the
# robot, and not in the actor (the student could not copy actions that depend on it).
##


def _foot_pos_vel_b(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]
  data = asset.data
  quat = data.root_link_quat_w[:, None].expand(-1, len(asset_cfg.site_ids), -1)
  rel = data.site_pos_w[:, asset_cfg.site_ids] - data.root_link_pos_w[:, None]
  vel = data.site_lin_vel_w[:, asset_cfg.site_ids]
  return quat_apply_inverse(quat, rel), quat_apply_inverse(quat, vel)


def foot_pos_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Foot positions relative to the trunk, in the trunk frame [m]. Shape: [B, feet * 3]."""
  return _foot_pos_vel_b(env, asset_cfg)[0].flatten(1)


def foot_vel_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Foot velocities in the trunk frame [m/s], clipped to +-10. Shape: [B, feet * 3]."""
  return _foot_pos_vel_b(env, asset_cfg)[1].flatten(1).clamp(-10.0, 10.0)


def foot_clearance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Each foot's height above the ground directly under it [m] (height-scan rays), clipped
  to [-0.5, 1]. Shape: [B, feet]."""
  foot_z, ground_z, _, _ = _foot_terrain(env, asset_cfg.site_ids, sensor_name, 0.05, asset_cfg.name)
  return (foot_z - ground_z).clamp(-0.5, 1.0)


def base_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Trunk height above the ground under it [m]; 0 if no ray is near. Shape: [B, 1]."""
  height = base_height_above_terrain(env, sensor_name)
  return torch.nan_to_num(height, nan=0.0).clamp(-1.0, 1.0).unsqueeze(-1)


def foot_contact_time(env: ManagerBasedRlEnv, sensor_name: str, max_time: float = 2.0) -> torch.Tensor:
  """Per foot: seconds in the air, then seconds on the ground, each capped at `max_time`.
  Shape: [B, feet * 2]."""
  data = env.scene[sensor_name].data
  assert data.current_air_time is not None and data.current_contact_time is not None
  return torch.cat([data.current_air_time, data.current_contact_time], dim=-1).clamp(max=max_time)


def joint_torques(env: ManagerBasedRlEnv, asset_name: str = "robot") -> torch.Tensor:
  """True motor torques [Nm], with this robot's randomized gains, delay and strength.
  Shape: [B, motors]."""
  asset: Entity = env.scene[asset_name]
  return asset.data.actuator_force


class _MotorJoints:
  """Base for terms that need each motor's joint angle and speed, in motor order."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_name", "robot")]
    model = env.sim.mj_model
    ctrl = asset.indexing.ctrl_ids.cpu().numpy()
    joints = model.actuator_trnid[ctrl, 0]
    self._ctrl = asset.indexing.ctrl_ids
    self._q_adr = torch.as_tensor(model.jnt_qposadr[joints], device=env.device)
    self._v_adr = torch.as_tensor(model.jnt_dofadr[joints], device=env.device)

  def _state(self, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = env.sim.data
    return data.ctrl[:, self._ctrl], data.qpos[:, self._q_adr], data.qvel[:, self._v_adr]


class motor_target_error(_MotorJoints):
  """Position target the motor is actually tracking now (after the delay) minus the joint
  angle [rad]. Privileged: the delayed target exists only inside the sim. Shape: [B, motors]."""

  def __call__(self, env: ManagerBasedRlEnv, asset_name: str = "robot") -> torch.Tensor:
    target, q, _ = self._state(env)
    return target - q


class motor_torque_estimate(_MotorJoints):
  """Sim stand-in for the Go2's per-motor torque estimate (LowState motor_state.tau_est).

  On the robot the PD loop runs on the motor driver with the gains we send, and tau_est
  is the driver's current times the nominal torque constant: it reports the torque the
  driver commands, Kp (target - q) - Kd qd with the nominal Kp/Kd, clipped to the nominal
  limit. It does not show a motor that is really weaker or stronger. So this computes the
  same from the delayed target, the joint state and the model's nominal gains, never from
  the sim's randomized ones (the true torque, joint_torques, would let the student read
  the randomized gains off this input, which the robot's tau_est cannot do).
  Shape: [B, motors] in Nm, motor order.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    sim = env.sim
    self._gain = sim.get_default_field("actuator_gainprm")[self._ctrl, 0]
    self._bias_q = sim.get_default_field("actuator_biasprm")[self._ctrl, 1]
    self._bias_v = sim.get_default_field("actuator_biasprm")[self._ctrl, 2]
    self._limit = sim.get_default_field("actuator_forcerange")[self._ctrl, 1]

  def __call__(self, env: ManagerBasedRlEnv, asset_name: str = "robot") -> torch.Tensor:
    target, q, qd = self._state(env)
    torque = self._gain * target + self._bias_q * q + self._bias_v * qd
    return torch.maximum(torch.minimum(torque, self._limit), -self._limit)


def foot_force_sensor(env: ManagerBasedRlEnv, sensor_name: str, asset_name: str = "robot") -> torch.Tensor:
  """Sim stand-in for the Go2 EDU's foot force sensors (LowState foot_force).

  Per foot, the contact force magnitude divided by robot weight / 4: the four sum to ~4
  when standing, 0 in the air. They are not equal: in the default stand the Go2 carries
  more on its hind legs (~0.66 per front foot, ~1.34 per hind foot). The real sensor
  reports raw counts; export_deploy.py calibrates each foot against this sim's standing
  value (deploy.yaml foot_force: offset, scale). Faults: FootForceNoiseModelCfg.
  Shape: [B, feet].
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  asset: Entity = env.scene[asset_name]
  mass = env.sim.get_default_field("body_mass")[asset.indexing.body_ids].sum()
  per_foot = mass * 9.81 / sensor.data.force.shape[1]
  return sensor.data.force.norm(dim=-1) / per_foot


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
