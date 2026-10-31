"""Unitree Go2 velocity environment configurations."""

from dataclasses import replace

import mujoco

from mjlab.asset_zoo.robots import (
  get_go2_robot_cfg,
)
from mjlab.actuator.delayed_actuator import DelayedActuatorCfg
from mjlab.asset_zoo.robots.unitree_go2 import go2_constants
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  LidarPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.terrain_profiles import set_terrain_profile
from mjlab.tasks.velocity.velocity_env_cfg import (
  make_pgtt_env_cfg,
  make_velocity_env_cfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def unitree_go2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  del cfg.curriculum["terrain_levels"]

  return cfg


##
# PGTT teacher-student (stairs).
##

_LEGS = ("FL", "FR", "RL", "RR")  # Same order as the robot joints and the gait command.


def _go2_spec_with_named_calves() -> mujoco.MjSpec:
  """Go2 model where the two unnamed calf collision cylinders of each leg get names.

  Contact sensors select geoms by name, so this is needed to detect the shank touching
  the ground (privileged leg_ground_contact). Only used by the PGTT task.
  """
  spec = go2_constants.get_spec()
  for leg in _LEGS:
    cylinders = [
      g
      for g in spec.body(f"{leg}_calf").geoms
      if not g.name and g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
    ]
    for geom, part in zip(cylinders, ("upper", "lower"), strict=True):
      geom.name = f"{leg}_calf_{part}_collision"
  return spec


def unitree_go2_pgtt_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Unitree Go2 PGTT stairs task, shared by the teacher (PPO) and student (distillation)."""
  cfg = make_pgtt_env_cfg()

  # Go2 with named calf geoms and DELAYED actuators: the position target can reach the
  # motor 0-20 ms late (sampled per episode by the actuator_delay event), as on the real
  # robot where commands travel over DDS. Gains from the official PGTT training config
  # (Kp 40, Kd 0.5; the stock Go2 task uses 20 / 1); the export writes them to deploy.yaml.
  go2 = get_go2_robot_cfg()
  delayed = DelayedActuatorCfg(
    base_cfg=replace(go2_constants.GO2_ACTUATOR_GO2HV, stiffness=40.0, damping=0.5),
    delay_target="position",
    delay_max_lag=4,
  )
  cfg.scene.entities = {
    "robot": replace(
      go2,
      spec_fn=_go2_spec_with_named_calves,
      articulation=replace(go2.articulation, actuators=(delayed,)),
    )
  }

  foot_geoms = tuple(f"{leg}_foot_collision" for leg in _LEGS)
  leg_geoms = tuple(
    f"{leg}_{part}_collision" for leg in _LEGS for part in ("thigh", "calf_upper", "calf_lower")
  )

  ##
  # Sensors
  ##

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force", "normal"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Separate sensor for normals: with reduce="netforce" the normal field is not
  # meaningful, "maxforce" reports the normal of the strongest contact.
  feet_normals_cfg = ContactSensorCfg(
    name="feet_ground_normals",
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "normal"),
    reduce="maxforce",
    num_slots=1,
  )
  leg_ground_cfg = ContactSensorCfg(
    name="leg_ground_contact",
    primary=ContactMatch(mode="geom", pattern=leg_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  # Trunk or hips touching the terrain ends the episode. Without it the robot learned to
  # lie on its belly and paddle its legs once the effort penalties ramped up (only
  # flipping over >90 deg ended an episode). Thighs are left out: in the default crouch
  # (base ~0.21 m) a thigh brushes the ground on landing; leg_contact penalises them.
  body_ground_cfg = ContactSensorCfg(
    name="body_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=("base_collision",) + tuple(f"{leg}_hip_collision" for leg in _LEGS),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  # 11 x 9 = 99 rays over 1.0 m x 0.8 m around the base, yaw-aligned (PGTT uses an
  # 11 x 9 robot-centric map). Origins start 0.5 m above the base so rays still start
  # above the terrain when the next steps are higher than the base. Only geom group 0
  # (terrain) is hit; the robot's own geoms are groups 2 and 3.
  height_scanner_cfg = RayCastSensorCfg(
    name="height_scanner",
    frame=ObjRef(type="body", name="base_link", entity="robot"),
    pattern=GridPatternCfg(size=(1.0, 0.8), resolution=0.1, z_offset=0.5),
    ray_alignment="yaw",
    max_distance=3.0,
    include_geom_groups=(0,),
    debug_vis=play,
  )
  # Simulated head LiDAR for the student's height map (see mdp.lidar_height_map).
  # Beams start at the mount point and tilt with the body (ray_alignment="base"), so the
  # map has the real sensor's blind spots and ring gaps. Mount position / tilt are an
  # approximation of the Go2 EDU head LiDAR: check them against your robot's TF
  # (base_link -> lidar frame) and update mount_pos / pitch_deg to match.
  lidar_cfg = RayCastSensorCfg(
    name="lidar",
    frame=ObjRef(type="body", name="base_link", entity="robot"),
    pattern=LidarPatternCfg(
      mount_pos=(0.28, 0.0, 0.03),
      azimuth_res_deg=4.0,
      elevation_range_deg=(-80.0, -10.0),
      num_rings=16,
      pitch_deg=0.0,
    ),
    ray_alignment="base",
    max_distance=5.0,
    include_geom_groups=(0,),
  )
  cfg.scene.sensors = (
    feet_ground_cfg,
    feet_normals_cfg,
    leg_ground_cfg,
    body_ground_cfg,
    height_scanner_cfg,
    lidar_cfg,
  )
  cfg.terminations["body_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": body_ground_cfg.name},
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  ##
  # Robot-specific names
  ##

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.commands["gait"].leg_names = _LEGS  # type: ignore[attr-defined]

  for event in ("base_mass", "base_com", "external_wrench"):
    cfg.events[event].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["link_mass"].params["asset_cfg"].body_names = (r".*_(hip|thigh|calf)",)

  privileged = cfg.observations["privileged"].terms
  privileged["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
  privileged["base_external_wrench"].params["asset_cfg"].body_names = ("base_link",)

  # Hips only, as in PGTT: keeps the legs from splaying sideways without fighting the
  # thigh/calf motion that foot_phase asks for (large on stairs).
  cfg.rewards["default_pose"].params["weights"] = {
    r".*_hip_joint": 1.0,
    r".*_thigh_joint": 0.0,
    r".*_calf_joint": 0.0,
  }
  # Foot site is the center of the 2.2 cm foot sphere, so it sits 2.2 cm above the
  # ground when standing.
  cfg.rewards["foot_phase"].params["stance_height"] = 0.022
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = _LEGS
  # Height grid layout (rows = y points, cols = x points) for the student's map-shift noise.
  cfg.observations["height_noisy"].terms["height_scan"].noise.grid_shape = (9, 11)  # type: ignore[union-attr]

  ##
  # Play mode
  ##

  if play:
    from mjlab.utils.checkpoint import nominal_eval_cfg
    cfg = nominal_eval_cfg(cfg)
    cfg.episode_length_s = int(1e9)
    for group in ("policy", "proprio_history", "height_noisy"):
      cfg.observations[group].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("external_wrench", None)
    for name in list(cfg.curriculum):  # no terrain or reward curriculum in play mode
      cfg.curriculum.pop(name)
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


##
# PGTT reproduction: the official PGTT training setup, as close as mjlab allows.
##


def _go2_spec_pgtt_joints() -> mujoco.MjSpec:
  """Go2 with PGTT's passive joint damping (= Kd, 0.5; our XML has 0.1)."""
  spec = _go2_spec_with_named_calves()
  for joint in spec.joints:
    if joint.type != mujoco.mjtJoint.mjJNT_FREE:
      joint.damping = 0.5
  return spec


def unitree_go2_pgtt_repro_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The official PGTT Go2 training config (github.com/NtagkasAlex/
  phase_guided_terrain_traversal, method "pgtt", task flat_terrain), ported to mjlab.

  A known-good baseline: v1-v4 of our teacher changed many things at once and never
  learned to walk, so this reproduces what is published to walk, and every extra
  (full collisions, delays, pushes, our reward terms) is then added one at a time.
  Same observation groups as the PGTT teacher, so the same network trains on it.

  Differences from the official code that remain: rsl_rl PPO instead of Brax PPO
  (hyperparameters matched in rl_cfg), 4096 robots instead of 8192, one global learned
  std instead of a state-dependent one, random initial gait phase, the teacher's extra
  observations (history, privileged state).
  """
  cfg = unitree_go2_pgtt_env_cfg(play=play)
  set_terrain_profile(cfg, "flat")

  # Robot: feet-only collisions (scene_mjx_feetonly.xml), Kp 40 with actuator Kd 0.5 plus
  # passive joint damping 0.5, no actuator delay.
  go2 = cfg.scene.entities["robot"]
  cfg.scene.entities = {
    "robot": replace(
      go2,
      collisions=(go2_constants.FEET_ONLY_COLLISION,),
      spec_fn=_go2_spec_pgtt_joints,
      articulation=replace(
        go2.articulation,
        actuators=(
          replace(go2_constants.GO2_ACTUATOR_GO2HV, stiffness=40.0, damping=0.5),
        ),
      ),
    )
  }
  # Only flipping over ends an episode.
  cfg.terminations.pop("body_contact", None)

  # Observations: PGTT noise (joint_pos +-0.03), no sensor delays.
  for group in cfg.observations.values():
    for term in group.terms.values():
      term.delay_min_lag = term.delay_max_lag = 0
  for group in ("policy", "proprio_history"):
    terms = cfg.observations[group].terms
    key = "joint_pos" if "joint_pos" in terms else "joint_pos_history"
    terms[key].noise = Unoise(n_min=-0.03, n_max=0.03)

  # Commands: uniform in [-1, 1], each axis non-zero with probability (0.9, 0.25, 0.5),
  # resampled every ~5 s.
  twist = cfg.commands["twist"]
  twist.axis_nonzero_prob = (0.9, 0.25, 0.5)  # type: ignore[attr-defined]
  twist.rel_standing_envs = 0.0  # type: ignore[attr-defined]
  twist.resampling_time_range = (2.0, 8.0)

  # Events: PGTT's domain_randomize_simple; no pushes, no motor delay.
  for name in ("push_robot", "external_wrench", "effort_limits", "actuator_delay",
               "joint_friction", "encoder_bias"):
    cfg.events.pop(name, None)
  cfg.events["reset_base"].params = {
    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    "velocity_range": {k: (-0.1, 0.1) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
  }
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  if "body_friction" in cfg.events:
    cfg.events["body_friction"].params["ranges"] = (0.4, 1.0)
  if "base_mass" in cfg.events:
    cfg.events["base_mass"].params["ranges"] = (-1.0, 1.0)
  if "joint_armature" in cfg.events:
    cfg.events["joint_armature"].params["ranges"] = (1.0, 1.05)
  if "pd_gains" in cfg.events:
    cfg.events["pd_gains"].params["kp_range"] = (0.9, 1.1)
    cfg.events["pd_gains"].params["kd_range"] = (1.0, 1.0)
  if not play:
    cfg.events["joint_damping"] = EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "operation": "scale",
        "field": "dof_damping",
        "ranges": (0.9, 1.1),
      },
    )

  # Rewards: exactly PGTT's (robots/go2 default_config + joystick_base._get_reward).
  # mjlab multiplies each by dt and clips the step's sum at 0, as PGTT does.
  hips_only = cfg.rewards["default_pose"]
  hips_only.weight = -0.5
  monitor = cfg.rewards["hacking_monitor"]
  cfg.rewards = {
    "track_lin_vel": RewardTermCfg(
      func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "twist", "sigma": 0.25}
    ),
    "track_ang_vel": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "twist", "sigma": 0.25}
    ),
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
    "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
    "default_pose": hips_only,
    # PGTT: -1 inside the clip (-0.02 per episode end); ours is added after the clip.
    # Either way the cost of falling is the positive reward the robot no longer gets.
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-1.0),
    "torques": RewardTermCfg(func=mdp.torques_pgtt, weight=-2e-6),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.001),
    "foot_phase": RewardTermCfg(
      func=mdp.foot_phase_tracking_body,
      weight=1.0,
      params={
        "command_name": "gait",
        "sensor_name": "height_scanner",
        "ref_site": "imu",
        # Feet 0.30 m below the IMU in stance, 0.20 m (+ terrain step) at the apex.
        "stance_height": -0.30,
        "swing_height": -0.20,
        "sigma": 0.05,
        "terrain_radius": 0.15,
        "max_terrain_delta": 0.15,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "swing_contact": RewardTermCfg(
      func=mdp.swing_foot_contact,
      weight=-0.25,
      params={"sensor_name": "feet_ground_contact", "command_name": "gait"},
    ),
    "hacking_monitor": monitor,
  }
  # No reward curriculum in PGTT.
  for name in [n for n in cfg.curriculum if n.startswith("ramp_")]:
    cfg.curriculum.pop(name)
  return cfg


def unitree_go2_pgtt_repro_smooth_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rung 0b: the reproduction plus one term, a penalty on the action's second difference.

  repro0 walks well but its actions carry a fast tremor on top of the gait: 85% of the
  action-rate cost and 98% of sum (a_t - 2a_(t-1) + a_(t-2))^2 is above 6 Hz, while the
  gait itself is 1-3 Hz. PGTT's action_rate (-0.001) costs 0.0015/s, too little to matter.

  The second difference of a sine of frequency f scales as (2 sin(pi f dt))^2, so it costs
  a 12 Hz tremor ~900x more than a 2 Hz gait of the same size (action_rate: ~30x), and it
  removes the tremor without fighting the steps. Weight -0.02: the largest that keeps the
  zero clip slack early in training (a random policy pays ~0.35/s of the ~1/s it earns;
  -0.05 would pay ~0.9/s). On repro0 it costs the tremor 0.061/s and the gait 0.001/s.
  """
  cfg = unitree_go2_pgtt_repro_env_cfg(play=play)
  cfg.rewards["action_acc"] = RewardTermCfg(func=mdp.action_acc_l2, weight=-0.02)
  return cfg
