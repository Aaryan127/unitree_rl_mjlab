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
from mjlab.tasks.velocity.velocity_env_cfg import (
  make_pgtt_env_cfg,
  make_velocity_env_cfg,
)


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
  # robot where commands travel over DDS.
  go2 = get_go2_robot_cfg()
  delayed = DelayedActuatorCfg(
    base_cfg=go2_constants.GO2_ACTUATOR_GO2HV, delay_target="position", delay_max_lag=4
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
    height_scanner_cfg,
    lidar_cfg,
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

  cfg.rewards["default_pose"].params["weights"] = {
    r".*_hip_joint": 1.0,
    r".*_thigh_joint": 0.25,
    r".*_calf_joint": 0.25,
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
