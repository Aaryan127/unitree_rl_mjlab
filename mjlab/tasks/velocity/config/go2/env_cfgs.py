"""Unitree Go2 velocity environment configurations."""
"""
NOTE
 to add any sensors:
 1. add the sensor by implementing the class unitree_rl_mjlab/mjlab/sensor/sensor.py
 2. import the sensor cfg in this file
 
 For lidar use raycastsensorcfg with ray pattern set to pin hole for lidar. (check raycastsensorcfg for more info)
"""

from mjlab.asset_zoo.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
<<<<<<< HEAD
from mjlab.sensor import ContactMatch, ContactSensorCfg , ObjRef, RayCastSensorCfg, PinholeCameraPatternCfg, GridPatternCfg
=======
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg, PinholeCameraPatternCfg
>>>>>>> ba122f926f2411a6cfe704e1394b5bb1689f813b
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise.noise_cfg import UniformNoiseCfg



def unitree_go2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)
  body_calf = tuple(f"{name}_calf" for name in foot_names)
  body_thigh = tuple(f"{name}_thigh" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force","normal"),
    #       BINARY, FORCE OF CONTACT,   NORMAL FORCE
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  shank_thigh_contact_cfg = ContactSensorCfg(
    name="shank_thigh_contact",
    primary=ContactMatch(
      mode="body", 
      pattern=body_calf,
      entity="robot"),
    secondary=ContactMatch(
      mode="body",
      pattern=".*_thigh",
      entity="robot"),
    fields=("found",),
    num_slots=1,
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
<<<<<<< HEAD
  )

  height_scanner_cfg = RayCastSensorCfg(
      name="height_scanner",
      frame=ObjRef(
          type="body",
          name="base_link",
          entity="robot",
      ),
      pattern= GridPatternCfg(
          size=(1.0, 1.0),     # 1m x1m
          resolution=0.1,      # 10cm spacing
      ),
      ray_alignment="yaw",     
      exclude_parent_body=True,
      debug_vis=True,
        viz=RayCastSensorCfg.VizCfg(
            hit_color=(0, 1, 0, 0.8),      # Green for hits
            miss_color=(1, 0, 0, 0.4),     # Red for misses
            show_rays=True,                 # Draw ray arrows
            show_normals=True,              # Draw surface normals
            normal_color=(1, 1, 0, 1),     # Yellow normals
        ),
  )
  

  cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg,height_scanner_cfg,shank_thigh_contact_cfg)
  

  print(f"\n----------Sensor------------\n{cfg.observations.keys()}\nPolicy{cfg.observations['policy'].terms.keys()}\nCritic: {cfg.observations['critic'].terms.keys()}\nExtero : {cfg.observations['extero'].terms.keys()}")
=======
    )
  thigh_shank_cfg = ContactSensorCfg(
    name="leg_segment_contact",
    primary=ContactMatch(
        mode="geom",
        entity="robot",
        pattern=r".*(thigh|calf).*_collision.*",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
    )

  # 3d lidar
  # lidar_cfg = RayCastSensorCfg(
  #   name="lidar",
  #   frame=ObjRef(
  #       type="body",
  #       name="base",
  #       entity="robot",
  #   ),
  #   pattern=PinholeCameraPatternCfg(
  #       fov=(360, 30),          # 360° horizontal, 30° vertical
  #       resolution=(1.0, 1.0), # 1° per rayh
  #   ),
  #   ray_alignment="base",
  #   exclude_parent_body=True,
  #   debug_vis=False,
  # )
#   height_scanner_cfg = RayCastSensorCfg(
#     name="height_scanner",
#     frame=ObjRef(
#         type="body",
#         name="base_link",
#         entity="robot",
#     ),
#     pattern=PinholeCameraPatternCfg(
#         width= 10,
#         height= 10,# CHECK PAPER FOR RES
#     ),
#     ray_alignment="down",   
#     exclude_parent_body=True,
#     debug_vis=False,
# )


  cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg,thigh_shank_cfg)
  # adding sensors to policy(actor) and critic
  # cfg.observations["policy"].terms["height_scan"] = {
  #     "func": mdp.sensor_data,
  #     "params": {
  #         "sensor_name": "height_scanner",
  #         "flatten": True,
  #     },
  #     "noise": UniformNoiseCfg(
  #       n_min=-0.01,
  #       n_max=0.01,
  #       operation="add",
  #   ),
  # }

  # cfg.observations["critic"].terms["height_scan"] = {
  #   "func": mdp.sensor_data,
  #   "params": {
  #       "sensor_name": "height_scanner",
  #       "flatten": True,
  #   },
  # }
  # cfg.observations["critic"].terms["feet_contact"] = {
  #     "func": mdp.foot_contact,
  #     "params": {
  #         "sensor_name": "feet_ground_contact",
  #         "flatten": True,
  #     },
  # }

  # cfg.observations["critic"].terms["nonfoot_contact"] = {
  #     "func": mdp.foot_contact,
  #     "params": {
  #         "sensor_name": "nonfoot_ground_touch",
  #         "flatten": True,
  #     },
  # }
  # cfg.observations["critic"].terms["leg_segment_contact"] = {
  #   "func": mdp.foot_contact,
  #   "params": {
  #       "sensor_name": "leg_segment_contact",
  #       "flatten": True,
  #   },
  # }
  # #check if legal in mujoco
  # cfg.observations["critic"].terms["external_forces"] = {
  #   "func": mdp.body_external_forces,
  #   "params": {
  #       "body_name": "base_link",
  #       "flatten": True,
  #   },
  # }

  print(f"\n----------Sensor------------\n{cfg.observations.keys()}\n-------Policy------\n{cfg.observations['policy'].terms.keys()}-------CritiC------\n{cfg.observations['critic'].terms.keys()}")
>>>>>>> ba122f926f2411a6cfe704e1394b5bb1689f813b

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
  # cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
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
