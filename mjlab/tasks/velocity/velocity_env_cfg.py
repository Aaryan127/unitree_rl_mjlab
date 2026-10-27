"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import PGTT_MIXED_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from mjlab.utils.noise import HeightScanNoiseModelCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Observations
  ##

  policy_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    **policy_terms,
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.05,
      rel_heading_envs=0.25,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 1.0),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.25, 0.25),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "body_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=".*_collision"),
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=mdp.randomize_encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-5.0),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "foot_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 0.3,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.12,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(50.0)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
          {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
          # {"step": 10000 * 24, "lin_vel_x": (-1.5, 3.0)},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )


def make_pgtt_env_cfg() -> ManagerBasedRlEnvCfg:
  """PGTT teacher-student task (Ntagkas et al. 2025 + Miki et al. 2022 distillation).

  Observation groups (who uses them is set by `obs_groups` in the RL config):

    policy          proprio + gait phase, noisy          teacher actor, student, ROBOT
    proprio_history joint pos/vel/action history, noisy  teacher actor
    height          clean height scan                    teacher actor, critic
    height_noisy    corrupted height scan                student, ROBOT
    privileged      contacts, friction, pushes, lin vel  teacher actor, critic
    critic          proprio + base lin vel, clean        critic

  Robot-specific values (sensor patterns, foot/geom/body names) are set in
  config/<robot>/env_cfgs.py, exactly like `make_velocity_env_cfg`.
  """

  ##
  # Observations
  ##

  # Everything in "policy" must be computable on the real robot. The order here is
  # the order of the deployed observation vector (see scripts/export_deploy.py).
  policy_terms = {
    # delay_max_lag=1: readings may be up to 1 control step (20 ms) old, as on the real
    # robot (sensor -> DDS -> controller latency). Sim-to-real robustness.
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      delay_max_lag=1,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      delay_max_lag=1,
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      delay_max_lag=1,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      delay_max_lag=1,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "gait_phase": ObservationTermCfg(
      func=mdp.gait_phase,
      params={"command_name": "gait"},
    ),
  }

  proprio_history_terms = {
    "joint_pos_history": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      history_length=3,
    ),
    "joint_vel_history": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      history_length=2,
    ),
    "action_history": ObservationTermCfg(func=mdp.last_action, history_length=2),
  }

  height_params = {"sensor_name": "height_scanner", "base_height": 0.25, "clip": 1.0}
  height_terms = {
    "height_scan": ObservationTermCfg(func=mdp.height_scan, params=height_params),
  }
  # The student's map is built from a simulated LiDAR with the robot bridge's exact
  # gridding (scripts/lidar_height_map.py), then corrupted further by the noise model
  # (drift, outliers, shift, dropouts). These settings are exported to deploy.yaml and
  # used by the bridge on the robot. To train on the ideal ray-cast map instead, set
  # func=mdp.height_scan, params=height_params.
  lidar_map_params = {
    "sensor_name": "lidar",
    "size_x": 1.0,
    "size_y": 0.8,
    "resolution": 0.1,
    "base_height": 0.25,
    "min_z": -0.9,
    "max_z": 0.45,
    "body_x": 0.30,
    "body_y": 0.15,
    "body_min_z": -0.10,
    "cell_radius": 0.075,
    "min_points": 2,
    "quantile": 0.2,
    "min_coverage": 0.5,
    "point_noise_std": 0.01,
  }
  height_noisy_terms = {
    "height_scan": ObservationTermCfg(
      func=mdp.lidar_height_map,
      params=lidar_map_params,
      noise=HeightScanNoiseModelCfg(),
      clip=(-1.0, 1.0),
      # LiDAR -> mapping -> UDP latency: the map may be up to 3 steps (60 ms) old.
      delay_max_lag=3,
    ),
  }

  privileged_terms = {
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_contact_normals": ObservationTermCfg(
      func=mdp.contact_normals, params={"sensor_name": "feet_ground_normals"}
    ),
    "foot_friction": ObservationTermCfg(
      func=mdp.geom_friction,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=())},  # Set per-robot.
    ),
    "leg_ground_contact": ObservationTermCfg(
      func=mdp.foot_contact, params={"sensor_name": "leg_ground_contact"}
    ),
    "base_external_wrench": ObservationTermCfg(
      func=mdp.body_external_wrench,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
      scale=0.1,
    ),
  }

  critic_terms = {
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    **policy_terms,
  }

  observations = {
    "policy": ObservationGroupCfg(terms=policy_terms, enable_corruption=True),
    "proprio_history": ObservationGroupCfg(
      terms=proprio_history_terms, enable_corruption=True
    ),
    "height": ObservationGroupCfg(terms=height_terms, enable_corruption=False),
    "height_noisy": ObservationGroupCfg(
      terms=height_noisy_terms, enable_corruption=True
    ),
    "privileged": ObservationGroupCfg(terms=privileged_terms, enable_corruption=False),
    "critic": ObservationGroupCfg(terms=critic_terms, enable_corruption=False),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
      clip_to_joint_limits=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    # PGTT: commands uniform in [-1, 1], resampled about once per episode.
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(8.0, 12.0),
      rel_standing_envs=0.05,
      heading_command=False,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
      ),
    ),
    "gait": mdp.GaitPhaseCommandCfg(
      leg_names=("FL", "FR", "RL", "RR"),  # Override per-robot if needed.
      frequency_range=(1.0, 3.0),
    ),
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14),
          "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1),  # start slightly tilted
        },
        # start moving a little, so the policy learns to take over from any state
        "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.3, 0.3)},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(4.0, 8.0),
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
    # A real force on the base (unlike push_robot), so the privileged
    # base_external_wrench observation is non-zero and the teacher can react to it.
    "external_wrench": EventTermCfg(
      func=mdp.apply_external_force_torque,
      mode="interval",
      interval_range_s=(3.0, 6.0),
      params={
        "force_range": (-15.0, 15.0),
        "torque_range": (-2.0, 2.0),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "body_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=".*_collision"),
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "field": "body_mass",
        "ranges": (-1.0, 2.0),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "field": "body_ipos",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
    ##
    # Sim-to-real robustness (motor and model uncertainty). Robot-specific names are set
    # in config/<robot>/env_cfgs.py.
    ##
    "pd_gains": EventTermCfg(  # real motors' effective kp/kd differ from the nominal 20/1
      func=mdp.randomize_pd_gains,
      mode="reset",
      model_fields=("actuator_gainprm", "actuator_biasprm"),
      params={"kp_range": (0.85, 1.15), "kd_range": (0.8, 1.2), "operation": "scale"},
    ),
    "effort_limits": EventTermCfg(  # weaker motors (heat, battery voltage)
      func=mdp.randomize_effort_limits,
      mode="reset",
      model_fields=("actuator_forcerange",),
      params={"effort_limit_range": (0.85, 1.0), "operation": "scale"},
    ),
    "actuator_delay": EventTermCfg(  # command latency in 5 ms physics steps: 0-20 ms
      func=mdp.sync_actuator_delays,
      mode="reset",
      params={"lag_range": (0, 4)},
    ),
    "link_mass": EventTermCfg(  # leg link masses +-10 %
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "scale",
        "field": "body_mass",
        "ranges": (0.9, 1.1),
      },
    ),
    "joint_friction": EventTermCfg(  # joint friction loss 0-0.2 Nm (nominal model has 0)
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "operation": "abs",
        "field": "dof_frictionloss",
        "ranges": (0.0, 0.2),
      },
    ),
    "joint_armature": EventTermCfg(  # rotor inertia x0.8..x1.2
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "operation": "scale",
        "field": "dof_armature",
        "ranges": (0.8, 1.2),
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=mdp.randomize_encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "bias_range": (-0.015, 0.015),
      },
    ),
  }

  ##
  # Rewards (PGTT Table I). mjlab multiplies every term by dt, so only the ratios
  # between weights matter.
  ##

  rewards = {
    "track_lin_vel": RewardTermCfg(
      func=mdp.track_lin_vel_xy_exp,
      weight=1.0,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "track_ang_vel": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp,
      weight=0.5,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
    "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
    "orientation": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.2),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-1.0),
    "joint_power": RewardTermCfg(func=mdp.joint_power, weight=-2e-5),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "default_pose": RewardTermCfg(
      func=mdp.weighted_default_pose,
      weight=-0.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "weights": {},  # Set per-robot.
      },
    ),
    "joint_torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-5),
    "foot_phase": RewardTermCfg(
      func=mdp.foot_phase_tracking,
      weight=1.0,
      params={
        "command_name": "gait",
        "sensor_name": "height_scanner",
        "stance_height": 0.0,  # Set per-robot (foot site height when standing).
        "swing_height": 0.08,
        # PGTT's sigma_f = 0.05 is read as a 5 cm standard deviation.
        "sigma": 0.05**2,
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
    ##
    # Optional terms (weight 0 = off and not computed). Not in PGTT; from the cpu
    # branch / Miki et al. 2022 for experiments. Enable from the command line, e.g.
    #   --env.rewards.swing-clearance.weight -0.5
    ##
    "swing_clearance": RewardTermCfg(  # swing foot lower than the terrain around it
      func=mdp.swing_terrain_clearance,
      weight=0.0,
      params={
        "command_name": "gait",
        "sensor_name": "height_scanner",
        "radius": 0.15,
        "margin": 0.02,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "stance_on_edge": RewardTermCfg(  # stance foot planted on a step edge
      func=mdp.stance_on_edge,
      weight=0.0,
      params={
        "command_name": "gait",
        "sensor_name": "height_scanner",
        "radius": 0.06,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "leg_contact": RewardTermCfg(  # thigh/shank touching the terrain
      func=mdp.leg_ground_contact_count,
      weight=0.0,
      params={"sensor_name": "leg_ground_contact"},
    ),
    "foot_slip": RewardTermCfg(  # foot sliding while in contact
      func=mdp.feet_slip,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "lin_vel_orthogonal": RewardTermCfg(  # sideways drift off the commanded direction
      func=mdp.lin_vel_orthogonal_l2,
      weight=0.0,
      params={"command_name": "twist"},
    ),
    "joint_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=0.0),
    "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=0.0),
    "action_smoothness": RewardTermCfg(func=mdp.action_acc_l2, weight=0.0),  # 2nd difference
  }

  ##
  # Terminations (PGTT: only when the robot turns over)
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(90.0)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
  }
  # Regularization curriculum (Miki et al. 2022): penalties start at 10 % of their weight
  # and reach 100 % after ~1500 iterations (x 24 steps), so the robot learns to walk
  # before it is pushed to walk smoothly. Disable one with
  #   --env.curriculum.ramp-joint-power.params.start-scale 1.0
  for reward_name in (
    "joint_power",
    "joint_torques",
    "action_rate",
    "action_smoothness",
    "joint_vel",
    "joint_acc",
    "foot_slip",
    "leg_contact",
  ):
    curriculum[f"ramp_{reward_name}"] = CurriculumTermCfg(
      func=mdp.reward_weight_ramp,
      params={"reward_name": reward_name, "start_scale": 0.1, "ramp_steps": 1500 * 24},
    )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=replace(PGTT_MIXED_TERRAINS_CFG),
        max_init_terrain_level=3,
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      # Stairs (box edges, shank contacts) create more contacts than rough terrain.
      nconmax=64,
      njmax=400,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
