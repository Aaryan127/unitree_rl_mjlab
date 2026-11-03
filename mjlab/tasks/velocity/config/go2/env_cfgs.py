"""Unitree Go2 velocity environment configurations."""

# How to read this file
# ---------------------
# Every function here returns a ManagerBasedRlEnvCfg: a big nested dataclass that says
# what the simulated world is (robot, terrain, sensors) and what the task is
# (observations, actions, commands, randomization "events", rewards, terminations,
# curriculum). Nothing is simulated here; train.py builds the environment from it.
# Each function starts from a more general config and overwrites parts of it, so read
# them as "the same as X, except ...". Most functions take `play`: True gives the
# version used for watching or checking a policy (no noise, no pushes, no curriculum).
#
#   unitree_go2_rough_env_cfg / _flat_env_cfg     stock mjlab Go2 tasks (not PGTT)
#   unitree_go2_pgtt_env_cfg                       the PGTT teacher/student task (4.1 in CODE_GUIDE)
#   unitree_go2_pgtt_repro_env_cfg                 rung 0: the official PGTT setup, walks
#   unitree_go2_pgtt_repro_smooth_env_cfg          rung 0b: + a tremor penalty
#   unitree_go2_pgtt_rung_env_cfg(rung)            rungs 1-7: + one teacher change each
#
# How to change it (the file you edit most)
# ----------------------------------------
# Words: the "ladder" is rung 0 (the official PGTT setup) plus rungs 1-7, each adding one change
# towards the full "teacher" (the policy with privileged sim-only inputs; the deployable
# "student" learns from it later). "Events" are resets and randomization; the "curriculum"
# changes the task during training. Details: doc/experiments.md and CODE_GUIDE.md.
#
# Everything the ladder trains with starts in unitree_go2_pgtt_repro_env_cfg (rung 0) and is
# inherited by every rung above it. unitree_go2_pgtt_rung_env_cfg then adds one block per rung
# (`if rung >= 1:` ... `if rung >= 7:`); a later block can overwrite what an earlier one set.
# - A reward weight, for every rung: edit its RewardTermCfg(weight=...) in the `cfg.rewards = {`
#   dict of unitree_go2_pgtt_repro_env_cfg.
# - The three rung 5+ terms (base_height -50, leg_contact -0.25, joint_limits -1) exist only
#   from rung 5 on. The `if rung >= 5:` block takes them, as objects, from the teacher config
#   (velocity_env_cfg.py). To change one for the ladder only, add a line AFTER that loop,
#   e.g. cfg.rewards["leg_contact"].weight = -0.5.
# - A weight for ONE run only, without editing code: add --env.rewards.<name>.weight=<value>
#   to the scripts/train.py command (COMMANDS.md 4b), e.g. for rung 7:
#     python scripts/train.py Mjlab-Velocity-PGTT-Rung7-Unitree-Go2 ... \
#       --env.rewards.leg-contact.weight=-0.5
#   In the reward's name, underscores become dashes (leg_contact -> leg-contact); the rest of
#   the flag stays as written. A negative value needs the `=`.
# - A new reward: write the function in mdp/rewards.py, then add
#   cfg.rewards["<name>"] = RewardTermCfg(func=mdp.<name>, weight=<w>, params={...}) here,
#   in rung 0 (all rungs) or inside the rung block that needs it.
# - Randomization ranges (friction, mass, Kp/Kd, motor strength): rung 0-3 use the PGTT values
#   set in the "Events" part of unitree_go2_pgtt_repro_env_cfg; rung 4+ use the teacher's,
#   defined in velocity_env_cfg.py (make_pgtt_env_cfg, `events = {`). To change one for the
#   ladder only, add a line after `cfg.events.update(teacher.events)` in the `if rung >= 4:`
#   block, e.g. cfg.events["body_friction"].params["ranges"] = (0.2, 1.4).
# - Terrain: rung 7 copies teacher.scene.terrain, which is mjlab/terrains/config.py's
#   PGTT_MIXED_TERRAINS_CFG. Change the mix there, not here.
# - Robot gains, collisions, motor delay: the robot blocks (rung 0, rung 1-2 blocks).
# After any change: a new rung or run name, so old and new results stay apart, and a line in
# doc/experiments.md.

import math
from dataclasses import replace  # copy of a dataclass with some fields changed

import mujoco  # the physics engine's Python API (used to edit the robot model, MjSpec)

from mjlab.asset_zoo.robots import (
  get_go2_robot_cfg,  # a fresh Go2 description: full collisions, stock motors
)
from mjlab.actuator.delayed_actuator import DelayedActuatorCfg  # motor whose target arrives late
from mjlab.asset_zoo.robots.unitree_go2 import go2_constants  # Go2 motor/collision presets
from mjlab.envs import ManagerBasedRlEnvCfg  # the config type every function returns
from mjlab.envs.mdp.actions import JointPositionActionCfg  # actions = joint position targets
from mjlab.managers.event_manager import EventTermCfg  # randomization/reset "events"
from mjlab.managers.reward_manager import RewardTermCfg  # one reward term: function + weight
from mjlab.managers.scene_entity_config import SceneEntityCfg  # robot/bodies/joints a term uses
from mjlab.managers.termination_manager import TerminationTermCfg  # when an episode ends
from mjlab.sensor import (
  ContactMatch,  # which geoms/bodies a contact sensor watches
  ContactSensorCfg,  # "is A touching B, and how hard"
  GridPatternCfg,  # ray pattern: a flat grid of downward rays (height map)
  LidarPatternCfg,  # ray pattern: a spinning LiDAR's beams
  ObjRef,  # names the body a sensor is attached to
  RayCastSensorCfg,  # sensor that shoots rays and reports where they hit
)
from mjlab.tasks.velocity import mdp  # all reward/observation/event/termination functions
from mjlab.tasks.velocity.terrain_profiles import set_terrain_profile  # "flat", "stairs", ...
from mjlab.tasks.velocity.velocity_env_cfg import (
  make_pgtt_env_cfg,  # robot-independent PGTT task recipe (velocity_env_cfg.py)
  make_velocity_env_cfg,  # robot-independent stock velocity task
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise  # noise drawn uniformly in [n_min, n_max]


##
# Stock mjlab Go2 tasks (Mjlab-Velocity-Rough/Flat-Unitree-Go2). Not used by PGTT.
##


def unitree_go2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()  # the generic velocity task, no robot yet

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}  # put one Go2 (per environment) in it

  # Leg order used to build names below. Geom = a collision shape in the model.
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")  # "sites" = marked points at the feet
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)  # the 4 foot spheres

  # Contact sensor: which feet touch the terrain, with how much force, and for how long
  # each foot has been in the air (used by air-time rewards).
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),  # the feet...
    secondary=ContactMatch(mode="body", pattern="terrain"),  # ...against the ground
    fields=("found", "force"),  # report "touching?" and the contact force
    reduce="netforce",  # several contact points per foot -> summed into one force
    num_slots=1,
    track_air_time=True,
  )
  # Contact sensor for everything that is NOT a foot (trunk, hips, legs).
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
    fields=("found",),  # only "touching or not"
    reduce="none",  # one answer per geom
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg)

  # Terrain curriculum: robots that walk well get moved to harder terrain rows.
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)  # sanity check of the action type

  # Where the viewer camera looks: at the trunk, 1.5 m away, 10 degrees from above.
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # The generic task leaves robot-specific names empty; fill in the Go2's.
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)  # shift the trunk's CoM

  # Posture reward: how far each joint may stray from the standing pose (std in rad),
  # tighter when standing, looser when walking/running. Hips tightest (no splaying).
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
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)  # trunk wobble
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names  # foot lift
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names  # feet sliding

  # Any non-foot part touching the ground ends the episode.
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False  # no sensor noise
    cfg.events.pop("push_robot", None)  # no shoves

    # A small terrain (5 x 5 tiles) of mixed difficulty, no curriculum.
    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)  # the rough task above, then:

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"  # an infinite flat plane
  cfg.scene.terrain.terrain_generator = None  # no generated tiles

  # Disable terrain curriculum.
  del cfg.curriculum["terrain_levels"]  # nothing harder to move to on a plane

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
  spec = go2_constants.get_spec()  # the Go2 model as an editable MjSpec (before compiling)
  for leg in _LEGS:
    # The calf's collision cylinders that have no name yet (the XML leaves them unnamed).
    cylinders = [
      g
      for g in spec.body(f"{leg}_calf").geoms
      if not g.name and g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
    ]
    # Exactly two per calf (strict=True fails loudly otherwise): name them upper/lower.
    for geom, part in zip(cylinders, ("upper", "lower"), strict=True):
      geom.name = f"{leg}_calf_{part}_collision"  # e.g. "FL_calf_upper_collision"
  return spec


def unitree_go2_pgtt_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Unitree Go2 PGTT stairs task, shared by the teacher (PPO) and student (distillation)."""
  # The robot-independent PGTT recipe: observation groups, gait command, randomization,
  # the (old) teacher reward set, terminations, curriculum, mixed terrain.
  cfg = make_pgtt_env_cfg()

  # Go2 with named calf geoms and DELAYED actuators: the position target can reach the
  # motor 0-20 ms late (sampled per episode by the actuator_delay event), as on the real
  # robot where commands travel over DDS. Gains from the official PGTT training config
  # (Kp 40, Kd 0.5; the stock Go2 task uses 20 / 1); the export writes them to deploy.yaml.
  go2 = get_go2_robot_cfg()  # full-body collisions (every *_collision geom can touch)
  delayed = DelayedActuatorCfg(
    # The Go2 motor model (23.5 Nm torque limit, rotor armature 0.01) with PGTT's gains.
    base_cfg=replace(go2_constants.GO2_ACTUATOR_GO2HV, stiffness=40.0, damping=0.5),
    delay_target="position",  # what arrives late: the position target
    delay_max_lag=4,  # up to 4 physics steps of 5 ms = 20 ms
  )
  cfg.scene.entities = {
    "robot": replace(
      go2,
      spec_fn=_go2_spec_with_named_calves,  # build the model with named calf cylinders
      articulation=replace(go2.articulation, actuators=(delayed,)),  # the delayed motors
    )
  }

  # Names of the geoms the sensors below watch.
  foot_geoms = tuple(f"{leg}_foot_collision" for leg in _LEGS)  # 4 foot spheres
  leg_geoms = tuple(  # 12 leg shapes: thigh + 2 calf cylinders per leg
    f"{leg}_{part}_collision" for leg in _LEGS for part in ("thigh", "calf_upper", "calf_lower")
  )

  ##
  # Sensors
  ##

  # Feet vs terrain: touching? total force? normal? air time? (foot rewards, privileged obs)
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force", "normal"),
    reduce="netforce",  # one summed force per foot
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
  # Thighs/calves vs terrain, one yes/no per geom (12 values). Read by the leg_contact
  # reward (kneeling, a shin on a step edge) and the teacher's privileged observation.
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
    frame=ObjRef(type="body", name="base_link", entity="robot"),  # moves with the trunk
    pattern=GridPatternCfg(size=(1.0, 0.8), resolution=0.1, z_offset=0.5),  # 10 cm spacing
    ray_alignment="yaw",  # the grid turns with the robot's heading, not its roll/pitch
    max_distance=3.0,  # rays longer than 3 m count as "no hit"
    include_geom_groups=(0,),
    debug_vis=play,  # draw the rays in the viewer when watching
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
      mount_pos=(0.28, 0.0, 0.03),  # 28 cm forward, 3 cm up from the trunk centre
      azimuth_res_deg=4.0,  # one beam every 4 degrees around
      elevation_range_deg=(-80.0, -10.0),  # beams aimed 10-80 degrees below horizontal
      num_rings=16,  # 16 beams spread over that elevation range
      pitch_deg=0.0,  # mount tilt
    ),
    ray_alignment="base",  # tilts with the trunk, like a real head-mounted LiDAR
    max_distance=5.0,
    include_geom_groups=(0,),  # sees only the terrain
  )
  cfg.scene.sensors = (
    feet_ground_cfg,
    feet_normals_cfg,
    leg_ground_cfg,
    body_ground_cfg,
    height_scanner_cfg,
    lidar_cfg,
  )
  # Episode ends (counts as a fall) when the trunk or a hip touches the ground.
  cfg.terminations["body_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,  # True if any watched geom is touching
    params={"sensor_name": body_ground_cfg.name},
  )

  # Terrain curriculum on (robots that walk far enough move to harder rows).
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  ##
  # Robot-specific names
  ##

  # Viewer camera: follows the trunk.
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.commands["gait"].leg_names = _LEGS  # type: ignore[attr-defined]  # phase clock leg order

  # Randomization events that need body names: trunk mass, trunk centre of mass, the
  # external push force (all on the trunk), and the leg links' masses.
  for event in ("base_mass", "base_com", "external_wrench"):
    cfg.events[event].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["link_mass"].params["asset_cfg"].body_names = (r".*_(hip|thigh|calf)",)

  # Privileged (simulator-only) observations that need names: the feet's friction and
  # the external force on the trunk. Only the teacher and critic see these.
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
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = _LEGS  # (optional term, weight 0)
  # Height grid layout (rows = y points, cols = x points) for the student's map-shift noise.
  cfg.observations["height_noisy"].terms["height_scan"].noise.grid_shape = (9, 11)  # type: ignore[union-attr]

  ##
  # Play mode
  ##

  if play:
    # nominal_eval_cfg removes every training perturbation (randomization, pushes, delays,
    # noise, curriculum), keeping what the policy sees and does. The hack check uses it too.
    from mjlab.utils.checkpoint import nominal_eval_cfg
    cfg = nominal_eval_cfg(cfg)
    cfg.episode_length_s = int(1e9)  # never time out while watching
    for group in ("policy", "proprio_history", "height_noisy"):
      cfg.observations[group].enable_corruption = False  # no observation noise
    cfg.events.pop("push_robot", None)
    cfg.events.pop("external_wrench", None)
    for name in list(cfg.curriculum):  # no terrain or reward curriculum in play mode
      cfg.curriculum.pop(name)
    # Small 5 x 5 terrain with all difficulties mixed.
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
    if joint.type != mujoco.mjtJoint.mjJNT_FREE:  # every leg joint (not the floating base)
      joint.damping = 0.5  # passive damping in Nm per rad/s, on top of the motor's Kd
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
  # Rung 0 = the teacher task above with the parts below replaced by PGTT's.
  cfg = unitree_go2_pgtt_env_cfg(play=play)
  set_terrain_profile(cfg, "flat")  # PGTT's flat_terrain task: an infinite plane

  # Robot: feet-only collisions (scene_mjx_feetonly.xml), Kp 40 with actuator Kd 0.5 plus
  # passive joint damping 0.5, no actuator delay.
  go2 = cfg.scene.entities["robot"]
  cfg.scene.entities = {
    "robot": replace(
      go2,
      collisions=(go2_constants.FEET_ONLY_COLLISION,),  # only the 4 feet touch anything
      spec_fn=_go2_spec_pgtt_joints,  # joint damping 0.5
      articulation=replace(
        go2.articulation,
        actuators=(
          # Plain (not delayed) motors with PGTT's gains.
          replace(go2_constants.GO2_ACTUATOR_GO2HV, stiffness=40.0, damping=0.5),
        ),
      ),
    )
  }
  # Only flipping over ends an episode.
  cfg.terminations.pop("body_contact", None)  # with feet-only collisions it can't fire anyway

  # Observations: PGTT noise (joint_pos +-0.03), no sensor delays.
  for group in cfg.observations.values():
    for term in group.terms.values():
      term.delay_min_lag = term.delay_max_lag = 0  # every observation arrives on time
  for group in ("policy", "proprio_history"):
    terms = cfg.observations[group].terms
    key = "joint_pos" if "joint_pos" in terms else "joint_pos_history"  # name differs per group
    terms[key].noise = Unoise(n_min=-0.03, n_max=0.03)  # +-0.03 rad joint-angle noise

  # No student LiDAR: its map (group height_noisy) is not a teacher input, but its 1440
  # rays per robot were tested against every terrain geom (4049 boxes on stairs) at every
  # step, ~75% of rung 7's step time. The student task builds its own config and keeps it.
  cfg.observations.pop("height_noisy", None)
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "lidar")
  # A robot whose physics turns NaN/Inf is reset (counts as a fall) instead of putting
  # NaN into the batch, where one robot poisons the whole update (rung 7 crashed at
  # iteration 4 with a NaN noise std).
  cfg.terminations["nan_state"] = TerminationTermCfg(func=mdp.nan_detection)
  # Same for a robot whose physics blew up without turning NaN (rung 7 on GPU: hip angles of
  # ~1e6 rad, rewards of -1e13): joints faster than 100 rad/s or the base faster than 20 m/s.
  cfg.terminations["physics_exploded"] = TerminationTermCfg(func=mdp.physics_exploded)

  # Commands: uniform in [-1, 1], each axis non-zero with probability (0.9, 0.25, 0.5),
  # resampled every ~5 s.
  twist = cfg.commands["twist"]  # the velocity command (vx, vy, yaw rate)
  twist.axis_nonzero_prob = (0.9, 0.25, 0.5)  # type: ignore[attr-defined]
  twist.rel_standing_envs = 0.0  # type: ignore[attr-defined]  # no robots told to stand on purpose
  twist.resampling_time_range = (2.0, 8.0)  # new command every 2-8 s

  # Events: PGTT's domain_randomize_simple; no pushes, no motor delay.
  for name in ("push_robot", "external_wrench", "effort_limits", "actuator_delay",
               "joint_friction", "encoder_bias"):
    cfg.events.pop(name, None)
  # Reset: anywhere within +-0.5 m, any heading, standing still (+-0.1 m/s), level.
  cfg.events["reset_base"].params = {
    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    "velocity_range": {k: (-0.1, 0.1) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
  }
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)  # joints at default
  # PGTT's narrower randomization ranges (the "if" guards: play mode has none of these).
  if "body_friction" in cfg.events:
    cfg.events["body_friction"].params["ranges"] = (0.4, 1.0)  # friction of the robot's geoms
  if "base_mass" in cfg.events:
    cfg.events["base_mass"].params["ranges"] = (-1.0, 1.0)  # kg added to the trunk
  if "joint_armature" in cfg.events:
    cfg.events["joint_armature"].params["ranges"] = (1.0, 1.05)  # rotor inertia x1.0-1.05
  if "pd_gains" in cfg.events:
    cfg.events["pd_gains"].params["kp_range"] = (0.9, 1.1)  # Kp x0.9-1.1
    cfg.events["pd_gains"].params["kd_range"] = (1.0, 1.0)  # Kd unchanged
  if not play:
    # Joint damping x0.9-1.1, sampled once per robot at startup.
    cfg.events["joint_damping"] = EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,  # one value per robot (not one for all)
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "operation": "scale",  # multiply the model's value
        "field": "dof_damping",
        "ranges": (0.9, 1.1),
      },
    )

  # Rewards: exactly PGTT's (robots/go2 default_config + joystick_base._get_reward).
  # mjlab multiplies each by dt and clips the step's sum at 0, as PGTT does.
  # Each term below: per step, reward += weight x func(...) x dt (dt = 0.02 s). Weights
  # are "per second", so a term worth -0.5/s costs 0.01 per step.
  hips_only = cfg.rewards["default_pose"]  # keep the teacher's hips-only pose term...
  hips_only.weight = -0.5  # ...at PGTT's weight
  monitor = cfg.rewards["hacking_monitor"]  # weight 0: only logs Metrics/hack/* for TensorBoard
  cfg.rewards = {  # replaces the teacher's whole reward set
    # +1/s at most: exp(-|commanded xy velocity - actual|^2 / 0.25).
    "track_lin_vel": RewardTermCfg(
      func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "twist", "sigma": 0.25}
    ),
    # +0.5/s at most: the same for the yaw (turning) rate.
    "track_ang_vel": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "twist", "sigma": 0.25}
    ),
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),  # -2 x vz^2: bouncing
    "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),  # roll/pitch rates^2
    "default_pose": hips_only,  # -0.5 x hip angles' squared distance from standing
    # PGTT: -1 inside the clip (-0.02 per episode end); ours is added after the clip.
    # Either way the cost of falling is the positive reward the robot no longer gets.
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-1.0),
    "torques": RewardTermCfg(func=mdp.torques_pgtt, weight=-2e-6),  # motor effort
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.001),  # (a_t - a_t-1)^2
    # +1/s at most: each foot follows the gait clock's target height (stance on the ground,
    # swing lifted), exp(-sum over feet of error^2 / sigma).
    "foot_phase": RewardTermCfg(
      func=mdp.foot_phase_tracking_body,  # foot heights measured from the IMU (PGTT code)
      weight=1.0,
      params={
        "command_name": "gait",  # the phase clock
        "sensor_name": "height_scanner",  # terrain under each foot (raises the swing target)
        "ref_site": "imu",
        # Feet 0.30 m below the IMU in stance, 0.20 m (+ terrain step) at the apex.
        "stance_height": -0.30,
        "swing_height": -0.20,
        "sigma": 0.05,  # m^2 on the summed squared error: how forgiving the match is
        "terrain_radius": 0.15,  # look for steps within 15 cm of each foot
        "max_terrain_delta": 0.15,  # raise the swing apex by at most 15 cm
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # -0.25/s per foot that touches the ground while the clock says it should swing.
    "swing_contact": RewardTermCfg(
      func=mdp.swing_foot_contact,
      weight=-0.25,
      params={"sensor_name": "feet_ground_contact", "command_name": "gait"},
    ),
    "hacking_monitor": monitor,
  }
  # No reward curriculum in PGTT.
  for name in [n for n in cfg.curriculum if n.startswith("ramp_")]:  # the penalty ramps
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
  cfg = unitree_go2_pgtt_repro_env_cfg(play=play)  # rung 0...
  # ...plus -0.02 x sum over joints of (a_t - 2 a_(t-1) + a_(t-2))^2.
  cfg.rewards["action_acc"] = RewardTermCfg(func=mdp.action_acc_l2, weight=-0.02)
  return cfg


##
# The ladder from rung 0b to the full teacher (doc/experiments.md, "The plan").
##

# Rung N's name is RUNG_NAMES[N - 1]; __init__.py registers Mjlab-Velocity-PGTT-Rung<N>-...
# and train_ladder.sh names the runs rung<N>_<name>.
RUNG_NAMES = ("collisions", "delays", "pushes", "randomization", "rewards", "commands", "terrain")


def unitree_go2_pgtt_rung_env_cfg(rung: int, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rung 0b plus the first `rung` changes towards the full teacher, each taken from it:

    1 collisions     full-body collisions; the trunk or a hip on the ground ends the episode
    2 delays         0-20 ms motor delay and the teacher's sensor delays
    3 pushes         velocity pushes and the external force on the base
    4 randomization  the teacher's wider domain randomization and reset states
    5 rewards        base_height, leg_contact and joint_limits (see below)
    6 commands       all three axes at once, 5% standing, resampled every 8-12 s
    7 terrain        mixed terrain (stairs, slopes, rough) with the terrain curriculum

  v1-v4 changed all of this at once and never walked; one change per rung means a failed
  rung points at exactly one change (scripts/train_ladder.sh stops there).

  Rung 5 takes only the teacher terms for what full collisions and terrain make possible:
  crouching (base_height), kneeling or dragging a shin (leg_contact) and pressing into
  the joint limits. Not its doubled tracking weights, -100 termination, power/torque
  terms or penalty ramps: the PGTT rewards walk, action_acc already smooths, and a ramp
  only delays a failure (lesson 7). No ramp here either, so a failure shows in 1500
  iterations. After rung 7 the remaining differences from the teacher task are these
  rewards, PGTT's joint damping and joint_pos noise, and the runner (Brax settings).
  """
  cfg = unitree_go2_pgtt_repro_smooth_env_cfg(play=play)  # start from rung 0b
  teacher = unitree_go2_pgtt_env_cfg(play=play)  # the full teacher: pieces are copied from it
  full = teacher.scene.entities["robot"]  # the teacher's robot (full collisions, delayed motors)

  # Each block adds one change; rung N runs blocks 1..N (">=").
  if rung >= 1:  # collisions
    robot = cfg.scene.entities["robot"]
    cfg.scene.entities["robot"] = replace(robot, collisions=full.collisions)  # every part touches
    cfg.terminations["body_contact"] = teacher.terminations["body_contact"]  # trunk/hip = fall
  if rung >= 2:  # delays
    robot = cfg.scene.entities["robot"]
    cfg.scene.entities["robot"] = replace(
      robot, articulation=replace(robot.articulation, actuators=full.articulation.actuators)
    )  # the delayed motors (0-20 ms)
    if "actuator_delay" in teacher.events:  # the event that samples each robot's delay
      cfg.events["actuator_delay"] = teacher.events["actuator_delay"]
    # Every observation term gets the teacher's delay range (IMU and joints 0-20 ms). The
    # student's LiDAR map (0-60 ms) is not in the rungs (see the repro config).
    for group_name, group in teacher.observations.items():
      if group_name not in cfg.observations:
        continue
      for term_name, term in group.terms.items():
        ours = cfg.observations[group_name].terms[term_name]
        ours.delay_min_lag, ours.delay_max_lag = term.delay_min_lag, term.delay_max_lag
  if rung >= 3:  # pushes
    for name in ("push_robot", "external_wrench"):  # velocity kicks every 4-8 s, force every 3-6 s
      if name in teacher.events:  # not in play mode
        cfg.events[name] = teacher.events[name]
  if rung >= 4:  # randomization
    cfg.events.update(teacher.events)  # keeps PGTT's joint_damping randomization
  if rung >= 5:  # rewards (weights: base_height -50, leg_contact -0.25, joint_limits -1)
    for name in ("base_height", "leg_contact", "joint_limits"):
      cfg.rewards[name] = teacher.rewards[name]
  if rung >= 6:  # commands
    cfg.commands["twist"] = teacher.commands["twist"]
  if rung >= 7:  # terrain
    cfg.scene.terrain = teacher.scene.terrain  # mixed: stairs, slopes, rough, waves, flat
    if "terrain_levels" in teacher.curriculum:  # not in play mode
      cfg.curriculum["terrain_levels"] = teacher.curriculum["terrain_levels"]
      # Heading commands for 80% of the robots, as legged_gym does on rough terrain: the
      # robot gets a direction and the yaw-rate command is 0.5 x the heading error, so it
      # turns, then walks straight. The curriculum promotes a robot that got > 4 m from its
      # start; with free yaw rates up to 1 rad/s it walks in circles, and even a perfect
      # walker was promoted in only 52% of episodes (demoted 43%), so the levels would
      # barely rise. With this: 84% / 14%. The other 20% keep direct yaw rates (the
      # joystick's), and the policy only ever sees a yaw rate (export sets heading None).
      twist = cfg.commands["twist"]
      twist.heading_command = True  # type: ignore[attr-defined]
      twist.heading_control_stiffness = 0.5  # type: ignore[attr-defined]
      twist.rel_heading_envs = 0.8  # type: ignore[attr-defined]
      twist.ranges.heading = (-math.pi, math.pi)  # type: ignore[attr-defined]
  return cfg
