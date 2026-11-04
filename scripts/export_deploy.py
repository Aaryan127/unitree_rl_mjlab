"""Export a trained PGTT student for the real Go2 and verify the export.

Writes, into the deploy policy folder (default deploy/robots/go2/config/policy/pgtt/v0):
  exported/policy.onnx   student network: (obs, h_in) -> (actions, h_out)
  params/deploy.yaml     everything the C++ controller needs, generated from the
                         training config: joint mapping, PD gains, default pose,
                         action scale, observation order and parameters.

Then it runs the simulator for a few hundred steps and checks that ONNX Runtime gives
the same actions as the PyTorch policy (including the GRU hidden state).

Usage:
  python scripts/export_deploy.py --checkpoint-file logs/rsl_rl/go2_pgtt_student/<run>/model_<it>.pt
"""

from __future__ import annotations

import os
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Export on CPU.

import numpy as np
import torch
import tyro
import yaml

import mjlab.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.rsl_rl.runners import DistillationRunner
from mjlab.utils.checkpoint import checkpoint_configs, nominal_eval_cfg

# Unitree SDK motor order for the Go2 (lowcmd / lowstate index 0..11).
GO2_SDK_JOINT_ORDER = [
  f"{leg}_{joint}_joint"
  for leg in ("FR", "FL", "RR", "RL")
  for joint in ("hip", "thigh", "calf")
]

# Python observation term name -> C++ observation name (deploy/include/.../observations.h).
CPP_OBS_NAMES = {
  "base_ang_vel": "base_ang_vel",
  "projected_gravity": "projected_gravity",
  "command": "velocity_commands",
  "joint_pos": "joint_pos_rel",
  "joint_vel": "joint_vel_rel",
  "actions": "last_action",
  "gait_phase": "gait_phase_pgtt",
  "height_scan": "height_scan",
  # student_sensors (the Go2 EDU's other sensors):
  "base_lin_acc": "imu_accelerometer",
  "motor_torque": "joint_tau_est",
  "foot_force": "foot_force",
}
# Go2 SDK foot order (lowstate foot_force index 0..3).
GO2_SDK_FOOT_ORDER = ("FR", "FL", "RR", "RL")


@dataclass
class ExportConfig:
  checkpoint_file: str
  """Student checkpoint (logs/rsl_rl/go2_pgtt_student/<run>/model_<it>.pt)."""
  task: str = "Mjlab-Velocity-PGTT-Student-Unitree-Go2"
  output_dir: str = "deploy/robots/go2/config/policy/pgtt/v0"
  gait_frequency: float = 2.0
  """Gait frequency used on the robot (training samples it from U[1, 3] Hz)."""
  height_map_timeout: float = 0.2
  """Seconds after which a height map is considered stale (robot falls back to 0s)."""
  height_map_udp_port: int = 9870
  """UDP port the controller listens on for height maps (0 = disabled, always zeros)."""
  verify_steps: int = 200
  tolerance: float = 1e-4
  allow_legacy_config: bool = False
  foot_force_offset: tuple[float, float, float, float] | None = None
  """Raw foot_force reading of each foot in the air, SDK order FR, FL, RR, RL."""
  foot_force_standing: tuple[float, float, float, float] | None = None
  """Raw foot_force reading of each foot standing still on flat ground in the policy's
  default pose, SDK order. Both are required when the student uses foot_force: calibrate
  on the robot first (doc/experiments.md, "Calibrating the foot force sensors")."""


def _term_cfg(params: dict, dim: int) -> dict:
  return {"params": params, "clip": None, "scale": [1.0] * dim, "history_length": 1}


def sim_standing_foot_load(env_cfg) -> list[float]:
  """Per sim foot (contact sensor order), mdp.foot_force_sensor while standing still on flat
  ground in the default pose (zero action), without noise: what the student saw for a
  standing robot. The feet are not equal (the hind legs carry more)."""
  from copy import deepcopy

  from mjlab.tasks.velocity import mdp
  from mjlab.tasks.velocity.terrain_profiles import set_terrain_profile

  cfg = nominal_eval_cfg(deepcopy(env_cfg))
  set_terrain_profile(cfg, "flat")
  cfg.scene.num_envs = 1
  cfg.episode_length_s = 1e6
  for name in [n for n in cfg.commands]:  # stand: zero velocity command
    term = cfg.commands[name]
    if hasattr(term, "rel_standing_envs"):
      term.rel_standing_envs = 1.0
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    sensor_name = env.observation_manager.get_term_cfg("student_sensors", "foot_force").params["sensor_name"]
    loads = []
    for step in range(150):  # 3 s: settle 1.5 s, average the rest
      env.step(torch.zeros(1, env.action_manager.total_action_dim))
      if step >= 75:
        loads.append(mdp.foot_force_sensor(env, sensor_name)[0])
    return torch.stack(loads).mean(0).tolist()
  finally:
    env.close()


def _foot_force_params(
  env: ManagerBasedRlEnv, term, cfg: ExportConfig, sim_standing: list[float] | None
) -> dict:
  """foot_ids_map (SDK index of each sim foot, in the contact sensor's order) and the
  per-foot calibration that turns raw counts into the training unit: each foot standing
  on the robot reads what it read standing in the sim."""
  if cfg.foot_force_offset is None or cfg.foot_force_standing is None:
    raise ValueError(
      "This student reads the foot force sensors: pass --foot-force-offset and "
      "--foot-force-standing from the robot's calibration (doc/experiments.md)."
    )
  if sim_standing is None:
    raise ValueError("The sim's standing foot loads are needed (sim_standing_foot_load).")
  sensor = env.scene[term.params["sensor_name"]]
  feet = [slot.primary_name for slot in sensor._slots if slot.field_name == "force"]
  ids = []
  for geom in feet:
    leg = next(leg for leg in GO2_SDK_FOOT_ORDER if geom.startswith(leg + "_"))
    ids.append(GO2_SDK_FOOT_ORDER.index(leg))
  scale = [0.0] * 4
  for sim_foot, sdk in enumerate(ids):
    span = cfg.foot_force_standing[sdk] - cfg.foot_force_offset[sdk]
    if span <= 0:
      raise ValueError(f"Foot {GO2_SDK_FOOT_ORDER[sdk]}: standing reading not above the offset")
    scale[sdk] = sim_standing[sim_foot] / span
  return {
    "foot_ids_map": ids,
    "offset": [float(x) for x in cfg.foot_force_offset],
    "scale": scale,
    "sim_standing": [float(x) for x in sim_standing],  # for reference, sim foot order
  }


def build_deploy_yaml(
  env: ManagerBasedRlEnv, cfg: ExportConfig, policy_groups: tuple[str, ...] = ("policy",),
  sim_standing: list[float] | None = None,
) -> dict:
  robot = env.scene["robot"]
  sim_joints = list(robot.joint_names)
  assert sorted(sim_joints) == sorted(GO2_SDK_JOINT_ORDER), sim_joints
  joint_ids_map = [GO2_SDK_JOINT_ORDER.index(name) for name in sim_joints]

  meta = get_base_metadata(env, run_path="")
  stiffness_sdk = [0.0] * len(sim_joints)
  damping_sdk = [0.0] * len(sim_joints)
  for i, sdk_id in enumerate(joint_ids_map):
    stiffness_sdk[sdk_id] = float(meta["joint_stiffness"][i])  # type: ignore[index]
    damping_sdk[sdk_id] = float(meta["joint_damping"][i])  # type: ignore[index]
  default_pos = [float(x) for x in meta["default_joint_pos"]]  # type: ignore[union-attr]
  joint_action = env.action_manager.get_term("joint_pos")
  if joint_action.target_names != sim_joints:
    raise ValueError("Export requires actions in robot joint order")
  if len(env.action_manager.active_terms) != 1 or joint_action.cfg.clip is not None:
    raise ValueError("Export supports one unclipped joint-position action term")
  offset = joint_action.offset
  action_offset = offset[0].cpu().tolist() if isinstance(offset, torch.Tensor) else [offset] * len(sim_joints)
  target_clip = None
  if joint_action.cfg.clip_to_joint_limits:
    target_clip = robot.data.joint_pos_limits[0, joint_action.target_ids].cpu().tolist()
  action_scale = meta["action_scale"]
  if not isinstance(action_scale, list):
    action_scale = [float(action_scale)] * len(sim_joints)

  twist = env.command_manager.get_term_cfg("twist")
  gait = env.command_manager.get_term_cfg("gait")
  if not gait.frequency_range[0] <= cfg.gait_frequency <= gait.frequency_range[1]:
    raise ValueError("Deployment gait frequency must lie inside the training range")

  # Observation vector = the student's proprio groups' terms in order (its obs_groups
  # "policy": policy, then student_sensors), then the height map: the order the ONNX
  # model's input is split in (StudentOnnxWrapper).
  obs_manager = env.observation_manager
  observations: dict = {}
  for group in policy_groups:
    for name, dims in zip(
      obs_manager.active_terms[group], obs_manager.group_obs_term_dim[group], strict=True
    ):
      if name not in CPP_OBS_NAMES:
        raise KeyError(f"Student observation '{group}/{name}' has no C++ implementation.")
      if CPP_OBS_NAMES[name] in observations:
        raise KeyError(f"Two student observations map to C++ '{CPP_OBS_NAMES[name]}'.")
      dim = int(np.prod(dims))
      params: dict = {}
      if name == "command":
        params = {"command_name": "base_velocity"}
      elif name == "gait_phase":
        params = {
          "frequency": cfg.gait_frequency,
          "offsets": [float(x) for x in gait.leg_offsets],
        }
      elif name == "foot_force":
        params = _foot_force_params(env, obs_manager.get_term_cfg(group, name), cfg, sim_standing)
      term = obs_manager.get_term_cfg(group, name)
      if term.history_length or term.scale is not None:
        raise ValueError(f"Export cannot silently discard history/scale for {name}")
      observations[CPP_OBS_NAMES[name]] = _term_cfg(params, dim)
      if term.clip is not None:
        observations[CPP_OBS_NAMES[name]]["clip"] = [float(term.clip[0]), float(term.clip[1])]

  height_dims = obs_manager.group_obs_term_dim["height_noisy"][0]
  num_rays = int(np.prod(height_dims))
  scanner = env.scene["height_scanner"].cfg
  observations["height_scan"] = _term_cfg(
    {
      "num_points": num_rays,
      "timeout": cfg.height_map_timeout,
      "udp_port": cfg.height_map_udp_port,
      # Layout the elevation-map bridge must produce (see height_map.h):
      "size_x": float(scanner.pattern.size[0]),
      "size_y": float(scanner.pattern.size[1]),
      "resolution": float(scanner.pattern.resolution),
      "base_height": float(
        obs_manager.get_term_cfg("height_noisy", "height_scan").params["base_height"]
      ),
    },
    num_rays,
  )

  # Map-building settings of the student's LiDAR map (mdp.lidar_height_map), so the robot's
  # bridge (scripts/lidar_height_map.py) builds the map exactly as in training.
  height_term = obs_manager.get_term_cfg("height_noisy", "height_scan")
  if getattr(height_term.func, "__name__", "") == "lidar_height_map":
    p = height_term.params
    observations["height_scan"]["params"]["lidar_map"] = {
      k: float(p[k]) if k != "min_points" else int(p[k])
      for k in ("min_z", "max_z", "body_x", "body_y", "body_min_z", "cell_radius",
                "min_points", "quantile", "min_coverage")
    }

  return {
    "joint_ids_map": joint_ids_map,
    "step_dt": float(env.step_dt),
    "stiffness": stiffness_sdk,
    "damping": damping_sdk,
    "default_joint_pos": default_pos,
    "commands": {
      "base_velocity": {
        "ranges": {
          "lin_vel_x": list(twist.ranges.lin_vel_x),
          "lin_vel_y": list(twist.ranges.lin_vel_y),
          "ang_vel_z": list(twist.ranges.ang_vel_z),
          "heading": None,
        }
      }
    },
    "actions": {
      "JointPositionAction": {
        "clip": target_clip,
        "joint_names": [".*"],
        "scale": [float(s) for s in action_scale],
        "offset": action_offset,
        "joint_ids": None,
      }
    },
    "observations": observations,
  }


def verify(
  env: RslRlVecEnvWrapper, policy, onnx_path: Path, cfg: ExportConfig,
  policy_groups: tuple[str, ...] = ("policy",),
) -> float:
  import onnxruntime as ort

  session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  h_shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[1].shape]
  h = np.zeros(h_shape, dtype=np.float32)

  policy.eval()
  policy.reset()
  obs = env.get_observations()
  max_err = 0.0
  squash = env.unwrapped.cfg.squash_actions
  with torch.inference_mode():
    for _ in range(cfg.verify_steps):
      torch_actions = policy.act_inference(obs)
      expected = torch.tanh(torch_actions) if squash else torch_actions  # what the env applies
      flat_obs = torch.cat(
        [obs[g] for g in policy_groups] + [obs["height_noisy"]], dim=-1
      ).cpu().numpy()
      onnx_actions, h = session.run(None, {"obs": flat_obs.astype(np.float32), "h_in": h})
      error = float(np.abs(onnx_actions - expected.cpu().numpy()).max())
      if not np.isfinite(error):
        raise RuntimeError("Non-finite policy actions during export verification")
      max_err = max(max_err, error)
      obs, _, dones, _ = env.step(torch_actions)
      policy.reset(dones)
      h[:, dones.cpu().numpy().astype(bool), :] = 0
  return max_err


def main(cfg: ExportConfig) -> None:
  if cfg.verify_steps < 1 or cfg.tolerance <= 0 or cfg.height_map_timeout <= 0:
    raise ValueError("Verification steps, tolerance and map timeout must be positive")
  if not 0 <= cfg.height_map_udp_port <= 65535:
    raise ValueError("Invalid UDP port")
  out_dir = Path(cfg.output_dir)
  checkpoint = torch.load(cfg.checkpoint_file, weights_only=False, map_location="cpu")
  env_cfg, agent_cfg = checkpoint_configs(checkpoint, cfg.task, allow_legacy=cfg.allow_legacy_config)
  policy_groups = tuple(agent_cfg["obs_groups"]["policy"])
  sim_standing = None
  if "student_sensors" in policy_groups:
    sim_standing = sim_standing_foot_load(env_cfg)
  env_cfg = nominal_eval_cfg(env_cfg)
  if agent_cfg.get("clip_actions") is not None:
    raise ValueError("Export of wrapper-level raw action clipping is not supported")
  env_cfg.scene.num_envs = 1
  # Exercise recurrent resets repeatedly as well as ordinary sequential inference.
  env_cfg.episode_length_s = 0.2
  # Numerical equivalence does not require building the entire training terrain.
  from mjlab.tasks.velocity.terrain_profiles import set_terrain_profile
  set_terrain_profile(env_cfg, "flat")
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cpu"))

  runner = DistillationRunner(env, agent_cfg, device="cpu")
  resumed = runner.alg.policy.load_state_dict(checkpoint["model_state_dict"])
  if not resumed:
    raise ValueError("That checkpoint is a teacher. Export a student checkpoint.")
  policy = runner.alg.policy

  deploy_cfg = build_deploy_yaml(env.unwrapped, cfg, policy_groups, sim_standing)
  obs_dim = sum(len(term["scale"]) for term in deploy_cfg["observations"].values())
  assert obs_dim == policy.student_obs_dim, (obs_dim, policy.student_obs_dim)

  out_dir.parent.mkdir(parents=True, exist_ok=True)
  try:
    with tempfile.TemporaryDirectory(prefix="pgtt-export-", dir=out_dir.parent) as temp:
      staged = Path(temp)
      policy.export_onnx(str(staged), "policy.onnx", squash=env.unwrapped.cfg.squash_actions)
      err = verify(env, policy, staged / "policy.onnx", cfg, policy_groups)
      if err > cfg.tolerance:
        raise RuntimeError(f"ONNX export mismatch {err:.2e} > {cfg.tolerance}")
      # Failed verification never replaces the controller's current model/config.
      (out_dir / "exported").mkdir(parents=True, exist_ok=True)
      (out_dir / "params").mkdir(parents=True, exist_ok=True)
      for file in staged.iterdir():
        os.replace(file, out_dir / "exported" / file.name)
      yaml_path = staged / "deploy.yaml"
      with yaml_path.open("w") as f:
        f.write(f"# Generated from {cfg.checkpoint_file}\n")
        yaml.safe_dump(deploy_cfg, f, sort_keys=False, default_flow_style=None)
      os.replace(yaml_path, out_dir / "params" / "deploy.yaml")
      manifest = {
        "task": cfg.task, "checkpoint": str(Path(cfg.checkpoint_file).resolve()),
        "iteration": int(checkpoint.get("iter", -1)),
        "config_snapshot": "env_cfg" in checkpoint,
        "numerical_max_action_error": err,
        "policy_sha256": hashlib.sha256((out_dir / "exported" / "policy.onnx").read_bytes()).hexdigest(),
        "behavior_validated_on_robot": False,
      }
      (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  finally:
    env.close()

  print("\n=== PGTT student export ===")
  print(f"ONNX:        {out_dir / 'exported' / 'policy.onnx'}")
  print(f"deploy.yaml: {out_dir / 'params' / 'deploy.yaml'}")
  print(f"obs dim:     {obs_dim}  ({list(deploy_cfg['observations'].keys())})")
  print(f"joint map:   {deploy_cfg['joint_ids_map']}")
  print(f"ONNX vs torch max |action diff| over {cfg.verify_steps} steps: {err:.2e}")
  if err > cfg.tolerance:
    raise RuntimeError(f"ONNX export mismatch {err:.2e} > {cfg.tolerance}")
  print("Export verified.")


if __name__ == "__main__":
  main(tyro.cli(ExportConfig, config=(tyro.conf.FlagConversionOff,)))
