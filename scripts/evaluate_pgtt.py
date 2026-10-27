"""Headless per-surface evaluation of a PGTT checkpoint (not a hardware certification).

Reports failures, command tracking, progress, and perception coverage separately for
flat, stairs and rough terrain. A short run only checks stack integrity. Use trained
checkpoints, many seeds and full episode horizons to assess locomotion quality.
"""

from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import json

import torch
import tyro

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rsl_rl.runners import OnPolicyRunner, DistillationRunner
from mjlab.tasks.velocity.terrain_profiles import TerrainProfile, set_terrain_profile
from mjlab.utils.checkpoint import checkpoint_configs, nominal_eval_cfg


@dataclass
class EvaluationConfig:
  checkpoint_file: str
  task: str = "Mjlab-Velocity-PGTT-Student-Unitree-Go2"
  terrains: tuple[TerrainProfile, ...] = ("flat", "stairs", "rough")
  num_envs: int = 64
  steps: int = 1000
  seeds: tuple[int, ...] = (42, 43, 44)
  forward_velocity: float = 0.3
  difficulty: float = 0.5
  device: str = "cpu"
  blind: bool = False
  randomized: bool = False
  allow_legacy_config: bool = False
  output: str = "logs/pgtt_evaluation.json"


def evaluate(cfg: EvaluationConfig) -> list[dict]:
  if cfg.steps < 1 or cfg.num_envs < 1 or not 0 <= cfg.difficulty <= 1:
    raise ValueError("Positive steps/env count and difficulty in [0,1] required")
  checkpoint = torch.load(cfg.checkpoint_file, weights_only=False, map_location="cpu")
  source_env, train_cfg = checkpoint_configs(checkpoint, cfg.task, allow_legacy=cfg.allow_legacy_config)
  reports = []
  for profile in cfg.terrains:
    for seed in cfg.seeds:
      env_cfg = deepcopy(source_env) if cfg.randomized else nominal_eval_cfg(source_env)
      env_cfg.curriculum = {}
      set_terrain_profile(env_cfg, profile)
      env_cfg.scene.num_envs = cfg.num_envs
      env_cfg.seed = seed
      env_cfg.episode_length_s = 20.0
      if env_cfg.scene.terrain.terrain_generator:
        gen = env_cfg.scene.terrain.terrain_generator
        gen.seed = seed
        gen.num_rows = 1
        gen.num_cols = 20
        gen.curriculum = True  # fixed columns cover each type, no runtime curriculum
        gen.difficulty_range = (cfg.difficulty, cfg.difficulty)
        env_cfg.scene.terrain.max_init_terrain_level = 0
      twist = env_cfg.commands["twist"]
      if not twist.ranges.lin_vel_x[0] <= cfg.forward_velocity <= twist.ranges.lin_vel_x[1]:
        raise ValueError("Requested evaluation velocity is outside training range")
      twist.ranges.lin_vel_x = (cfg.forward_velocity, cfg.forward_velocity)
      twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0.0, 0.0)
      twist.rel_standing_envs = 0.0
      env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device),
                               clip_actions=train_cfg.get("clip_actions"))
      try:
        cls = DistillationRunner if train_cfg["class_name"] == "DistillationRunner" else OnPolicyRunner
        runner = cls(env, train_cfg, device=cfg.device)
        runner.load(cfg.checkpoint_file, load_optimizer=False, map_location=cfg.device)
        policy = runner.get_inference_policy(cfg.device)
        policy.reset()
        obs = env.get_observations()
        fallen = torch.zeros(cfg.num_envs, dtype=torch.bool, device=cfg.device)
        failures = 0
        velocity_error = progress = coverage = 0.0
        with torch.no_grad():
          for _ in range(cfg.steps):
            if cfg.blind:
              obs["height_noisy"].zero_()
            actions = policy(obs)
            if not torch.isfinite(actions).all():
              raise RuntimeError("Non-finite policy action")
            obs, reward, dones, _ = env.step(actions)
            if not torch.isfinite(reward).all() or any(not torch.isfinite(v).all() for v in obs.values()):
              raise RuntimeError("Non-finite environment reward/observation")
            terminated = env.unwrapped.reset_terminated
            fallen |= terminated
            failures += int(terminated.sum())
            robot = env.unwrapped.scene["robot"]
            velocity = robot.data.root_link_lin_vel_b[:, :2]
            target = env.unwrapped.command_manager.get_command("twist")[:, :2]
            velocity_error += float((velocity - target).square().sum(-1).mean())
            progress += float(velocity[:, 0].mean()) * env.unwrapped.step_dt
            coverage += float(env.unwrapped.extras.get("log", {}).get("Metrics/lidar_map_coverage", 0))
            policy.reset(dones)
        report = dict(terrain=profile, seed=seed, steps=cfg.steps, num_envs=cfg.num_envs,
                      difficulty=cfg.difficulty, blind=cfg.blind,
                      randomized=cfg.randomized,
                      fall_resets=failures, fraction_without_fall=float((~fallen).float().mean()),
                      mean_velocity_error_m2_s2=velocity_error / cfg.steps,
                      mean_body_forward_distance_m=progress,
                      mean_lidar_coverage=coverage / cfg.steps)
        reports.append(report)
        print(json.dumps(report), flush=True)
      finally:
        env.close()
  output = Path(cfg.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps({"checkpoint": cfg.checkpoint_file,
                                "iteration": checkpoint.get("iter"), "results": reports}, indent=2) + "\n")
  return reports


if __name__ == "__main__":
  evaluate(tyro.cli(EvaluationConfig, config=(tyro.conf.FlagConversionOff,)))
