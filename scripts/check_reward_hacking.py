"""Check a PGTT checkpoint for reward hacking: what the robot does, not what it scores.

Runs the policy headless with the training command ranges (no perturbations) and
reports behaviour (mdp/hacking_monitor.py), falls and where the reward comes from, each
with an OK / WARN / HACK verdict. Works on old checkpoints too.

  python scripts/check_reward_hacking.py --checkpoint-file logs/.../model_3000.pt
  python scripts/check_reward_hacking.py --watch logs/rsl_rl/go2_pgtt_teacher/<run>

--watch checks every new checkpoint of a running training (CPU, so it can run next to
it in another tmux window) and appends one line per checkpoint to <run>/hacking_check.csv.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import csv
import re
import time

import torch
import tyro

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rsl_rl.runners import DistillationRunner, OnPolicyRunner
from mjlab.tasks.velocity.mdp.hacking_monitor import behaviour_metrics
from mjlab.tasks.velocity.terrain_profiles import TerrainProfile, set_terrain_profile
from mjlab.utils.checkpoint import nominal_eval_cfg

TEACHER = "Mjlab-Velocity-PGTT-Teacher-Unitree-Go2"
STUDENT = "Mjlab-Velocity-PGTT-Student-Unitree-Go2"


@dataclass
class HackCheckConfig:
  checkpoint_file: str | None = None
  watch: str | None = None
  """Run folder to follow; checks each new model_*.pt as training saves it."""
  terrain: TerrainProfile | None = None
  """Default: the terrain the checkpoint was trained on."""
  num_envs: int = 32
  steps: int = 500
  """Policy steps (50 per second)."""
  device: str = "cpu"
  seed: int = 0
  poll_s: float = 60.0


# (metric, warn if, hack if, what it means); thresholds are for the Go2 (walks at ~0.27 m).
CHECKS = [
  ("base_height", lambda v: v < 0.2, lambda v: v < 0.15, "body low: crouching or lying down"),
  ("low_base_frac", lambda v: v > 0.02, lambda v: v > 0.1, "time spent with the body below 0.15 m"),
  ("body_ground_frac", lambda v: v > 0.005, lambda v: v > 0.05, "trunk/hips on the ground"),
  ("leg_ground_frac", lambda v: v > 0.1, lambda v: v > 0.3, "thighs/calves dragging or kneeling"),
  ("speed_ratio", lambda v: v < 0.8, lambda v: v < 0.5, "fraction of the commanded speed achieved"),
  ("feet_down_moving", lambda v: v > 0.5, lambda v: v > 0.8, "all feet down while moving: shuffling/sliding"),
  ("standing_drift", lambda v: v > 0.15, lambda v: v > 0.3, "moving when told to stand [m/s]"),
  ("action_jitter", lambda v: v > 0.3, lambda v: v > 0.8, "action change per step: vibrating joints"),
  ("action_saturation", lambda v: v > 0.2, lambda v: v > 0.5, "actions stuck at +-1: bang-bang control"),
  ("falls_per_min", lambda v: v > 0.2, lambda v: v > 1.0, "episodes ended early, per robot-minute"),
]


def check(checkpoint_file: str, cfg: HackCheckConfig) -> dict:
  checkpoint = torch.load(checkpoint_file, weights_only=False, map_location="cpu")
  train_cfg = deepcopy(checkpoint["train_cfg"])
  env_cfg = nominal_eval_cfg(checkpoint["env_cfg"])
  if cfg.terrain is not None:
    set_terrain_profile(env_cfg, cfg.terrain)
  gen = env_cfg.scene.terrain.terrain_generator if env_cfg.scene.terrain else None
  if gen is not None:
    gen.num_rows, gen.num_cols = 3, 5
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device), clip_actions=train_cfg.get("clip_actions")
  )
  try:
    cls = DistillationRunner if train_cfg["class_name"] == "DistillationRunner" else OnPolicyRunner
    runner = cls(env, train_cfg, device=cfg.device)
    runner.load(checkpoint_file, load_optimizer=False, map_location=cfg.device)
    policy = runner.get_inference_policy(cfg.device)
    policy.reset()
    base = env.unwrapped
    sums: dict[str, list[float]] = {}
    rewards = torch.zeros(len(base.reward_manager.active_terms))
    terminations: dict[str, int] = {}
    obs = env.get_observations()
    with torch.no_grad():
      for _ in range(cfg.steps):
        obs, _, dones, _ = env.step(policy(obs))
        policy.reset(dones)
        for name, value in behaviour_metrics(base).items():
          valid = value[~torch.isnan(value)]
          if valid.numel():
            sums.setdefault(name, []).append(float(valid.mean()))
        rewards += base.reward_manager._step_reward.mean(0).cpu()
        for name in base.termination_manager.active_terms:
          if name != "time_out":
            terminations[name] = terminations.get(name, 0) + int(base.termination_manager.get_term(name).sum())
  finally:
    env.close()

  metrics = {name: sum(v) / len(v) for name, v in sums.items()}
  robot_minutes = cfg.num_envs * cfg.steps * base.step_dt / 60
  metrics["falls_per_min"] = sum(terminations.values()) / robot_minutes
  reward_share = dict(zip(base.reward_manager.active_terms, (rewards / cfg.steps).tolist()))
  return dict(checkpoint=checkpoint_file, iteration=checkpoint.get("iter"), metrics=metrics,
              terminations=terminations, rewards=reward_share)


def verdicts(metrics: dict) -> list[tuple[str, str, float, str]]:
  rows = []
  for name, warn, hack, meaning in CHECKS:
    if name in metrics:
      v = metrics[name]
      rows.append((name, "HACK" if hack(v) else "WARN" if warn(v) else "OK", v, meaning))
  return rows


def report(result: dict) -> None:
  print(f"\n== {result['checkpoint']}  (iteration {result['iteration']})")
  for name, verdict, value, meaning in verdicts(result["metrics"]):
    print(f"  {verdict:4}  {name:17} {value:8.3f}   {meaning}")
  if result["terminations"]:
    print("  episode endings:", ", ".join(f"{k}={v}" for k, v in result["terminations"].items()))
  rewards = {k: v for k, v in result["rewards"].items() if abs(v) > 1e-9}
  positive = sum(v for v in rewards.values() if v > 0) or 1.0
  print("  reward per second (share of the positive reward):")
  for name, value in sorted(rewards.items(), key=lambda kv: -abs(kv[1])):
    share = f"{100 * value / positive:5.1f}%" if value > 0 else "      "
    print(f"    {name:22} {value:+8.4f} {share}")
  tracking = sum(v for k, v in rewards.items() if k.startswith("track") and v > 0)
  ratio = result["metrics"].get("speed_ratio", 1.0)
  if ratio < 0.5 and tracking < 0.5 * positive:
    top = max((k for k, v in rewards.items() if v > 0 and not k.startswith("track")),
              key=rewards.get, default=None)
    print(f"  !! it ignores the command but still scores: most positive reward is from '{top}',"
          " not tracking")
  worst = [v for _, v, _, _ in verdicts(result["metrics"])]
  print("  overall:", "HACK" if "HACK" in worst else "WARN" if "WARN" in worst else "OK")


def _iteration(path: Path) -> int:
  return int(re.search(r"model_(\d+)\.pt$", path.name).group(1))


def watch(cfg: HackCheckConfig) -> None:
  run = Path(cfg.watch)
  out = run / "hacking_check.csv"
  done = set()
  if out.exists():
    done = {row["checkpoint"] for row in csv.DictReader(out.open())}
  print(f"watching {run} (Ctrl+C to stop), results in {out}")
  while True:
    for path in sorted(run.glob("model_*.pt"), key=_iteration):
      if str(path) in done:
        continue
      result = check(str(path), cfg)
      report(result)
      row = {"checkpoint": str(path), "iteration": result["iteration"],
             "overall": max((v for _, v, _, _ in verdicts(result["metrics"])),
                            key=["OK", "WARN", "HACK"].index),
             **{k: round(v, 4) for k, v in result["metrics"].items()}}
      new = not out.exists()
      with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
          writer.writeheader()
        writer.writerow(row)
      done.add(str(path))
    time.sleep(cfg.poll_s)


def main(cfg: HackCheckConfig) -> None:
  if cfg.watch:
    watch(cfg)
  elif cfg.checkpoint_file:
    report(check(cfg.checkpoint_file, cfg))
  else:
    raise SystemExit("pass --checkpoint-file <model.pt> or --watch <run folder>")


if __name__ == "__main__":
  main(tyro.cli(HackCheckConfig, config=(tyro.conf.FlagConversionOff,)))
