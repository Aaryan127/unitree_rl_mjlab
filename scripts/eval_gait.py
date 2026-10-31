"""Measure how a velocity checkpoint walks, under fixed commands (deterministic policy).

check_reward_hacking.py asks "does it cheat?"; this asks "how good is the gait?". Every
robot gets one fixed (vx, vy, wz, gait frequency) scenario for the whole rollout, and the
table reports per scenario:

  vx vy wz      achieved velocity (compare with the command in the name)
  swing/s       real swings (foot in the air >= 60 ms) per foot-second; should equal f
  blip/s        air gaps < 60 ms per foot-second: feet bouncing or tapping
  clock         share of time each foot's contact matches the gait clock (stance/swing)
  apex          median swing apex above the ground [cm]
  slip          mean horizontal speed of feet in contact [m/s] (includes the ball foot
                rolling, so not exactly zero even without slip)
  tilt          mean body tilt [deg]
  power         mechanical power sum |tau * qd| [W]
  tremor        mean |a_t - 2a_(t-1) + a_(t-2)| (<= 0.09 for any motion of at most 3 Hz)
  hf            share of action power above 6 Hz (above the gait and its harmonics)
  tau_max       peak joint torque [Nm]

Below it: the lowest any non-foot body origin gets (knee clearance; with feet-only
collisions nothing else would ever report this).

  python scripts/eval_gait.py --checkpoint-file logs/.../model_2500.pt
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import tyro

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rsl_rl.runners import OnPolicyRunner
from mjlab.utils.checkpoint import nominal_eval_cfg

# name, vx, vy, wz, gait frequency [Hz]
SCENARIOS = [
  ("stand 2Hz", 0.0, 0.0, 0.0, 2.0),
  ("fwd 0.3 2Hz", 0.3, 0.0, 0.0, 2.0),
  ("fwd 0.6 2Hz", 0.6, 0.0, 0.0, 2.0),
  ("fwd 1.0 2Hz", 1.0, 0.0, 0.0, 2.0),
  ("fwd 0.6 1Hz", 0.6, 0.0, 0.0, 1.0),
  ("fwd 0.6 3Hz", 0.6, 0.0, 0.0, 3.0),
  ("fwd 1.0 3Hz", 1.0, 0.0, 0.0, 3.0),
  ("back 0.5 2Hz", -0.5, 0.0, 0.0, 2.0),
  ("side 0.5 2Hz", 0.0, 0.5, 0.0, 2.0),
  ("turn 1.0 2Hz", 0.0, 0.0, 1.0, 2.0),
  ("diag 2Hz", 0.5, 0.3, 0.5, 2.0),
]
FOOT_RADIUS = 0.022
SHORT = 3  # policy steps (60 ms): shorter air gaps are bounces, not swings


@dataclass
class GaitEvalConfig:
  checkpoint_file: str
  seconds: float = 12.0
  settle: float = 3.0
  """Seconds ignored at the start (getting into the gait)."""
  robots_per_scenario: int = 3
  device: str = "cpu"
  seed: int = 0


def _segments(contact: np.ndarray) -> tuple[int, int]:
  """(swings, bounces) in one foot's contact series, partial first/last segments dropped."""
  x = contact.astype(int)
  segs = np.split(x, np.flatnonzero(np.diff(x)) + 1)[1:-1]
  air = [len(s) for s in segs if s[0] == 0]
  return sum(n >= SHORT for n in air), sum(n < SHORT for n in air)


def run(cfg: GaitEvalConfig) -> dict[str, np.ndarray]:
  checkpoint = torch.load(cfg.checkpoint_file, weights_only=False, map_location="cpu")
  train_cfg = deepcopy(checkpoint["train_cfg"])
  env_cfg = nominal_eval_cfg(checkpoint["env_cfg"])
  env_cfg.scene.num_envs = len(SCENARIOS) * cfg.robots_per_scenario
  env_cfg.seed = cfg.seed
  env_cfg.episode_length_s = 1e6
  twist_cfg = env_cfg.commands["twist"]
  twist_cfg.resampling_time_range = (1e9, 1e9)
  twist_cfg.heading_command = False
  twist_cfg.ranges.heading = None
  twist_cfg.rel_standing_envs = 0.0
  twist_cfg.init_velocity_prob = 0.0
  twist_cfg.axis_nonzero_prob = None
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device), clip_actions=train_cfg.get("clip_actions")
  )
  runner = OnPolicyRunner(env, train_cfg, device=cfg.device)
  runner.load(cfg.checkpoint_file, load_optimizer=False, map_location=cfg.device)
  policy = runner.get_inference_policy(cfg.device)

  base = env.unwrapped
  robot = base.scene["robot"]
  twist = base.command_manager.get_term("twist")
  gait = base.command_manager.get_term("gait")
  feet = base.scene["feet_ground_contact"]
  site_ids, _ = robot.find_sites(list(gait.cfg.leg_names), preserve_order=True)
  rep = cfg.robots_per_scenario
  command = torch.tensor([s[1:4] for s in SCENARIOS for _ in range(rep)], device=cfg.device)
  frequency = torch.tensor([s[4] for s in SCENARIOS for _ in range(rep)], device=cfg.device)

  log: dict[str, list] = {}
  settle = int(cfg.settle / base.step_dt)
  with torch.no_grad():
    for t in range(int(cfg.seconds / base.step_dt)):
      twist.vel_command_b[:] = command
      gait.frequency[:] = frequency
      _, _, dones, _ = env.step(policy(env.get_observations()))
      if t < settle:
        continue
      sample = {
        "done": dones,
        "vel": torch.cat([robot.data.root_link_lin_vel_b[:, :2],
                          robot.data.root_link_ang_vel_b[:, 2:]], dim=-1),
        "tilt": torch.rad2deg(torch.acos(-robot.data.projected_gravity_b[:, 2].clamp(-1, 1))),
        "contact": feet.data.found.flatten(1).bool(),
        "swing": gait.swing_mask,
        "foot_z": robot.data.site_pos_w[:, site_ids, 2] - FOOT_RADIUS,
        "foot_v": robot.data.site_lin_vel_w[:, site_ids, :2].norm(dim=-1),
        "tau": robot.data.actuator_force,
        "qd": robot.data.joint_vel,
        "action": base.action_manager.action,
        "body_z": robot.data.body_link_pos_w[:, 1:, 2],
      }
      for k, v in sample.items():
        log.setdefault(k, []).append(v.clone().cpu())
  env.close()
  out = {k: torch.stack(v).numpy() for k, v in log.items()}
  out["body_names"] = np.array(robot.body_names[1:])
  out["dt"] = np.array(base.step_dt)
  return out


def report(cfg: GaitEvalConfig, L: dict[str, np.ndarray]) -> None:
  dt = float(L["dt"])
  T = L["vel"].shape[0]
  rep = cfg.robots_per_scenario
  freqs = np.fft.rfftfreq(T, dt)
  print(f"\n== {cfg.checkpoint_file}: {T * dt:.0f} s per robot after {cfg.settle:.0f} s settle, "
        f"{rep} robots per row, falls: {int(L['done'].sum())}")
  cols = ("vx", "vy", "wz", "swing/s", "blip/s", "clock", "apex", "slip", "tilt", "power",
          "tremor", "hf", "tau_max")
  print(f"{'scenario':>13} " + " ".join(f"{c:>7}" for c in cols))
  for i, (name, *_cmd) in enumerate(SCENARIOS):
    ids = slice(i * rep, (i + 1) * rep)
    contact = L["contact"][:, ids]
    swings = bounces = 0
    apexes = []
    for r in range(contact.shape[1]):
      for leg in range(4):
        s, b = _segments(contact[:, r, leg])
        swings, bounces = swings + s, bounces + b
        air = ~contact[:, r, leg]
        edges = np.flatnonzero(np.diff(air.astype(int))) + 1
        for seg in np.split(np.arange(T), edges):
          if air[seg[0]] and len(seg) >= SHORT:
            apexes.append(L["foot_z"][seg, r, leg].max())
    foot_seconds = contact.shape[1] * 4 * T * dt
    action = L["action"][:, ids]
    spectrum = np.abs(np.fft.rfft(action - action.mean(0), axis=0)) ** 2
    vel = L["vel"][:, ids].mean((0, 1))
    row = (
      *vel,
      swings / foot_seconds,
      bounces / foot_seconds,
      (contact == ~L["swing"][:, ids]).mean(),
      100 * float(np.median(apexes)) if apexes else 0.0,
      (L["foot_v"][:, ids] * contact).sum() / max(contact.sum(), 1),
      L["tilt"][:, ids].mean(),
      np.abs(L["tau"][:, ids] * L["qd"][:, ids]).sum(-1).mean(),
      np.abs(np.diff(action, 2, axis=0)).mean(),
      spectrum[freqs > 6].sum() / spectrum.sum(),
      np.abs(L["tau"][:, ids]).max(),
    )
    print(f"{name:>13} " + " ".join(f"{x:7.2f}" for x in row))
  low = L["body_z"].min((0, 1))
  order = np.argsort(low)[:4]
  print("lowest body origins [m]:", ", ".join(f"{L['body_names'][j]} {low[j]:.3f}" for j in order))


def main(cfg: GaitEvalConfig) -> None:
  report(cfg, run(cfg))


if __name__ == "__main__":
  main(tyro.cli(GaitEvalConfig, config=(tyro.conf.FlagConversionOff,)))
