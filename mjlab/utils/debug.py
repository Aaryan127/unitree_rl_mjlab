"""Opt-in debugging hooks: prints, NaN checks and pdb stops inside the training loop.

This is the cpu branch's debugging style (prints and ipdb stops inside the managers),
made safe: everything is OFF unless you set an environment variable, and when off
each hook costs one boolean check. Works with any script (train.py, play.py,
debug_env.py) because it is configured through environment variables.

  MJLAB_DEBUG=1
      Print a summary of every observation term and every reward term (shape,
      min/max/mean) on the first step, the observation wiring when a runner starts,
      and a reward-balance table every MJLAB_DEBUG_EVERY steps (default 1000).
  MJLAB_DEBUG_NAN=1
      Check every observation and reward term every step; on NaN/Inf, print which
      term and which robots, then stop in pdb.
  MJLAB_BREAK=<target>[,<target>...]
      Stop in pdb at a hook. Targets:
        step                       start of env.step()
        obs:<group>                after any term of an observation group
        obs:<group>/<term>         after one observation term
        reward:<term>              after one reward term
        reward                     after any reward term
        update                     after each learning update (losses ready)
  MJLAB_BREAK_AFTER=<n>   skip the first n matching hits (default 0)
  MJLAB_BREAK_COUNT=<n>   stop at most n times (default 1, 0 = every time)
  MJLAB_DEBUG_ENV=<i>     which robot's values to print (default 0)

Examples:
  MJLAB_BREAK=reward:foot_phase python scripts/debug_env.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2
  MJLAB_BREAK=obs:height_noisy MJLAB_BREAK_AFTER=100 python scripts/debug_env.py ... --steps 200
  MJLAB_DEBUG=1 MJLAB_DEBUG_NAN=1 python scripts/train.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2 ...

In the pdb prompt: the variables `env`, `name`, `value` (and `group` for observations)
describe where you stopped; type `u` to go up into the manager that called the term.
"""

from __future__ import annotations

import os
from typing import Any

import torch

DEBUG = os.environ.get("MJLAB_DEBUG", "0") not in ("", "0")
CHECK_NAN = os.environ.get("MJLAB_DEBUG_NAN", "0") not in ("", "0")
BREAK_TARGETS = tuple(
  t.strip() for t in os.environ.get("MJLAB_BREAK", "").split(",") if t.strip()
)
BREAK_AFTER = int(os.environ.get("MJLAB_BREAK_AFTER", "0"))
BREAK_COUNT = int(os.environ.get("MJLAB_BREAK_COUNT", "1"))
PRINT_EVERY = int(os.environ.get("MJLAB_DEBUG_EVERY", "1000"))
ENV_ID = int(os.environ.get("MJLAB_DEBUG_ENV", "0"))

ENABLED = DEBUG or CHECK_NAN or bool(BREAK_TARGETS)
"""Single flag the hooks in the managers check. False -> hooks are skipped."""

_hits = 0
_stops = 0
_printed_first_step = set()
_reward_sums: dict[str, float] = {}
_reward_steps = 0


def _stats(x: torch.Tensor) -> str:
  x = x.detach().float()
  return f"min {x.min().item():+9.3f} max {x.max().item():+9.3f} mean {x.mean().item():+9.3f}"


def _matches(kind: str, group: str | None, name: str) -> bool:
  for target in BREAK_TARGETS:
    if target == kind:
      return True
    if kind == "obs" and group is not None:
      if target in (f"obs:{group}", f"obs:{group}/{name}"):
        return True
    elif target == f"{kind}:{name}":
      return True
  return False


def _maybe_break(kind: str, group: str | None, name: str, message: str) -> bool:
  """True if the caller should stop in pdb now."""
  global _hits, _stops
  if not _matches(kind, group, name):
    return False
  _hits += 1
  if _hits <= BREAK_AFTER:
    return False
  if BREAK_COUNT and _stops >= BREAK_COUNT:
    return False
  _stops += 1
  print(f"\n[mjlab.debug] STOP {message}")
  return True


def _nan_report(kind: str, label: str, value: torch.Tensor) -> bool:
  bad = ~torch.isfinite(value)
  if not bool(bad.any()):
    return False
  rows = bad.reshape(bad.shape[0], -1).any(dim=1).nonzero().flatten().tolist()
  print(f"\n[mjlab.debug] NaN/Inf in {kind} '{label}' for robots {rows[:20]}")
  return True


def step_hook(env: Any, action: torch.Tensor) -> None:
  """Called at the start of env.step()."""
  if DEBUG and env.common_step_counter > 0 and env.common_step_counter % PRINT_EVERY == 0:
    print_reward_balance()
  if BREAK_TARGETS and _maybe_break("step", None, "step", f"at env.step(), step {env.common_step_counter}"):
    print("  variables: env, action")
    breakpoint()


def observation_hook(env: Any, group: str, name: str, value: torch.Tensor) -> None:
  """Called after each observation term is computed (after noise/clip/scale/history)."""
  if DEBUG and (group, name) not in _printed_first_step:
    _printed_first_step.add((group, name))
    print(
      f"[mjlab.debug] obs {group:16s}/{name:24s} {tuple(value.shape)!s:12s} {_stats(value)}"
      f"  env{ENV_ID}: {value[ENV_ID].flatten()[:4].tolist()}"
    )
  stop = CHECK_NAN and _nan_report("observation", f"{group}/{name}", value)
  if BREAK_TARGETS and _maybe_break("obs", group, name, f"after observation {group}/{name}"):
    stop = True
  if stop:
    print(f"  variables: env, group, name, value {tuple(value.shape)}; 'u' = observation manager")
    breakpoint()


def reward_hook(env: Any, name: str, value: torch.Tensor, scale: float) -> None:
  """Called after each reward term (value = weight * term * dt)."""
  global _reward_steps
  if DEBUG:
    if name not in _printed_first_step:
      _printed_first_step.add(name)
      print(f"[mjlab.debug] reward (first value) {name:24s} (weighted, per second) {_stats(value / scale)}")
    _reward_sums[name] = _reward_sums.get(name, 0.0) + float((value / scale).mean())
    if name == next(iter(_reward_sums)):
      _reward_steps += 1
  stop = CHECK_NAN and _nan_report("reward", name, value)
  if BREAK_TARGETS and _maybe_break("reward", None, name, f"after reward {name}, step {env.common_step_counter}"):
    stop = True
  if stop:
    print(f"  variables: env, name, value {tuple(value.shape)} (= weight * term * dt); 'u' = reward manager")
    breakpoint()


def update_hook(loss_dict: dict, iteration: int) -> None:
  """Called after each learning update."""
  if CHECK_NAN:
    bad = {k: v for k, v in loss_dict.items() if v != v or abs(v) == float("inf")}
    if bad:
      print(f"\n[mjlab.debug] NaN/Inf loss at iteration {iteration}: {bad}")
      breakpoint()
  if BREAK_TARGETS and _maybe_break("update", None, "update", f"after update, iteration {iteration}"):
    print("  variables: loss_dict, iteration; 'u' = runner (self.alg, self.env)")
    breakpoint()


def print_reward_balance() -> None:
  """Average weighted reward per second of each term since the last print."""
  global _reward_steps
  if not _reward_sums or _reward_steps == 0:
    return
  print(f"[mjlab.debug] reward balance over last {_reward_steps} steps (weighted, per second):")
  total = sum(abs(v) for v in _reward_sums.values()) or 1.0
  for name, s in sorted(_reward_sums.items(), key=lambda kv: -abs(kv[1])):
    mean = s / _reward_steps
    share = 100.0 * abs(s) / total
    print(f"    {name:24s} {mean:+10.4f}  ({share:5.1f}% of |total|)")
  _reward_sums.clear()
  _reward_steps = 0


def print_obs_wiring(obs: Any, obs_groups: dict) -> None:
  """Observation groups from the env and which network input reads which."""
  if not DEBUG:
    return
  print("[mjlab.debug] observation groups from the environment:")
  for key in obs.keys():
    print(f"    {key:16s} {tuple(obs[key].shape)}  {obs[key].dtype}")
  print("[mjlab.debug] network inputs (obs_groups):")
  for key, groups in obs_groups.items():
    dim = sum(obs[g].shape[-1] for g in groups)
    print(f"    {key:20s} <- {list(groups)}  = {dim}")
