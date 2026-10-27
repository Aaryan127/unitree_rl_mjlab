"""Task registry system for managing environment registration and creation."""

import os
from copy import deepcopy
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg


@dataclass
class _TaskCfg:
  env_cfg: ManagerBasedRlEnvCfg
  play_env_cfg: ManagerBasedRlEnvCfg
  rl_cfg: RslRlOnPolicyRunnerCfg
  runner_cls: type | None


# Private module-level registry: task_id -> task config.
_REGISTRY: dict[str, _TaskCfg] = {}


def register_mjlab_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  play_env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlOnPolicyRunnerCfg,
  runner_cls: type | None = None,
) -> None:
  """Register an environment task.

  Args:
    task_id: Unique task identifier (e.g., "Mjlab-Velocity-Rough-Unitree-Go1").
    env_cfg: Environment configuration used for training.
    play_env_cfg: Environment configuration in "play" mode.
    rl_cfg: RL runner configuration.
    runner_cls: Optional custom runner class. If None, uses OnPolicyRunner.
  """
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered")
  _REGISTRY[task_id] = _TaskCfg(env_cfg, play_env_cfg, rl_cfg, runner_cls)


def list_tasks() -> list[str]:
  """List all registered task IDs."""
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Load environment configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  Terms listed in the MJLAB_DROP environment variable are removed (see _apply_drops).
  """
  cfg = deepcopy(
    _REGISTRY[task_name].env_cfg if not play else _REGISTRY[task_name].play_env_cfg
  )
  drops = [d.strip() for d in os.environ.get("MJLAB_DROP", "").split(",") if d.strip()]
  if drops:
    _apply_drops(cfg, drops)
  return cfg


def _apply_drops(cfg: ManagerBasedRlEnvCfg, drops: list[str]) -> None:
  """Remove terms from a config, for terminal experiments (tyro can't delete entries).

  MJLAB_DROP="reward:foot_slip,obs:privileged/foot_friction,event:push_robot,sensor:lidar"
  Kinds: reward, obs (<group>/<term>), event, curriculum, termination, sensor.
  Dropping a reward also drops its curriculum term "ramp_<reward>" if there is one.
  """
  tables = {
    "reward": cfg.rewards,
    "event": cfg.events,
    "curriculum": cfg.curriculum,
    "termination": cfg.terminations,
  }
  dropped_sensors = []
  for item in drops:
    kind, _, name = item.partition(":")
    if kind in tables:
      if name not in tables[kind]:
        raise ValueError(f"MJLAB_DROP: no {kind} '{name}'. Available: {sorted(tables[kind])}")
      del tables[kind][name]
      if kind == "reward":
        cfg.curriculum.pop(f"ramp_{name}", None)
    elif kind == "obs":
      group, _, term = name.partition("/")
      if group not in cfg.observations or term not in cfg.observations[group].terms:
        available = [f"{g}/{t}" for g, grp in cfg.observations.items() for t in grp.terms]
        raise ValueError(f"MJLAB_DROP: no obs '{name}'. Available: {available}")
      del cfg.observations[group].terms[term]
    elif kind == "sensor":
      names = [s.name for s in cfg.scene.sensors]
      if name not in names:
        raise ValueError(f"MJLAB_DROP: no sensor '{name}'. Available: {names}")
      cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != name)
      dropped_sensors.append(name)
    else:
      raise ValueError(
        f"MJLAB_DROP: '{item}' should be reward:, obs:<group>/, event:, curriculum:, "
        "termination: or sensor:<name>"
      )
  # A dropped sensor must not still be read by a remaining term.
  users = [
    f"{kind}:{label}"
    for kind, terms in (
      ("reward", cfg.rewards.items()),
      ("termination", cfg.terminations.items()),
      ("obs", [(f"{g}/{n}", t) for g, grp in cfg.observations.items() for n, t in grp.terms.items()]),
    )
    for label, term in terms
    if term.params.get("sensor_name") in dropped_sensors
  ]
  if users:
    raise ValueError(f"MJLAB_DROP: dropped sensor(s) {dropped_sensors} still used by {users}; drop those too.")
  print(f"[MJLAB_DROP] removed: {', '.join(drops)}")


def load_rl_cfg(task_name: str) -> RslRlOnPolicyRunnerCfg:
  """Load RL configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  """
  return deepcopy(_REGISTRY[task_name].rl_cfg)


def load_runner_cls(task_name: str) -> type | None:
  """Load the runner class for a task.

  If None, the default OnPolicyRunner will be used.
  """
  return _REGISTRY[task_name].runner_cls
