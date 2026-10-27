"""Restore the configuration that produced a policy, including CLI overrides."""

from copy import deepcopy
import warnings


def checkpoint_configs(checkpoint: dict, task: str, *, allow_legacy: bool = False):
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from dataclasses import asdict

  if "env_cfg" in checkpoint and "train_cfg" in checkpoint:
    agent = deepcopy(checkpoint["train_cfg"])
    saved_task = agent.get("task_id")
    if saved_task and saved_task != task:
      raise ValueError(f"Checkpoint task {saved_task!r} differs from requested {task!r}")
    return deepcopy(checkpoint["env_cfg"]), agent
  if not allow_legacy:
    raise ValueError(
      "Checkpoint has no saved environment/training configuration. Retrain with config "
      "snapshots, or explicitly use --allow-legacy-config True after checking the old "
      "run's params against the current task. CLI overrides cannot be recovered automatically."
    )
  warnings.warn("Legacy checkpoint: using current task defaults; training overrides are unknown.",
                stacklevel=2)
  return load_env_cfg(task), asdict(load_rl_cfg(task))


def nominal_eval_cfg(cfg):
  """Remove training perturbations, retaining observation/action conventions."""
  cfg = deepcopy(cfg)
  cfg.curriculum = {}
  for group in cfg.observations.values():
    group.enable_corruption = False
    for term in group.terms.values():
      term.delay_min_lag = term.delay_max_lag = 0
      if "point_noise_std" in term.params:
        term.params["point_noise_std"] = 0.0
  perturbations = {
    "push_robot", "external_wrench", "body_friction", "base_mass", "base_com",
    "pd_gains", "effort_limits", "actuator_delay", "link_mass", "joint_friction",
    "joint_armature", "encoder_bias",
  }
  for name in perturbations:
    cfg.events.pop(name, None)
  if "reset_base" in cfg.events:
    cfg.events["reset_base"].params["velocity_range"] = {}
  if "reset_robot_joints" in cfg.events:
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
  for entity in cfg.scene.entities.values():
    if entity.articulation:
      for actuator in entity.articulation.actuators:
        if hasattr(actuator, "delay_max_lag"):
          actuator.delay_min_lag = actuator.delay_max_lag = 0
  if cfg.scene.terrain and cfg.scene.terrain.terrain_generator:
    cfg.scene.terrain.terrain_generator.curriculum = False
  return cfg


class InferencePolicy:
  """Callable policy with an explicit recurrent-state reset for viewers/evaluators."""

  def __init__(self, module):
    self.module = module

  def __call__(self, obs):
    return self.module.act_inference(obs)

  def reset(self, dones=None):
    self.module.reset(dones)


def assert_resume_compatible(checkpoint, env_cfg, train_cfg, *, allow_legacy=False):
  """Reject changes to a resumed policy's inputs, action meaning, or architecture."""
  if "env_cfg" not in checkpoint or "train_cfg" not in checkpoint:
    if not allow_legacy:
      raise ValueError("Resume checkpoint has no config snapshot; explicitly allow legacy config to proceed")
    warnings.warn("Legacy resume: input/action compatibility could not be checked", stacklevel=2)
    return
  from dataclasses import fields, is_dataclass
  import torch

  def freeze(value):
    if is_dataclass(value):
      return tuple((f.name, freeze(getattr(value, f.name))) for f in fields(value))
    if isinstance(value, dict):
      return tuple((k, freeze(v)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
      return tuple(freeze(v) for v in value)
    if isinstance(value, torch.Tensor):
      return freeze(value.detach().cpu().tolist())
    if callable(value):
      return value.__module__, value.__qualname__
    return value

  old = checkpoint["env_cfg"]
  comparisons = {
    "policy architecture": (checkpoint["train_cfg"]["policy"], train_cfg["policy"]),
    "observation routing": (checkpoint["train_cfg"]["obs_groups"], train_cfg["obs_groups"]),
    "action configuration": (old.actions, env_cfg.actions),
    "observation configuration": (old.observations, env_cfg.observations),
    "robot configuration": (old.scene.entities, env_cfg.scene.entities),
  }
  for name, (saved, current) in comparisons.items():
    if freeze(saved) != freeze(current):
      raise ValueError(f"Resume changed {name}; use the original training settings or start a new run")
