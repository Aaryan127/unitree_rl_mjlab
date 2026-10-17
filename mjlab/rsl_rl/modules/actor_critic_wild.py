"""Teacher actor-critic for PPO (stage 1 of PGTT teacher-student training).

Same interface as ActorCritic, so the stock PPO algorithm and OnPolicyRunner work
unchanged. The difference is that the actor is a TeacherMLP that takes three inputs,
selected from the environment observation groups with these `obs_groups` keys:

    "policy"     -> proprioception (+ history)   e.g. ["policy", "proprio_history"]
    "height"     -> clean height map             e.g. ["height"]
    "privileged" -> privileged simulator state   e.g. ["privileged"]
    "critic"     -> value function input         e.g. ["critic", "height", "privileged"]
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.distributions import Normal

from mjlab.rsl_rl.networks import MLP, EmpiricalNormalization
from mjlab.rsl_rl.networks.teacher_mlp import TeacherMLP


def _group_dim(obs, groups) -> int:
  dim = 0
  for group in groups:
    assert len(obs[group].shape) == 2, f"Observation group '{group}' must be 1D."
    dim += obs[group].shape[-1]
  return dim


class ActorCriticWild(nn.Module):
  is_recurrent = False

  def __init__(
    self,
    obs,
    obs_groups,
    num_actions,
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    actor_hidden_dims=(512, 256, 128),
    critic_hidden_dims=(512, 256, 128),
    activation="elu",
    init_noise_std=1.0,
    noise_std_type: str = "scalar",
    height_latent_dim: int = 32,
    privileged_latent_dim: int = 16,
    height_encoder_dims=(128, 64),
    privileged_encoder_dims=(64, 32),
    **kwargs,
  ):
    if kwargs:
      print(f"ActorCriticWild ignores unexpected arguments: {list(kwargs.keys())}")
    super().__init__()
    for key in ("policy", "height", "privileged", "critic"):
      if key not in obs_groups:
        raise ValueError(f"ActorCriticWild needs obs_groups['{key}'], got {obs_groups}")
    self.obs_groups = obs_groups

    proprio_dim = _group_dim(obs, obs_groups["policy"])
    height_dim = _group_dim(obs, obs_groups["height"])
    privileged_dim = _group_dim(obs, obs_groups["privileged"])
    critic_dim = _group_dim(obs, obs_groups["critic"])
    print(
      f"[ActorCriticWild] proprio={proprio_dim} height={height_dim} "
      f"privileged={privileged_dim} critic={critic_dim} actions={num_actions}"
    )

    self.actor = TeacherMLP(
      proprio_dim,
      height_dim,
      privileged_dim,
      num_actions,
      hidden_dims=actor_hidden_dims,
      height_latent_dim=height_latent_dim,
      privileged_latent_dim=privileged_latent_dim,
      height_encoder_dims=height_encoder_dims,
      privileged_encoder_dims=privileged_encoder_dims,
      activation=activation,
      obs_normalization=actor_obs_normalization,
    )
    print(f"Actor (TeacherMLP): {self.actor}")

    self.critic = MLP(critic_dim, 1, critic_hidden_dims, activation)
    self.critic_obs_normalization = critic_obs_normalization
    self.critic_obs_normalizer = (
      EmpiricalNormalization(critic_dim) if critic_obs_normalization else nn.Identity()
    )
    print(f"Critic MLP: {self.critic}")

    self.noise_std_type = noise_std_type
    if noise_std_type == "scalar":
      self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    elif noise_std_type == "log":
      self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
    else:
      raise ValueError(f"Unknown noise_std_type '{noise_std_type}' (scalar | log)")

    self.distribution = None
    Normal.set_default_validate_args(False)

  # -- observation helpers

  def _cat(self, obs, key: str) -> torch.Tensor:
    return torch.cat([obs[group] for group in self.obs_groups[key]], dim=-1)

  def get_actor_obs(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return self._cat(obs, "policy"), self._cat(obs, "height"), self._cat(obs, "privileged")

  def get_critic_obs(self, obs) -> torch.Tensor:
    return self._cat(obs, "critic")

  # -- interface used by PPO / OnPolicyRunner

  def reset(self, dones=None):
    pass

  def forward(self):
    raise NotImplementedError

  @property
  def action_mean(self):
    return self.distribution.mean

  @property
  def action_std(self):
    return self.distribution.stddev

  @property
  def entropy(self):
    return self.distribution.entropy().sum(dim=-1)

  def update_distribution(self, obs):
    mean = self.actor(*self.get_actor_obs(obs))
    if self.noise_std_type == "scalar":
      std = self.std.expand_as(mean)
    else:
      std = torch.exp(self.log_std).expand_as(mean)
    self.distribution = Normal(mean, std)

  def act(self, obs, **kwargs):
    self.update_distribution(obs)
    return self.distribution.sample()

  def act_inference(self, obs):
    return self.actor(*self.get_actor_obs(obs))

  def evaluate(self, obs, **kwargs):
    return self.critic(self.critic_obs_normalizer(self.get_critic_obs(obs)))

  def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)

  def update_normalization(self, obs):
    proprio, _, privileged = self.get_actor_obs(obs)
    self.actor.update_normalization(proprio, privileged)
    if self.critic_obs_normalization:
      self.critic_obs_normalizer.update(self.get_critic_obs(obs))  # type: ignore[operator]

  def load_state_dict(self, state_dict, strict=True):
    super().load_state_dict(state_dict, strict=strict)
    return True  # training resumes

  def export_onnx(self, path: str, filename: str = "teacher.onnx") -> None:
    """Export the teacher actor (for inspection only; it needs privileged inputs)."""
    os.makedirs(path, exist_ok=True)
    actor = _TeacherOnnx(self.actor).to("cpu").eval()
    torch.onnx.export(
      actor,
      (
        torch.zeros(1, actor.teacher.proprio_dim),
        torch.zeros(1, actor.teacher.height_dim),
        torch.zeros(1, actor.teacher.privileged_dim),
      ),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      input_names=["proprio", "height", "privileged"],
      output_names=["actions"],
      dynamo=False,
    )


class _TeacherOnnx(nn.Module):
  def __init__(self, teacher: TeacherMLP):
    super().__init__()
    import copy

    self.teacher = copy.deepcopy(teacher)

  def forward(self, proprio, height, privileged):
    return self.teacher(proprio, height, privileged)
