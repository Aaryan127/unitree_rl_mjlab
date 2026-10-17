"""Student-teacher module for distillation (stage 2 of PGTT teacher-student training).

- teacher: the TeacherMLP trained with PPO (ActorCriticWild.actor), frozen.
- student: StudentPolicyWild (belief encoder + GRU), trained by DistillationWild to
  copy the teacher's actions and to reconstruct the clean height map / privileged state.

`obs_groups` keys:

    "policy"             -> student proprioception        e.g. ["policy"]
    "student_height"     -> noisy height map              e.g. ["height_noisy"]
    "teacher"            -> teacher proprioception        e.g. ["policy", "proprio_history"]
    "teacher_height"     -> clean height map              e.g. ["height"]
    "teacher_privileged" -> privileged state              e.g. ["privileged"]

The GRU hidden state is handled here (not with rsl_rl's Memory) so it can be reset and
detached without in-place edits of tensors that are part of the autograd graph.
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn
from torch.distributions import Normal

from mjlab.rsl_rl.networks.belief_encoder import StudentPolicyWild
from mjlab.rsl_rl.networks.teacher_mlp import TeacherMLP


def _group_dim(obs, groups) -> int:
  return sum(obs[group].shape[-1] for group in groups)


class StudentTeacherWild(nn.Module):
  is_recurrent = True

  def __init__(
    self,
    obs,
    obs_groups,
    num_actions,
    # Teacher architecture: must match the ActorCriticWild that was trained.
    actor_obs_normalization=True,
    actor_hidden_dims=(512, 256, 128),
    activation="elu",
    height_latent_dim: int = 32,
    privileged_latent_dim: int = 16,
    height_encoder_dims=(128, 64),
    privileged_encoder_dims=(64, 32),
    # Student.
    student_obs_normalization=True,
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
    belief_dims=(64, 64),
    init_noise_std: float = 0.1,
    **kwargs,
  ):
    unused = {k: v for k, v in kwargs.items() if k not in ("critic_obs_normalization", "critic_hidden_dims", "noise_std_type")}
    if unused:
      print(f"StudentTeacherWild ignores unexpected arguments: {list(unused.keys())}")
    super().__init__()
    for key in ("policy", "student_height", "teacher", "teacher_height", "teacher_privileged"):
      if key not in obs_groups:
        raise ValueError(f"StudentTeacherWild needs obs_groups['{key}'], got {obs_groups}")
    self.obs_groups = obs_groups
    self.loaded_teacher = False

    student_proprio_dim = _group_dim(obs, obs_groups["policy"])
    student_height_dim = _group_dim(obs, obs_groups["student_height"])
    teacher_proprio_dim = _group_dim(obs, obs_groups["teacher"])
    height_dim = _group_dim(obs, obs_groups["teacher_height"])
    privileged_dim = _group_dim(obs, obs_groups["teacher_privileged"])
    assert student_height_dim == height_dim, "Noisy and clean height maps must match."
    print(
      f"[StudentTeacherWild] student proprio={student_proprio_dim} height={student_height_dim} | "
      f"teacher proprio={teacher_proprio_dim} privileged={privileged_dim}"
    )

    self.teacher = TeacherMLP(
      teacher_proprio_dim,
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
    self.teacher.requires_grad_(False)
    self.teacher.eval()

    self.student = StudentPolicyWild(
      student_proprio_dim,
      student_height_dim,
      privileged_dim,
      num_actions,
      hidden_dims=actor_hidden_dims,
      height_latent_dim=height_latent_dim,
      privileged_latent_dim=privileged_latent_dim,
      height_encoder_dims=height_encoder_dims,
      rnn_hidden_dim=rnn_hidden_dim,
      rnn_num_layers=rnn_num_layers,
      belief_dims=belief_dims,
      activation=activation,
      obs_normalization=student_obs_normalization,
    )
    print(f"Student: {self.student}")

    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=False)
    self.hidden_states: torch.Tensor | None = None
    self._rnn_out: torch.Tensor | None = None
    self.distribution = None
    Normal.set_default_validate_args(False)

  # -- observation helpers

  def _cat(self, obs, key: str) -> torch.Tensor:
    return torch.cat([obs[group] for group in self.obs_groups[key]], dim=-1)

  # -- student forward

  def forward_student(self, obs) -> torch.Tensor:
    proprio = self._cat(obs, "policy")
    height = self._cat(obs, "student_height")
    actions, self.hidden_states, self._rnn_out = self.student(
      proprio, height, self.hidden_states
    )
    return actions

  def reconstruction_loss(self, obs) -> torch.Tensor:
    """MSE between decoded and true (clean height, privileged). Call after forward_student."""
    assert self._rnn_out is not None
    recon_height, recon_privileged = self.student.decoder(
      self._rnn_out, self._cat(obs, "student_height")
    )
    true_height = self._cat(obs, "teacher_height")
    true_privileged = self.teacher.privileged_normalizer(self._cat(obs, "teacher_privileged"))
    return nn.functional.mse_loss(recon_height, true_height) + nn.functional.mse_loss(
      recon_privileged, true_privileged
    )

  # -- interface used by Distillation / DistillationRunner

  def reset(self, dones=None, hidden_states=None):
    if dones is None:
      self.hidden_states = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    elif self.hidden_states is not None:
      keep = (dones != 1).to(self.hidden_states.dtype).view(1, -1, 1)
      self.hidden_states = self.hidden_states * keep

  def get_hidden_states(self):
    return self.hidden_states, None

  def detach_hidden_states(self, dones=None):
    if self.hidden_states is None:
      return
    if dones is None:
      self.hidden_states = self.hidden_states.detach()
    else:
      done = (dones == 1).view(1, -1, 1)
      self.hidden_states = torch.where(done, self.hidden_states.detach(), self.hidden_states)

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

  def act(self, obs):
    mean = self.forward_student(obs)
    self.distribution = Normal(mean, self.std.expand_as(mean))
    return self.distribution.sample()

  def act_inference(self, obs):
    return self.forward_student(obs)

  def evaluate(self, obs):
    """Teacher actions (labels for distillation)."""
    with torch.no_grad():
      return self.teacher(
        self._cat(obs, "teacher"),
        self._cat(obs, "teacher_height"),
        self._cat(obs, "teacher_privileged"),
      )

  def update_normalization(self, obs):
    self.student.update_normalization(self._cat(obs, "policy"))

  def train(self, mode=True):
    super().train(mode)
    self.teacher.eval()  # The teacher is frozen, its normalizers must not update.
    return self

  def load_state_dict(self, state_dict, strict=True):
    """Load either a teacher checkpoint (from PPO) or a student checkpoint (resume).

    Returns True if training resumes (student checkpoint), False for a teacher.
    """
    if any(key.startswith("actor.") for key in state_dict):
      teacher_state = {
        key.removeprefix("actor."): value
        for key, value in state_dict.items()
        if key.startswith("actor.")
      }
      self.teacher.load_state_dict(teacher_state, strict=strict)
      # Start the student's height encoder from the teacher's: same input, same latent.
      self.student.height_encoder.load_state_dict(self.teacher.height_encoder.state_dict())
      self.loaded_teacher = True
      self.teacher.eval()
      print("[StudentTeacherWild] Loaded teacher weights.")
      return False
    if any(key.startswith("student.") for key in state_dict):
      super().load_state_dict(state_dict, strict=strict)
      self.loaded_teacher = True
      self.teacher.eval()
      return True
    raise ValueError("state_dict has neither teacher ('actor.*') nor student ('student.*') keys.")

  # -- deployment

  @property
  def student_obs_dim(self) -> int:
    return self.student.proprio_dim + self.student.height_dim

  def export_onnx(self, path: str, filename: str = "policy.onnx") -> None:
    """Export the deployable student.

    inputs:  obs   [1, proprio + height]  (deploy.yaml observation order)
             h_in  [rnn_layers, 1, rnn_hidden]
    outputs: actions [1, num_actions], h_out [rnn_layers, 1, rnn_hidden]
    The C++ runner feeds h_out back as h_in on the next step.
    """
    os.makedirs(path, exist_ok=True)
    model = StudentOnnxWrapper(self.student).to("cpu").eval()
    h_in = torch.zeros(self.student.rnn_num_layers, 1, self.student.rnn_hidden_dim)
    torch.onnx.export(
      model,
      (torch.zeros(1, self.student_obs_dim), h_in),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      input_names=["obs", "h_in"],
      output_names=["actions", "h_out"],
      dynamo=False,
    )


class StudentOnnxWrapper(nn.Module):
  """Single-step deployable student: (obs, h_in) -> (actions, h_out)."""

  def __init__(self, student: StudentPolicyWild):
    super().__init__()
    self.student = copy.deepcopy(student)
    self.proprio_dim = student.proprio_dim

  def forward(self, obs: torch.Tensor, h_in: torch.Tensor):
    proprio = obs[:, : self.proprio_dim]
    height = obs[:, self.proprio_dim :]
    actions, h_out, _ = self.student(proprio, height, h_in)
    return actions, h_out
