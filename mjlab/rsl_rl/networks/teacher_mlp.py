"""Teacher actor network (Miki et al. 2022, teacher policy).

    proprio ──────────────────────────────┐
    height (99) ──> height encoder ──> 32 ├─> policy MLP ──> 12 joint actions
    privileged ──> privileged encoder ─> 16┘

The teacher sees the clean height map and privileged simulator state. The student
(belief_encoder.py) has to infer the same latents from noisy data, so the latent sizes
here define what the student's belief state looks like.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mjlab.rsl_rl.networks.mlp import MLP
from mjlab.rsl_rl.networks.normalization import EmpiricalNormalization


class TeacherMLP(nn.Module):
  def __init__(
    self,
    proprio_dim: int,
    height_dim: int,
    privileged_dim: int,
    num_actions: int,
    hidden_dims=(512, 256, 128),
    height_latent_dim: int = 32,
    privileged_latent_dim: int = 16,
    height_encoder_dims=(128, 64),
    privileged_encoder_dims=(64, 32),
    activation: str = "elu",
    obs_normalization: bool = True,
  ):
    super().__init__()
    self.proprio_dim = proprio_dim
    self.height_dim = height_dim
    self.privileged_dim = privileged_dim

    # Heights are already in [-1, 1] (see mdp.height_scan), so only proprio and
    # privileged observations are normalized.
    self.obs_normalization = obs_normalization
    if obs_normalization:
      self.proprio_normalizer = EmpiricalNormalization(proprio_dim)
      self.privileged_normalizer = EmpiricalNormalization(privileged_dim)
    else:
      self.proprio_normalizer = nn.Identity()
      self.privileged_normalizer = nn.Identity()

    self.height_encoder = MLP(height_dim, height_latent_dim, height_encoder_dims, activation)
    self.privileged_encoder = MLP(
      privileged_dim, privileged_latent_dim, privileged_encoder_dims, activation
    )
    self.policy = MLP(
      proprio_dim + height_latent_dim + privileged_latent_dim,
      num_actions,
      hidden_dims,
      activation,
    )

  def forward(
    self, proprio: torch.Tensor, height: torch.Tensor, privileged: torch.Tensor
  ) -> torch.Tensor:
    proprio = self.proprio_normalizer(proprio)
    privileged = self.privileged_normalizer(privileged)
    latent = torch.cat(
      [self.height_encoder(height), self.privileged_encoder(privileged)], dim=-1
    )
    return self.policy(torch.cat([proprio, latent], dim=-1))

  def update_normalization(self, proprio: torch.Tensor, privileged: torch.Tensor) -> None:
    if self.obs_normalization:
      self.proprio_normalizer.update(proprio)  # type: ignore[operator]
      self.privileged_normalizer.update(privileged)  # type: ignore[operator]
