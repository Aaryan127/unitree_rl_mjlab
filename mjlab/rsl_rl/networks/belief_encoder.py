"""Student network with a recurrent belief encoder (Miki et al. 2022, Fig. 3).

The student only gets what the real robot has: proprioception + gait phase and a
noisy height map. A GRU keeps memory over time and estimates a "belief" that replaces
the teacher's clean height latent and privileged latent:

    noisy height ──> height encoder ──> e (32) ──────────────┐ (gated)
                                         │                    v
    proprio ─────────────────────────────┴──> GRU ──> h ──> belief = [state(h) + sigmoid(gate(h)) * e,
                                                                        privileged_state(h)]
    proprio + belief ──> policy MLP ──> actions

The gate lets the network ignore the height map when it is unreliable (noise, drift,
missing data) and rely on memory instead. A decoder reconstructs the clean height map
and the privileged state from h; that reconstruction loss forces h to really contain
the information the teacher had.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mjlab.rsl_rl.networks.mlp import MLP
from mjlab.rsl_rl.networks.normalization import EmpiricalNormalization


class BeliefEncoder(nn.Module):
  def __init__(
    self,
    proprio_dim: int,
    height_latent_dim: int,
    privileged_latent_dim: int,
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
    belief_dims=(64, 64),
    activation: str = "elu",
  ):
    super().__init__()
    self.rnn = nn.GRU(proprio_dim + height_latent_dim, rnn_hidden_dim, rnn_num_layers)
    self.gate = MLP(rnn_hidden_dim, height_latent_dim, belief_dims, activation)
    self.height_state = MLP(rnn_hidden_dim, height_latent_dim, belief_dims, activation)
    self.privileged_state = MLP(
      rnn_hidden_dim, privileged_latent_dim, belief_dims, activation
    )

  def forward(
    self,
    proprio: torch.Tensor,
    height_latent: torch.Tensor,
    hidden: torch.Tensor | None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One time step. proprio [B, P], height_latent [B, E], hidden [layers, B, H].

    Returns (belief [B, E + priv_latent], next hidden, rnn output [B, H]).
    """
    x = torch.cat([proprio, height_latent], dim=-1).unsqueeze(0)
    out, hidden = self.rnn(x, hidden)
    out = out.squeeze(0)
    alpha = torch.sigmoid(self.gate(out))
    belief_height = self.height_state(out) + alpha * height_latent
    belief = torch.cat([belief_height, self.privileged_state(out)], dim=-1)
    return belief, hidden, out


class BeliefDecoder(nn.Module):
  """Reconstructs the clean height map and privileged state from the GRU output."""

  def __init__(
    self,
    rnn_hidden_dim: int,
    height_dim: int,
    privileged_dim: int,
    decoder_dims=(64, 64),
    activation: str = "elu",
  ):
    super().__init__()
    self.height_gate = MLP(rnn_hidden_dim, height_dim, decoder_dims, activation)
    self.height_decoder = MLP(rnn_hidden_dim, height_dim, decoder_dims, activation)
    self.privileged_decoder = MLP(rnn_hidden_dim, privileged_dim, decoder_dims, activation)

  def forward(
    self, rnn_out: torch.Tensor, noisy_height: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    height = self.height_decoder(rnn_out) + torch.sigmoid(
      self.height_gate(rnn_out)
    ) * noisy_height
    return height, self.privileged_decoder(rnn_out)


class StudentPolicyWild(nn.Module):
  """The full deployable student: normalizer + height encoder + belief encoder + policy.

  Everything the robot runs lives in this module, so exporting it to ONNX exports
  the complete deployed network (decoder excluded, it is only used for training).
  """

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
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
    belief_dims=(64, 64),
    activation: str = "elu",
    obs_normalization: bool = True,
  ):
    super().__init__()
    self.proprio_dim = proprio_dim
    self.height_dim = height_dim
    self.rnn_hidden_dim = rnn_hidden_dim
    self.rnn_num_layers = rnn_num_layers

    self.obs_normalization = obs_normalization
    self.proprio_normalizer = (
      EmpiricalNormalization(proprio_dim) if obs_normalization else nn.Identity()
    )
    self.height_encoder = MLP(height_dim, height_latent_dim, height_encoder_dims, activation)
    self.belief_encoder = BeliefEncoder(
      proprio_dim,
      height_latent_dim,
      privileged_latent_dim,
      rnn_hidden_dim,
      rnn_num_layers,
      belief_dims,
      activation,
    )
    self.policy = MLP(
      proprio_dim + height_latent_dim + privileged_latent_dim,
      num_actions,
      hidden_dims,
      activation,
    )
    self.decoder = BeliefDecoder(
      rnn_hidden_dim, height_dim, privileged_dim, belief_dims, activation
    )

  def forward(
    self,
    proprio: torch.Tensor,
    height: torch.Tensor,
    hidden: torch.Tensor | None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (actions, next hidden, rnn output)."""
    proprio = self.proprio_normalizer(proprio)
    height_latent = self.height_encoder(height)
    belief, hidden, rnn_out = self.belief_encoder(proprio, height_latent, hidden)
    actions = self.policy(torch.cat([proprio, belief], dim=-1))
    return actions, hidden, rnn_out

  def update_normalization(self, proprio: torch.Tensor) -> None:
    if self.obs_normalization:
      self.proprio_normalizer.update(proprio)  # type: ignore[operator]
