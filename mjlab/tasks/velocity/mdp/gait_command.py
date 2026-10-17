"""Gait phase generator for PGTT (Phase-Guided Terrain Traversal, Ntagkas et al. 2025).

Every leg i has a phase phi_i in [0, 2*pi) that advances at the gait frequency f:

    phi_i(t) = (phi_0 + offset_i + 2*pi*f*t) mod 2*pi

    phi in [0, pi)      -> stance (foot should be on the ground)
    phi in [pi, 2*pi)   -> swing  (foot should follow the swing-height spline)

The phase is not an action: it is a clock the rewards use to shape the gait, and the
policy observes it as [cos(phi_i), sin(phi_i), f]. On the real robot the same clock is
computed in C++ (`gait_phase_pgtt` in deploy/.../observations.h), so the ordering of the
legs and of the output vector must stay identical in both places.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class GaitPhaseCommand(CommandTerm):
  cfg: GaitPhaseCommandCfg

  def __init__(self, cfg: GaitPhaseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if len(cfg.leg_names) != len(cfg.leg_offsets):
      raise ValueError("leg_names and leg_offsets must have the same length.")

    self.num_legs = len(cfg.leg_names)
    self.offsets = torch.tensor(cfg.leg_offsets, device=self.device, dtype=torch.float32)
    self.frequency = torch.zeros(self.num_envs, device=self.device)
    self.base_phase = torch.zeros(self.num_envs, device=self.device)

    self.metrics["frequency"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def phase(self) -> torch.Tensor:
    """Per-leg phase in [0, 2*pi). Shape: [num_envs, num_legs]."""
    return torch.remainder(self.base_phase.unsqueeze(1) + self.offsets, 2.0 * math.pi)

  @property
  def swing_mask(self) -> torch.Tensor:
    """True for legs in swing (phase in [pi, 2*pi)). Shape: [num_envs, num_legs]."""
    return self.phase >= math.pi

  @property
  def command(self) -> torch.Tensor:
    """[cos(phi_1..L), sin(phi_1..L), f]. Shape: [num_envs, 2 * num_legs + 1]."""
    phase = self.phase
    return torch.cat(
      [torch.cos(phase), torch.sin(phase), self.frequency.unsqueeze(1)], dim=1
    )

  def _update_metrics(self) -> None:
    self.metrics["frequency"][:] = self.frequency

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Called on episode reset (resampling_time_range is very long), so frequency is
    # constant within an episode and every episode starts at a random point of the cycle.
    r = torch.empty(len(env_ids), device=self.device)
    self.frequency[env_ids] = r.uniform_(*self.cfg.frequency_range)
    if self.cfg.random_initial_phase:
      self.base_phase[env_ids] = r.uniform_(0.0, 2.0 * math.pi)
    else:
      self.base_phase[env_ids] = 0.0

  def _update_command(self) -> None:
    self.base_phase = torch.remainder(
      self.base_phase + 2.0 * math.pi * self.frequency * self._env.step_dt,
      2.0 * math.pi,
    )


@dataclass(kw_only=True)
class GaitPhaseCommandCfg(CommandTermCfg):
  leg_names: tuple[str, ...] = ("FL", "FR", "RL", "RR")
  """Foot site names, in the order used by observations, rewards and the C++ deploy."""

  leg_offsets: tuple[float, ...] = (0.0, math.pi, math.pi, 0.0)
  """Phase offset per leg. Default is a trot: diagonal pairs (FL, RR) and (FR, RL)."""

  frequency_range: tuple[float, float] = (1.0, 3.0)
  """Gait frequency f in Hz, sampled once per episode (PGTT: U[1, 3])."""

  random_initial_phase: bool = True

  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  """Only resample on reset."""

  def build(self, env: ManagerBasedRlEnv) -> GaitPhaseCommand:
    return GaitPhaseCommand(self, env)
