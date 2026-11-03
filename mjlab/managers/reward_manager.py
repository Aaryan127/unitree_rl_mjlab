"""Reward manager for computing reward signals."""

# What this file does
# -------------------
# The environment's config has a dict of reward terms, e.g. {"track_lin_vel":
# RewardTermCfg(func=..., weight=1.0), ...}. Once per policy step (every 0.02 s),
# ManagerBasedRlEnv.step() calls RewardManager.compute(dt), which runs every term's
# function for all robots at once and adds them up into one number per robot: the reward
# PPO learns from. It also keeps per-episode totals for TensorBoard (Episode_Reward/<name>).
#
# For PGTT tasks (only_positive_rewards=True in the env config) compute() also clips each
# step's sum at 0 and adds the `termination` term after the clip. That is the part that
# decides whether falling can ever pay (doc/experiments.md, "How the reward
# works here").

from __future__ import annotations  # type hints are not evaluated at runtime

from copy import deepcopy  # full independent copy of the config dict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from prettytable import PrettyTable  # for the printed table of active terms

from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from mjlab.utils import debug as _debug  # opt-in debug hooks (MJLAB_DEBUG, MJLAB_BREAK, ...)

if TYPE_CHECKING:  # only imported for type checkers, avoids a circular import at runtime
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)  # fields must be given by name: RewardTermCfg(func=..., weight=...)
class RewardTermCfg(ManagerTermBaseCfg):  # the base class adds `params` (extra arguments)
  """Configuration for a reward term."""

  func: Any
  """The callable that computes this reward term's value."""
  # Called as func(env, **params); returns a tensor [num_envs], one raw value per robot
  # (e.g. a squared error, a count of feet, an exp() score between 0 and 1).

  weight: float
  """Weight multiplier for this reward term."""
  # Positive = reward, negative = penalty. Its unit is "per second" because the value is
  # also multiplied by dt (see compute). Weight 0 = the term is skipped entirely.


class RewardManager(ManagerBase):
  """Manages reward computation by aggregating weighted reward terms.

  Reward Scaling Behavior:
    By default, rewards are scaled by the environment step duration (dt). This
    normalizes cumulative episodic rewards across different simulation frequencies.
    The scaling can be disabled via the ``scale_by_dt`` parameter.

    When ``scale_by_dt=True`` (default):
      - ``reward_buf`` (returned by ``compute()``) = raw_value * weight * dt
      - ``_episode_sums`` (cumulative rewards) are scaled by dt
      - ``Episode_Reward/*`` logged metrics are scaled by dt

    When ``scale_by_dt=False``:
      - ``reward_buf`` = raw_value * weight (no dt scaling)

    Regardless of the scaling setting:
      - ``_step_reward`` (via ``get_active_iterable_terms()``) always contains
        the unscaled reward rate (raw_value * weight)
  """

  _env: ManagerBasedRlEnv  # the environment this manager belongs to (set by ManagerBase)

  def __init__(
    self,
    cfg: dict[str, RewardTermCfg],  # the env config's `rewards` dict
    env: ManagerBasedRlEnv,
    *,  # everything after this must be passed by name
    scale_by_dt: bool = True,
    only_positive: bool = False,  # = ManagerBasedRlEnvCfg.only_positive_rewards (True for PGTT)
  ):
    self._term_names: list[str] = list()  # e.g. ["track_lin_vel", "foot_phase", ...]
    self._term_cfgs: list[RewardTermCfg] = list()  # same order as the names
    self._class_term_cfgs: list[RewardTermCfg] = list()  # terms whose func is a class with reset()
    self._scale_by_dt = scale_by_dt
    self._only_positive = only_positive

    self.cfg = deepcopy(cfg)  # own copy, so changing it doesn't change the caller's config
    super().__init__(env=env)  # ManagerBase.__init__ calls _prepare_terms() (bottom of file)
    # Running total of each term over the current episode, per robot: {name: [num_envs]}.
    self._episode_sums = dict()
    for term_name in self._term_names:
      self._episode_sums[term_name] = torch.zeros(
        self.num_envs, dtype=torch.float, device=self.device
      )
    # This step's total reward per robot, [num_envs]. Returned by compute().
    self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    # This step's value of every term per robot, [num_envs, num_terms], without dt
    # (a rate per second). Shown live in the viewer.
    self._step_reward = torch.zeros(
      (self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device
    )

  def __str__(self) -> str:
    # The "Active Reward Terms" table printed when the environment is built.
    msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = "Active Reward Terms"
    table.field_names = ["Index", "Name", "Weight"]
    table.align["Name"] = "l"
    table.align["Weight"] = "r"
    for index, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      table.add_row([index, name, term_cfg.weight])
    msg += table.get_string()
    msg += "\n"
    return msg

  # Properties.

  @property
  def active_terms(self) -> list[str]:
    return self._term_names  # names of all terms, including weight-0 ones

  # Methods.

  def reset(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> dict[str, torch.Tensor]:
    # Called when some robots' episodes end (env_ids = which robots). Logs their episode
    # totals and starts new ones.
    if env_ids is None:
      env_ids = slice(None)  # all robots
    extras = {}
    for key in self._episode_sums.keys():
      # Average over the robots that just ended, of each term's episode total...
      episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
      # ...divided by the MAXIMUM episode length (20 s), not the real one. So
      # Episode_Reward/<name> is "per second of a full episode": short episodes (falls)
      # lower it even if the robot did well while it was up.
      extras["Episode_Reward/" + key] = (
        episodic_sum_avg / self._env.max_episode_length_s
      )
      self._episode_sums[key][env_ids] = 0.0  # those robots start a new total
    for term_cfg in self._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)  # class terms clear their own per-robot memory
    return extras  # goes to TensorBoard via the runner

  def compute(self, dt: float) -> torch.Tensor:
    # The reward for this step, one number per robot. dt = 0.02 s (one policy step).
    self._reward_buf[:] = 0.0  # running total of all terms except `termination`
    after_clip = torch.zeros_like(self._reward_buf)  # `termination`, added after the clip
    scale = dt if self._scale_by_dt else 1.0
    for term_idx, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      if term_cfg.weight == 0.0:  # switched-off term: not even computed (costs nothing)
        self._step_reward[:, term_idx] = 0.0
        continue
      # The term's contribution this step: raw value x weight x 0.02 s, for all robots.
      value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale
      # NaN/Inf can occur from corrupted physics state; zero them to avoid policy crash.
      if _debug.ENABLED:  # opt-in prints / NaN checks / pdb stops (mjlab/utils/debug.py)
        _debug.reward_hook(self._env, name, value, scale)
      value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
      if self._only_positive and name == "termination":
        after_clip += value  # kept aside: it must not be hidden by the clip below
      else:
        self._reward_buf += value
      self._episode_sums[name] += value  # for Episode_Reward/<name>
      self._step_reward[:, term_idx] = value / scale  # the rate per second, for display
    if self._only_positive:
      # THE ZERO CLIP (PGTT). If this step's penalties outweigh its positive terms, the
      # robot gets 0, not a negative number. So living never costs more than dying, which
      # stopped v2 from falling on purpose. The flip side: on a clipped step no term
      # changes anything, so a robot stuck below 0 gets no signal at all.
      self._reward_buf.clamp_(min=0.0)
      # Then the fall penalty, outside the clip, so a fall always costs |weight| x dt
      # (repro/rungs: 1 x 0.02 = 0.02; old teacher: 100 x 0.02 = 2).
      self._reward_buf += after_clip
    return self._reward_buf  # [num_envs]: what PPO stores as this step's reward

  def get_active_iterable_terms(self, env_idx):
    # Every term's current rate for one robot, for the viewer's live reward plot.
    terms = []
    for idx, name in enumerate(self._term_names):
      terms.append((name, [self._step_reward[env_idx, idx].cpu().item()]))
    return terms

  def get_term_cfg(self, term_name: str) -> RewardTermCfg:
    # The live config of one term. Changing its weight here changes training immediately
    # (used by the reward-weight ramp curriculum and in pdb).
    if term_name not in self._term_names:
      raise ValueError(f"Term '{term_name}' not found in active terms.")
    return self._term_cfgs[self._term_names.index(term_name)]

  def _prepare_terms(self):
    # Turns the config dict into the lists used by compute(). Called once, from __init__.
    for term_name, term_cfg in self.cfg.items():
      term_cfg: RewardTermCfg | None
      if term_cfg is None:  # a term set to None in the config is left out
        print(f"term: {term_name} set to None, skipping...")
        continue
      # Resolves the names in its SceneEntityCfg params (joint/body regexes -> indices)
      # and, if func is a class (e.g. foot_phase_tracking_body), creates its instance.
      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      # A class term that remembers something per robot between steps gets reset()
      # calls at episode ends.
      if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
        self._class_term_cfgs.append(term_cfg)
