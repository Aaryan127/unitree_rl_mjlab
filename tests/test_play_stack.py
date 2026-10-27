"""Headless checks for dummy and recurrent trained playback without video."""

import contextlib
import io
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.rsl_rl.modules.student_teacher_wild import StudentTeacherWild
from mjlab.utils.checkpoint import nominal_eval_cfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from play import PlayConfig, run_play

TASK = "Mjlab-Velocity-PGTT-Student-Unitree-Go2"


class OneStepViewer:
  runs = 0

  def __init__(self, env, policy):
    self.env, self.policy = env, policy

  def run(self):
    assert self.env.render_mode is None
    with torch.no_grad():
      action = self.policy(self.env.get_observations())
      obs, rewards, dones, _ = self.env.step(action)
      assert torch.isfinite(rewards).all() and all(torch.isfinite(v).all() for v in obs.values())
      if hasattr(self.policy, "reset"):
        self.policy.reset(dones)
    OneStepViewer.runs += 1


class PlayStackTests(unittest.TestCase):
  def test_dummy_agent_needs_no_checkpoint_and_no_video_renderer(self):
    with contextlib.redirect_stdout(io.StringIO()), patch("play.NativeMujocoViewer", OneStepViewer):
      run_play(TASK, PlayConfig(agent="zero", terrain="flat", num_envs=1, device="cpu", viewer="native"))

  def test_trained_agent_restores_snapshot_without_video(self):
    from tensordict import TensorDict
    cfg = asdict(load_rl_cfg(TASK))
    cfg["task_id"] = TASK
    obs = TensorDict({name: torch.zeros(1, size) for name, size in {
      "policy": 54, "proprio_history": 84, "height": 99,
      "height_noisy": 99, "privileged": 53, "critic": 57}.items()}, [1])
    with contextlib.redirect_stdout(io.StringIO()):
      policy = StudentTeacherWild(obs, cfg["obs_groups"], 12)
    with tempfile.TemporaryDirectory() as temp:
      checkpoint = Path(temp) / "model.pt"
      torch.save(dict(model_state_dict=policy.state_dict(), iter=0, infos=None,
                      train_cfg=cfg, env_cfg=load_env_cfg(TASK)), checkpoint)
      with contextlib.redirect_stdout(io.StringIO()), patch("play.NativeMujocoViewer", OneStepViewer):
        run_play(TASK, PlayConfig(checkpoint_file=str(checkpoint), terrain="flat",
                                 num_envs=1, device="cpu", viewer="native"))

  def test_viewer_resets_recurrent_policy_on_done_and_manual_reset(self):
    from types import SimpleNamespace as S
    from mjlab.viewer.base import BaseViewer

    class Viewer(BaseViewer):
      def setup(self): pass
      def sync_env_to_viewer(self): pass
      def sync_viewer_to_env(self): pass
      def close(self): pass
      def is_running(self): return True

    calls = []

    class Policy:
      def __call__(self, obs): return torch.zeros(2, 12)
      def reset(self, dones=None): calls.append(dones)

    dones = torch.tensor([True, False])
    env = S(cfg=S(viewer=S()), get_observations=lambda: {}, reset=lambda: None,
            step=lambda actions: ({}, torch.zeros(2), dones, {}))
    viewer = Viewer(env, Policy())
    viewer.step_simulation()
    self.assertTrue(torch.equal(calls[0], dones))
    viewer.reset_environment()
    self.assertIsNone(calls[1])


if __name__ == "__main__":
  unittest.main()
