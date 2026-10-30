import contextlib
import io
import unittest
from dataclasses import asdict
from unittest.mock import patch

import torch

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.terrain_profiles import set_terrain_profile
from mjlab.utils.checkpoint import checkpoint_configs, nominal_eval_cfg, assert_resume_compatible
from mjlab.rsl_rl.modules.student_teacher_wild import StudentTeacherWild
from mjlab.rsl_rl.algorithms.distillation_wild import DistillationWild
from mjlab.utils.gpu import select_gpus

TASK = "Mjlab-Velocity-PGTT-Student-Unitree-Go2"


class AuditRegressionTests(unittest.TestCase):
  def test_snapshot_preserves_overrides_and_does_not_alias(self):
    env = load_env_cfg(TASK)
    env.actions["joint_pos"].scale = 0.17
    agent = asdict(load_rl_cfg(TASK))
    agent["task_id"] = TASK
    restored, config = checkpoint_configs({"env_cfg": env, "train_cfg": agent}, TASK)
    self.assertEqual(restored.actions["joint_pos"].scale, 0.17)
    config["policy"].pop("class_name")
    self.assertIn("class_name", agent["policy"])
    assert_resume_compatible({"env_cfg": env, "train_cfg": agent}, env, agent)
    restored.actions["joint_pos"].scale = 0.2
    with self.assertRaisesRegex(ValueError, "action configuration"):
      assert_resume_compatible({"env_cfg": env, "train_cfg": agent}, restored, agent)
    with self.assertRaises(ValueError):
      checkpoint_configs({}, TASK)

  def test_all_surfaces_and_nominal_eval_are_isolated(self):
    source = load_env_cfg(TASK)
    mix = source.scene.terrain.terrain_generator.sub_terrains
    self.assertIn("random_rough", mix)
    self.assertIn("stairs_up_30", mix)
    self.assertAlmostEqual(sum(x.proportion for x in mix.values()), 1.0)
    nominal = nominal_eval_cfg(source)
    self.assertNotIn("pd_gains", nominal.events)
    self.assertIn("pd_gains", source.events)
    self.assertTrue(all(t.delay_max_lag == 0 for g in nominal.observations.values() for t in g.terms.values()))
    set_terrain_profile(nominal, "flat")
    self.assertEqual(nominal.scene.terrain.terrain_type, "plane")
    self.assertIsNotNone(source.scene.terrain.terrain_generator)

  def test_no_gpu_defaults_to_cpu(self):
    with patch.dict("os.environ", {}, clear=True), patch("torch.cuda.device_count", return_value=0):
      self.assertEqual(select_gpus([0]), (None, 0))

  def test_student_update_handles_short_final_chunk_and_resets_memory(self):
    cfg = asdict(load_rl_cfg(TASK))
    obs = {key: torch.randn(2, size) for key, size in {
      "policy": 54, "proprio_history": 84, "height": 99,
      "height_noisy": 99, "privileged": 53, "critic": 57}.items()}
    from tensordict import TensorDict
    obs = TensorDict(obs, batch_size=[2])
    with contextlib.redirect_stdout(io.StringIO()):
      policy = StudentTeacherWild(obs, cfg["obs_groups"], 12)
    algorithm = DistillationWild(policy, gradient_length=2)
    algorithm.init_storage("distillation", 2, 3, obs, [12])
    with torch.inference_mode():
      for i in range(3):
        algorithm.act(obs)
        algorithm.process_env_step(obs, torch.zeros(2), torch.tensor([i == 1, False]), {})
    before = policy.student.policy[0].weight.detach().clone()
    loss = algorithm.update()
    self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in loss.values()))
    self.assertFalse(torch.equal(before, policy.student.policy[0].weight))
    self.assertTrue(all(p.grad is None for p in policy.teacher.parameters()))
    policy.reset(torch.tensor([True, False]))
    self.assertTrue(torch.all(policy.hidden_states[:, 0] == 0))

  def test_squashed_entropy_penalizes_saturation(self):
    """Brax-style entropy of tanh(z): saturating the mean lowers it, a raw Gaussian
    entropy would not notice (flat_v4 went bang-bang with std pinned at its cap)."""
    from tensordict import TensorDict
    from mjlab.rsl_rl.modules.actor_critic_wild import ActorCriticWild
    obs = TensorDict({k: torch.randn(4096, 5) for k in ("p", "h", "x", "c")}, batch_size=[4096])
    groups = {"policy": ["p"], "height": ["h"], "privileged": ["x"], "critic": ["c"]}
    with contextlib.redirect_stdout(io.StringIO()):
      policy = ActorCriticWild(obs, groups, 12, squashed_entropy=True, noise_std_type="log",
                               init_noise_std=1.0, height_encoder_dims=(8,), privileged_encoder_dims=(8,),
                               height_latent_dim=4, privileged_latent_dim=4)
    policy.update_distribution(obs)
    mean = policy.distribution.mean.detach()
    torch.manual_seed(0)
    policy.distribution = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    centred = policy.entropy.mean()
    policy.distribution = torch.distributions.Normal(torch.full_like(mean, 3.0), torch.ones_like(mean))
    saturated = policy.entropy.mean()
    # Per dimension: 0.67 nats centred vs -3.2 saturated (raw Gaussian: 1.42 for both).
    self.assertAlmostEqual(centred.item() / 12, 0.669, delta=0.02)
    self.assertLess(saturated.item() / 12, -3.0)
    # On average (one sample per action, as in Brax) the entropy gradient pulls a
    # saturated mean back towards 0: d/dmu ~ -2 per dimension.
    mu = torch.full_like(mean, 3.0).requires_grad_()
    policy.distribution = torch.distributions.Normal(mu, torch.ones_like(mean))
    policy.entropy.sum().backward()
    self.assertAlmostEqual(mu.grad.mean().item(), -1.94, delta=0.05)


if __name__ == "__main__":
  unittest.main()
