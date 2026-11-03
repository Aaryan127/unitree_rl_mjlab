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

  def test_effort_limits_do_not_compound_over_resets(self):
    # Rung 4's effort_limits event (x0.85-1.0, every reset) used to scale the current
    # limit, so the motors weakened with each reset (~11% after 30) and rung 4 collapsed.
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.velocity.config.go2.env_cfgs import unitree_go2_pgtt_rung_env_cfg

    cfg = unitree_go2_pgtt_rung_env_cfg(4)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      for _ in range(20):
        env.reset(env_ids=torch.arange(2))
      upper = env.sim.model.actuator_forcerange[:, :, 1]
      self.assertGreaterEqual(upper.min().item(), 0.85 * 23.5 - 1e-3)
      self.assertLessEqual(upper.max().item(), 23.5 + 1e-3)
    finally:
      env.close()


  def test_rungs_skip_the_student_lidar_and_reset_nan_physics(self):
    # The student-only LiDAR cost ~75% of rung 7's step time; one NaN robot crashed rung 7.
    from mjlab.tasks.velocity.config.go2.env_cfgs import unitree_go2_pgtt_rung_env_cfg

    for rung in (1, 7):
      cfg = unitree_go2_pgtt_rung_env_cfg(rung)
      self.assertNotIn("height_noisy", cfg.observations)
      self.assertFalse(any(s.name == "lidar" for s in cfg.scene.sensors))
      self.assertIn("nan_state", cfg.terminations)
      self.assertIn("physics_exploded", cfg.terminations)
    self.assertIn("height_noisy", load_env_cfg(TASK).observations)  # the student keeps it

  def test_rung7_trains_with_heading_commands(self):
    # Free yaw rates make robots walk in circles, so the terrain curriculum (> 4 m from the
    # start) could barely promote even a perfect walker.
    from mjlab.tasks.velocity.config.go2.env_cfgs import unitree_go2_pgtt_rung_env_cfg

    twist = unitree_go2_pgtt_rung_env_cfg(7).commands["twist"]
    self.assertTrue(twist.heading_command)
    self.assertEqual(twist.rel_heading_envs, 0.8)
    self.assertFalse(unitree_go2_pgtt_rung_env_cfg(6).commands["twist"].heading_command)

  def test_physics_exploded_flags_only_the_blown_up_robot(self):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.velocity import mdp
    from mjlab.tasks.velocity.config.go2.env_cfgs import unitree_go2_pgtt_rung_env_cfg

    cfg = unitree_go2_pgtt_rung_env_cfg(6)
    cfg.scene.num_envs = 3
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      env.reset()
      self.assertFalse(mdp.physics_exploded(env).any())
      robot = env.scene["robot"]
      vel = robot.data.joint_vel.clone()
      vel[1, 0] = 500.0  # robot 1's first joint spins at 500 rad/s
      robot.write_joint_state_to_sim(robot.data.joint_pos.clone(), vel)
      self.assertEqual(mdp.physics_exploded(env).tolist(), [False, True, False])
    finally:
      env.close()

  def test_hard_terrain_grows_with_the_row(self):
    import mujoco
    import numpy as np

    from mjlab.terrains import BoxBouldersTerrainCfg, HfRandomUniformTerrainCfg

    def tallest(cfg, difficulty):
      spec = mujoco.MjSpec()
      spec.worldbody.add_body(name="terrain")
      out = cfg.function(difficulty, spec, np.random.default_rng(0))
      return out, spec

    rough = HfRandomUniformTerrainCfg(
      size=(8.0, 8.0), noise_range=(0.0, 0.10), noise_step=0.01, border_width=0.25,
      downsampled_scale=0.2, scale_with_difficulty=True,
    )
    peaks = []
    for d in (0.0, 1.0):
      _, spec = tallest(rough, d)
      peaks.append(float(np.max(spec.hfields[-1].userdata)) * spec.hfields[-1].size[2])
    self.assertLess(peaks[0], 0.03)  # row 0: nearly flat
    self.assertGreater(peaks[1], 0.06)  # last row: real bumps

    boulders = BoxBouldersTerrainCfg(size=(8.0, 8.0), height_range=(0.03, 0.15))
    out, _ = tallest(boulders, 1.0)
    boxes = out.geometries[1:]  # [0] is the floor
    self.assertEqual(len(boxes), 30)
    for g in boxes:
      x, y, z = g.geom.pos
      self.assertLessEqual(2 * z, 0.15 + 1e-6)  # never taller than the range
      self.assertFalse(abs(x - 4.0) < 1.0 and abs(y - 4.0) < 1.0)  # spawn platform clear
    self.assertTrue(np.allclose(out.origin, (4.0, 4.0, 0.0)))

  def test_rungs_8_and_9(self):
    from mjlab.tasks.velocity.config.go2.env_cfgs import unitree_go2_pgtt_rung_env_cfg

    self.assertNotIn("dynamics", unitree_go2_pgtt_rung_env_cfg(7).observations["privileged"].terms)
    self.assertIn("dynamics", unitree_go2_pgtt_rung_env_cfg(8).observations["privileged"].terms)
    nine = unitree_go2_pgtt_rung_env_cfg(9)
    self.assertIn("boulders", nine.scene.terrain.terrain_generator.sub_terrains)
    self.assertTrue(nine.scene.terrain.terrain_generator.curriculum)
    self.assertGreater(nine.rewards["foot_phase"].params["max_terrain_delta"], 0.15)

  def test_raycast_cache_matches_a_fresh_cast(self):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.velocity.config.go2.env_cfgs import unitree_go2_pgtt_rung_env_cfg

    cfg = unitree_go2_pgtt_rung_env_cfg(6)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      env.reset()
      env.step(torch.zeros(2, 12))
      scan = env.scene["height_scanner"]
      cached = scan.data.hit_pos_w.clone()
      scan._cast_poses = None  # force a new cast
      self.assertTrue(torch.equal(cached, scan.data.hit_pos_w))
      env.step(torch.ones(2, 12))  # the robots move: the cache must not be reused
      self.assertFalse(torch.equal(cached, scan.data.hit_pos_w))
    finally:
      env.close()


if __name__ == "__main__":
  unittest.main()
