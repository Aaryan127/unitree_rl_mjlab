"""RL configuration for Unitree Go2 velocity task."""

from mjlab.rl import (
  RslRlDistillationRunnerCfg,
  RslRlDistillationWildAlgorithmCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoActorCriticWildCfg,
  RslRlPpoAlgorithmCfg,
  RslRlStudentTeacherWildCfg,
)


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


##
# PGTT teacher-student.
##

# Which environment observation groups each network input uses.
# (Groups are defined in velocity_env_cfg.make_pgtt_env_cfg.)
PGTT_TEACHER_OBS_GROUPS = {
  "policy": ("policy", "proprio_history"),
  "height": ("height",),
  "privileged": ("privileged",),
  "critic": ("critic", "height", "privileged"),
}
PGTT_STUDENT_OBS_GROUPS = {
  "policy": ("policy",),
  "student_height": ("height_noisy",),
  "teacher": PGTT_TEACHER_OBS_GROUPS["policy"],
  "teacher_height": PGTT_TEACHER_OBS_GROUPS["height"],
  "teacher_privileged": PGTT_TEACHER_OBS_GROUPS["privileged"],
}


def unitree_go2_pgtt_teacher_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Stage 1: PPO teacher with clean heights + privileged state (PGTT rewards)."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticWildCfg(
      init_noise_std=1.0,
      # The first flat_v2 teacher's std grew 1 -> 2.5 (noise past the joint limits is
      # free), so the penalties under that noise outgrew the task reward. Can only shrink.
      max_noise_std=1.0,
      actor_hidden_dims=(512, 256, 128),  # PGTT
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups=PGTT_TEACHER_OBS_GROUPS,
    experiment_name="go2_pgtt_teacher",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10000,
  )


def unitree_go2_pgtt_student_runner_cfg() -> RslRlDistillationRunnerCfg:
  """Stage 2: distill the teacher into the deployable recurrent student."""
  return RslRlDistillationRunnerCfg(
    policy=RslRlStudentTeacherWildCfg(
      actor_hidden_dims=(512, 256, 128),  # Must match the teacher.
      activation="elu",
    ),
    algorithm=RslRlDistillationWildAlgorithmCfg(),
    obs_groups=PGTT_STUDENT_OBS_GROUPS,
    experiment_name="go2_pgtt_student",
    teacher_experiment_name="go2_pgtt_teacher",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5000,
  )
