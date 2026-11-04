"""RL configuration for Unitree Go2 velocity task."""

# What this file does
# -------------------
# The environment config (env_cfgs.py) says WHAT is learned; these runner configs say HOW:
# the network sizes and the PPO (or distillation) settings. Each task in __init__.py pairs
# one env config with one runner config. The runner trains for `max_iterations`; each
# iteration collects `num_steps_per_env` steps from every robot (4096 robots x 20 steps =
# 81920 transitions), then updates the networks on them.
#
# PPO in one paragraph: the actor (policy) outputs a mean action plus Gaussian noise
# (std); the critic predicts how much discounted reward the robot will still collect from
# each state (the "value"). After each collection, the advantage of each action (did it
# turn out better or worse than the critic expected?) is computed, and the actor is moved
# towards better-than-expected actions, but only a little per update (the clip), while an
# entropy bonus keeps it exploring.

from mjlab.rl import (
  RslRlDistillationRunnerCfg,  # runner for stage 2: teacher -> student imitation
  RslRlDistillationWildAlgorithmCfg,  # distillation loss settings (action + reconstruction)
  RslRlOnPolicyRunnerCfg,  # runner for PPO
  RslRlPpoActorCriticCfg,  # plain MLP actor-critic (stock tasks)
  RslRlPpoActorCriticWildCfg,  # PGTT teacher: height + privileged encoders + MLP (Miki 2022)
  RslRlPpoAlgorithmCfg,  # the PPO hyperparameters
  RslRlStudentTeacherWildCfg,  # PGTT student: GRU belief encoder, frozen teacher inside
)


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
  # Stock mjlab Go2 tasks (rough/flat). Not used by PGTT.
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,  # starting exploration noise on each action
      actor_obs_normalization=True,  # scale observations to mean 0 / std 1 (running stats)
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),  # three hidden layers
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,  # weight of the critic's error in the loss
      use_clipped_value_loss=True,  # the critic may not jump far from its old prediction either
      clip_param=0.2,  # the actor's action probabilities may change by at most +-20% per update
      entropy_coef=0.01,  # bonus for keeping the actions random (exploration)
      num_learning_epochs=5,  # passes over each batch of collected data
      num_mini_batches=4,  # each pass split into 4 gradient steps
      learning_rate=1.0e-3,
      schedule="adaptive",  # lr goes down/up to keep the policy change near desired_kl
      gamma=0.99,  # discount: a reward 1 s (50 steps) ahead counts 0.99^50 = 0.6
      lam=0.95,  # GAE lambda: how far advantages look ahead before trusting the critic
      desired_kl=0.01,  # target size of each policy update (only with "adaptive")
      max_grad_norm=1.0,  # gradient clipping, against occasional huge updates
    ),
    experiment_name="go2_velocity",  # logs/rsl_rl/go2_velocity/
    save_interval=100,  # checkpoint every 100 iterations
    num_steps_per_env=24,  # steps per robot per iteration
    max_iterations=10001,
  )


##
# PGTT teacher-student.
##

# Which environment observation groups each network input uses.
# (Groups are defined in velocity_env_cfg.make_pgtt_env_cfg.)
PGTT_TEACHER_OBS_GROUPS = {
  # Actor's main input: IMU, joints, command, gait clock, plus their recent history.
  "policy": ("policy", "proprio_history"),
  "height": ("height",),  # actor's height encoder: the clean 11 x 9 map
  "privileged": ("privileged",),  # actor's privileged encoder: contacts, friction, push force
  "critic": ("critic", "height", "privileged"),  # the critic sees everything, noise-free
}
PGTT_STUDENT_OBS_GROUPS = {
  # Only what the real robot measures: IMU, joints, command, gait clock, plus the
  # accelerometer, motor torque estimates and foot force sensors...
  "policy": ("policy", "student_sensors"),
  "student_height": ("height_noisy",),  # ...and the LiDAR-built, noisy map
  # The frozen teacher inside the student still gets its own inputs, to produce the
  # actions the student imitates.
  "teacher": PGTT_TEACHER_OBS_GROUPS["policy"],
  "teacher_height": PGTT_TEACHER_OBS_GROUPS["height"],
  "teacher_privileged": PGTT_TEACHER_OBS_GROUPS["privileged"],
}


def unitree_go2_pgtt_teacher_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Stage 1: PPO teacher with clean heights + privileged state (PGTT rewards)."""
  # Used by the old Mjlab-Velocity-PGTT-Teacher task (runs v1-v4). The ladder uses
  # unitree_go2_pgtt_repro_runner_cfg below instead.
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
    algorithm=RslRlPpoAlgorithmCfg(  # rsl_rl's usual legged-robot settings (not PGTT's)
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
    experiment_name="go2_pgtt_teacher",  # logs/rsl_rl/go2_pgtt_teacher/
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
    # Defaults: imitate the teacher's actions + reconstruct the map and privileged state.
    algorithm=RslRlDistillationWildAlgorithmCfg(),
    obs_groups=PGTT_STUDENT_OBS_GROUPS,
    experiment_name="go2_pgtt_student",  # logs/rsl_rl/go2_pgtt_student/
    # Where train.py looks for the newest teacher checkpoint if --teacher-checkpoint is not
    # given. The ladder's teachers are in go2_pgtt_repro: pass --teacher-checkpoint for them.
    teacher_experiment_name="go2_pgtt_teacher",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5000,
  )


def unitree_go2_pgtt_repro_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PGTT teacher on the reproduced official setup, with Brax PPO's settings
  (PGTT train.py: gamma 0.97, lr 3e-4 fixed, 8 epochs, clip 0.3, entropy 0.01 on the
  tanh-squashed action, unroll 20, value loss 0.5, initial std softplus(0) ~ 0.7)."""
  # Used by the Repro, Repro-Smooth and Rung1-7 tasks (the ladder), experiment go2_pgtt_repro.
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticWildCfg(
      init_noise_std=0.7,  # Brax: softplus(0) = 0.69
      noise_std_type="log",  # the network learns log(std), so the std can never go negative
      # The entropy bonus is computed on the action AFTER tanh (what the robot gets).
      # On the raw Gaussian it paid for ever-larger noise that tanh flattens to +-1:
      # v4's bang-bang actions.
      squashed_entropy=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=0.5,
      use_clipped_value_loss=True,
      clip_param=0.3,  # +-30% per update, larger steps than rsl_rl's usual 0.2
      # Exploration bonus. If the reward signal weakens (rungs 4-5: more falls), this
      # constant bonus can win and the std grows; r4_ent001 tests 0.001.
      entropy_coef=0.01,
      num_learning_epochs=8,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="fixed",  # the learning rate never changes (desired_kl below is then unused)
      gamma=0.97,  # shorter horizon: a reward 1 s ahead counts 0.97^50 = 0.22
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups=PGTT_TEACHER_OBS_GROUPS,  # same inputs as the teacher, so the networks match
    experiment_name="go2_pgtt_repro",  # logs/rsl_rl/go2_pgtt_repro/
    save_interval=100,
    num_steps_per_env=20,  # Brax's unroll length
    max_iterations=3000,  # default; train_ladder.sh and the commands set --agent.max-iterations
  )


def unitree_go2_pgtt_rung_student_runner_cfg() -> RslRlDistillationRunnerCfg:
  """Stage 2 for a ladder teacher (Rung<N>-Student tasks): the same student, distilled from
  a go2_pgtt_repro checkpoint. Pass --teacher-checkpoint: the newest run in go2_pgtt_repro
  is not necessarily the rung you mean."""
  cfg = unitree_go2_pgtt_student_runner_cfg()
  cfg.teacher_experiment_name = "go2_pgtt_repro"
  cfg.experiment_name = "go2_pgtt_rung_student"  # logs/rsl_rl/go2_pgtt_rung_student/
  return cfg
