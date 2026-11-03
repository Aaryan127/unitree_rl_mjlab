from mjlab.rsl_rl.runners import DistillationRunner, OnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  RUNG_NAMES,
  unitree_go2_flat_env_cfg,
  unitree_go2_pgtt_env_cfg,
  unitree_go2_pgtt_repro_env_cfg,
  unitree_go2_pgtt_repro_smooth_env_cfg,
  unitree_go2_pgtt_rung_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .rl_cfg import (
  unitree_go2_pgtt_repro_runner_cfg,
  unitree_go2_pgtt_student_runner_cfg,
  unitree_go2_pgtt_teacher_runner_cfg,
  unitree_go2_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go2",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go2",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# PGTT teacher-student. Both stages use the same environment; they differ only in the
# network, the algorithm and which observation groups the networks read.
register_mjlab_task(
  task_id="Mjlab-Velocity-PGTT-Teacher-Unitree-Go2",
  env_cfg=unitree_go2_pgtt_env_cfg(),
  play_env_cfg=unitree_go2_pgtt_env_cfg(play=True),
  rl_cfg=unitree_go2_pgtt_teacher_runner_cfg(),
  runner_cls=OnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-PGTT-Student-Unitree-Go2",
  env_cfg=unitree_go2_pgtt_env_cfg(),
  play_env_cfg=unitree_go2_pgtt_env_cfg(play=True),
  rl_cfg=unitree_go2_pgtt_student_runner_cfg(),  # type: ignore[arg-type]
  runner_cls=DistillationRunner,
)

# The official PGTT training setup reproduced (flat ground): the known-good baseline.
register_mjlab_task(
  task_id="Mjlab-Velocity-PGTT-Repro-Unitree-Go2",
  env_cfg=unitree_go2_pgtt_repro_env_cfg(),
  play_env_cfg=unitree_go2_pgtt_repro_env_cfg(play=True),
  rl_cfg=unitree_go2_pgtt_repro_runner_cfg(),
  runner_cls=OnPolicyRunner,
)

# Rung 0b: the reproduction plus a penalty on fast action changes (tremor).
register_mjlab_task(
  task_id="Mjlab-Velocity-PGTT-Repro-Smooth-Unitree-Go2",
  env_cfg=unitree_go2_pgtt_repro_smooth_env_cfg(),
  play_env_cfg=unitree_go2_pgtt_repro_smooth_env_cfg(play=True),
  rl_cfg=unitree_go2_pgtt_repro_runner_cfg(),
  runner_cls=OnPolicyRunner,
)

# Rungs 1-9: rung 0b plus one more change each, up to the full teacher (Rung9 is it).
# scripts/train_ladder.sh trains them in order.
for _rung in range(1, len(RUNG_NAMES) + 1):
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-PGTT-Rung{_rung}-Unitree-Go2",
    env_cfg=unitree_go2_pgtt_rung_env_cfg(_rung),
    play_env_cfg=unitree_go2_pgtt_rung_env_cfg(_rung, play=True),
    rl_cfg=unitree_go2_pgtt_repro_runner_cfg(),
    runner_cls=OnPolicyRunner,
  )
