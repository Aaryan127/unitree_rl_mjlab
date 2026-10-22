#!/usr/bin/env bash
# PGTT shortcuts (the portable version of the cpu branch's train.sh).
#
#   scripts/pgtt.sh config                    reward / curriculum / observation config
#   scripts/pgtt.sh debug   [debug_env args]  inspect obs, rewards, sensors (CPU, 2 robots)
#   scripts/pgtt.sh smoke   [teacher|student] 3-iteration CPU training run (small terrain)
#   scripts/pgtt.sh teacher [train.py args]   full teacher training (GPU)
#   scripts/pgtt.sh student [train.py args]   full student distillation (GPU, latest teacher)
#   scripts/pgtt.sh play    teacher|student [checkpoint]   watch a policy (default: latest)
#   scripts/pgtt.sh latest  teacher|student   print the latest checkpoint path
#   scripts/pgtt.sh tb                        tensorboard on logs/
#   scripts/pgtt.sh export  [checkpoint]      export the student for the robot (default: latest)
#
# Any extra arguments are passed through, e.g.
#   scripts/pgtt.sh teacher --agent.run-name sigma004 --env.rewards.foot-phase.params.sigma 0.004
# Debug switches work with every command, e.g.
#   MJLAB_DEBUG=1 scripts/pgtt.sh smoke teacher
#   MJLAB_BREAK=reward:foot_phase scripts/pgtt.sh debug
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

TEACHER=Mjlab-Velocity-PGTT-Teacher-Unitree-Go2
STUDENT=Mjlab-Velocity-PGTT-Student-Unitree-Go2
SMALL_TERRAIN=(--env.scene.terrain.terrain-generator.num-rows 3 --env.scene.terrain.terrain-generator.num-cols 5)

task_for() {
  case "$1" in
    teacher) echo "$TEACHER" ;;
    student) echo "$STUDENT" ;;
    *) echo "expected 'teacher' or 'student', got '$1'" >&2; exit 1 ;;
  esac
}

latest_checkpoint() {
  local dir="logs/rsl_rl/go2_pgtt_$1"
  local run
  run=$(ls -1d "$dir"/*/ 2>/dev/null | sort | tail -n 1)
  [[ -n "$run" ]] || { echo "no runs in $dir" >&2; exit 1; }
  ls -1 "$run"model_*.pt | sort -t_ -k2 -n | tail -n 1
}

cmd="${1:-help}"
shift || true
case "$cmd" in
  config)  python scripts/debug_env.py "$TEACHER" --config ;;
  debug)   python scripts/debug_env.py "$TEACHER" "$@" ;;
  smoke)
    which="${1:-teacher}"; shift || true
    CUDA_VISIBLE_DEVICES="" python scripts/train.py "$(task_for "$which")" \
      --env.scene.num-envs=16 --agent.max-iterations=3 "${SMALL_TERRAIN[@]}" "$@" ;;
  teacher) python scripts/train.py "$TEACHER" --env.scene.num-envs=4096 "$@" ;;
  student) python scripts/train.py "$STUDENT" --env.scene.num-envs=4096 "$@" ;;
  play)
    which="${1:-teacher}"; shift || true
    ckpt="${1:-$(latest_checkpoint "$which")}"; shift || true
    echo "playing $ckpt"
    python scripts/play.py "$(task_for "$which")" --checkpoint-file "$ckpt" "$@" ;;
  latest)  latest_checkpoint "${1:-teacher}" ;;
  tb)      tensorboard --logdir logs ;;
  export)
    ckpt="${1:-$(latest_checkpoint student)}"; shift || true
    python scripts/export_deploy.py --checkpoint-file "$ckpt" "$@" ;;
  help|*)  sed -n '2,18p' "$0" ;;
esac
