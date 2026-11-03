#!/usr/bin/env bash
# Train the ladder from rung 0b to the full teacher: rungs 1-9 one after another, each
# from scratch (env_cfgs.unitree_go2_pgtt_rung_env_cfg says what each rung adds). After
# each rung it runs the reward-hacking check on the last checkpoint and stops at the first
# rung with a HACK verdict, so a failure points at exactly one change.
#
#   bash scripts/train_ladder.sh        # rungs 1-9
#   bash scripts/train_ladder.sh 7      # rungs 7, 8, 9 (terrain, dynamics, hard terrain)
#   bash scripts/train_ladder.sh 4      # start at rung 4 (after a stop, a failure or a crash)
#
# Run it from the repo root with the venv active, inside tmux. It uses the RTX 3070 on the
# GPU machine unless CUDA_VISIBLE_DEVICES is already set. Rungs 1-6 train 1500 iterations
# (~2-3 h each at 4-7 s per iteration), rung 7 (terrain) 6000 (~15 h at ~9 s), rung 8
# (dynamics) 3000 (~8 h), rung 9 (hard terrain, the teacher candidate) 6000 (~15 h). Run folders:
# logs/rsl_rl/go2_pgtt_repro/<date>_<time>_rung<N>_<name>/, with hacking_check.txt inside.
set -euo pipefail

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
NAMES=(collisions delays pushes randomization rewards commands terrain dynamics hard_terrain)
FIRST=${1:-1}

for rung in $(seq "$FIRST" "${#NAMES[@]}"); do
  name="rung${rung}_${NAMES[$((rung - 1))]}"
  iterations=1500
  if [ "$rung" -eq 7 ] || [ "$rung" -eq 9 ]; then iterations=6000; fi
  if [ "$rung" -eq 8 ]; then iterations=3000; fi
  echo "=== $(date '+%F %T') rung $rung ($name): training $iterations iterations"
  python scripts/train.py "Mjlab-Velocity-PGTT-Rung${rung}-Unitree-Go2" \
    --env.scene.num-envs=4096 --agent.max-iterations="$iterations" --agent.run-name "$name"

  run=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_"$name" | tail -1)
  checkpoint=$(ls "$run"/model_*.pt | sort -V | tail -1)
  echo "=== $(date '+%F %T') rung $rung: checking $checkpoint"
  python scripts/check_reward_hacking.py --checkpoint-file "$checkpoint" --device cuda:0 \
    | tee "$run/hacking_check.txt"
  if grep -q "overall: HACK" "$run/hacking_check.txt"; then
    echo "=== rung $rung ($name) FAILED the check, stopping. Details: $run/hacking_check.txt"
    exit 1
  fi
done
echo "=== $(date '+%F %T') all rungs passed. Full teacher: $run"
