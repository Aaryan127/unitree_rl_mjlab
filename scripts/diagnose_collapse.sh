#!/usr/bin/env bash
# Find out why rungs 4 and 5 collapse (doc/reward_hacking_runs_1-3.md, "Rungs 4 and 5
# collapse"): four 1500-iteration runs back to back, each followed by the reward-hacking
# check, then one table comparing them with rungs 3-5 and rung4_term100.
#
#   bash scripts/diagnose_collapse.sh        # all four (~12 h)
#   bash scripts/diagnose_collapse.sh 3      # start at the third (after a stop or a crash)
#
# Unlike train_ladder.sh it does not stop at a HACK: every run answers its own question.
# Run it from the repo root with the venv active, inside tmux. The table is also saved to
# logs/rsl_rl/go2_pgtt_repro/diagnosis.txt.
set -euo pipefail

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
FIRST=${1:-1}
TERM100="--env.rewards.termination.weight=-100"  # a fall costs 2 instead of 0.02
ENT001="--agent.algorithm.entropy-coef=0.001"    # a tenth of the entropy bonus

# name|rung|extra arguments
EXPERIMENTS=(
  # Price the fall and the kneel together. The candidate fix.
  "r5_term100|5|$TERM100"
  # Cheap falls as in rung 4, but a small entropy bonus. Is the bonus the root cause?
  "r4_ent001|4|$ENT001"
  # Both. Needed if r5_term100 still drifts with a rising noise std.
  "r5_term100_ent001|5|$TERM100 $ENT001"
  # The candidate fix plus the teacher's commands: the last rung before terrain.
  "r6_term100|6|$TERM100"
)

for i in $(seq "$FIRST" "${#EXPERIMENTS[@]}"); do
  IFS="|" read -r name rung extra <<< "${EXPERIMENTS[$((i - 1))]}"
  echo "=== $(date '+%F %T') experiment $i/${#EXPERIMENTS[@]} ($name): rung $rung $extra"
  # $extra is unquoted on purpose: it holds zero or more arguments.
  python scripts/train.py "Mjlab-Velocity-PGTT-Rung${rung}-Unitree-Go2" \
    --env.scene.num-envs=4096 --agent.max-iterations=1500 --agent.run-name "$name" $extra

  run=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_"$name" | tail -1)
  checkpoint=$(ls "$run"/model_*.pt | sort -V | tail -1)
  echo "=== $(date '+%F %T') $name: checking $checkpoint"
  python scripts/check_reward_hacking.py --checkpoint-file "$checkpoint" --device cuda:0 \
    | tee "$run/hacking_check.txt"
done

names=(rung3_pushes rung4_randomization rung5_rewards rung4_term100)
for experiment in "${EXPERIMENTS[@]}"; do names+=("${experiment%%|*}"); done
python scripts/summarize_runs.py "${names[@]}" | tee logs/rsl_rl/go2_pgtt_repro/diagnosis.txt
echo "=== $(date '+%F %T') done. Table: logs/rsl_rl/go2_pgtt_repro/diagnosis.txt"
