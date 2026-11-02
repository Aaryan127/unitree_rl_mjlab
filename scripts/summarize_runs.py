"""One table comparing PGTT runs: the hack check verdict and the training curves.

For each run name it takes the newest logs/rsl_rl/go2_pgtt_repro/*_<name> folder, reads
its hacking_check.txt (written by train_ladder.sh / diagnose_collapse.sh) and its
TensorBoard log, and prints one row per run. The curves show what the check can't: the
check runs without randomization, so a policy that falls apart under it still looks fine
there (doc/reward_hacking_runs_1-3.md, "Rungs 4 and 5 collapse").

  python scripts/summarize_runs.py rung3_pushes rung4_randomization r5_term100
"""

import glob
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = "logs/rsl_rl/go2_pgtt_repro"
CHECK_METRICS = ("speed_ratio", "falls_per_min", "leg_ground_frac", "base_height", "action_tremor")
# (column title, TensorBoard tag); each is shown at iteration 200, 800 and the last one.
CURVES = (
  ("episode length", "Train/mean_episode_length"),
  ("noise std", "Policy/mean_noise_std"),
  ("body_contact ends", "Episode_Termination/body_contact"),
)
ITERS = (200, 800)


def read_check(run: Path) -> dict[str, str]:
  path = run / "hacking_check.txt"
  if not path.exists():
    return {}
  text = path.read_text()
  values = {
    name: value for name, value in re.findall(r"^\s*(?:OK|WARN|HACK)\s+(\w+)\s+([-\d.]+)", text, re.M)
  }
  overall = re.search(r"overall:\s*(\w+)", text)
  values["verdict"] = overall.group(1) if overall else "?"
  return values


def read_curves(run: Path) -> dict[str, str]:
  ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
  ea.Reload()
  tags = ea.Tags()["scalars"]
  out = {}
  for title, tag in CURVES:
    if tag not in tags:
      out[title] = "-"
      continue
    v = {e.step: e.value for e in ea.Scalars(tag)}
    last = max(v)
    out[title] = " / ".join(f"{v[i]:.3g}" if i in v else "-" for i in (*ITERS, last))
    out["last iteration"] = str(last)
  return out


def main(names: list[str]) -> None:
  for name in names:
    runs = sorted(glob.glob(f"{LOG_ROOT}/*_{name}"))
    print(f"\n== {name}")
    if not runs:
      print("  no run folder found")
      continue
    run = Path(runs[-1])
    print(f"  folder: {run}")
    check = read_check(run)
    if check:
      print(f"  hack check: {check['verdict']}   " + "   ".join(
        f"{m} {check.get(m, '-')}" for m in CHECK_METRICS))
    else:
      print("  hack check: none (no hacking_check.txt)")
    curves = read_curves(run)
    print(f"  curves at iteration 200 / 800 / {curves.get('last iteration', 'last')}:")
    for title, _ in CURVES:
      print(f"    {title:18s} {curves[title]}")


if __name__ == "__main__":
  main(sys.argv[1:])
