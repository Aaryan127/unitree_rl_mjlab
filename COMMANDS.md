# Commands: train, find, check and watch the Go2 PGTT policies

Every command here is meant to be copied exactly. Where a command contains `<SOMETHING>`,
the lines under it say what goes there and which command prints it. Sections 0-3 are needed
every time; after that, jump to what you want to do.

**Contents**
0. [The two machines](#0-the-two-machines)
1. [Log in and get ready (GPU machine)](#1-log-in-and-get-ready-gpu-machine)
2. [tmux: keep things running after you log out](#2-tmux-keep-things-running-after-you-log-out)
3. [Names: tasks, run names, run folders, checkpoints](#3-names-tasks-run-names-run-folders-checkpoints)
4. [Train](#4-train)
5. [Is it running? How far is it?](#5-is-it-running-how-far-is-it)
6. [Stop, restart, continue](#6-stop-restart-continue)
7. [Find a run and its checkpoints](#7-find-a-run-and-its-checkpoints)
8. [Watch the robot (viewer)](#8-watch-the-robot-viewer)
9. [Graphs (TensorBoard)](#9-graphs-tensorboard)
10. [Reward-hacking check](#10-reward-hacking-check)
11. [Gait numbers](#11-gait-numbers)
12. [Copy runs to the laptop](#12-copy-runs-to-the-laptop)
13. [When something goes wrong](#13-when-something-goes-wrong)

---

## 0. The two machines

| | GPU machine | Laptop |
|---|---|---|
| Where | `ssh manas@10.68.98.4` (asks for the password) | your own terminal |
| Repo folder | `/mnt/sdb1/Research/neptune/unitree_rl_mjlab` | `~/unitree_rl_mjlab` |
| Python | `python`, after `source .venv/bin/activate` | `.venv/bin/python` (no activation needed) |
| GPU | use only the RTX 3070: put `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` in front of every `python` command (the GTX 1080s fail with `no kernel image`) | none: everything runs on the CPU, slowly |
| Use it for | training, checks, viewer in the browser | looking at copied runs |

All commands run **from the repo folder** (the paths like `logs/...` and `scripts/...` are
relative to it). Installing from zero on a new machine: [doc/setup_en.md](doc/setup_en.md).

What a run must pass, and why: [doc/reward_hacking_runs_1-3.md](doc/reward_hacking_runs_1-3.md).

---

## 1. Log in and get ready (GPU machine)

On the **laptop**, open a terminal and log in. The two `-L` parts forward the GPU machine's
TensorBoard (port 6006) and 3D viewer (port 8080) to the laptop, so
`http://localhost:6006` and `http://localhost:8080` open them in the laptop's browser. They
only work while this ssh window stays open.
```bash
ssh -L 6006:localhost:6006 -L 8080:localhost:8080 manas@10.68.98.4
```
If it says `bind: Address already in use`, another ssh with the same `-L` is still open on
the laptop. Use that one, or close it and log in again.

Now on the **GPU machine**:
```bash
ls /mnt/sdb1/Research/neptune/unitree_rl_mjlab > /dev/null || (sudo mkdir -p /mnt/sdb1 && sudo mount --bind /mnt/ubuntu /mnt/sdb1)
cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && source .venv/bin/activate
```
- The first line only does something after a reboot, when the disk is mounted at `/mnt/ubuntu`
  but not at `/mnt/sdb1`. It may ask for the password.
- Every new terminal, tmux session or tmux window needs the `cd ... && source ...` line again.
  You know it worked when the prompt starts with `(.venv)` (or the repo's venv name).

**Get the newest code** (do this before starting a training run):
```bash
git pull
git log --oneline -1
```
The second line shows which version you are on. It should match the last commit on GitHub
(github.com/Aaryan127/unitree_rl_mjlab, branch `pgtt`). If `git pull` complains about local
changes, don't force it: `git status` shows which files, and ask before throwing them away.

---

## 2. tmux: keep things running after you log out

A program started in a plain ssh window dies when the window closes or the WiFi drops. Inside
tmux it keeps running, and you can come back to it later, even from another computer.

| I want to | Command |
|---|---|
| start a new session called `train` | `tmux new -s train` |
| leave it running and get back to the normal terminal (detach) | press **Ctrl+B**, let go, then press **D** |
| see which sessions exist | `tmux ls` |
| go back into the session (attach) | `tmux attach -t train` |
| open a second window in the same session | **Ctrl+B**, then **C** |
| switch windows | **Ctrl+B**, then **N** (next) or **P** (previous), or **Ctrl+B** then the window number **0**, **1**, ... |
| scroll up to read older output | **Ctrl+B**, then **[**, then arrow keys / PgUp; press **q** to stop scrolling |
| close the current window | type `exit` (only when nothing is running in it) |
| delete a whole session (kills what runs in it!) | `tmux kill-session -t train` |

- `duplicate session: train`: that session already exists. Go into it with `tmux attach -t train`,
  or use another name, e.g. `tmux new -s train2`.
- `no sessions` / `can't find session`: nothing is running in tmux (the machine may have
  rebooted). Check with `tmux ls`.
- `sessions should be nested with care`: you're already inside tmux. Detach first (**Ctrl+B**, **D**).
- The ssh connection dropped? Training in tmux is fine. Log in again (section 1) and
  `tmux attach -t train`.

A good layout: session `train` with window 0 for training, window 1 for TensorBoard
(section 9), window 2 for checks and the viewer. Each new window: run the `cd ... && source ...`
line of section 1 first.

---

## 3. Names: tasks, run names, run folders, checkpoints

- **Task**: which training setup to use, e.g. `Mjlab-Velocity-PGTT-Rung1-Unitree-Go2`. To list
  all Go2 tasks: `grep task_id mjlab/tasks/velocity/config/go2/__init__.py` (the rung tasks are
  made in a loop at the bottom, `Rung1` to `Rung7`).
- **Run name**: a label you give a training run (`--agent.run-name ...`). Used to find it later.
- **Run folder**: where a run saves everything:
  `logs/rsl_rl/<experiment>/<date>_<time>_<run name>/`, e.g.
  `logs/rsl_rl/go2_pgtt_repro/2026-11-01_09-12-40_rung1_collisions/`.
  The experiment is `go2_pgtt_repro` for every task in the table below. Inside the folder:
  - `model_0.pt`, `model_100.pt`, ... : **checkpoints** (the policy's weights), one every 100
    iterations, plus the last one (`model_1499.pt` for a 1500-iteration run, because counting
    starts at 0);
  - `events.out.tfevents...` : the graphs for TensorBoard;
  - `params/env.yaml`, `params/agent.yaml` : the exact settings it trained with;
  - `hacking_check.txt` : the check's result (ladder runs only, section 4).

**The tasks and the run names used for them:**

| Task | Run name | What it is | Iterations |
|---|---|---|---|
| `Mjlab-Velocity-PGTT-Repro-Unitree-Go2` | `repro0` | rung 0: official PGTT setup, flat (done, walks) | 3000 |
| `Mjlab-Velocity-PGTT-Repro-Smooth-Unitree-Go2` | `smooth0b` | rung 0b: + smoothness penalty (done, walks) | 1500 |
| `Mjlab-Velocity-PGTT-Rung1-Unitree-Go2` | `rung1_collisions` | + full-body collisions; trunk/hip on the ground ends the episode | 1500 |
| `Mjlab-Velocity-PGTT-Rung2-Unitree-Go2` | `rung2_delays` | + motor and sensor delays | 1500 |
| `Mjlab-Velocity-PGTT-Rung3-Unitree-Go2` | `rung3_pushes` | + pushes and external force | 1500 |
| `Mjlab-Velocity-PGTT-Rung4-Unitree-Go2` | `rung4_randomization` | + wider domain randomization | 1500 |
| `Mjlab-Velocity-PGTT-Rung5-Unitree-Go2` | `rung5_rewards` | + penalties for crouching, kneeling, joint limits | 1500 |
| `Mjlab-Velocity-PGTT-Rung6-Unitree-Go2` | `rung6_commands` | + full command range, 5% standing | 1500 |
| `Mjlab-Velocity-PGTT-Rung7-Unitree-Go2` | `rung7_terrain` | + stairs, slopes, rough ground: **the full teacher** | 6000 |

Each rung contains everything above it in the table. Old teacher runs (v1-v4, don't walk) are in
`logs/rsl_rl/go2_pgtt_teacher/`.

---

## 4. Train

All on the **GPU machine**, inside tmux (section 2), in the repo with the venv active (section 1).
One training at a time: two at once both run at half speed or run out of GPU memory.

### 4a. The whole ladder, rung 1 to 7 (the normal way to get the full teacher)

```bash
tmux new -s train
cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && source .venv/bin/activate && git pull
bash scripts/train_ladder.sh
```
- Trains rung 1, checks it, trains rung 2, and so on. About 9 h for rungs 1-6 and 6 h for
  rung 7.
- After each rung it runs the reward-hacking check and saves the result as
  `hacking_check.txt` in that rung's run folder. If a rung gets a HACK verdict, the script
  **stops** there and prints `FAILED`, so the problem is that rung's one change.
- It picks the 3070 by itself (no `CUDA_...` prefix needed).
- When you see the first `Learning iteration 0/1500` box, detach: **Ctrl+B**, **D**.

**Start from a later rung** (after a crash, after you stopped it, or after fixing a failed rung):
```bash
bash scripts/train_ladder.sh <FIRST_RUNG>
```
- **`<FIRST_RUNG>`**: the first rung that hasn't finished, a number 1-7. To see which rungs
  finished: `ls logs/rsl_rl/go2_pgtt_repro/*_rung*/hacking_check.txt`. A rung with that
  file finished training and was checked; the next number is the one to start from. E.g. it
  lists `..._rung1_collisions/...` and `..._rung2_delays/...`: run
  `bash scripts/train_ladder.sh 3`.
- A rung that was interrupted starts again from zero (the half-finished folder stays; ignore it).

### 4b. One task on its own

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py <TASK> --env.scene.num-envs=4096 --agent.max-iterations=<ITERATIONS> --agent.run-name <RUN_NAME>
```
- **`<TASK>`**: a task from the table in section 3.
- **`<ITERATIONS>`**: from the same table (about 1 h per 1000 on the 3070).
- **`<RUN_NAME>`**: from the same table. For a second try of the same task, add something,
  e.g. `rung1_collisions_seed2`.

Example, rung 1 alone:
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung1-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=1500 --agent.run-name rung1_collisions
```
A different random seed (same task, to check a result wasn't luck): add `--agent.seed 2`.

The run folder is printed near the start, on the line
`[INFO] Logging experiment in directory: ...`.

---

## 5. Is it running? How far is it?

**Look at it directly:** `tmux attach -t train`. The box at the bottom shows
`Learning iteration <done>/<total>` and the time left (`ETA`). Leave with **Ctrl+B**, **D**
(not Ctrl+C, that stops it).

**Without attaching:**
```bash
nvidia-smi
```
A `python` process using the 3070 (several GB of memory, high `GPU-Util`) means something is
training. No process: nothing is training.

```bash
ls -t logs/rsl_rl/go2_pgtt_repro/ | head -5
```
The five newest runs, newest first: the top one is the run in progress.

```bash
ls logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME>/model_*.pt | sort -V | tail -1
```
- **`<RUN_NAME>`**: from the table in section 3, e.g. `rung1_collisions`.
- Prints the newest checkpoint; its number is roughly how far it got (saved every 100 iterations).

**Ladder progress:** which rungs are done, and their verdicts:
```bash
grep -H "overall" logs/rsl_rl/go2_pgtt_repro/*_rung*/hacking_check.txt
```

---

## 6. Stop, restart, continue

**Stop a training (or the ladder):** `tmux attach -t train`, then **Ctrl+C**. Press it once
and wait; the ladder may need a second **Ctrl+C** to stop starting the next step. Checkpoints
already saved stay.

**Restart the ladder from a rung:** section 4a.

**Continue a stopped single run from its last checkpoint** (instead of from zero):
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py <TASK> --env.scene.num-envs=4096 --agent.max-iterations=<MORE_ITERATIONS> --agent.run-name <RUN_NAME>_resumed --agent.resume True --agent.load-run '.*_<RUN_NAME>$' --agent.load-checkpoint <CHECKPOINT_FILE>
```
- **`<TASK>`**, **`<RUN_NAME>`**: the same as when the run was started (section 3).
- **`<MORE_ITERATIONS>`**: how many iterations to add.
- **`<CHECKPOINT_FILE>`**: only the file name of its newest checkpoint, e.g. `model_1000.pt`:
  ```bash
  ls logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME>/model_*.pt | sort -V | tail -1 | xargs basename
  ```
  Always give it: without it, "latest" is chosen alphabetically and `model_900.pt` would win
  over `model_1500.pt`.
- It saves into a new folder ending in `_<RUN_NAME>_resumed`.

---

## 7. Find a run and its checkpoints

Same commands on both machines, from the repo folder. On the **GPU machine** runs are in
`logs/rsl_rl/go2_pgtt_repro/`; on the **laptop**, copied runs are in `logs/remote/<RUN_NAME>/`
(section 12).

| I want | Command (GPU machine) | Example output |
|---|---|---|
| all runs, newest first | `ls -t logs/rsl_rl/go2_pgtt_repro/` | `2026-11-01_09-12-40_rung1_collisions` ... |
| the folder of one run | `ls -d logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME>` | `logs/rsl_rl/go2_pgtt_repro/2026-11-01_09-12-40_rung1_collisions` |
| all its checkpoints, in order | `ls logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME>/model_*.pt \| sort -V` | `.../model_0.pt` ... `.../model_1499.pt` |
| its newest checkpoint | `ls logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME>/model_*.pt \| sort -V \| tail -1` | `logs/rsl_rl/go2_pgtt_repro/2026-11-01_09-12-40_rung1_collisions/model_1499.pt` |
| one particular checkpoint | `ls logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME>/model_1000.pt` | `.../model_1000.pt` |
| its hacking check (ladder) | `cat logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME>/hacking_check.txt` | the OK / WARN / HACK table |

| I want | Command (laptop) |
|---|---|
| which runs are copied | `ls logs/remote/` |
| newest checkpoint of one | `ls logs/remote/<RUN_NAME>/model_*.pt \| sort -V \| tail -1` |

- **`<RUN_NAME>`**: from the table in section 3, e.g. `rung1_collisions`. Don't know it? The
  part after the time in `ls -t logs/rsl_rl/go2_pgtt_repro/`
  (`2026-11-01_09-12-40_`**`rung1_collisions`**).
- **`sort -V`** sorts by number, so `model_1400` comes after `model_900` (plain `ls` puts it
  before).
- `ls -d ...` prints **two lines**: the run name was used twice. The later date is the newer
  run. Use the full folder name instead of `*_<RUN_NAME>`.
- `no such file or directory` / `no matches found`: the run name is spelled differently, or the
  run has no checkpoint yet. Check with `ls -t logs/rsl_rl/go2_pgtt_repro/`.

**Shortcut: keep the paths in variables** so you don't have to copy them by hand:
```bash
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME> | tail -1); echo $RUN
CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
```
After this, `$RUN` is the run folder and `$CKPT` the newest checkpoint, and you can write
`--checkpoint-file $CKPT` in the commands below. Laptop: `RUN=logs/remote/<RUN_NAME>` instead
of the first line. Variables are forgotten in a new terminal or tmux window: run the two lines
again there.

Below, **`<CHECKPOINT>`** always means a checkpoint path from this section (or `$CKPT`), and
**`<TASK>`** the task the run was trained with (section 3: run name → task).

---

## 8. Watch the robot (viewer)

**Always keep `--num-envs 1`.** Without it, the viewer simulates the 4096 robots the run
trained with; on the laptop that froze the whole desktop.

### 8a. On the GPU machine, shown in the laptop's browser

Needs the ssh login with `-L ... 8080` (section 1). In a free tmux window (section 2):
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py <TASK> --checkpoint-file <CHECKPOINT> --num-envs 1 --viewer viser
```
Give it about half a minute to start, then open **http://localhost:8080** in the laptop's
browser (reload if it's blank). **Ctrl+C** in the terminal stops it.

Example, the newest rung 1 checkpoint:
```bash
CKPT=$(ls logs/rsl_rl/go2_pgtt_repro/*_rung1_collisions/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung1-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser
```

### 8b. On the laptop, in a window

Copy the run first (section 12). Then:
```bash
cd ~/unitree_rl_mjlab && .venv/bin/python scripts/play.py <TASK> --checkpoint-file <CHECKPOINT> --num-envs 1
```
Example:
```bash
cd ~/unitree_rl_mjlab && .venv/bin/python scripts/play.py Mjlab-Velocity-PGTT-Repro-Smooth-Unitree-Go2 --checkpoint-file logs/remote/smooth0b/model_1499.pt --num-envs 1
```
It may run slower than real time (CPU). Don't take screenshots while it runs (GNOME crashed doing
that); use your phone.

### 8c. Options

| I want | Add |
|---|---|
| a particular terrain (stairs etc.) | `--terrain stairs` (or `flat`, `rough`, `mixed`) |
| an older checkpoint | a different `--checkpoint-file`, e.g. `.../model_500.pt` |
| a robot that does nothing / random actions (for comparison) | `--agent zero` / `--agent random` (no checkpoint needed) |

**What a good gait looks like:** a trot (diagonal feet move together), clear foot lifts of
about 7-9 cm, a steady body, no buzzing or trembling legs, following the command arrows.
Rung 1+: no kneeling, no trunk touching the ground. Rung 7: lifts its feet onto steps, doesn't
hit step edges with its shins.

---

## 9. Graphs (TensorBoard)

### 9a. GPU machine, viewed on the laptop

Needs the ssh login with `-L 6006...` (section 1). In a free tmux window:
```bash
cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && source .venv/bin/activate
tensorboard --logdir logs/rsl_rl --port 6006
```
Open **http://localhost:6006** on the laptop. It shows all runs; type the run name (e.g.
`rung1`) in the filter box on the left to see only those. Leave it running in its window; it
picks up new data by itself (reload the page).

### 9b. Laptop, for copied runs
```bash
cd ~/unitree_rl_mjlab && .venv/bin/tensorboard --logdir logs/remote --port 6007
```
Open **http://localhost:6007** (6007, because 6006 may be taken by the ssh tunnel).

`--logdir` is the folder that contains the runs; nothing else to fill in. Only one run?
Use `--logdir logs/remote/<RUN_NAME>`.

### 9c. Graphs that matter

| Graph | Healthy |
|---|---|
| `Train/mean_episode_length` | climbs to ~1000 by iteration ~100 and stays (rung 1+: dips mean falls) |
| `Metrics/hack/speed_ratio` | above 0.9 (rung 7 on stairs: above 0.65) |
| `Policy/mean_noise_std` | falls (0.7 → ~0.3); stuck at 1.0 was v4's entropy bug |
| `Metrics/hack/action_saturation` | below 0.05 |
| `Metrics/hack/action_tremor` | below 0.15 (repro0, without the smoothness penalty: ~0.35) |
| `Metrics/hack/body_ground_frac`, `leg_ground_frac` | ~0 (rung 1+) |
| `Episode_Termination/fell_over`, `body_contact` | ~0 |
| `Episode_Reward/*` | track_lin near 1.0, track_ang near 0.5, foot_phase near 1.0 |
| `Curriculum/terrain_levels` | rises over training (rung 7 only) |

---

## 10. Reward-hacking check

Runs the policy without noise and prints OK / WARN / HACK for each behaviour (lying down,
kneeling, ignoring the command, vibrating, falling, ...), plus where the reward comes from.
About 1 min on the GPU, 2 min on the laptop. The ladder runs it by itself after each rung
(`hacking_check.txt`, section 7).

**One checkpoint, GPU machine** (free tmux window):
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/check_reward_hacking.py --checkpoint-file <CHECKPOINT> --device cuda:0
```
Without `--device cuda:0` it runs on the CPU and looks stuck for an hour.

**One checkpoint, laptop:**
```bash
cd ~/unitree_rl_mjlab && .venv/bin/python scripts/check_reward_hacking.py --checkpoint-file <CHECKPOINT>
```

**Every new checkpoint while a run trains** (GPU machine, own tmux window, **Ctrl+C** to stop):
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/check_reward_hacking.py --watch <RUN_FOLDER> --device cuda:0
```
- **`<RUN_FOLDER>`**: the folder of the run (section 7, "the folder of one run"), e.g.
  `logs/rsl_rl/go2_pgtt_repro/2026-11-01_09-12-40_rung1_collisions`. One line per checkpoint
  is added to `<RUN_FOLDER>/hacking_check.csv`.

**On another terrain** (e.g. to see how a flat rung does on stairs): add `--terrain stairs`.

---

## 11. Gait numbers

Each robot gets one fixed command (stand, forward 0.3/0.6/1.0 m/s, gait 1/2/3 Hz, backward,
sideways, turning, diagonal) and the table shows tracking, steps per second against the gait
clock, bounces, foot lift, slip, tilt, power, tremor and knee clearance. Each column is
explained at the top of `scripts/eval_gait.py`. Flat ground only.

**GPU machine:**
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/eval_gait.py --checkpoint-file <CHECKPOINT> --device cuda:0
```
**Laptop** (about 2.5 min):
```bash
cd ~/unitree_rl_mjlab && .venv/bin/python scripts/eval_gait.py --checkpoint-file <CHECKPOINT>
```
Compare a rung with the rung before it at the same iteration, e.g. `rung2_delays/model_1499.pt`
with `rung1_collisions/model_1499.pt`.

---

## 12. Copy runs to the laptop

Run these on the **laptop**. They ask for the GPU machine's password.

**One whole run** (all checkpoints and graphs, about 100 MB per 1500 iterations):
```bash
mkdir -p ~/unitree_rl_mjlab/logs/remote/<RUN_NAME>
rsync -av --exclude='videos' 'manas@10.68.98.4:/mnt/sdb1/Research/neptune/unitree_rl_mjlab/logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME>/' ~/unitree_rl_mjlab/logs/remote/<RUN_NAME>/
```
- **`<RUN_NAME>`**: from section 3, in all three places. Example for rung 1:
  ```bash
  mkdir -p ~/unitree_rl_mjlab/logs/remote/rung1_collisions
  rsync -av --exclude='videos' 'manas@10.68.98.4:/mnt/sdb1/Research/neptune/unitree_rl_mjlab/logs/rsl_rl/go2_pgtt_repro/*_rung1_collisions/' ~/unitree_rl_mjlab/logs/remote/rung1_collisions/
  ```
- Running it again later copies only what's new (e.g. while the run is still training).
- If the run name was used twice, `*_<RUN_NAME>` matches both and mixes them: use the full
  folder name (`2026-11-01_09-12-40_rung1_collisions`) instead.

**Only the graphs and a few checkpoints** (smaller):
```bash
rsync -av --include='events*' --include='params/***' --include='*.txt' --include='*.csv' --include='model_500.pt' --include='model_1000.pt' --include='model_1499.pt' --exclude='*' 'manas@10.68.98.4:/mnt/sdb1/Research/neptune/unitree_rl_mjlab/logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME>/' ~/unitree_rl_mjlab/logs/remote/<RUN_NAME>/
```
Change the `model_...` names to the checkpoints you want.

**All ladder runs at once** (each into its own folder):
```bash
for name in rung1_collisions rung2_delays rung3_pushes rung4_randomization rung5_rewards rung6_commands rung7_terrain; do mkdir -p ~/unitree_rl_mjlab/logs/remote/$name; rsync -av --exclude='videos' "manas@10.68.98.4:/mnt/sdb1/Research/neptune/unitree_rl_mjlab/logs/rsl_rl/go2_pgtt_repro/*_$name/" ~/unitree_rl_mjlab/logs/remote/$name/; done
```
It asks for the password once per run. Runs that don't exist yet print an error; ignore it.

---

## 13. When something goes wrong

| Symptom | Cause and fix |
|---|---|
| `cd: no such file or directory: /mnt/sdb1/...` | after a reboot the disk is only at `/mnt/ubuntu`: run the first line of section 1 |
| `python: command not found` / `No module named mjlab` | venv not active: `source .venv/bin/activate` in the repo folder (laptop: use `.venv/bin/python`) |
| `no kernel image is available for execution on the device` | ran on a GTX 1080: put `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` in front |
| `CUDA out of memory` | another training or viewer is using the 3070: `nvidia-smi` shows it; stop it first |
| `Unknown task` / task name not accepted | typo, or the code is old: `git pull`, then check the name with `grep task_id mjlab/tasks/velocity/config/go2/__init__.py` |
| `duplicate session: ...` | `tmux attach -t <name>`, or pick another name |
| `no matches found` / `No such file` for `*_<RUN_NAME>` | the run name is different or it has no checkpoint yet: `ls -t logs/rsl_rl/go2_pgtt_repro/` |
| the hacking check prints nothing for a long time | it's on the CPU: **Ctrl+C**, add `--device cuda:0`. Still there (`T` in `ps aux \| grep check_reward`)? `kill -9 <pid>` |
| laptop crawls or the desktop freezes in the viewer | forgot `--num-envs 1` |
| `localhost:6006` / `:8080` doesn't load | the ssh window with `-L` is closed, or TensorBoard / play isn't running on the GPU machine, or play hasn't finished starting |
| `Address already in use` (port 8080 or 6006) on the GPU machine | an old viewer / TensorBoard is still running: find it in your tmux windows and **Ctrl+C** it, or `pkill -f "scripts/play.py"` / `pkill tensorboard` |
| ladder stopped with `FAILED` | read that rung's `hacking_check.txt` (section 7); the change of that rung (section 3) caused it. Look at it in the viewer before changing anything |
| the GPU machine rebooted during the ladder | section 1, then `bash scripts/train_ladder.sh <FIRST_RUNG>` (section 4a) |
