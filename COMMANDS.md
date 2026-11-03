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

What a run must pass, and why: [doc/experiments.md](doc/experiments.md).

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

A session has a name you choose. These are the ones used so far: `train` (single runs), `ladder`
(`train_ladder.sh`), `diag` (`diagnose_collapse.sh`), `teacher` (the full teacher). In the
commands below, **`<SESSION>`** is that name; `tmux ls` lists the ones that exist.

| I want to | Command |
|---|---|
| start a new session and go into it | `tmux new -s <SESSION>`, e.g. `tmux new -s train` |
| start a command in a new session **without going in** (one line, safest) | `tmux new -d -s <SESSION> '<COMMAND>; exec bash'` (section 4 has the exact lines) |
| leave it running and get back to the normal terminal (detach) | press **Ctrl+B**, let go, then press **D** |
| see which sessions exist | `tmux ls` |
| go back into the session (attach) | `tmux attach -t <SESSION>`, e.g. `tmux attach -t diag` |
| open a second window in the same session | **Ctrl+B**, then **C** |
| switch windows | **Ctrl+B**, then **N** (next) or **P** (previous), or **Ctrl+B** then the window number **0**, **1**, ... |
| scroll up to read older output | **Ctrl+B**, then **[**, then arrow keys / PgUp; press **q** to stop scrolling |
| close the current window | type `exit` (only when nothing is running in it) |
| delete a whole session (kills what runs in it!) | `tmux kill-session -t <SESSION>` |

- `duplicate session: <SESSION>`: that session already exists (often: the first command worked).
  Go into it with `tmux attach -t <SESSION>`, or use another name.
- `no sessions` / `can't find session`: no session with that name. `tmux ls` lists the ones that
  exist; attach to one of those.
- `no server running`: **no** tmux session exists at all. Almost always the machine rebooted,
  which kills every session and the training in it. Section 13, "rebooted".
- `sessions should be nested with care`: you're already inside tmux. Detach first (**Ctrl+B**, **D**).
- The ssh connection dropped? Training in tmux is fine. Log in again (section 1) and
  `tmux attach -t <SESSION>`.
- **Ctrl+C** inside a session stops what runs there. To leave it running, always **Ctrl+B**, **D**.

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
  - `hacking_check.txt` : the check's result (written by the ladder and the diagnosis, or by you
    with section 10).

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
| `Mjlab-Velocity-PGTT-Rung7-Unitree-Go2` | `rung7_terrain` | + stairs, slopes, rough ground | 6000 |
| `Mjlab-Velocity-PGTT-Rung8-Unitree-Go2` | `rung8_dynamics` | + the teacher sees every randomized dynamics value | 3000 |
| `Mjlab-Velocity-PGTT-Rung9-Unitree-Go2` | `rung9_hard_terrain` | + taller stairs (to 18 cm), bigger rough ground, boulders: **the full teacher** | 6000 |

Each rung contains everything above it in the table.

**Runs with a setting changed on the command line** (section 4b, "Change a setting"): same task,
own run name.

| Run name | Task | Changed | Started by |
|---|---|---|---|
| `rung4_term100` | Rung4 | `termination` −100 | by hand |
| `r5_term100` | Rung5 | `termination` −100 | `diagnose_collapse.sh` (4c) |
| `r4_ent001` | Rung4 | entropy bonus 0.001 | `diagnose_collapse.sh` |
| `r5_term100_ent001` | Rung5 | both | `diagnose_collapse.sh` (never run) |
| `r6_term100` | Rung6 | `termination` −100 | `diagnose_collapse.sh` (never run) |
| `rung6_seed2` | Rung6 | `--agent.seed 2` (is rung 6 reliable?) | by hand |
| `teacher_full` | Rung9 | 10000 iterations, nothing else changed | by hand (4d) |

Runs from before 2 Nov 23:07 (`a21cf12`) with rung 4 or later were trained with the motor bug
(the motors weakened at every reset). Their results describe the bug, not the setup.

The settings a run really used are in its `params/env.yaml` and `params/agent.yaml`. Old teacher runs (v1-v4, don't walk) are in
`logs/rsl_rl/go2_pgtt_teacher/`.

---

## 4. Train

All on the **GPU machine**, inside tmux (section 2), in the repo with the venv active (section 1).
One training at a time: two at once both run at half speed or run out of GPU memory. Check
first that nothing is training: `nvidia-smi` (section 5).

**The machine is shared** (same `manas` account). Other people's runs show up in `nvidia-smi` as
`python` on GPU 1 too. On 3 Nov, Chirag's `Unitree-Go2-Flat` run (from
`/mnt/ubuntu/chirag/marl_quadruped`) used 1.3 GB, which was enough to push rung 7 out of memory.
- Whose is a process: `ps -fp <PID>` and `ls -l /proc/<PID>/cwd`.
- **Don't kill someone else's run.** Agree on GPU time instead.
- To start yours as soon as theirs ends:
  `while kill -0 <PID> 2>/dev/null; do sleep 60; done; nvidia-smi; <YOUR COMMAND>`.

**How long:** about **7 s per iteration** on the 3070, so **~2 h per 1000 iterations** (1500
take ~3 h). The box printed after each iteration shows `Iteration time:` and the time left
(`ETA:`). Much slower than 7 s (e.g. 20 s) means something else is using the machine.

**Two ways to start anything in this section** (the scripts and single runs alike):
- **One line, safest.** It makes the tmux session, runs the command in it and keeps the window
  open afterwards (`exec bash`), so you can read the end even after a crash:
  ```bash
  tmux new -d -s <SESSION> 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && <COMMAND>; exec bash'
  tmux attach -t <SESSION>
  ```
  **`<COMMAND>`**: the command from 4a-4d, e.g. `bash scripts/train_ladder.sh`. A command that
  itself contains `'` can't go inside the quotes; use the other way.
- **By hand.** `tmux new -s <SESSION>`, then type the `cd ... && source ...` line of section 1,
  then the command.

Either way, when the first `Learning iteration 0/...` box appears, detach: **Ctrl+B**, then **D**.
Starting it in a plain ssh window (without tmux) kills it when the window closes or you press
Ctrl+C.

### 4a. The whole ladder, rung 1 to 7

```bash
tmux new -d -s ladder 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && bash scripts/train_ladder.sh; exec bash'
tmux attach -t ladder
```
- Trains rung 1, checks it, trains rung 2, and so on. About 3 h per rung for rungs 1-6 and
  12 h or more for rung 7 (6000 iterations, with terrain).
- After each rung it runs the reward-hacking check and saves the result as
  `hacking_check.txt` in that rung's run folder. If a rung gets a HACK verdict, the script
  **stops** there and prints `FAILED`, so the problem is that rung's one change.
- It picks the 3070 by itself (no `CUDA_...` prefix needed).
- When you see the first `Learning iteration 0/1500` box, detach: **Ctrl+B**, **D**.

**Start from a later rung** (after a crash, after you stopped it, or after fixing a failed rung):
```bash
tmux new -d -s ladder 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && bash scripts/train_ladder.sh <FIRST_RUNG>; exec bash'
tmux attach -t ladder
```
- **`<FIRST_RUNG>`**: the first rung that hasn't finished, a number 1-7. To see which rungs
  finished: `ls logs/rsl_rl/go2_pgtt_repro/*_rung*/hacking_check.txt`. A rung with that
  file finished training and was checked; the next number is the one to start from. E.g. it
  lists `..._rung1_collisions/...` and `..._rung2_delays/...`: run
  `bash scripts/train_ladder.sh 3`.
- A rung that was interrupted starts again from zero (the half-finished folder stays; ignore it).
- **Rung 7 again after the terrain fixes (3 Nov):** includes heading commands for 80% of the
  robots, so the terrain curriculum can promote them. `git pull`, then
  `bash scripts/train_ladder.sh 7`. It runs rung 7 (6000 iterations), rung 8 (3000) and rung 9
  (6000), each followed by the check, and stops at the first HACK: ~38 h in all.
  - Check the **Iteration time** in the first boxes: before the fix it was ~80 s.
  - The **ETA** wraps every 24 h (it prints h:m:s only). So `ETA: 12:54:48` can mean
    5 days + 12:54. Work it out: iteration time × iterations left.
- **After the motor fix (2 Nov):** rungs 4+ have to run again, because the old ones were trained with
  motors that weakened at every reset (`doc/experiments.md`, "Root cause"). Start at
  4: `bash scripts/train_ladder.sh 4`, after a `git pull`. The new run folders get the same names
  with a newer date, and every command here takes the newest.

### 4b. One task on its own

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py <TASK> --env.scene.num-envs=4096 --agent.max-iterations=<ITERATIONS> --agent.run-name <RUN_NAME>
```
- **`<TASK>`**: a task from the table in section 3.
- **`<ITERATIONS>`**: from the same table (~2 h per 1000 on the 3070).
- **`<RUN_NAME>`**: from the same table. For a second try of the same task, add something,
  e.g. `rung1_collisions_seed2`.

Example, rung 1 alone, started in tmux in one line:
```bash
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung1-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=1500 --agent.run-name rung1_collisions; exec bash'
tmux attach -t train
```
A different random seed (same task, to check a result wasn't luck): add `--agent.seed 2`.

The run folder is printed near the start, on the line
`[INFO] Logging experiment in directory: ...`.

A single run does **not** run the hack check by itself. Run it when training ends (section 10)
and save the result into the run folder, like the scripts do.

**Change a setting** without touching the code: add it to the end of the command. Always give
the run a new name that says what changed.

| Change | Add |
|---|---|
| a reward weight | `--env.rewards.<TERM>.weight=<VALUE>`, e.g. `--env.rewards.termination.weight=-100` |
| the entropy bonus | `--agent.algorithm.entropy-coef=<VALUE>`, e.g. `0.001` (default 0.01) |
| the learning rate | `--agent.algorithm.learning-rate=<VALUE>` (default 3e-4) |
| the random seed | `--agent.seed <N>` |
| number of robots | `--env.scene.num-envs=<N>` (4096 normally; fewer only for a quick test) |

- **`<TERM>`**: a reward name as listed in `doc/experiments.md`, "How the reward is
  computed", **with every `_` written as `-`**: `termination`, `leg-contact`, `base-height`,
  `foot-phase`, `action-acc`. E.g. `--env.rewards.leg-contact.weight=-0.5`.
- Every other setting works the same way. To find the exact name of one:
  `python scripts/train.py <TASK> --help | grep -i <WORD>`, e.g. `grep -i entropy`.
- Use `=` between the option and a negative value (`weight=-100`); with a space the `-100` is
  read as a new option.
- A misspelled option stops the run straight away with an error that lists the valid ones.
- **Did it take effect?** `grep -A3 "termination:" $RUN/params/env.yaml` (section 7 for `$RUN`)
  shows the weight the run used. In this file the names keep their `_`.

### 4c. The collapse diagnosis (four runs back to back, ~12 h)

**Done, no need to run it again (2 Nov).** Two of its runs were enough, and the cause turned out
to be a bug (the motors weakened at every reset), now fixed. Kept here for reference.

```bash
tmux new -d -s diag 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && bash scripts/diagnose_collapse.sh; exec bash'
tmux attach -t diag
```
- Runs `r5_term100`, `r4_ent001`, `r5_term100_ent001` and `r6_term100` (1500 iterations
  each) and checks each one. It does **not** stop at a HACK.
- At the end it prints one table comparing them with rungs 3-5. It also saves the table to
  `logs/rsl_rl/go2_pgtt_repro/diagnosis.txt`; read it later with
  `cat logs/rsl_rl/go2_pgtt_repro/diagnosis.txt`.
- **Start from a later experiment** (after a crash, a stop or a reboot): put the number after the
  script, `bash scripts/diagnose_collapse.sh <N>`, in the same one-line tmux command. **`<N>`** is
  the first experiment without a `hacking_check.txt`:
  ```bash
  ls logs/rsl_rl/go2_pgtt_repro/*_r[456]_*/hacking_check.txt
  ```
  Order: 1 `r5_term100`, 2 `r4_ent001`, 3 `r5_term100_ent001`, 4 `r6_term100`. E.g. only
  `..._r5_term100/hacking_check.txt` is listed: `<N>` is 2. Nothing listed: 1.
- **The table again at any time**, for any runs: section 9d.
- What each run tests and how to read the result: `doc/experiments.md`,
  "The diagnosis batch".

### 4d. The full teacher (10000 iterations, a day or more)

Only after rung 7 has passed (4a, `bash scripts/train_ladder.sh 7`). No extra settings: the
`termination` −100 of the diagnosis is not needed (the collapse was the motor bug).
```bash
tmux new -d -s teacher 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung9-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=10000 --agent.run-name teacher_full; exec bash'
tmux attach -t teacher
```
- How long: 10000 × the iteration time of the rung 7 run. That was 80 s before the terrain
  fixes; measure it again in rung 7. At ~10 s, it's ~28 h. A reboot in that time kills it:
  continue from its last checkpoint (section 6) instead of starting again.
- Watch the curves in the first ~1000 iterations (sections 9c, 9d). If episodes get shorter
  again or the noise std turns up, stop it there rather than waiting a day.
- When it ends: hack check (section 10) on the last checkpoint, plus once with
  `--terrain stairs`, then `eval_gait.py` (section 11).

### 4e. Every run: start it, watch it (GPU machine)

Copy-paste commands for each run, with nothing to fill in. **One training at a time**: check
`nvidia-smi` first (section 5). Every *start* line makes a tmux session `train`. If one
already exists, `tmux kill-session -t train` (only if nothing in it still trains). Every
*watch* line takes the newest checkpoint of the newest folder with that run name, and opens
the viewer at **http://localhost:8080** (ssh with `-L 8080:...`, section 1). **Ctrl+C**
stops it. Run the watch lines in a free tmux window (**Ctrl+B**, **C**) after
`cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && source .venv/bin/activate`.

**`repro0`**: rung 0: the official PGTT setup, flat. Done, walks.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Repro-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=3000 --agent.run-name repro0; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_repro0 | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Repro-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser
```

**`smooth0b`**: rung 0b: + the tremor penalty. Done, walks.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Repro-Smooth-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=1500 --agent.run-name smooth0b; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_smooth0b | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Repro-Smooth-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser
```

**`rung1_collisions`**: + full-body collisions. Done, WARN.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung1-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=1500 --agent.run-name rung1_collisions; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_rung1_collisions | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung1-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser
```

**`rung2_delays`**: + motor and sensor delays. Done, WARN.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung2-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=1500 --agent.run-name rung2_delays; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_rung2_delays | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung2-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser
```

**`rung3_pushes`**: + pushes. Done, WARN.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung3-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=1500 --agent.run-name rung3_pushes; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_rung3_pushes | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung3-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser
```

**`rung4_randomization`**: + wide randomization. Done (motor fix), WARN.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung4-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=1500 --agent.run-name rung4_randomization; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_rung4_randomization | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung4-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser
```

**`rung5_rewards`**: + anti-kneeling rewards. Done (motor fix), WARN.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung5-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=1500 --agent.run-name rung5_rewards; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_rung5_rewards | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung5-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser
```

**`rung6_commands`**: + the teacher's commands: **the flat-ground teacher**. Done, WARN.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung6-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=1500 --agent.run-name rung6_commands; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_rung6_commands | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung6-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser
```

**`rung6_seed2`**: rung 6 again with another seed: is rung 6 reliable? To run.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung6-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=1500 --agent.run-name rung6_seed2 --agent.seed 2; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_rung6_seed2 | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung6-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser
```

**`rung7_terrain`**: + terrain (50% stairs). **Next.** Better started with the ladder script (4a: `bash scripts/train_ladder.sh 7`), which also runs the check.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung7-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=6000 --agent.run-name rung7_terrain; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_rung7_terrain | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung7-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser --terrain stairs
```
- Without `--terrain stairs` it shows the mixed terrain. `--terrain rough` and `--terrain flat` also work.

**`rung8_dynamics`**: + the teacher sees every randomized dynamics value. Runs by itself after rung 7 with `bash scripts/train_ladder.sh 7`.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung8-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=3000 --agent.run-name rung8_dynamics; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_rung8_dynamics | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung8-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser --terrain stairs
```

**`rung9_hard_terrain`**: + stairs to 18 cm, rough ground to 10 cm, boulders to 15 cm, each harder by row: **the teacher candidate**. Runs by itself after rung 8.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung9-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=6000 --agent.run-name rung9_hard_terrain; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_rung9_hard_terrain | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung9-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser --terrain stairs
```

**`teacher_full`**: the teacher: rung 9 for 10000 iterations. Only after rung 9 passes.
```bash
# start
tmux new -d -s train 'cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && git pull && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Rung9-Unitree-Go2 --env.scene.num-envs=4096 --agent.max-iterations=10000 --agent.run-name teacher_full; exec bash'
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_teacher_full | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Rung9-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser --terrain stairs
```
- Without `--terrain stairs` it shows the mixed terrain. `--terrain rough` and `--terrain flat` also work.

**`student`**: stage 2, distills a teacher into the deployable student. Only after `teacher_full`.
The student task is built separately from the rungs, so first check that it loads a rung-7
teacher: start it and see that the first iteration box appears.
```bash
# start (TEACHER = the teacher checkpoint to distill)
TEACHER=$(ls $(ls -d logs/rsl_rl/go2_pgtt_repro/*_teacher_full | tail -1)/model_*.pt | sort -V | tail -1); echo $TEACHER
tmux new -d -s train "cd /mnt/sdb1/Research/neptune/unitree_rl_mjlab && . .venv/bin/activate && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Velocity-PGTT-Student-Unitree-Go2 --env.scene.num-envs=4096 --agent.run-name student --teacher-checkpoint $TEACHER; exec bash"
tmux attach -t train
# watch
RUN=$(ls -d logs/rsl_rl/go2_pgtt_student/*_student | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/play.py Mjlab-Velocity-PGTT-Student-Unitree-Go2 --checkpoint-file $CKPT --num-envs 1 --viewer viser --terrain stairs
```
- The student command uses double quotes, so `$TEACHER` is filled in before tmux starts.

**Starting a run again with the same name** makes a new folder with a newer date. The watch
lines take the newest. For an older one: `ls -d logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME>` and
set `RUN=` to that folder by hand.

---

## 5. Is it running? How far is it?

**Look at it directly:** `tmux ls` to see the session names, then `tmux attach -t <SESSION>`.
The box at the bottom shows `Learning iteration <done>/<total>`, `Iteration time:` (~4-7 s on
flat ground) and the time left (`ETA`). **The ETA wraps every 24 h**: it prints only h:m:s,
so `12:54:48` can mean 5 days and 12:54. Work it out instead: iteration time × iterations left. Leave with **Ctrl+B**, **D** (not Ctrl+C, that stops it).
Scripts print a line starting with `===` at every step (which experiment or rung, when it
started).

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

**Which runs are finished and checked**, with their verdicts (ladder, diagnosis and any run you
checked into its folder):
```bash
grep -H "overall" logs/rsl_rl/go2_pgtt_repro/*/hacking_check.txt
```
A run folder without `hacking_check.txt` is still training, was interrupted, or was never
checked.

**Is anything running at all?**
```bash
ps aux | grep -c "[s]cripts/train.py"
```
`0`: nothing is training. `1` or more: that many trainings are running.

---

## 6. Stop, restart, continue

**Stop a training (or a script):** `tmux attach -t <SESSION>`, then **Ctrl+C**. Press it once
and wait; a script may need a second **Ctrl+C** to stop starting the next step. Checkpoints
already saved stay.

**Restart the ladder from a rung:** section 4a. **The diagnosis from an experiment:** section 4c.
A run inside a script that was interrupted starts again from zero.

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
- Add the same changed settings the run was started with (4b), e.g.
  `--env.rewards.termination.weight=-100` for `rung4_term100`. They are not taken from the
  checkpoint.

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
| its hacking check (if it has one, section 10) | `cat logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME>/hacking_check.txt` | the OK / WARN / HACK table |

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
| `Train/mean_episode_length` | climbs to ~1000 by iteration ~100 and **stays** (rung 1+: dips mean falls; rung 4 fell from 811 to 173 = collapse) |
| `Metrics/hack/speed_ratio` | above 0.9 (rung 7 on stairs: above 0.65) |
| `Policy/mean_noise_std` | falls (0.7 → ~0.2-0.3) and **keeps falling**; turning back up (rung 4: 0.29 → 0.39) means the policy is getting worse; stuck at 1.0 was v4's entropy bug |
| `Metrics/hack/action_saturation` | below 0.05 |
| `Metrics/hack/action_tremor` | below 0.15 (repro0, without the smoothness penalty: ~0.35) |
| `Metrics/hack/body_ground_frac`, `leg_ground_frac` | ~0 (rung 1+) |
| `Episode_Termination/fell_over`, `body_contact` | ~0 |
| `Episode_Reward/*` | track_lin near 1.0, track_ang near 0.5, foot_phase near 1.0 |
| `Curriculum/terrain_levels` | rises over training (rung 7 only) |

### 9d. The important curves in the terminal (no browser needed)

GPU machine, repo folder, venv active:
```bash
python scripts/summarize_runs.py <RUN_NAME> <RUN_NAME> ...
```
- **`<RUN_NAME>`**: one or more run names (section 3), e.g.
  `python scripts/summarize_runs.py rung3_pushes r5_term100`. Put a healthy run (`rung3_pushes`)
  next to the one you're judging.
- Per run: its hack check verdict (if it has a `hacking_check.txt`) and episode length, noise
  std and body-contact endings at iteration 200, 800 and the newest one. Works while a run
  still trains. It takes the newest folder with that name.
- Healthy looks like rung 3: episode length stays above 900, noise std keeps falling, body-contact
  endings go towards 0.

---

## 10. Reward-hacking check

Runs the policy without noise and prints OK / WARN / HACK for each behaviour (lying down,
kneeling, ignoring the command, vibrating, falling, ...), plus where the reward comes from.
About 1 min on the GPU, 2 min on the laptop. The ladder and the diagnosis run it by themselves
after each run (`hacking_check.txt`, section 7).

**It tests the robot without randomization** (nominal mass, motors and friction, no pushes). A
policy that only falls apart on heavy, weak or slippery robots passes it anyway: rung 4 got WARN
while its training episodes had shrunk from 811 to 173 steps. So always look at the training
curves too (section 9d).

**The newest checkpoint of a run, GPU machine** (free tmux window), saved into the run folder:
```bash
RUN=$(ls -d logs/rsl_rl/go2_pgtt_repro/*_<RUN_NAME> | tail -1); CKPT=$(ls $RUN/model_*.pt | sort -V | tail -1); echo $CKPT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/check_reward_hacking.py --checkpoint-file $CKPT --device cuda:0 | tee $RUN/hacking_check.txt
```
- **`<RUN_NAME>`**: section 3, e.g. `r5_term100`.
- The last line of the output, `overall: OK / WARN / HACK`, is the verdict. Under it: which
  metric failed and where the reward comes from.
- Without `--device cuda:0` it runs on the CPU and looks stuck for an hour.

**Any one checkpoint, GPU machine:**
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/check_reward_hacking.py --checkpoint-file <CHECKPOINT> --device cuda:0
```

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

**Under the training conditions, with falls per robot parameter** (T1/T2): add
`--randomized True --num-envs 1024`. It keeps the domain randomization, pushes, delays and
noise, and adds a table of falls, legs on the ground and speed for the low / middle / high
third of each robot parameter (motor strength, Kp/Kd, trunk mass, friction, ...). A parameter
whose thirds differ a lot is what the falls come from. A few minutes on the GPU:
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/check_reward_hacking.py --checkpoint-file $CKPT --device cuda:0 --randomized True --num-envs 1024 | tee $RUN/hacking_check_randomized.txt
```
(`--randomized` needs the `True`. Without it, tyro complains about a missing value.)

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
| `CUDA out of memory` / `Graph launch error: Warp CUDA error 2: out of memory` | another training, viewer or check is holding the 3070. On 3 Nov rung 7 failed this way twice (16:55, 17:34): someone else's run was using 1.3 GB. Rung 7 at 4096 robots needs nearly all 8 GB. `nvidia-smi` lists the processes; stop the leftovers with `pkill -f scripts/play.py; pkill -f scripts/check_reward_hacking.py`, or `kill <PID>`. Check that the 3070's memory is near 0, then start again. If the process is someone else's, see "The machine is shared" in section 4 |
| `Unknown task` / task name not accepted | typo, or the code is old: `git pull`, then check the name with `grep task_id mjlab/tasks/velocity/config/go2/__init__.py` |
| `duplicate session: ...` | `tmux attach -t <name>`, or pick another name |
| `no server running on /tmp/tmux-...` | no tmux session exists: the machine most likely rebooted. See "rebooted" below |
| training started in a plain ssh window, then Ctrl+C or the window closed (`KeyboardInterrupt`) | it's dead. Start it again with the one-line tmux command (section 4) |
| `no matches found` / `No such file` for `*_<RUN_NAME>` | the run name is different or it has no checkpoint yet: `ls -t logs/rsl_rl/go2_pgtt_repro/` |
| the hacking check prints nothing for a long time | it's on the CPU: **Ctrl+C**, add `--device cuda:0`. Still there (`T` in `ps aux \| grep check_reward`)? `kill -9 <pid>` |
| laptop crawls or the desktop freezes in the viewer | forgot `--num-envs 1` |
| `localhost:6006` / `:8080` doesn't load | the ssh window with `-L` is closed, or TensorBoard / play isn't running on the GPU machine, or play hasn't finished starting |
| `Address already in use` (port 8080 or 6006) on the GPU machine | an old viewer / TensorBoard is still running: find it in your tmux windows and **Ctrl+C** it, or `pkill -f "scripts/play.py"` / `pkill tensorboard` |
| ladder stopped with `FAILED` | read that rung's `hacking_check.txt` (section 7); the change of that rung (section 3) caused it. Look at it in the viewer before changing anything |
| a run is much slower than ~7 s per iteration | something else uses the GPU or CPU: `nvidia-smi`, `top` (press `q` to leave) |
| terrain (rung 7) much slower than flat | ray casts cost the most on terrain (every ray is tested against every terrain box). Before 3 Nov the rungs also computed the student's LiDAR: `git pull` |
| `Assertion ... expects all elements of std >= 0.0 failed` / `CUDA error: device-side assert triggered` | the policy's noise std became NaN: a robot's physics blew up and poisoned the update. Since 3 Nov the rungs reset such robots (`Episode_Termination/nan_state` counts them): `git pull` and start the run again |
| `Episode_Reward/...` values in the millions or more (e.g. `default_pose: -19595920408576`), `speed_ratio` far outside 0-1 | some robots' physics blew up without turning NaN. Since 4 Nov (`18e7e21`) the rungs reset them (`Episode_Termination/physics_exploded` counts them). A run started before that is poisoned (the teacher's input normalizer kept the outliers): stop it, `git pull`, start again |
| `EOFError: Ran out of input` when loading a checkpoint | the file is empty, cut off by a crash. Use the newest non-empty one: `find $RUN -name 'model_*.pt' -size +0 \| sort -V \| tail -1` |
| **rebooted** during a training | see below |

**The GPU machine rebooted during a training.** A reboot kills every tmux session and the
training in it. Step by step:
1. Was it a reboot? `uptime` shows how long the machine has been up (`up 1:05` = 1 h 5 min).
   Shorter than the training has been going: it rebooted. Nothing trains:
   `ps aux | grep -c "[s]cripts/train.py"` prints `0`.
2. Mount the disk and go to the repo: section 1 (the `ls ... || (sudo ... mount ...)` line).
3. See what finished: `grep -H "overall" logs/rsl_rl/go2_pgtt_repro/*/hacking_check.txt`.
4. Start again from what didn't finish:
   - the ladder: section 4a, "Start from a later rung";
   - the diagnosis: section 4c, "Start from a later experiment";
   - a single run or the full teacher: continue from its last checkpoint, section 6.
5. Optional, why it rebooted: `last -x reboot shutdown | head -4`. A `shutdown` just before the
   `reboot`: someone or an update restarted it. A `reboot` with no `shutdown` before it: a crash
   or power cut.
