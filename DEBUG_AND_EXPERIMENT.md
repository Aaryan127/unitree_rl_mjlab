# Debugging and experimenting from the terminal (PGTT, Go2)

Everything here runs from the repo root with the venv active:

```bash
cd ~/unitree_rl_mjlab && source .venv/bin/activate
T=Mjlab-Velocity-PGTT-Teacher-Unitree-Go2      # stage 1: teacher (PPO)
S=Mjlab-Velocity-PGTT-Student-Unitree-Go2      # stage 2: student (distillation)
```

## 0. The three ways to change things

| You want to... | How | Works in |
|---|---|---|
| **Change a value** (weight, noise, terrain size, gains, learning rate) | a CLI flag: `--env.<path> <value>` or `--agent.<path> <value>` | `train.py`, `pgtt.sh teacher/student/smoke` |
| **Remove something** (reward, obs term, privileged info, event, sensor) | the `MJLAB_DROP` environment variable | every script |
| **Add something new** (new reward, new sensor, new obs) | a code edit (section 9) | — |

**Finding the exact flag name.** Every config value has a flag. Search the help text:

```bash
python scripts/train.py $T --help | grep -i foot-phase      # all flags for one reward
python scripts/train.py $T --help | grep -i "terrain-generator"
python scripts/train.py $T --help | grep -i "privileged"
```

Rules: `_` in names becomes `-` (`foot_phase` → `foot-phase`). Tuples take several values
(`--env.commands.gait.frequency-range 1.5 2.5`). Booleans inside `--env`/`--agent` take
`True` or `False`; a script's own on/off options are switches (`--breakpoint`, `--no-small-terrain`).

**See the whole current config** (rewards with weights and params, curriculum, observations):

```bash
scripts/pgtt.sh config            # = python scripts/debug_env.py $T --config
```

**CPU vs GPU.** `CUDA_VISIBLE_DEVICES=""` in front of any command forces CPU. `debug_env.py`,
`pgtt.sh smoke` and `export_deploy.py` already do this.

---

## 1. Fast checks (no training)

```bash
scripts/pgtt.sh debug                                  # build env, 2 robots, 50 zero-action steps, print everything
python scripts/debug_env.py $T --agent random --steps 200   # random actions: do robots fall? rewards sane?
python scripts/debug_env.py $T --play                  # play config (no noise, no pushes)
python scripts/debug_env.py $T --num-envs 8 --env-id 3 # print robot 3's values
python scripts/debug_env.py $T --no-small-terrain      # full 10 x 20 terrain (slow on CPU)
```

What it prints: every observation term (shape, min/max/mean, NaNs), every reward term
(average weighted value per second), commands, sensors, and how many episodes reset.
**Run this after every change, before training.** It takes about a minute.

`debug_env.py` only takes its own options (above) plus `MJLAB_DROP` / `MJLAB_BREAK`. To test
a CLI value change, use a smoke run (next section).

---

## 2. Short training runs to test a change

```bash
scripts/pgtt.sh smoke teacher                          # 3 iterations, 16 robots, small terrain, CPU
scripts/pgtt.sh smoke student                          # same for distillation (uses the latest teacher)
scripts/pgtt.sh smoke teacher --env.rewards.foot-phase.weight 2.0   # extra flags pass through
```

A smoke run proves the change builds, trains, and saves. It tells you nothing about how well
the policy walks. For that you need a real GPU run (section 8).

---

## 3. pdb: stopping inside the running code

### 3.1 Stop after building the env
```bash
python scripts/debug_env.py $T --breakpoint
```
You land in pdb with `env`, `obs`, `robot` ready.

### 3.2 Stop at a precise point while it runs (`MJLAB_BREAK`)
Works with `debug_env.py`, `train.py`, `play.py` and `pgtt.sh`.

```bash
MJLAB_BREAK=reward:foot_phase  scripts/pgtt.sh debug    # after the foot_phase reward is computed
MJLAB_BREAK=obs:height_noisy   scripts/pgtt.sh debug    # after any term of the student height map
MJLAB_BREAK=obs:privileged/foot_friction scripts/pgtt.sh debug   # after one obs term
MJLAB_BREAK=step               scripts/pgtt.sh debug    # start of every env.step()
MJLAB_BREAK=update             scripts/pgtt.sh smoke teacher     # after a PPO update (losses ready)
MJLAB_BREAK=reward:foot_phase,obs:policy scripts/pgtt.sh debug   # several targets
```

| Extra switch | What it does |
|---|---|
| `MJLAB_BREAK_AFTER=100` | skip the first 100 matches (stop later in the episode) |
| `MJLAB_BREAK_COUNT=5` | stop up to 5 times (`0` = every time); default 1 |
| `MJLAB_DEBUG_ENV=3` | which robot's numbers get printed |

Variables at the stop:
- **reward stop:** `env`, `name`, `value`. `value` is weight × term × dt, one value per robot.
- **obs stop:** `env`, `group`, `name`, `value`. `value` is after noise, clip, scale and delay.
- **step stop:** `env`, `action`.
- **update stop:** `loss_dict`, `iteration`. Type `u` to reach the runner (`self.alg`, `self.env`).

### 3.3 Automatic stops and prints
```bash
MJLAB_DEBUG=1 scripts/pgtt.sh smoke teacher       # first-step stats of every obs/reward term,
                                                   # obs wiring per network, reward-balance table
MJLAB_DEBUG=1 MJLAB_DEBUG_EVERY=200 ...            # reward-balance table every 200 steps
MJLAB_DEBUG_NAN=1 scripts/pgtt.sh smoke teacher   # check every term each step; on NaN/Inf print
                                                   # which term + which robots and stop in pdb
```

### 3.4 Stop inside your own function
Put `breakpoint()` as the first line of a reward or observation function (for example in
`mjlab/tasks/velocity/mdp/rewards.py`), then run `scripts/pgtt.sh debug`. Delete it afterwards.

### 3.5 pdb commands
| Command | Does |
|---|---|
| `p x` / `pp x` | print / pretty-print |
| `n` | next line, without stepping into calls |
| `s` | step into a call |
| `c` | continue to the next stop |
| `u` / `d` | move up / down the call stack (`u` from a term = its manager) |
| `l` / `ll` | show code around the line / the whole function |
| `w` | where am I (stack trace) |
| `interact` | full Python shell with the current variables (Ctrl-D to return) |
| `q` | quit the program |

### 3.6 Useful things to type at a stop
```python
value.shape, value[0]                                   # this term, robot 0
robot = env.scene["robot"]
robot.data.joint_pos[0], robot.data.joint_vel[0]        # joint state, sim order (FL, FR, RL, RR)
robot.data.root_link_pos_w[0]                           # base position (world)
env.command_manager.get_command("twist")[0]             # velocity command [vx, vy, wz]
env.command_manager.get_command("gait")[0]              # [cos phi x4, sin phi x4, frequency]
env.scene["lidar"].data.distances[0]                    # LiDAR ray distances (-1 = miss)
env.scene["height_scanner"].data.hit_pos_w[0, :, 2]     # ground height under the 99 rays
env.scene["feet_ground_contact"].data.found[0]          # feet in contact
env.reward_manager.active_terms, env.observation_manager.active_terms
env.sim.model.actuator_gainprm[:, 0, 0]                 # kp of joint 0 for every robot (randomized)
env.extras["log"]                                       # what goes to tensorboard
```

**Change things live.** The change lasts until the program exits:
```python
env.reward_manager.get_term_cfg("foot_phase").weight = 2.0
env.reward_manager.get_term_cfg("foot_phase").params["sigma"] = 0.004
env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["x"] = (-1.0, 1.0)
```
then `c` and watch what happens. When something helps, make it permanent with a CLI flag or
in the code. Exception: rewards with a `ramp_<name>` curriculum term (`joint_power`,
`action_rate`, ...) get their weight overwritten at the next episode reset. Change those
with CLI flags instead.

---

## 4. Terrain

```bash
scripts/pgtt.sh teacher --terrain mixed      # default: flat + stairs + rough + slopes + waves
scripts/pgtt.sh teacher --terrain stairs     # PGTT paper setup: stairs up/down + some flat
scripts/pgtt.sh teacher --terrain rough      # flat, random rough, slopes, waves (no stairs)
scripts/pgtt.sh teacher --terrain flat       # infinite plane, no terrain curriculum
python scripts/play.py $S --checkpoint-file <model.pt> --terrain stairs   # watch on one surface
```

Tune the terrain (without `--terrain`; see the warning below):
```bash
--env.scene.terrain.terrain-generator.num-rows 10        # difficulty levels (rows)
--env.scene.terrain.terrain-generator.num-cols 20        # tiles per level (columns)
--env.scene.terrain.terrain-generator.difficulty-range 0.0 1.0
--env.scene.terrain.terrain-generator.curriculum False   # random difficulty instead of a curriculum
--env.scene.terrain.max-init-terrain-level 3             # hardest row robots start on
--env.scene.terrain.terrain-generator.sub-terrains.stairs-up-30.step-height-range 0.02 0.18
--env.scene.terrain.terrain-generator.sub-terrains.stairs-up-30.step-width 0.25
--env.scene.terrain.terrain-generator.sub-terrains.flat.proportion 0.1
--env.scene.terrain.terrain-generator.sub-terrains.random-rough.noise-range 0.0 0.08
--env.scene.terrain.terrain-generator.sub-terrains.slope-up.slope-range 0.0 0.35
```
Sub-terrain names: `flat`, `stairs-up-30`, `stairs-up-40`, `stairs-down-30`, `stairs-down-40`,
`random-rough`, `slope-up`, `slope-down`, `waves`.

> **Warning:** `--terrain <profile>` rebuilds the terrain generator after the other flags are
> read. It keeps `num-rows`, `num-cols`, `curriculum`, `seed`, `border-width` and
> `difficulty-range`, but resets everything under `sub-terrains.*` (stair heights, slope
> ranges, proportions) to the profile's values. To change sub-terrain settings, leave out
> `--terrain` (the default terrain is `mixed`).

Check curriculum progress in tensorboard: `Curriculum/terrain_levels` (average row).

---

## 5. Rewards

**Change a weight or a parameter:**
```bash
scripts/pgtt.sh teacher --env.rewards.foot-phase.weight 1.5
scripts/pgtt.sh teacher --env.rewards.foot-phase.params.sigma 0.004
scripts/pgtt.sh teacher --env.rewards.track-lin-vel.weight 2.0
scripts/pgtt.sh teacher --env.rewards.action-rate.weight -0.02
```
Find a reward's parameters: `python scripts/train.py $T --help | grep "rewards.<name>"`.

**Turn on an optional reward.** These are already written but have weight 0, so they are skipped:
`swing_clearance`, `stance_on_edge`, `leg_contact`, `foot_slip`, `lin_vel_orthogonal`,
`joint_vel`, `joint_acc`, `action_smoothness`.
```bash
scripts/pgtt.sh teacher --env.rewards.swing-clearance.weight 0.5 --env.rewards.foot-slip.weight -0.1
```

**Remove a reward completely:**
```bash
MJLAB_DROP=reward:default_pose scripts/pgtt.sh teacher
```
This also removes its `ramp_<name>` curriculum term.

**Penalty ramp.** Some penalties start at 10 % and grow to full strength:
```bash
--env.curriculum.ramp-joint-power.params.start-scale 0.3
--env.curriculum.ramp-joint-power.params.ramp-steps 20000
MJLAB_DROP=curriculum:ramp_action_rate ...              # full penalty from the start
```

**While tuning, watch** (tensorboard, section 8):
- `Episode_Reward/<name>`: each term's contribution. Watch for one term swamping the others.
- `Episode_Reward/foot_phase` against `Episode_Reward/track_lin_vel`: gait shape vs actually moving.
- `Curriculum/terrain_levels`: is the robot getting onto harder terrain?
- `MJLAB_DEBUG=1`: prints the same balance in the terminal.

---

## 6. Observations, sensors and privileged info

Current groups (`scripts/pgtt.sh config` shows them):

| Group | Who sees it | Terms |
|---|---|---|
| `policy` | teacher + student (the robot) | base_ang_vel, projected_gravity, command, joint_pos, joint_vel, actions, gait_phase |
| `proprio_history` | teacher | joint_pos/joint_vel/action history |
| `height` | teacher (clean ray-cast map) | height_scan |
| `height_noisy` | student (LiDAR-built map) | height_scan |
| `privileged` | teacher only (sim-only truth) | base_lin_vel, foot_contact, foot_contact_forces, foot_contact_normals, foot_friction, leg_ground_contact, base_external_wrench |
| `critic` | value function only | proprio + base_lin_vel |

Sensors: `feet_ground_contact`, `feet_ground_normals`, `leg_ground_contact`,
`height_scanner`, `lidar`.

**Drop privileged info** (does the teacher need it?):
```bash
MJLAB_DROP=obs:privileged/foot_friction,obs:privileged/base_external_wrench scripts/pgtt.sh teacher
```
The networks resize themselves.

**Drop a sensor.** You must also drop every term that reads it. If you forget, it stops and
lists them:
```bash
MJLAB_DROP=sensor:leg_ground_contact,obs:privileged/leg_ground_contact,reward:leg_contact scripts/pgtt.sh teacher
```

**Noise and delay:**
```bash
--env.observations.policy.enable-corruption False            # no sensor noise at all (policy group)
--env.observations.height-noisy.enable-corruption False      # perfect LiDAR map for the student
--env.observations.policy.terms.joint-vel.delay-max-lag 2    # 0-40 ms joint-velocity delay
--env.observations.height-noisy.terms.height-scan.delay-max-lag 5
--env.observations.height-noisy.terms.height-scan.noise.blind-prob 0.1    # episodes with no map
--env.observations.height-noisy.terms.height-scan.noise.outlier-prob 0.02
--env.observations.height-noisy.terms.height-scan.params.point-noise-std 0.02   # LiDAR point noise (m)
--env.observations.height-noisy.terms.height-scan.params.min-coverage 0.4      # LiDAR map settings
```
See every option: `python scripts/train.py $S --help | grep height-noisy`.

> **Rules that keep training and the robot consistent:**
> - **The student must use the same `MJLAB_DROP` as its teacher**, at least for the
>   teacher's groups. Otherwise the teacher checkpoint doesn't fit (shape mismatch on load).
> - **Changing the `policy` group or `height_noisy` changes what the robot must provide.**
>   `export_deploy.py` writes the new list into deploy.yaml. The C++ controller only knows the
>   terms in `deploy/include/isaaclab/envs/mdp/observations/observations.h`, and export stops
>   if a term has no C++ version.
> - Changing a LiDAR map setting changes the robot bridge too. It is exported automatically;
>   re-run `python scripts/check_lidar_map.py` after a change.

---

## 7. Randomization, disturbances, commands

```bash
MJLAB_DROP=event:push_robot,event:external_wrench ...       # no shoves (is it the pushes?)
MJLAB_DROP=event:actuator_delay,event:pd_gains ...          # no motor randomization
--env.events.push-robot.interval-range-s 2.0 5.0            # push more often
--env.events.push-robot.params.velocity-range.x -1.0 1.0    # push harder
--env.events.pd-gains.params.kp-range 0.7 1.3               # wider kp randomization
--env.events.effort-limits.params.effort-limit-range 0.7 1.0
--env.commands.twist.ranges.lin-vel-x -0.5 1.5              # commanded speed range (m/s)
--env.commands.twist.ranges.ang-vel-z -0.5 0.5
--env.commands.gait.frequency-range 1.5 2.5                 # gait clock (Hz); deploy uses 2.0
```
Event names: `reset_base`, `reset_robot_joints`, `push_robot`, `external_wrench`,
`body_friction`, `base_mass`, `base_com`, `pd_gains`, `effort_limits`, `actuator_delay`,
`link_mass`, `joint_friction`, `joint_armature`, `encoder_bias`.
Termination names: `time_out`, `fell_over`.

If you change the twist ranges, the robot's joystick limits follow (they are exported). The
deploy gait frequency must stay inside `frequency-range`; export checks this.

---

## 8. Running and comparing experiments

**Name every run** so you can compare later:
```bash
scripts/pgtt.sh teacher --agent.run-name sigma004 --env.rewards.foot-phase.params.sigma 0.004
scripts/pgtt.sh teacher --agent.run-name no_friction_priv   # with MJLAB_DROP=...
```
Runs go to `logs/rsl_rl/go2_pgtt_teacher/<date>_<run-name>/`.

**Watch live while it trains** (second terminal):
```bash
scripts/pgtt.sh tb                       # tensorboard; open http://localhost:6006
scripts/pgtt.sh play teacher             # newest checkpoint of the newest run, in a viewer
python scripts/play.py $T --checkpoint-file logs/rsl_rl/go2_pgtt_teacher/<run>/model_1500.pt --viewer viser
python scripts/play.py $T --agent random     # env only, no policy (check terrain, spawning)
```

**Training settings:**
```bash
--env.scene.num-envs 4096                 # robots in parallel (pgtt.sh teacher already sets 4096)
--agent.max-iterations 20000
--agent.save-interval 100
--agent.algorithm.learning-rate 5e-4
--agent.algorithm.entropy-coef 0.005
--agent.policy.init-noise-std 0.8
--agent.seed 1
--agent.logger wandb                      # log to Weights & Biases instead of tensorboard
--agent.resume True --agent.load-run <run folder name>      # continue a stopped run
```
Student: `scripts/pgtt.sh student --teacher-checkpoint logs/rsl_rl/go2_pgtt_teacher/<run>/model_X.pt`
(default: the newest teacher).

**Measure, don't eyeball:**
```bash
python scripts/evaluate_pgtt.py --checkpoint-file <student model.pt>          # flat/stairs/rough, 3 seeds
python scripts/evaluate_pgtt.py --checkpoint-file <model.pt> --terrains stairs --difficulty 0.8
python scripts/evaluate_pgtt.py --checkpoint-file <model.pt> --blind          # no height map: does it still walk?
```
This reports failures, velocity tracking, progress and map coverage per terrain, and writes
`logs/pgtt_evaluation.json`.

**Then export and check the deploy files:**
```bash
scripts/pgtt.sh export                   # newest student -> deploy/robots/go2/config/policy/pgtt/v0
python scripts/check_lidar_map.py        # robot LiDAR bridge == training map
```

A good loop for one idea: `debug` → `smoke` → GPU run with a `--agent.run-name` → tensorboard
→ `evaluate_pgtt.py` → keep it (put it in the code) or drop it.

---

## 9. Adding something new (needs code)

The terminal can change and remove what exists. New things need a few lines of code.
CODE_GUIDE.md walks through each one.

| New thing | Where |
|---|---|
| reward | function in `mjlab/tasks/velocity/mdp/rewards.py` + an entry in `make_pgtt_env_cfg()` rewards (`mjlab/tasks/velocity/velocity_env_cfg.py`) with weight 0 → then turn it on from the CLI |
| observation / privileged term | function in `mjlab/tasks/velocity/mdp/observations.py` + a term in the right group in `make_pgtt_env_cfg()` |
| sensor | `cfg.scene.sensors` in `unitree_go2_pgtt_env_cfg()` (`mjlab/tasks/velocity/config/go2/env_cfgs.py`) |
| terrain type | `mjlab/terrains/config.py` (+ `terrain_profiles.py` for a new `--terrain` name) |
| randomization | an `EventTermCfg` in `make_pgtt_env_cfg()` events (functions in `mjlab/envs/mdp/events.py`) |

Tip: add new rewards with weight 0. They then cost nothing until you switch them on with
`--env.rewards.<name>.weight`.

---

## 10. One-page cheat sheet

| Goal | Command |
|---|---|
| see config | `scripts/pgtt.sh config` |
| sanity check | `scripts/pgtt.sh debug` |
| pdb after build | `python scripts/debug_env.py $T --breakpoint` |
| pdb in a reward | `MJLAB_BREAK=reward:foot_phase scripts/pgtt.sh debug` |
| pdb in an obs | `MJLAB_BREAK=obs:height_noisy scripts/pgtt.sh debug` |
| catch NaNs | `MJLAB_DEBUG_NAN=1 scripts/pgtt.sh smoke teacher` |
| reward balance | `MJLAB_DEBUG=1 scripts/pgtt.sh smoke teacher` |
| change a weight | `--env.rewards.<name>.weight X` |
| optional reward on | `--env.rewards.swing-clearance.weight 0.5` |
| remove anything | `MJLAB_DROP=reward:x,obs:group/term,event:x,sensor:x,curriculum:x,termination:x` |
| terrain type | `--terrain flat\|stairs\|rough\|mixed` |
| stairs height | `--env.scene.terrain.terrain-generator.sub-terrains.stairs-up-30.step-height-range a b` |
| no noise | `--env.observations.<group>.enable-corruption False` |
| quick test | `scripts/pgtt.sh smoke teacher <flags>` |
| real run | `scripts/pgtt.sh teacher --agent.run-name <name> <flags>` |
| watch | `scripts/pgtt.sh tb` / `scripts/pgtt.sh play teacher` |
| measure | `python scripts/evaluate_pgtt.py --checkpoint-file <pt>` |
| find a flag | `python scripts/train.py $T --help \| grep <word>` |
