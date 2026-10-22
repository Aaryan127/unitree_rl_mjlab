# PGTT teacher-student on the Unitree Go2

This branch trains a terrain-aware locomotion policy for the Go2 and deploys it on the
real robot. It combines two papers:

- **PGTT** (Ntagkas et al. 2025, arXiv 2510.18348): the gait is shaped by rewards. Each
  leg follows a phase clock, and its swing height adapts to the terrain under it,
  measured with a height map. The policy outputs plain joint targets.
- **Teacher-student** (Miki et al. 2022, "Learning robust perceptive locomotion in the
  wild"): a teacher is trained with RL using a clean height map and privileged simulator
  state. A student, which only has what the real robot has (proprioception and a
  *noisy* height map), is then trained to imitate it, using a GRU belief encoder.

```
Stage 1  Mjlab-Velocity-PGTT-Teacher-Unitree-Go2   PPO          -> logs/rsl_rl/go2_pgtt_teacher
Stage 2  Mjlab-Velocity-PGTT-Student-Unitree-Go2   distillation -> logs/rsl_rl/go2_pgtt_student
Stage 3  scripts/export_deploy.py                  ONNX + deploy.yaml -> deploy/robots/go2/config/policy/pgtt/v0
Stage 4  deploy/robots/go2 (C++)                   go2_ctrl on the robot
```

## 1. Setup (once per machine)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch                       # GPU machine; on a CPU-only laptop use:
                                        # pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "mujoco-warp@git+https://github.com/google-deepmind/mujoco_warp@9491175b7cbea87e28d3e3e67733095317c33398"
pip install -e .                        # setup.py pins mujoco 3.4.0 / warp-lang 1.11.1
```
CPU-only machines: put `CUDA_VISIBLE_DEVICES=""` in front of `train.py` commands.

## 2. Debug before you train (CPU is fine)

```bash
python scripts/debug_env.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2                 # shapes + values of everything
python scripts/debug_env.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2 --agent random --steps 200
python scripts/debug_env.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2 --breakpoint    # pdb with `env`, `obs` ready
python scripts/play.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2 --agent zero         # look at terrain + robots
```
Run `debug_env.py` after **every** change to an observation, reward or sensor. It prints
each observation term (min/max/mean, NaN check), each reward term's average weighted
value, the commands and all sensors. To stop inside a single function, put `breakpoint()`
as its first line and run `debug_env.py` (2 robots, so tensors are small). Remove it
afterwards: one breakpoint in one place, never left in the managers.

A 3-iteration training smoke test (about 1 minute on CPU):
```bash
CUDA_VISIBLE_DEVICES="" python scripts/train.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2 --env.scene.num-envs=16 --agent.max-iterations=3
```

### Shortcuts and debug switches
`scripts/pgtt.sh` wraps the common commands (`config`, `debug`, `smoke`, `teacher`, `student`,
`play`, `latest`, `tb`, `export`; `scripts/pgtt.sh help`). Opt-in debug switches work with any
command: `MJLAB_DEBUG=1` (term summaries, network wiring, reward-balance table),
`MJLAB_DEBUG_NAN=1` (find the term that produced a NaN), `MJLAB_BREAK=reward:foot_phase`
(stop in pdb after a term; also `obs:<group>[/<term>]`, `step`, `update`). Details:
`mjlab/utils/debug.py` and CODE_GUIDE Part 7.

### How-to (the cpu branch's checklists, for this branch)
**Add a sensor:** add a `ContactSensorCfg` / `RayCastSensorCfg` in `unitree_go2_pgtt_env_cfg`
(`config/go2/env_cfgs.py`) and put it in `cfg.scene.sensors`. Contact sensors select geoms by
name (`ContactMatch(mode="geom", pattern=...)`); use `debug_env.py` to check `found`/`force`.

**Add an observation:** write `def my_obs(env, ...) -> Tensor [B, D]` in
`mjlab/tasks/velocity/mdp/observations.py`; add an `ObservationTermCfg(func=mdp.my_obs, ...)` to
a group in `make_pgtt_env_cfg`. Robot-measurable → `policy` (and the C++ side needs it too);
simulator-only → `privileged`. The network input size changes, so train from scratch.

**Add a reward:** write it in `mjlab/tasks/velocity/mdp/rewards.py` (terrain-aware: copy
`stance_on_edge`, use `_foot_terrain`); return a positive amount `[B]`; register it in the
`rewards` dict with `weight=0.0`; check with `debug_env.py`; enable with
`--env.rewards.<name>.weight <w>`.

**Ramp a penalty in over training:** add
`curriculum["ramp_<name>"] = CurriculumTermCfg(func=mdp.reward_weight_ramp, params={"reward_name": "<name>", "start_scale": 0.1, "ramp_steps": 1500 * 24})`
(see the loop at the end of `make_pgtt_env_cfg`). It scales the configured weight, so
command-line weights still work.

## 3. Train (GPU)

```bash
# Stage 1: teacher (PPO). Watch Episode_Reward/*, Curriculum/terrain_levels in tensorboard.
python scripts/train.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2 --env.scene.num-envs=4096

# Stage 2: student (distillation). Uses the latest teacher checkpoint automatically,
# or pass one explicitly with --teacher-checkpoint logs/rsl_rl/go2_pgtt_teacher/<run>/model_<it>.pt
python scripts/train.py Mjlab-Velocity-PGTT-Student-Unitree-Go2 --env.scene.num-envs=4096

tensorboard --logdir logs
```
Student losses to watch: `behavior` (action MSE vs teacher, should fall well below 0.01)
and `reconstruction` (height map + privileged state decoded from the GRU).

Watch a policy in simulation:
```bash
python scripts/play.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2 --checkpoint-file logs/rsl_rl/go2_pgtt_teacher/<run>/model_<it>.pt
python scripts/play.py Mjlab-Velocity-PGTT-Student-Unitree-Go2 --checkpoint-file logs/rsl_rl/go2_pgtt_student/<run>/model_<it>.pt
```

## 4. Export for the robot

```bash
python scripts/export_deploy.py --checkpoint-file logs/rsl_rl/go2_pgtt_student/<run>/model_<it>.pt
```
Writes `deploy/robots/go2/config/policy/pgtt/v0/exported/policy.onnx` and
`params/deploy.yaml` (generated from the training config, so joint order, gains,
default pose, action scale and observation order always match training), then replays
the simulator through ONNX Runtime and PyTorch and fails if the actions differ.

## 5. Deploy on the Go2 EDU

Dependencies on the computer that runs the controller (the Go2's Jetson or a laptop on
the robot's Ethernet): `sudo apt install libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev`,
then build/install [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds) and
[unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2).

```bash
cd deploy/robots/go2 && mkdir -p build && cd build && cmake .. && make
./go2_ctrl --network=<ethernet interface, e.g. enp5s0>
```
Joystick: `LT + up` stand up, `RT + X` start the PGTT policy, `LT + B` passive (damping),
`LT + up` back to standing. Always start with the robot hanging in a harness.

**Height map on the robot.** The controller listens on UDP port 9870 for the height map
(99 float32 values, layout documented in `deploy/include/isaaclab/devices/height_map/height_map.h`).
Without it, the policy receives zeros (= "flat ground / no map"), a case the student was
trained on, so it walks as a blind policy. Test the link with synthetic maps:
```bash
python scripts/send_height_map.py --mode flat --host <controller ip>
python scripts/send_height_map.py --mode step --step-height 0.1 --host <controller ip>
```
For real terrain, an elevation-mapping node (from the Go2 LiDAR) must fill the same grid
and send it the same way; that node is not part of this repo yet (see updates.md).

## 6. What each file does

Training (Python):

| File | Role |
|---|---|
| `mjlab/tasks/velocity/velocity_env_cfg.py` | `make_pgtt_env_cfg()`: the task recipe (observation groups, actions, commands, events, PGTT rewards, terminations, curriculum). `make_velocity_env_cfg()` is the untouched stock task. |
| `mjlab/tasks/velocity/config/go2/env_cfgs.py` | `unitree_go2_pgtt_env_cfg()`: Go2 sensors (foot contacts, foot normals, leg-ground contacts, 11x9 height scanner) and Go2-specific names. |
| `mjlab/tasks/velocity/config/go2/rl_cfg.py` | Teacher PPO and student distillation configs, and `obs_groups`: which observation groups each network input uses. |
| `mjlab/tasks/velocity/config/go2/__init__.py` | Registers the two task names. |
| `mjlab/tasks/velocity/mdp/gait_command.py` | Gait phase clock (per-leg phase, trot offsets, frequency). |
| `mjlab/tasks/velocity/mdp/observations.py` | `height_scan`, `gait_phase`, `geom_friction`, `body_external_wrench`, `contact_normals`. |
| `mjlab/tasks/velocity/mdp/rewards.py` | PGTT rewards; `foot_phase_tracking` (terrain-aware swing spline) and `swing_foot_contact` are the paper-specific ones. |
| `mjlab/terrains/config.py` | `PGTT_STAIRS_TERRAINS_CFG`: flat + stairs up/down, 1-13 cm steps, 10 difficulty rows. |
| `mjlab/sensor/raycast_sensor.py` | `GridPatternCfg.z_offset`: ray origins above the base so rays work on stairs going up. |
| `mjlab/utils/noise/` | `HeightScanNoiseModel`: offset, drift, jitter, outliers, missing cells, blind episodes for the student. |
| `mjlab/rsl_rl/networks/teacher_mlp.py` | Teacher network: height encoder + privileged encoder + policy MLP. |
| `mjlab/rsl_rl/networks/belief_encoder.py` | Student network: GRU belief encoder with gated height fusion, decoder, policy MLP. |
| `mjlab/rsl_rl/modules/actor_critic_wild.py` | Teacher actor-critic for PPO. |
| `mjlab/rsl_rl/modules/student_teacher_wild.py` | Frozen teacher + student, hidden-state handling, ONNX export. |
| `mjlab/rsl_rl/algorithms/distillation_wild.py` | Distillation loss = action imitation + reconstruction. |
| `mjlab/rsl_rl/runners/*.py`, `mjlab/rl/config.py` | Runners know the new classes; config dataclasses for them. |
| `scripts/train.py` | Picks PPO or distillation from the config; loads the teacher for the student. |
| `scripts/debug_env.py` | Debugging tool (section 2); `--config` prints all rewards/curriculum/obs groups. |
| `scripts/pgtt.sh` | Shortcuts for config, debug, smoke, train, play, export. |
| `mjlab/utils/debug.py` | Opt-in debug mode (`MJLAB_DEBUG`, `MJLAB_DEBUG_NAN`, `MJLAB_BREAK`) hooked into the managers, env step and runners. |
| `mjlab/tasks/velocity/mdp/curriculums.py` | `reward_weight_ramp`: regularization curriculum. |
| `scripts/export_deploy.py` | ONNX + deploy.yaml export and verification. |
| `scripts/send_height_map.py` | Test sender for the robot's height map link. |

Deployment (C++):

| File | Role |
|---|---|
| `deploy/include/isaaclab/algorithms/algorithms.h` | ONNX runner; feeds the GRU state `h_out` back into `h_in`, resets it when the policy starts. |
| `deploy/include/isaaclab/envs/mdp/observations/observations.h` | `gait_phase_pgtt` and `height_scan` observations (must match the Python ones). |
| `deploy/include/isaaclab/devices/height_map/height_map.h` | Thread-safe height map + UDP receiver, zeros when stale. |
| `deploy/include/isaaclab/envs/manager_based_rl_env.h` | Resets the policy state on start. |
| `deploy/robots/go2/config/config.yaml` | Adds the `PGTT` controller state. |

## 7. Suggested learning order

1. `scripts/train.py` and `mjlab/envs/manager_based_rl_env.py` (`step()`): the loop.
2. `make_pgtt_env_cfg()` top to bottom, with `debug_env.py` output next to it.
3. `gait_command.py`, then `foot_phase_tracking` in `rewards.py`: the PGTT idea.
4. `teacher_mlp.py` + `actor_critic_wild.py`, then `belief_encoder.py` + `student_teacher_wild.py`.
5. `scripts/export_deploy.py`, then the C++ files above.
