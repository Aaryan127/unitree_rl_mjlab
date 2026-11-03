# PGTT teacher-student on Go2: progress log

How to use the branch and what each file does: see [instruction.md](instruction.md).
Every command (train, tmux, find runs, check, view, copy): [COMMANDS.md](COMMANDS.md).
Every training run, change and result (the experiment log):
[doc/experiments.md](doc/experiments.md).

## Status

- [x] Stairs terrain + curriculum (`PGTT_STAIRS_TERRAINS_CFG`)
- [x] Height-scan sensor (11 x 9 rays, 1.0 x 0.8 m, yaw-aligned) + observation
- [x] Gait phase command + phase observations
- [x] PGTT rewards (Table I), incl. terrain-aware foot-phase spline and swing-contact penalty
- [x] Privileged observations + teacher (`ActorCriticWild`, PPO)
- [x] Student (`StudentTeacherWild`: GRU belief encoder + decoder) + distillation with reconstruction loss
- [x] Height-map noise model for the student (offset, drift, jitter, outliers, missing cells, blind episodes)
- [x] Export: ONNX + generated deploy.yaml, verified against PyTorch in closed loop
- [x] C++: recurrent ONNX runner, `gait_phase_pgtt` + `height_scan` observations, UDP height map, PGTT FSM state
- [x] Debugging tool `scripts/debug_env.py`
- [x] Verified on CPU: teacher and student training run, checkpoints save, export verifies,
      C++ runner matches ONNX Runtime exactly, UDP height map link works
- [x] Reward-hacking checks: `hacking_monitor` metrics in TensorBoard, `scripts/check_reward_hacking.py`,
      `scripts/eval_gait.py`
- [x] Official PGTT setup reproduced on our Go2 (`repro0`): walks, 95% of the commanded speed, no falls
- [x] Ladder from the reproduction to the full teacher (rungs 1-9, `scripts/train_ladder.sh`):
      rungs 1-3 walk
- [x] Rung 4/5 collapse: a bug (motors weakened at every reset), fixed; rungs 4-6 pass
- [x] Flat-ground teacher: rung 6 (`2026-11-03_02-53-15_rung6_commands`), check WARN
- [ ] Rung 6 with a second seed (`rung6_seed2`)
- [ ] Rung 7 (terrain): running again with the fixes (9 s/iteration)
- [x] Rung 8 (privileged dynamics) and rung 9 (hard terrain gradient, boulders): built and tested
- [ ] Rungs 8 and 9 on GPU (they follow rung 7 automatically)
- [ ] Full teacher training on GPU (rung 9's setup, 10k iterations, 4096 envs)
- [ ] Student task with the rung 8+ teacher inputs (`dynamics`), and a per-part reconstruction check
- [ ] Full student distillation on GPU (check first that the student task loads a rung-7 teacher)
- [ ] Play both in simulation; check stairs success per terrain level
- [ ] Build `go2_ctrl` on the robot (needs unitree_sdk2 + cyclonedds; not buildable on the dev laptop)
- [ ] Real robot: blind test on flat ground first (no height map -> zeros)
- [x] ROS 2 PointCloud2 bridge: LiDAR points -> robot-centric 11 x 9 grid -> UDP
- [ ] Confirm LiDAR topic, TF, coverage, and timing on the physical Go2

## 2026-11-04: rung 7 rerun, the explosion guard, rungs 8 and 9
- Rung 7 rerun with the fixes: **9.26 s per iteration** (was 80 s), so ~15.5 h for 6000. `nan_state` 0.
  Chirag's run had held the 3070 at first; it caused the two out-of-memory starts.
- New problem: some robots' physics blew up without turning NaN (hip angles ~10⁶ rad,
  `default_pose` −2×10¹³ per episode). Fix: a `physics_exploded` termination (a joint faster than
  100 rad/s or the base faster than 20 m/s) (`18e7e21`).
- Correction: the teacher's observation normalization is **on** (`RslRlPpoActorCriticWildCfg`
  defaults it to True). So those robots also poisoned its running statistics. Runs started
  before `18e7e21` are not usable.
- Rung 8 "dynamics": the teacher also sees 90 randomized dynamics values (mass, CoM, motor
  strength, Kp/Kd, armature, damping, joint friction, encoder bias, delay). The privileged
  group grows from 53 to 143 (`ea62402`).
- Rung 9 "hard terrain": `PGTT_HARD_TERRAINS_CFG`. Stairs 2 → 18 cm, rough ground scaling with
  the row 0 → 10 cm (20 cm cells), a new boulders tile 3 → 15 cm, swing cap 21 cm. Same box
  count as rung 7 (`ea62402`).
- Research (`doc/research_terrain_and_privileged_state.md`): a trotting Go2 handles 10-15 cm
  comfortably and 18-20 cm as a stretch; 25-30 cm needs jumping. No paper gives its teacher
  PD gains or delays, so before distilling, check the student can reconstruct each part.
- Header guides ("what this file is for, how to change it") in 12 reward, terrain and env
  files (`8eb379f`).

## 2026-11-03: rungs 4-6 pass, rung 7 crash fixed
- With the motor fix the ladder restarted at rung 4:
  - rung 4: WARN, legs down 0.297 (slight kneeling, nothing prices it yet);
  - rung 5: WARN; at iteration 1260 its episodes last 996 steps, speed ratio 0.91, legs
    down 0.0005, noise std 0.21;
  - rung 6: WARN. That makes it the flat-ground teacher.
- `base_height` target raised 0.25 → 0.265 m, weight kept at −50 (`4bb3280`). Applies from rung 7.
- Rung 7 (terrain) crashed at iteration 4: the noise std became NaN, at 80 s per iteration.
  The ETA of 12:54:48 wraps every 24 h and really meant ~5.5 days.
- Profiled: the physics was ~2% of the step time. ~75% was the student-only LiDAR map, cast
  against 4049 stair boxes; most of the rest was the height scanner re-cast ~4× per step.
- Fixes:
  - the rungs drop the LiDAR (the student keeps it);
  - `RayCastSensor` caches its casts until a body moves;
  - a `nan_state` termination resets robots with NaN physics.

  Stairs step 2971 → 239 ms on the CPU. Tests added.
- Details: doc/experiments.md, "Rung 7: the crash and the slowness".

## 2026-11-02: diagnosing the rung 4/5 collapse
- **Result (evening): it was a bug.** `randomize_effort_limits` multiplied the *current*
  torque limit at every reset, so the motors faded: 23.5 Nm → 11.4 after 10 resets → 2.6 after
  30. Fixed to scale the defaults (`a21cf12`), with a regression test.
  - Two diagnosis runs had already failed: `r5_term100` and `r4_ent001`, with the noise std
    flat and the falls still rising.
  - New: `check_reward_hacking.py --randomized True`, with falls per robot parameter (`413f2d1`).
  - Corrected: the legs are not frictionless (they grip with μ 1.0).
- Rungs 4 and 5 learn to walk by iteration 100-200, then get worse. Rung 4's episodes go from
  811 to 173 steps; rung 5's robots end up falling about once a second.
- What moved first: trunk/hip falls rise from iteration ~200. The noise std turns up later
  (rung 4: 0.29 → 0.39, rung 5: → 0.58) and amplifies it. The critic is fine.
- A fall costs only −0.02 (`termination` −1 × dt). With rung 4's heavy, weak or slippery robots,
  dropping them costs the policy little.
- `rung4_term100` (fall cost −2): it stops falling but kneels in place (legs on the ground 65%
  of the time, speed ratio 0.005), because kneeling is free in rung 4.
- The hack check said only WARN for rung 4: it runs without randomization. Training curves
  are now part of every judgement (`scripts/summarize_runs.py`).
- `scripts/diagnose_collapse.sh`: four runs (`r5_term100`, `r4_ent001`, `r5_term100_ent001`,
  `r6_term100`) with pass rules fixed in advance. Started 14:35; a GPU-machine reboot killed it;
  restarted 16:36.
- COMMANDS.md rewritten around tmux one-liners, real times (~7 s per iteration, ~3 h per
  1500), command-line overrides, the full-teacher command and recovery after a reboot.
- Details: doc/experiments.md, "Rungs 4 and 5 collapse".

## 2026-11-01: the ladder from the reproduction to the full teacher
- Rung 0b (`smooth0b`): repro0 plus `action_acc` −0.02 (penalty on the action's second
  difference), against a 12 Hz tremor on top of the gait. Walks.
- Rungs 1-7 (`Mjlab-Velocity-PGTT-Rung<N>-Unitree-Go2`, `unitree_go2_pgtt_rung_env_cfg` in
  `config/go2/env_cfgs.py`): each adds one change from the teacher task.
  1. full-body collisions
  2. motor and sensor delays
  3. pushes and external force
  4. our wider randomization
  5. base_height, leg_contact, joint_limits
  6. the teacher's commands
  7. terrain, 6000 iterations
- Verified by diffing the flattened configs: each rung differs from the one before only by
  its change.
- `scripts/train_ladder.sh` trains them in order, checks each, and stops at the first HACK.
- Results: rungs 1-3 walk (WARN: kneeling grows 0.001 → 0.25 of the time, nothing prices it
  before rung 5); rung 4 collapses; rung 5 HACK.
- `check_reward_hacking.py` counts each fall once (per robot), as falls per robot-minute.

## 2026-10-31: the official PGTT setup reproduced, and it walks
- After four failed flat teacher runs (v1 lay down, v2 fell on purpose, v3 vibrated, v4
  bang-bang), the approach changed: reproduce the published setup first, then add one thing at
  a time.
- `Mjlab-Velocity-PGTT-Repro-Unitree-Go2` (experiment `go2_pgtt_repro`):
  - PGTT's feet-only collisions, rewards, gains and commands;
  - Brax PPO's settings (γ 0.97, lr 3e-4, 8 epochs, clip 0.3, entropy 0.01 on the
    tanh-squashed action).
- `repro0` walks: speed ratio 0.95, no falls, no saturation. Its one flaw is a 12 Hz action
  tremor.
- Fixed: the entropy bonus is now taken on the squashed action (`squashed_entropy`). On the raw
  Gaussian it pushed the std to its cap and the actions into saturation.
- `scripts/eval_gait.py`: per-command gait numbers (tracking, swing frequency, foot lift,
  slip, tremor).

## 2026-10-27 to 10-30: four flat teacher runs, four reward hacks
- v1: lay on the belly and paddled. Fixes: trunk/hip contact ends the episode, rebalanced
  rewards, `hacking_monitor` and `check_reward_hacking.py`.
- v2: walked, then fell on purpose. Fixes: each step's reward clipped at 0 (as PGTT does),
  noise std capped.
- v3: vibrated its joints between the limits. Fixes: tanh-bounded actions, PGTT's gains and
  foot-phase formula.
- v4: bang-bang vibration. Cause: the entropy bonus on the unsquashed action (fixed 10-31).

## 2026-10-26: student trained on a simulated-LiDAR map, identical to the robot bridge
- `LidarPatternCfg` (raycast_sensor.py): head-mounted spinning-LiDAR beams (1440 rays), tilting
  with the body. `mdp.lidar_height_map` builds the student's map from them with the bridge's
  gridding (nearest cell, 20 % lowest point, >= 2 points, "no map" below 50 % coverage).
- Bridge switched to the same gridding; settings exported in deploy.yaml (`lidar_map`) and
  loaded by the bridge. `scripts/check_lidar_map.py`: bridge == training map (0.0 m) on 32
  robots standing and moving; LiDAR covers ~80 % of the grid, blind spots behind the robot.
- Teacher/critic keep the clean ray-cast map (privileged). LiDAR mount position/tilt are an
  approximation: set them from the real robot's TF.

## 2026-10-26: LiDAR height-map bridge checked and hardened
- Bridge output equals the simulator's `height_scan` (0.000 m) on platforms and in front of
  rising stairs (fixed: max_z cut removed stair tops; trunk self-filter box instead).
- Rolling 0.4 s point buffer in the world frame, sampled at the latest pose; min coverage 0.5;
  pose from TF or a nav_msgs/Odometry topic; `--use-receive-time` for unsynced clocks.
- 9 unit tests (tests/test_lidar_height_map.py); ROS node logic exercised with fake ROS modules.
- Still to do on the robot: real topic/frame names, deskewing, coverage and latency.

## 2026-10-22: debugging and tuning features from the cpu branch, done safely
- Opt-in debug mode `mjlab/utils/debug.py` (MJLAB_DEBUG / MJLAB_DEBUG_NAN / MJLAB_BREAK), one guarded
  hook line in the observation manager, reward manager, env.step and both runners. Off = no cost.
- Regularization curriculum `reward_weight_ramp` (Miki; cpu's reward_weight stages): joint_power,
  joint_torques, action_rate (+ optional penalties) start at 10 % and reach 100 % at 1500 iterations.
- Optional rewards, weight 0 by default: swing_clearance (cpu's inverted feet_terrain_clearence,
  fixed), stance_on_edge, leg_contact (cpu's shank_knee_collision, now leg-vs-ground), foot_slip,
  lin_vel_orthogonal (Miki), joint_vel, joint_acc, action_smoothness (cpu's target_smoothness).
- Shared `_foot_terrain` helper for terrain-aware rewards (foot_phase_tracking refactored onto it).
- Student height-map noise: noisy episodes (Miki's NOISY type) and horizontal map shift (cpu's
  Map_noise_cfg).
- `debug_env.py --config` (live version of cpu's current_reward_cfg.md), `scripts/pgtt.sh` (portable
  train.sh), how-tos in instruction.md.
- Verified on CPU: debug prints/breaks/NaN checks, teacher with an optional term on, student,
  export (ONNX matches PyTorch).
- Not taken from cpu: FTG/IK (PGTT replaces it), MJX/JAX port (unfinished), .agent docs.

## Observation spec (as implemented)

| group | terms | dims | used by |
|---|---|---|---|
| `policy` | base_ang_vel 3, projected_gravity 3, command 3, joint_pos 12, joint_vel 12, actions 12, gait_phase 9 | 54 | teacher, student, **robot** |
| `proprio_history` | joint_pos x3, joint_vel x2, actions x2 | 84 | teacher |
| `height` | height_scan (clean) | 99 | teacher, critic, reconstruction target |
| `height_noisy` | height_scan (HeightScanNoiseModel) | 99 | student, **robot** |
| `privileged` | base_lin_vel 3, foot_contact 4, foot_contact_forces 12, foot_contact_normals 12, foot_friction 4, leg_ground_contact 12, base_external_wrench 6 | 53 | teacher, critic, reconstruction target |
| `critic` | base_lin_vel + policy terms, no noise | 57 | critic |

Deployed observation vector (153) = `policy` (54) + `height_noisy` (99), in that order.

## Design decisions (where the papers leave a choice)

These describe the original teacher task, `Mjlab-Velocity-PGTT-Teacher-Unitree-Go2` (experiment
`go2_pgtt_teacher`, runs v1-v4, never walked). The teacher is now rebuilt from the reproduction
through the ladder (`Repro` and `Rung<N>` tasks, experiment `go2_pgtt_repro`). The ladder takes
the teacher's collisions, delays, randomization, three of its rewards, commands and terrain, but
keeps PGTT's reward set (tracking ×1, no orientation/power terms), joint damping and PPO settings.
In the reproduction the foot-phase target is measured from the IMU (stance 0.30 m, swing apex
0.20 m below it), as in the official code. Exact differences: doc/experiments.md, "The
ladder in detail".

- **Rewards** start from the official PGTT code (github.com/NtagkasAlex/phase_guided_terrain_traversal),
  with tracking x2 and extra terms for what full collisions allow (body_contact termination,
  base_height, leg_contact). Each step's sum is clipped at 0 as in PGTT; the termination
  penalty is added after the clip. Reasoning per term: comments in `make_pgtt_env_cfg`.
- **Foot-phase reward**: as in the PGTT code, `exp(-sum_feet err^2 / 0.05)` (one exponential
  of the summed squared error, max 1). An earlier per-foot version with sigma 0.0025 paid
  planted feet most of a 4x larger reward.
- **Swing trajectory**: stance ratio 0.5, stance height 2.2 cm (foot sphere radius),
  swing apex 8 cm + terrain variation (max - min height within 15 cm of the foot, capped at 15 cm).
  Foot height is measured relative to the ground under the foot (nearest height-scan ray).
- **Gait**: trot (FL+RR, FR+RL), frequency U[1, 3] Hz per episode in training, fixed
  2.0 Hz on the robot (`deploy.yaml`, `--gait-frequency` in export).
- **Teacher architecture**: Miki-style encoders (height 99 -> 32, privileged 53 -> 16)
  feeding PGTT's 512-256-128 MLP. Student: same encoders + 256-unit GRU + gated belief.
- **Actions / PD gains**: actions are tanh-bounded to [-1, 1] (`squash_actions`, as Brax PPO's
  tanh-normal policy in the PGTT code), target = default + 0.5 x action, kp 40 / kd 0.5
  (PGTT training config; its real-robot script uses 60 / 3). The exported ONNX ends with the
  tanh, and deploy.yaml carries the training gains. An unbounded Gaussian with targets
  clamped to the joint limits learned to slam its joints between the limits (flat_v3).
- **Height map value**: `(base_z - terrain_z) - 0.25`, clipped to [-1, 1]. 0 = flat ground
  at walking height, so "no map" (zeros) is a meaningful, trained-for input.
- **Termination**: turned over (> 90 deg), as in PGTT, or trunk/hips touching the terrain
  (PGTT's feet-only collision model cannot lie down; ours can).
- **Calf collision geoms** are unnamed in the Go2 XML; the PGTT task names them at load
  time (`_go2_spec_with_named_calves`) so leg-ground contact can be sensed. Stock tasks unchanged.
- **Contact normals** come from a separate `maxforce` sensor: with `netforce` the normal
  field is not meaningful (it reads [1, 0, 0] on flat ground).

## Fixed compared with the cpu branch

- Critic sees proprioception, heights and privileged state (cpu: only privileged).
- Actor gets the gyro and no world-frame / linear velocity (not measurable on the robot).
- External-force observation is non-zero (real force push event; cpu pushed by setting velocity).
- Leg contact is leg-vs-ground (cpu: thigh-vs-shank self contact).
- No `ipdb` / prints inside managers; debugging goes through `scripts/debug_env.py`.
- No merge-conflict markers; stock G1/H1/Go2 tasks still register and train.

## Sim-to-real robustness (added)

- Actuator command delay 0-20 ms (`DelayedActuatorCfg`), per-robot kp/kd, torque limits,
  link masses, joint friction and armature; IMU/joint observation delay 0-20 ms, LiDAR map
  delay 0-60 ms; tilted/moving resets. Details: CODE_GUIDE 4.10.
- `EventTermCfg.model_fields` expands extra model fields per robot (fixes PD-gain and
  effort randomization silently being identical for every robot).
- `randomize_effort_limits` unwraps `DelayedActuator` like `randomize_pd_gains`.
- `check_lidar_map.py` disables the map delay so bridge and training compare the same frame.

## Open items / known limitations

- **Real height map**: `scripts/lidar_height_map.py` converts a ROS 2 PointCloud2 and TF
  into the 11 x 9 UDP map. The Go2 LiDAR driver, localization/TF publisher, real topic and
  frame names, and live sensor tuning must be supplied and checked on the robot. If the
  cloud is stale or too sparse, the bridge stops sending and the student runs blind.
- **CPU speed**: ~15 env-steps/s on the full stairs terrain (4100 box geoms + 99 rays per
  robot). Use `debug_env.py` (small terrain) on CPU; train on the GPU.
- Reward weights: the PGTT set walks on flat ground (repro0, rungs 0-3). With our wider
  randomization (rung 4+) it collapses unless falls are priced; being diagnosed (2026-11-02 above).
- The hack check runs without randomization, so it can miss a policy that only fails under it.
  Judge runs by the check **and** the training curves (`scripts/summarize_runs.py`).
- The student task is built on the old teacher task and finds its teacher in
  `go2_pgtt_teacher`. A rung-7 teacher (in `go2_pgtt_repro`) must be given with
  `--teacher-checkpoint`, and it is not yet checked that the student task's teacher network and
  observations match it.
- The GPU machine has rebooted during runs (29 Oct, 2 Nov), which kills tmux sessions. Recovery
  steps: COMMANDS.md section 13.
- C++ observation code was not compiled here (needs unitree_sdk2); `algorithms.h` and
  `height_map.h` were compiled and tested standalone.
