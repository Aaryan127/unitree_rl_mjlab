# PGTT teacher-student on Go2: progress log

How to use the branch and what each file does: see [instruction.md](instruction.md).

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
- [ ] Full teacher training on GPU (~10k iterations, 4096 envs) and reward tuning
- [ ] Full student distillation on GPU
- [ ] Play both in simulation; check stairs success per terrain level
- [ ] Build `go2_ctrl` on the robot (needs unitree_sdk2 + cyclonedds; not buildable on the dev laptop)
- [ ] Real robot: blind test on flat ground first (no height map -> zeros)
- [x] ROS 2 PointCloud2 bridge: LiDAR points -> robot-centric 11 x 9 grid -> UDP
- [ ] Confirm LiDAR topic, TF, coverage, and timing on the physical Go2

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

- **Rewards** follow PGTT Table I weights. mjlab multiplies all rewards by dt, so only ratios matter.
- **Foot-phase sigma**: the paper's `sigma_f = 0.05` is read as a 5 cm standard deviation
  (`sigma = 0.0025` in `exp(-err^2 / sigma)`). Taken literally (0.05) the reward barely
  changes between a 1 cm and a 10 cm error. First knob to tune if the gait looks wrong.
- **Swing trajectory**: stance ratio 0.5, stance height 2.2 cm (foot sphere radius),
  swing apex 8 cm + terrain variation (max - min height within 15 cm of the foot, capped at 15 cm).
  Foot height is measured relative to the ground under the foot (nearest height-scan ray).
- **Gait**: trot (FL+RR, FR+RL), frequency U[1, 3] Hz per episode in training, fixed
  2.0 Hz on the robot (`deploy.yaml`, `--gait-frequency` in export).
- **Teacher architecture**: Miki-style encoders (height 99 -> 32, privileged 53 -> 16)
  feeding PGTT's 512-256-128 MLP. Student: same encoders + 256-unit GRU + gated belief.
- **PD gains / action scale**: kp 20, kd 1 (this repo's Go2 actuator config and stock
  deploy), action scale 0.25. PGTT used kp 60 / kd 3 on its robot.
- **Height map value**: `(base_z - terrain_z) - 0.25`, clipped to [-1, 1]. 0 = flat ground
  at walking height, so "no map" (zeros) is a meaningful, trained-for input.
- **Termination**: only when the robot turns over (> 90 deg), as in PGTT.
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

## Open items / known limitations

- **Real height map**: `scripts/lidar_height_map.py` converts a ROS 2 PointCloud2 and TF
  into the 11 x 9 UDP map. The Go2 LiDAR driver, localization/TF publisher, real topic and
  frame names, and live sensor tuning must be supplied and checked on the robot. If the
  cloud is stale or too sparse, the bridge stops sending and the student runs blind.
- **CPU speed**: ~15 env-steps/s on the full stairs terrain (4100 box geoms + 99 rays per
  robot). Use `debug_env.py` (small terrain) on CPU; train on the GPU.
- Reward weights are untested beyond short CPU runs; expect to tune on GPU
  (watch `Episode_Reward/foot_phase` vs `track_lin_vel` and `Curriculum/terrain_levels`).
- C++ observation code was not compiled here (needs unitree_sdk2); `algorithms.h` and
  `height_map.h` were compiled and tested standalone.
