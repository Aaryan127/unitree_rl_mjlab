# Reframed Instructions — unitree_rl_mjlab

This file maps the original refactor checklist (Miki et al. 2022) to the actual locations in this repository and records quick smoke-check findings. Add `# REFACTOR: <section-number>` comments next to edits made to source files.

## Mapping (roles -> repo paths)
- ENV: mjlab/envs/manager_based_rl_env.py
  - Path: mjlab/envs/manager_based_rl_env.py
- MuJoCo XML assets: mjlab/asset_zoo/robots/unitree_go2/xmls (contains `assets/` and `go2.xml`)
- TEACHER_POLICY: mjlab/rsl_rl/networks/teacher_mlp.py
- STUDENT_POLICY / BELIEF_ENCODER: mjlab/rsl_rl/networks/belief_encoder.py
- TRAINER_TEACHER: scripts/train.py (entrypoint: `launch_training` / `main`)
- TRAINER_STUDENT: Not present (needs implementation)
- TERRAIN utilities (heightfields, primitives): mjlab/terrains/* (see `terrain_generator.py`, `heightfield_terrains.py`, `primitive_terrains.py`)
- TERRAIN: velocity task adds `height_map` observation in `mjlab/tasks/velocity/velocity_env_cfg.py` (observation registration)
- NOISE: mjlab/utils/noise (implements `NoiseModel` and configs)
- REWARD functions: mjlab/tasks/velocity/mdp/rewards.py
- CPG: FTG implementation in mjlab/rsl_rl/networks/ftg.py (FTG used instead of CPG)
- Inverse kinematics: mjlab/rsl_rl/networks/aik.py
- CONFIGS / hyperparams: mjlab/tasks/velocity/config/go2 (env_cfgs.py, rl_cfg.py)

## Smoke-check summary
- Verified existence/readable:
  - `mjlab/envs/manager_based_rl_env.py` — OK
  - `mjlab/rsl_rl/networks/teacher_mlp.py` — OK
  - `mjlab/rsl_rl/networks/belief_encoder.py` — OK
  - `scripts/train.py` — OK
  - `mjlab/tasks/velocity/velocity_env_cfg.py` — OK (height_map registered)
  - `mjlab/tasks/velocity/mdp/rewards.py` — OK (contains `track_linear_velocity`, `track_angular_velocity`, orthogonal penalty, etc.)
  - `mjlab/rsl_rl/networks/ftg.py` — OK (FTG implementation)
  - `mjlab/rsl_rl/networks/aik.py` — OK (Go2 IK)
  - `mjlab/utils/noise/noise_model.py` — OK (base noise classes implemented)
  - `mjlab/asset_zoo/robots/unitree_go2/xmls` — OK (`go2.xml` present)
  - `mjlab/tasks/velocity/config/go2` — OK (env & rl config files found)

Notes from quick content peek:
- `ManagerBasedRlEnv` is a general manager-based environment: observations are configured via `ObservationTermCfg` entries; height_map is provided by `mdp.height_map` in velocity task.
- The teacher network (`Teacher_wild`) uses a `RecurrentAttentionPolicy.BeliefEncoder` and separate priviledged encoder; outputs an `action` and `next_hidden` — aligns with teacher architecture in instructions.
- The `RecurrentAttentionPolicy` class (belief encoder) provides belief encoder/decoder and GRU-like recurrent handling; check hidden-state lifetime semantics in student code paths.
- Noise model implements additive bias and per-episode reset hooks — supports drift/dropout modes via config (verify configs).
- Reward module includes velocity tracking & orthogonal term implementations similar to instructions.
- No explicit `TRAINER_STUDENT` script found — student training loop (imitation) is missing and must be implemented if required.

## Suggested next actions (proposed order)
1. Confirm we should implement `TRAINER_STUDENT` (imitation + reconstruction). If yes, decide target path/name (suggest: `scripts/train_student.py` or `mjlab/rsl_rl/train_student.py`).
- Yes we can implement, target - scripts/train_student.py
2. Inspect `mdp.height_map` implementation to ensure exteroceptive sampling per-foot / radii. File: `mjlab/tasks/velocity/mdp` (open `height_map` function).
- Yes
3. Verify `ManagerBasedRlEnv` observation plumbing: ensure `o_p`, `o_e`, `s_p` are exposed as separate groups (policy/critic/extero groups in `velocity_env_cfg.py`). Add `# REFACTOR: 1` comments where you change behavior.
- This design choice is acrried forward to the encoder in teacher_mlp so the change is mostly not required.
4. Confirm FTG usage vs. CPG and whether phases are persisted across steps (FTG.phi stored per-env). If needed, add phase checks (`# REFACTOR: 2`).
- Yes
5. Create `unitree_rl_mjlab/.agent/reframed_instructions.md` (this file) and iterate with you to apply section-by-section changes.
- Yes
## Where I will write changes and docs
- Reframed instruction file: `.agent/reframed_instructions.md` (this file)
- Per-edit comments: add `# REFACTOR: <section-number>` near modifications in source files as requested.

## Next steps for me (confirm before I run):
- Option A (recommended): Run a focused inspection on `mjlab/tasks/velocity/mdp/*` to check `height_map`, `contact_forces`, and `projected_gravity` implementations (I will report mismatches against the instruction checklist).
- Option B: Start implementing `TRAINER_STUDENT` scaffold and tests.

Please confirm which next step you want, and whether I should start applying fixes now. If you want specific files scanned first, list them (or say "scan recommended files").

## Actions taken
- `mjlab/tasks/velocity/mdp/observations.py`: updated `height_map` to return per-foot 5-radius height samples (relative to each foot's z). Added `# REFACTOR: 1.2` comment.
- `scripts/train_student.py`: added scaffold integrated with `OnPolicyRunnerWild` and `RslRlVecEnvWrapper` to load a teacher checkpoint, collect rollouts into a sequence buffer, and save the collected dataset for offline student training. Marked `# REFACTOR: 10.2` in the file.
