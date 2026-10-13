# Refactoring Instructions: Miki et al. (2022) Perceptive Locomotion in MuJoCo

**Paper reference:** "Learning robust perceptive locomotion for quadrupedal robots in the wild"  
Miki et al., Science Robotics, 2022. arXiv:2201.08117  

**Target simulator:** MuJoCo (plain, CPU-based). MJX parallelization is a future milestone — do NOT introduce JAX, MJX, or functional transforms at this stage.

---

## Agent Instructions — How to Use This Document

1. Read this document in full before touching any file.
2. For each numbered section, open the relevant source file(s), verify whether the described issue is present, then apply the fix.
3. After every fix, run the smoke-test command listed in that section (if provided) before proceeding.
4. Do not change hyperparameters unless a section explicitly permits it.
5. Preserve all existing interfaces (function signatures, class APIs) unless a section instructs otherwise.
6. Add a one-line comment `# REFACTOR: <section-number>` near every change so diffs are traceable.

---

## 0. Codebase Reconnaissance (Do This First)

Before making changes, map the codebase:

```bash
find . -name "*.py" | sort
```

Identify and note the file responsible for each of the following roles — the section instructions reference these roles by name:

| Role label | What to look for |
|---|---|
| `ENV` | MuJoCo environment class (`gym.Env` or similar) |
| `TEACHER_POLICY` | Teacher MLP definition |
| `STUDENT_POLICY` | Student policy + belief encoder definition |
| `TRAINER_TEACHER` | PPO training loop for teacher |
| `TRAINER_STUDENT` | Imitation / distillation training loop |
| `TERRAIN` | Terrain generation / height-map utilities |
| `NOISE` | Observation noise model |
| `REWARD` | Reward function(s) |
| `CPG` | Central pattern generator / phase logic |
| `CONFIG` | Hyperparameter config file or dataclass |

---

## 1. Observation Space

### 1.1 Proprioceptive observation vector

The proprioceptive observation `o_p_t` must contain **exactly** the following, concatenated in this order:

```
[body_lin_vel (3),
 body_ang_vel (3),
 projected_gravity (3),          # R^T * g_world, normalised
 joint_pos_history (n_joints * H),  # H = 1 is minimum; paper uses history
 joint_vel_history (n_joints * H),
 action_history (n_actions * H),
 phase_per_leg (4 * 2 = 8)]      # sin(phi_l), cos(phi_l) per leg
```

**Common bugs to fix in `ENV`:**
- Body velocity reported in world frame — must be in robot body frame. Use `data.qvel[:3]` rotated by `data.xmat[root_body_id].reshape(3,3).T`.
- `projected_gravity` missing or computed as raw IMU reading rather than `R^T * [0,0,-1]`.
- Phase stored as raw angle rather than `[sin(phi), cos(phi)]` pairs (discontinuity at 2*pi breaks learning).
- History window not actually buffered — verify a `collections.deque` or circular buffer exists and is updated every step.

### 1.2 Exteroceptive observation vector

The exteroceptive observation `o_e_t` is a **flattened vector of height samples** around each foot at multiple radii.

Paper spec:
- Sampled around each of the 4 feet.
- 5 concentric radii per foot.
- Heights expressed **relative to the foot's current z-position** (not absolute world height).
- During **teacher training**: use ground-truth height map (query MuJoCo terrain geometry directly).
- During **student training and inference**: pass through the noise model (Section 5).

**Common bugs to fix in `ENV` / `TERRAIN`:**
- Heights returned as absolute world-frame z — subtract foot z.
- All samples taken around the body centre rather than each foot individually.
- Only one radius ring instead of five. Typical radii: `[0.05, 0.10, 0.15, 0.20, 0.25]` m.
- Height scan not recomputed every step (stale from previous step).

### 1.3 Privileged state vector (teacher only)

The privileged state `s_p_t` must include:

```
[contact_state_per_foot (4, binary),
 contact_force_per_foot (4 * 3),
 contact_normal_per_foot (4 * 3),
 friction_coefficient (1),
 thigh_contact (4, binary),
 shank_contact (4, binary),
 external_force_on_body (3),
 external_torque_on_body (3),
 swing_phase_duration_per_foot (4)]
```

**Common bugs to fix in `ENV`:**
- Contact forces read from `data.cfrc_ext` without selecting the correct body IDs for feet/thighs/shanks — enumerate body names and map them explicitly.
- Friction coefficient hard-coded to a constant rather than read from the MuJoCo model's `geom_friction` (which is randomised per episode — Section 7).
- External force/torque not stored during `apply_disturbance()` and therefore unavailable at observation-collection time.

---

## 2. Action Space and Central Pattern Generator (CPG)

The paper uses a CPG-based action space. Each leg `l in {0,1,2,3}` has a phase variable `phi_l`. The policy outputs:

```
action = [delta_phi_0, delta_phi_1, delta_phi_2, delta_phi_3,    # phase increments (4)
          delta_q_0, ..., delta_q_11]                             # residual joint targets (12)
```

### 2.1 Phase update

```python
# In CPG / ENV step():
phi[l] = (phi[l] + (omega_0 + delta_phi[l]) * dt) % (2 * pi)
```

`omega_0` is the nominal CPG frequency (typically 2-3 Hz, so `omega_0 ≈ 2*pi*2.5` rad/s). **Verify `omega_0` is non-zero and plausible.**

### 2.2 Nominal foot trajectory

The nominal foot tip position in the leg's local frame follows a stepping ellipse parameterised by `phi_l`. Compute nominal joint targets `q_nom(phi_l)` via inverse kinematics (IK). The final joint target is:

```
q_target[i] = q_nom_i(phi[l]) + delta_q[i]
```

**Common bugs to fix in `CPG`:**
- `delta_q` applied to absolute joint position rather than to `q_nom`.
- IK not implemented — `q_nom` is a constant default pose, so the policy cannot produce a gait.
- Phase not persisted between steps (re-initialised each call).
- Action clipped before adding to `q_nom` — clipping should happen on the **final** `q_target`, not on `delta_q` alone.

### 2.3 PD controller

Joint targets are tracked by a PD controller inside MuJoCo. Set actuator gains in the XML model:

```xml
<actuator>
  <position name="..." joint="..." kp="80" />
</actuator>
```

Verify that `kp` (and `kd` via `damping` on the joint) are set to physically reasonable values. A common mistake is leaving them at MuJoCo defaults (kp=1), which makes the robot collapse.

---

## 3. Network Architecture

### 3.1 Teacher policy

```
exteroceptive encoder g_e:   MLP([o_e_t])          -> l_e_t    (output dim 64)
privileged encoder g_p:      MLP([s_p_t])           -> l_priv_t (output dim 64)
main network:                MLP([o_p_t, l_e_t, l_priv_t]) -> action mean (16)
```

Each MLP: 2 hidden layers, 256 units, ELU activation. Output layer is linear (no activation).

**Common bugs to fix in `TEACHER_POLICY`:**
- Encoders missing entirely — `o_e_t` and `s_p_t` concatenated directly into one big MLP. Split them out.
- Output activation is `tanh` — remove it. The action is a Gaussian mean; variance is a separate learned log-std parameter.
- Gradient flow blocked between encoders and main network (detach called incorrectly).

### 3.2 Student policy and belief encoder

```
exteroceptive encoder g_e:  MLP([n(o_e_t)])               -> l_e_t   (same arch as teacher, weights NOT shared initially)
recurrent belief encoder:   GRU(input=[o_p_t, l_e_t], hidden_dim=256) -> h_t  (belief state)
main network:               MLP([h_t])                    -> action mean (16)
belief decoder (auxiliary): MLP([h_t])                    -> reconstructed privileged state  (training only)
```

**Critical architectural requirements:**
- The GRU hidden state `h_t` must be **carried across timesteps** within an episode. Reset only at episode boundaries.
- The belief decoder is used **only during training** (reconstruction loss). It must not be called during rollout/inference.
- The student shares **no weights** with the teacher during distillation. The teacher is frozen (`.eval()`, `requires_grad=False`).

**Common bugs to fix in `STUDENT_POLICY`:**
- GRU hidden state reset every step — belief never accumulates, policy behaves as memoryless.
- Belief decoder outputs fed into the action — decoder is auxiliary only.
- Teacher weights not frozen during student training.
- `h_t` not detached from computation graph across rollout steps (memory leak and incorrect BPTT).

### 3.3 Sequence length for student training

The student is trained on **fixed-length sequences** sampled from rollout buffers. Use sequence length `T = 50` steps. Zero the hidden state at the start of each sequence, not carried from a previous sequence in the buffer.

---

## 4. Reward Function

Implement the following reward terms in `REWARD`. All terms are summed each step; total reward `r = sum(w_i * r_i)`.

### 4.1 Command-following reward (positive)

```python
def r_command(v_des, v_actual):
    """v_des, v_actual: 2D horizontal velocity in body frame."""
    proj = dot(v_des, v_actual)
    v_des_norm = norm(v_des)
    if proj > v_des_norm:
        return 1.0
    else:
        return exp(-(proj - v_des_norm)**2)
```

Apply the same formula for yaw velocity independently.

Penalise the **orthogonal** velocity component:
```python
r_orth = -norm(v_actual - (proj / (v_des_norm**2 + 1e-8)) * v_des)
```

**Common bugs:**
- `dot(v_des, v_actual)` computed in world frame rather than body frame.
- No guard against zero division when `v_des = 0`.
- Yaw term missing.

### 4.2 Constraint-violation penalties (negative)

| Term | Description |
|---|---|
| `r_torque` | `-sum(tau^2)` — penalise high joint torques |
| `r_joint_vel` | `-sum(dq^2)` — penalise high joint velocities |
| `r_joint_acc` | `-sum(ddq^2)` — penalise high joint accelerations |
| `r_slip` | `-sum(v_foot^2)` for feet in contact — penalise foot slippage |
| `r_orientation` | `-||R_body_z - z_world||^2` — penalise body tilt |
| `r_shank_contact` | `-1` per shank/thigh body contacting terrain |

### 4.3 Reward scaling and curriculum

Each penalty term is multiplied by curriculum factor `c_k` (see Section 6). At the start of training `c_0 ≈ 0.01`; penalties are small so the policy first learns to walk, then to walk smoothly.

**Common bug:** All reward weights set to final values from iteration 0 — the policy is crushed by penalties before learning any locomotion.

---

## 5. Observation Noise Model (Student Training Only)

Apply `n(o_e_t)` to height samples **only during student training and evaluation**, never during teacher training.

Implement the following noise modes, applied stochastically per episode or per step:

| Mode | Description | Application |
|---|---|---|
| `gaussian` | Add `N(0, sigma^2)` to each height sample | Per-step, always on |
| `drift` | Slowly varying per-area offset that accumulates over time | Per-episode, reset at episode start |
| `outlier` | Replace a random subset of samples with uniform random heights | Per-step, probability `p_out = 0.05` |
| `dropout` | Zero out entire scan regions (simulates sensor failure) | Per-episode, probability `p_drop = 0.1` |

Reference value: `sigma = 0.02 m` for Gaussian noise.

**Common bugs in `NOISE`:**
- Noise module is a no-op (all modes return input unchanged).
- Noise applied to proprioceptive observations as well as exteroceptive — only height samples should be noised.
- Same noise mask reused for the entire training run (not re-sampled each step/episode).

---

## 6. Curriculum

Two separate curricula must run in parallel.

### 6.1 Terrain difficulty curriculum (adaptive)

Maintain a per-terrain-type success metric. After each episode, update terrain parameters (step height, slope angle, stair dimensions) using a particle filter that keeps difficulty at the frontier of the policy's capability.

Minimum implementation (if particle filter is too complex):
- Track a rolling mean of episode success rate `rho` (survived without termination).
- If `rho > 0.7`: increase terrain difficulty by a fixed increment.
- If `rho < 0.4`: decrease difficulty.

**Common bug:** Terrain parameters fixed at maximum difficulty from episode 1 — policy never learns basic locomotion.

### 6.2 Regularisation curriculum (logistic)

```python
# In TRAINER_TEACHER, each iteration k:
c_k = c_{k-1} ** d        # d in (0, 1), e.g. d = 0.999
```

Multiply the following penalty weights by `c_k`:
- `r_joint_vel`, `r_joint_acc`, `r_orientation`, `r_slip`, `r_shank_contact`

**Common bug:** `d > 1` — curriculum factor diverges instead of converging to 1.

---

## 7. Domain Randomisation

Apply the following randomisations at the **start of each episode** in `ENV.reset()`:

| Parameter | Distribution |
|---|---|
| Body mass | `U(0.8, 1.2) * nominal_mass` |
| Leg link masses | `U(0.8, 1.2) * nominal_mass` |
| Initial joint position | `N(0, 0.1)` rad offset from default |
| Initial joint velocity | `N(0, 0.5)` rad/s |
| Initial body orientation | `N(0, 0.05)` rad per axis |
| Foot friction coefficient | `U(0.3, 1.2)` — occasionally `U(0.05, 0.15)` for slippery episodes |

Additionally, apply a **random external force/torque** to the robot body at a random time within each episode:
- Force magnitude: `U(0, 50)` N, random direction in horizontal plane.
- Duration: 0.1–0.5 s.

**Common bugs:**
- Randomisation applied only to mass, not friction — policy never learns to handle slippery terrain.
- External disturbances not applied at all.
- MuJoCo model XML geom friction not modified at runtime. Use `model.geom_friction[geom_id, 0] = sampled_value` after `mj_resetData`. Modifying the XML file and reloading is too slow.

---

## 8. Termination Conditions

Terminate the episode (call `reset()`) when any of the following occur. Do **not** add a large negative reward on the final step — just terminate:

1. Body (torso) contacts the ground plane.
2. Body tilt exceeds 60 degrees from vertical (either roll or pitch).
3. Any joint torque exceeds the actuator torque limit for more than 3 consecutive steps.

**Common bugs:**
- Termination only on body contact but not on tilt — policy learns to lean at extreme angles without falling.
- Joint torque termination checked instantaneously (noisy) instead of over a 3-step window.

---

## 9. Training Loop — Teacher (PPO)

Use standard PPO with the following settings:

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO (clipped surrogate, Schulman et al. 2017) |
| Clip ratio epsilon | 0.2 |
| Discount gamma | 0.99 |
| GAE lambda | 0.95 |
| Mini-batch size | 4096 (or nearest power of 2 that fits memory) |
| PPO epochs per rollout | 5 |
| Learning rate | 1e-4 with linear decay |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Gradient clip norm | 1.0 |
| Rollout steps per update | 24 steps * N_envs |

For CPU-parallel environments without MJX: start with N=16 workers using `multiprocessing.Pool`, scale based on available cores.

**Common bugs in `TRAINER_TEACHER`:**
- Advantage not normalised per mini-batch — training instability.
- Old log-probs not recomputed correctly when batching sequences.
- Value function shares the same forward pass as the policy without a separate value head.
- Learning rate not decayed.

---

## 10. Training Loop — Student (Imitation + Reconstruction)

### 10.1 Data collection

Collect rollouts using the **frozen teacher policy** in the training environment (with noise applied to exteroception). Store:
- `(o_p_t, noisy_o_e_t, teacher_action_t)` tuples in a replay buffer.
- Group into fixed-length sequences of T=50 steps for GRU training.

### 10.2 Loss functions

```
L_total = lambda_bc * L_behavior + lambda_rec * L_reconstruction
```

**Behavior cloning loss** (imitate teacher actions):
```
L_behavior = MSE(student_action, teacher_action)
```

**Reconstruction loss** (belief encoder must reconstruct privileged state):
```
L_reconstruction = MSE(belief_decoder(h_t), s_p_t_ground_truth)
```

Reference weights: `lambda_bc = 1.0`, `lambda_rec = 0.5`.

**Common bugs in `TRAINER_STUDENT`:**
- Reconstruction target is the noisy observation rather than the ground-truth privileged state `s_p_t`.
- `L_total` is only `L_behavior` — reconstruction loss missing — encoder produces a degenerate belief state.
- Teacher generates actions using the student's noisy inputs rather than its own privileged inputs — wrong supervision signal.
- GRU hidden state not zeroed at sequence boundaries within a batch.

### 10.3 Optimiser

Adam, lr=1e-4, weight decay=1e-5. Apply gradient clipping (max norm 1.0).

---

## 11. Terrain Generation

Implement the following terrain types (parameterised, difficulty controlled by curriculum):

| Type | Key parameters |
|---|---|
| Flat | — |
| Random rough | height variance `sigma_h`, spatial frequency |
| Slopes | inclination angle theta (0–35 degrees) |
| Discrete steps | step height `h_s` (0–35 cm), step width |
| Stairs (standard) | riser height, tread depth |
| Stairs (open) | as standard but no riser surface |
| Stairs (ledged) | overhanging lip on each step |

**Critical note from paper:** Do NOT model stair risers as height-map discontinuities. Use **box primitives** (`geom type="box"` in MuJoCo XML) for stairs. Height-map vertical walls create non-physical edges that the policy exploits, destroying the quality of the learned behaviour. This applies even for MuJoCo-only training.

In `TERRAIN`, verify:
- Stairs are built from box geoms, not from height-map spikes.
- Terrain is regenerated (or randomly repositioned under the robot) at each episode reset.
- The height scan query correctly intersects rays/points with MuJoCo geom surfaces, not just a height-map buffer.

---

## 12. MuJoCo-Specific Implementation Details

### 12.1 Simulation timestep

Set `<option timestep="0.005"/>` in the XML (200 Hz physics). The policy runs at 50 Hz (every 4 physics steps). The same PD target is held for 4 consecutive physics steps before the next policy query.

**Common bug:** Policy called every physics step at 200 Hz — PD controller has no time to track targets, producing erratic motion.

### 12.2 Contact model

Use:
```xml
<option cone="pyramidal" impratio="1" />
```

Set `solimp` and `solref` on foot geoms for realistic contact compliance:
```xml
<geom ... solimp="0.9 0.95 0.001" solref="0.02 1" />
```

### 12.3 Reading per-foot contact forces

```python
# Correct approach: iterate active contacts
for i in range(data.ncon):
    con = data.contact[i]
    force = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, force)
    # Map con.geom1 / con.geom2 to the appropriate foot body
```

Do not use `data.cfrc_ext` for per-foot contact forces — it gives net external forces per body, conflating multiple contact points and non-foot contacts.

### 12.4 Parallel environments

Use `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`. Each worker must own its own `mujoco.MjModel` / `mujoco.MjData` copy. Do **not** share `MjData` across processes — MuJoCo data structs are not thread-safe.

---

## 13. Sanity Checks and Debugging Tools

Add (or verify existence of) the following diagnostic utilities:

### 13.1 Observation sanity check

```python
def check_obs(obs_dict):
    for k, v in obs_dict.items():
        assert not np.any(np.isnan(v)), f"NaN in {k}"
        assert not np.any(np.isinf(v)), f"Inf in {k}"
        print(f"{k}: min={v.min():.3f}  max={v.max():.3f}  mean={v.mean():.3f}")
```

Run after every `env.step()` during debug mode.

### 13.2 Phase plausibility check

```python
# Phase values must wrap correctly
assert np.all(phi >= 0) and np.all(phi < 2 * np.pi)
# In a trot gait, diagonal legs (FL-HR, FR-HL) should be ~pi apart.
# Check at the end of the first 200 training steps.
```

### 13.3 Reward component logging

Log each reward term separately (not just the total) via TensorBoard or W&B. A flat or near-zero component indicates a bug in that term.

### 13.4 Convergence smoke test (flat terrain, no noise)

Before full training:
1. Set terrain to flat only.
2. Disable the noise model.
3. Disable external disturbances.
4. Train teacher for 500 PPO iterations.

**Expected result:** The robot should learn to walk forward within 200–300 iterations. Mean episode reward must trend upward. If it does not, there is a fundamental bug in the reward, action space, or physics setup — fix it before proceeding to rough terrain or student training.

---

## 14. MJX Migration Readiness (Future — Do Not Implement Now)

While implementing the MuJoCo version, annotate the following with `# MJX-FUTURE:` comments for the upcoming migration:

- Python-level loops over environment steps (will become `jax.lax.scan`).
- NumPy operations on observation/action arrays (will become `jnp` operations).
- `multiprocessing`-based parallelism (will become `jax.vmap` over `mjx.step`).
- `mujoco.mj_contactForce` calls (MJX contact API differs).

Do not convert these now. Just mark them.

---

## 15. File-Level Checklist

After completing all sections, verify the following:

- [ ] `ENV.reset()` applies domain randomisation (Section 7) and resets CPG phases.
- [ ] `ENV.step()` calls physics at 200 Hz and policy at 50 Hz (Section 12.1).
- [ ] `ENV._get_obs()` returns `o_p`, `o_e`, `s_p` as separate tensors/arrays (not one concatenated blob).
- [ ] `CPG.step()` updates phases and returns `q_nom` via IK (Section 2).
- [ ] `TEACHER_POLICY` has three sub-networks: `g_e`, `g_p`, and the main MLP (Section 3.1).
- [ ] `STUDENT_POLICY` GRU hidden state is an instance variable, reset only on episode boundary (Section 3.2).
- [ ] `REWARD` logs all individual terms to the experiment tracker (Section 13.3).
- [ ] `NOISE` implements all four noise modes with correct application scope (Section 5).
- [ ] `TERRAIN` uses box geoms for stairs (Section 11).
- [ ] `TRAINER_TEACHER` clips gradients and decays learning rate (Section 9).
- [ ] `TRAINER_STUDENT` computes both `L_behavior` and `L_reconstruction` (Section 10.2).
- [ ] Curriculum factor `c_k` updated each iteration in both training loops (Section 6).

---

*End of Instructions.md*