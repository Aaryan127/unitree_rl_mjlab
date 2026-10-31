# Reward hacking in the PGTT teacher: runs 1-4, the fixes, and the first walking policy

Four flat-ground teacher runs (27-31 Oct 2026) each found a different way to score without
walking: v1 lay down and paddled, v2 fell on purpose, v3 vibrated its joints, v4 (with
tanh-bounded actions) vibrated with bang-bang actions and fell every 1.2 s. Each fix closed
one loophole and exposed the next.

After run 4 the approach changed (see [Why four runs failed](#why-four-runs-failed) and
[The plan](#the-plan-reproduce-then-add-one-thing-at-a-time)):
- the loss we train with pushes the policy towards saturated, bang-bang actions. This is
  proven below, and the official PGTT code (Brax PPO) avoids it;
- every run changed ~15 things against the published setup at once, so no failure could be
  attributed to a cause;
- the next run reproduces the official PGTT setup, which is published to walk. Our
  additions then go back in one at a time.

**Result (31 Oct):** the reproduction (`repro0`) walked: speed ratio 0.95, no falls, no
saturation, noise std 0.70 → 0.28. Run 4's own logs then confirmed the entropy diagnosis: its
noise std was exactly 1.000 (the cap) at every one of 6944 iterations. See
[Rung 0 result](#rung-0-result-the-first-walking-policy).

| Run | Folder (`logs/rsl_rl/go2_pgtt_teacher/`) | What it learned | Fix (commit) |
|---|---|---|---|
| v1 | `2026-10-27_20-18-35_flat` → `2026-10-28_01-14-38_flat` (to 11000) | lie on the belly and paddle | rebalanced rewards, body-contact termination, hacking checker (`5981fe3`) |
| v2 | `2026-10-29_04-12-16_flat_v2` (to 4300, machine reboot) | walk at 500, then fall on purpose | reward clip at 0, noise std cap (`6a11dc1`) |
| v3 | `*_flat_v3` (10000) | vibrate the joints between their limits | tanh-bounded actions, PGTT gains and foot-phase formula (`3164449`) |
| v4 | `2026-10-30_16-23-42_flat_v4` (6300 checked) | bang-bang vibration, a fall every 1.2 s | entropy on the squashed action, PGTT reproduction as baseline (`15ba559`) |
| repro0 | `go2_pgtt_repro/*_repro0` (3000) | **walks**: trot, 95% of commanded speed, no falls | baseline for the ladder |

## How the reward works here

- Every term is `weight × value × dt` (dt = 0.02 s), summed each step. Below, rewards are given
  per second (`weight × value`), which is what the scripted measurements report.
- PPO maximises the discounted sum of future rewards (γ = 0.99). An episode ends at 20 s
  (time-out, the future is bootstrapped) or early on a failure (no future reward).
- So two things decide behaviour: which behaviour earns the most per second, and whether
  ending the episode early is cheaper than staying alive.
- "Measured" numbers come from scripted robots on flat ground (standing, stepping in place,
  lying, vibrating, random actions, 16-64 robots, 5 s), not from a trained policy.

## Run 1: lay down and paddled

**Setup.** PGTT Table I weights: tracking 1.0 (linear) and 0.5 (yaw), `foot_phase` 1.0,
termination −1. Episodes ended only when the robot flipped over (> 90°). Leg-contact penalty off.

**Symptom.** After the student was trained, every robot in the viewer lay on the ground and
wriggled. The teacher did the same. Mean reward peaked at ~80 around iteration 1000-1300, then
fell to ~15 while episode length stayed at the maximum.

**Cause.** Doing nothing paid almost as much as walking, and lying down never ended the episode.

| Behaviour (measured) | Reward per second |
|---|---|
| standing still | +3.20 |
| stepping in place | +3.25 |
| lying down, legs folded | +1.91 |
| lying down, paddling | +1.32 |
| extra that perfect tracking could add over standing | only +1.1 |

- `foot_phase` was computed per foot and summed over 4 feet (max 4.0), with a tight
  sigma (0.0025, from reading the paper's 0.05 as a 5 cm standard deviation). Planted feet already
  matched the stance height, so standing scored +3.4 of the 4.0.
- The fall penalty was −1 × dt = −0.02, i.e. nothing.
- Episode length looked perfect *because* lying never ends an episode.

**Fix (`5981fe3`).**
- Tracking 2.0 / 1.0, `foot_phase` 0.5, termination −100 (= −2 per fall).
- New `base_height` penalty below 0.25 m; `leg_contact` −0.25 switched on.
- **Trunk or hips touching the terrain ends the episode** (`body_contact`). Thighs are left out:
  in the normal crouch a thigh brushes the ground on landing (it fired 99 times on standing robots).
- Reward-hacking detection: `Metrics/hack/*` in TensorBoard and `scripts/pgtt.sh hacks`.

Measured after: standing +1.34, stepping +1.58, lying −0.81 to −1.07 (and the episode ends),
walking estimated +4.5 to +4.8.

## Run 2: walked, then fell on purpose

**Symptom.** The hacks check at iteration 4300: 130 episode endings per robot-minute
(628 of 693 by `body_contact`), `speed_ratio` 0.04. The training curves show it walked first:

| Iteration | Episode length (max 1000) | Noise std | `speed_ratio` | Mean reward |
|---|---|---|---|---|
| 500 | 810 | 2.03 | 0.60 | 32.8 |
| 1000 | 336 | 2.41 | 0.03 | −0.4 |
| 1500 | 47.5 | 2.25 | ~0 | −4.2 |
| 4374 | 35.4 | 2.52 | ~0 | −3.8 |

**Cause.** Two things combined around the end of the penalty ramp (10 % → 100 % over 1500
iterations):

1. **The action noise std grew from 1.0 to 2.5.** Joint targets were clamped to the joint
   limits (`clip_to_joint_limits`, added in `98f41eb`), so noise past the limits had no physical
   effect, and PPO's entropy bonus pushed the std up for free.
2. **`action_rate` grows with the noise squared.** Between iterations 500 and 1500 the ramp
   (×2.5) and the std growth (×1.6) made it ~4× larger, ~−0.5/s → ~−2/s.

Staying alive then cost about −2.5/s while one fall cost −2. Falling within a second became
the best option. Shorter episodes meant less walking practice, which made living even more
expensive (810 → 336 → 35 steps). The `body_contact` termination added in the fix for run 1
became the exit door.

**Fix (`6a11dc1`).**
- `only_positive_rewards`: each step's summed reward is clipped at 0, and the termination
  penalty is added after the clip. Staying alive can never cost anything, so falling always
  loses the future plus −2 (legged_gym does the same).
- `max_noise_std = 1.0`: the std parameter itself is clamped in place, so it can still shrink.
  A first version clamped the output, which would have frozen the std at 1.0; a unit test caught it.

## Run 3: vibrated its joints

**Symptom.** The hacks check on the final checkpoint, run without noise:

| Check | Value |
|---|---|
| `action_rate` | **−16.2/s** (v2: −4.8): each joint's action jumped ~11 units every step |
| episode endings | 82 per robot-minute (`fell_over` 200, `body_contact` 236) |
| `standing_drift` | 0.41 m/s |
| reward share | 60 % from `foot_phase`, 26 % linear tracking: ignores the command |

In the viewer: episodes a bit longer than v2, but erratic motion, no walking.

**Cause.** The policy was an unbounded Gaussian and the targets were clamped to the joint limits.

- An output of 10 and an output of 2 both put the joint at its limit, so the network could
  output huge values. Far past the limits, its own exploration noise had no effect, and it could
  slam joints from limit to limit every step.
- That made `action_rate` large, the step's sum went negative, and the clip from run 2's fix
  turned it into 0. Mild and violent vibration then scored the same, so nothing pushed back
  towards smooth motion.
- The vibrating legs still collected `foot_phase` (per-foot, summed), and falls cost only −2.
- v2's deterministic check already showed −4.8/s of `action_rate`, so the vibration started
  before the clip. The clip removed the last thing that held it back.

**What the official PGTT code does differently**
([github.com/NtagkasAlex/phase_guided_terrain_traversal](https://github.com/NtagkasAlex/phase_guided_terrain_traversal)):

| | Official PGTT | Ours in v3 |
|---|---|---|
| Action distribution | tanh-squashed Gaussian (Brax PPO default): actions in [−1, 1] | unbounded Gaussian, targets clamped to joint limits |
| Action scale, Kp, Kd (training) | 0.5, 40, 0.5 | 0.25, 20, 1 |
| Reward clip at 0 | yes, with the termination inside the clip | yes, termination after the clip |
| `action_rate` | −0.001 | −0.01 (ramped) |
| `foot_phase` | `exp(−Σ_feet err² / 0.05)`, max 1 | `Σ_feet exp(−err² / 0.0025)`, max 4 × weight |
| default pose penalty | hips only | hips 1, thigh and calf 0.25 |
| Collisions / falls | feet-only collision model; ends only when flipped | full collisions; flipped or trunk/hips down |
| PPO | γ 0.97, lr 3e-4, 8 epochs | γ 0.99, lr 1e-3 adaptive, 5 epochs |

PGTT's real-robot script uses Kp 60, Kd 3, unlike its training config. We train with 40 / 0.5,
and the export writes the training gains into `deploy.yaml`.

**Fix (`3164449`).**
- `squash_actions`: tanh on every action before the action terms, as Brax does. PPO is unchanged
  (the tanh cancels in the probability ratio). The exported ONNX ends with the same tanh
  (verified to 1.5e-7).
- Action scale 0.5, Kp 40, Kd 0.5: targets stay within 0.5 rad of the standing pose, inside the
  joint limits.
- `foot_phase` = PGTT's formula, weight 1.0. `default_pose` on the hips only.
- New check `action_jitter` (mean |a_t − a_(t−1)|): WARN > 0.3, HACK > 0.8.

## Current reward design (v4)

| Term | Weight | Why |
|---|---|---|
| `track_lin_vel` / `track_ang_vel` | +2.0 / +1.0 | the task; PGTT 1.0 / 0.5, ×2 so it clearly beats stepping in place (max 3.0/s vs `foot_phase` 1.0/s) |
| `foot_phase` | +1.0 | PGTT gait shaping, max 1.0/s |
| `swing_contact` | −0.25 | PGTT "contact": a foot on the ground when it should swing |
| `lin_vel_z` / `ang_vel_xy` | −2.0 / −0.05 | PGTT: no bouncing or wobbling |
| `default_pose` | −0.5, hips only | PGTT: legs don't splay sideways |
| `termination` | −100 (= −2 per fall) | after the clip, so a fall always costs |
| `base_height` | −50 below 0.25 m | ours: crouching costs (PGTT's robot cannot lie down) |
| `leg_contact` | −0.25, ramped | ours: kneeling costs |
| `action_rate` | −0.01, ramped | vibrating ±1 costs ~0.5/s; PGTT's −0.001 would make it free |
| `orientation`, `joint_limits` | −0.2, −1.0 | small safety terms; PGTT 0 |
| `joint_power`, `joint_torques` | −2e-5, −1e-5, ramped | energy (Miki 2022), negligible early |
| Episode end | flipped, or trunk/hips touch the terrain | ours (full collisions) |

Measured per second, what PPO receives after the clip:

| Behaviour | v4 |
|---|---|
| walking well (estimate: standing + the extra tracking) | ≈ +3.4 |
| standing still | +1.09 |
| stepping in place (crude script) | +1.04 |
| crouching at 0.17 m | +0.84 |
| vibrating ±1 every step | −0.19, plus 9 falls per robot-minute |
| random actions (start of training) | −1.19 |

Every shortcut now scores below standing still, and walking is roughly 3× standing.

## Run 4: bang-bang vibration with bounded actions

**Setup.** Commit `3164449`: tanh-bounded actions, Kp 40 / Kd 0.5, action scale 0.5, PGTT
`foot_phase`, the v4 rewards in the table above, std capped at 1.0.

**Symptom.** Hacks check on `model_6300.pt`, no noise, no pushes, 32 robots × 10 s:

| Check | Value | |
|---|---|---|
| `speed_ratio` | 0.039 | HACK: ignores the command |
| `falls_per_min` | 51 | HACK: an episode ends every ~1.2 s (`body_contact` 221, `fell_over` 51) |
| `action_jitter` | 0.789 | WARN (0.8 = HACK) |
| `standing_drift` | 0.42 m/s | HACK |
| `leg_ground_frac` | 0.27 | WARN |
| `base_height` | 0.272 m | OK: it is not lying down |

Reward per second: positive terms +1.31 (`foot_phase` 0.62, tracking 0.70), costs −2.31
(`ang_vel_xy` −1.08, `lin_vel_z` −0.49, `default_pose` −0.22, `action_rate` −0.17, ...),
plus −1.59 from falls. Most steps sum below zero, so the clip turns them into 0.

**What these numbers prove (no assumptions).**
- `action_rate` is −0.01 × Σ_j (Δa_j)², so −0.1686/s means Σ_j (Δa_j)² = 16.9 per step: the
  mean over 12 joints is E[x²] = 1.41, with x = |a_t − a_(t−1)| ∈ [0, 2]. `action_jitter` gives
  E[x] = 0.79.
- x² ≤ x when x ≤ 1, and x² ≤ 2x when x ≤ 2, so E[x²] ≤ E[x] + E[x·1(x>1)]. Hence
  E[x·1(x>1)] ≥ 0.62, and since x ≤ 2, **P(x > 1) ≥ 0.31**.
- At least 31% of joint-steps jump by more than half the action range in 20 ms.
- A periodic gait at f ≤ 3 Hz with amplitude ≤ 1 changes by at most 2π·f·dt = 0.38 per step,
  so these jumps are not walking. A jump close to 2 needs the action near −1 then near +1:
  **the policy outputs saturated, bang-bang actions.**

## Why four runs failed

### 1. The entropy bonus rewards saturated actions (proven, and confirmed on run 4's logs)

PPO's loss includes `−0.01 × entropy`. rsl_rl computes the entropy of the Gaussian z, but the
robot executes a = tanh(z) (v4), or a clamp of z (v2, v3: targets clamped to the joint limits).
Per dimension (Monte Carlo, 400k samples):

| mean μ | std σ | raw H(z) (ours) | ∂H/∂μ | ∂H/∂σ | H(tanh z) (Brax) | ∂H/∂μ | ∂H/∂σ | std of a | share \|a\| > 0.95 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.0 | 1.42 | 0 | +1.0 | 0.67 | 0 | −0.21 | 0.63 | 7% |
| 1 | 1.0 | 1.42 | 0 | +1.0 | 0.10 | −1.10 | +0.10 | 0.50 | 20% |
| 2 | 1.0 | 1.42 | 0 | +1.0 | −1.36 | −1.73 | +0.63 | 0.25 | 57% |
| 3 | 1.0 | 1.42 | 0 | +1.0 | −3.22 | −1.94 | +0.91 | 0.09 | 88% |
| 3 | 0.3 | 0.22 | 0 | +3.3 | −4.40 | −1.99 | +3.33 | 0.004 | 100% |

What this means:
- Ours: ∂H/∂μ = 0 everywhere, so **saturating costs nothing**, and ∂H/∂σ = 1/σ > 0 always, so
  **the bonus always pushes the std up**. In v2 the std grew 1.0 → 2.0 by iteration 500 and
  → 2.5 later (it never fell, which is unusual in locomotion). In v3 and v4 it sits at the cap.
- With σ pinned at 1, the executed action at μ = 0 has std 0.63: ±0.3 rad of random target
  every 20 ms. The only way the policy can get precise actions is |μ| ≥ 3, where tanh
  saturates (88% of actions at ±1). The deterministic policy then outputs ±1:
  **bang-bang control, what run 4's numbers show.**
- Brax (PGTT's trainer) adds log(1 − tanh² z) to the entropy, evaluated at one sample (source:
  `brax/training/distribution.py`, `ParametricDistribution.entropy`). That term pulls a
  saturated mean back (∂H/∂μ ≈ −2), its maximum is at μ = 0, σ = 0.85, and the std can shrink
  (Brax: softplus + 0.001, no cap).

Fix: `squashed_entropy=True` in `ActorCriticWild` (Brax's formula, unit test in
`tests/test_pgtt_audit.py`).

**Confirmed on run 4's TensorBoard log:** `Policy/mean_noise_std` = 1.000 and `Loss/entropy` =
17.027 at every logged iteration from 0 to 6944. 17.027 = 12 × ½ log(2πe), the entropy of 12
unit Gaussians, so the std sat exactly on the cap for the whole run. The policy never got
more precise. In repro0, with the corrected entropy, the std fell 0.70 → 0.31 within 300
iterations and to 0.28 by the end.

### 2. Every run changed many things against the published setup at once

The official PGTT config (github.com/NtagkasAlex/phase_guided_terrain_traversal: `go2` robot
config, `joystick_base.py`, `base.py`, `randomize_simple.py`, `train.py`) against v4:

| | Official PGTT (flat) | v4 |
|---|---|---|
| Collisions | feet only | full body |
| Episode ends | flipped over | flipped, or trunk/hips touch the ground (81% of v4's endings) |
| Termination cost | −1 × dt inside the clip (≈ 0) | −2 after the clip |
| Pushes / external force | none (`pert_config.enable=False`) | ±0.5 m/s every 4-8 s, ±15 N every 3-6 s |
| Motor / sensor delay | none | 0-20 ms / up to 20 ms |
| Passive joint damping | 0.5 (= Kd, set in `base.py`) | 0.1 (our XML). **Missed in `3164449`.** |
| Commands | each axis non-zero with p = (0.9, 0.25, 0.5) | uniform on all 3 axes |
| Domain randomization | mild (Kp ±10%, mass, CoM, armature) | wider (+ effort limits, encoder bias, friction) |
| Extra rewards | none | base_height, leg_contact, orientation, joint_limits, energy, ramp |
| PPO | γ 0.97, lr 3e-4 fixed, 8 epochs, clip 0.3, value 0.5 | γ 0.99, lr 1e-3 adaptive (up to 1e-2), 5 epochs, clip 0.2, value 1.0 |
| Entropy | of tanh(z) | of z (section 1) |
| Std | learned, can shrink, init ~0.7 | capped at 1.0, init 1.0 |
| Foot height in `foot_phase` | relative to the body (feet 0.30 m below the IMU) | relative to the ground under the foot |

We never had a run that reproduced a known-good result. Every failure therefore had a dozen
possible causes, and each fix was a guess at one of them.

### 3. The reward table is not the main difference (measured)

Scripted behaviours, flat ground, 16 robots × 5 s, full penalty weights.
"Clip binds" = share of steps where the costs exceed the positive terms, so PPO receives 0:

| Behaviour | v4: PPO gets /s | v4: clip binds | PGTT repro: PPO gets /s | PGTT repro: clip binds |
|---|---|---|---|---|
| stand still | +1.09 | 3% | +1.33 | 2% |
| step in place | +1.04 | 6% | +0.91 | 4% |
| random actions (start of training) | −1.19 (43 falls/min) | 64% | +0.27 (53 falls/min) | 55% |
| vibrate ±1 | −0.19 (9 falls/min) | 78% | +0.02 (48 falls/min) | 89% |
| fold the legs | +0.84 | 4% | +0.93 | 4% |

With PGTT's own rewards, standing also scores at least as high as stepping, and PGTT still
learns to walk. So the reward shape alone does not explain v4, and a fifth redesign of the
weights would be another guess. Note: with action scale 0.5 the robot cannot fold its legs
far enough to lie down (±0.5 rad around the standing pose), so v1's exploit is closed by the
action space, not the reward.

### 4. Smaller bugs fixed

- A robot that fell on the last step of its episode was also flagged as a time-out, and PPO
  added the value of continuing (`vecenv_wrapper.py`: now `truncated & ~terminated`).
- The hacks check had no saturation metric. New: `action_saturation` (share of actions beyond
  ±0.95), WARN > 0.2, HACK > 0.5.

### 5. What I got wrong earlier

- In `3164449` I wrote that we matched PGTT's gains. PGTT also sets the passive joint damping
  to Kd, so its joints have 1.0 total damping and ours 0.6.
- I treated the raw Gaussian log-probability as correct under tanh. That is right for the PPO
  ratio (the tanh Jacobian cancels), but the entropy does not cancel (section 1).
- Run 4 was meant to "match PGTT" but kept a dozen of our own changes (section 2).

## The plan: reproduce, then add one thing at a time

**Rung 0: `Mjlab-Velocity-PGTT-Repro-Unitree-Go2`.** The table in section 2, left column,
ported to mjlab (`unitree_go2_pgtt_repro_env_cfg`, `unitree_go2_pgtt_repro_runner_cfg`).
- Same teacher network and observation groups as before.
- The robot has feet-only collisions and PGTT's damping.
- Episodes end only on a flip.
- PGTT rewards, with `foot_phase_tracking_body` measuring foot height relative to the IMU.
- PGTT commands, PGTT domain randomization, no pushes or delays.
- PPO with Brax's settings and the corrected entropy.
- Remaining differences: rsl_rl instead of Brax, 4096 robots instead of 8192, one global std
  instead of a state-dependent one.

**Pass criteria, fixed before the run:** the hacks check on the iteration-2500 checkpoint shows
- `speed_ratio` ≥ 0.65 (PGTT's own stop criterion is 65% tracking);
- `falls_per_min` < 0.2;
- `action_saturation` < 0.2 and `action_jitter` < 0.3;
- `Policy/mean_noise_std` falling below its starting 0.7.

PGTT trains 200M steps, which is about 2440 iterations here.

- **If rung 0 fails**, the cause is in our training stack (rsl_rl, mjlab), not in rewards. The
  next test is the stock `Mjlab-Velocity-Flat-Unitree-Go2` task.
- **If it passes**, add one change per run (~1500 iterations each), in this order:
  0b. a penalty on fast action changes (added after repro0's gait analysis, see
     [Rung 0 result](#rung-0-result-the-first-walking-policy)),
  1. full-body collisions + trunk/hip termination,
  2. motor and sensor delays,
  3. pushes and external force,
  4. our wider domain randomization,
  5. our extra reward terms,
  6. full command range,
  7. stairs and mixed terrain.

  Each run uses the same pass criteria, so a failure points at exactly one change.
- Then the student is distilled from the last teacher that passed.

**Run 4 data:** copied and checked (section 1): the std was pinned at 1.0 for the whole run.

## Rung 0 result: the first walking policy

Run `go2_pgtt_repro/*_repro0`: 3000 iterations on the 3070 (about 3 hours), checked on the
laptop with `scripts/check_reward_hacking.py` and the new `scripts/eval_gait.py`.

### Pass criteria (fixed before the run)

| Check (model_2500, deterministic) | Result | Criterion |
|---|---|---|
| `speed_ratio` | 0.949 | ≥ 0.65 |
| `falls_per_min` | 0 | < 0.2 |
| `action_saturation` | 0.003 | < 0.2 |
| `action_jitter` | 0.240 | < 0.3 |
| `Policy/mean_noise_std` | 0.70 → 0.31 by iteration 300, 0.28 at the end | falls below 0.7 |

It passed every criterion, at iteration 1000 as well. The short check's zero falls (5.3
robot-minutes) can't by itself show a rate below 0.2/min. The training log can:
`Episode_Termination/fell_over` is about 0 over 4096 robots for the last 1500 iterations.
This is one seed only.

### How it walks (`eval_gait.py`, fixed commands, 3 robots each, 9 s after a 3 s settle)

| Scenario | Achieved | swing/s | blip/s | apex [cm] | slip [m/s] | tilt [°] | power [W] | tremor | power > 6 Hz |
|---|---|---|---|---|---|---|---|---|---|
| stand, 2 Hz | 0.01 m/s | 1.94 | 1.27 | 8.5 | 0.14 | 1.4 | 77 | 0.32 | 21% |
| forward 0.6, 2 Hz | 0.60 | 1.94 | 1.22 | 7.7 | 0.16 | 1.4 | 117 | 0.31 | 20% |
| forward 1.0, 2 Hz | 1.01 | 1.93 | 1.73 | 8.5 | 0.25 | 1.3 | 159 | 0.35 | 22% |
| forward 0.6, 1 Hz | 0.59 | 0.94 | 2.33 | 8.5 | 0.19 | 2.3 | 98 | 0.31 | 22% |
| forward 1.0, 3 Hz | 1.00 | 2.93 | 1.58 | 6.5 | 0.28 | 1.1 | 188 | 0.41 | 32% |
| backward 0.5 | −0.49 | 1.94 | 1.88 | 7.1 | 0.25 | 0.5 | 93 | 0.38 | 27% |
| sideways 0.5 | 0.48 | 1.94 | 2.31 | 6.5 | 0.26 | 1.4 | 111 | 0.39 | 29% |
| turn 1.0 rad/s | 1.01 rad/s | 1.93 | 1.31 | 7.7 | 0.17 | 1.7 | 83 | 0.34 | 23% |

- **Tracking is excellent** in every direction: 95-101% of the command, no overspeed, cross
  errors ≤ 0.01.
- **It trots exactly on the clock.** Swings per second equal the commanded 1, 2 or 3 Hz.
  Diagonal feet agree 89-96% of the time, and 83-90% of contacts match the clock.
- **Swing apex 6.5-8.5 cm** (the target is 10 cm above stance).
- **The body is steady:** height 0.28 m, tilt 0.5-2.3°.
- **Knees never come close to the ground.** The lowest knee origin is 6.7 cm, so full
  collisions (rung 1) shouldn't collide anything new.
- **Torque** peaks at 19-22 Nm (limit 23.5). No action is saturated.
- **It steps in place when told to stand**: the clock never stops, so it uses 77 W.
- **Tremor.** The actions are a clean trot plus fast tremor. 85% of `Σ(Δa)²` and 98% of
  `Σ(a_t − 2a_(t−1) + a_(t−2))²` lie above 6 Hz, and 20-32% of the action power is above
  6 Hz. Its tremor (0.29-0.43) is 4× what any motion of at most 3 Hz within the action range
  can produce (0.09).
- **Bounces.** Feet make 1-2.3 short air gaps (< 60 ms) per foot-second.
- **Training longer does not fix the tremor.** At iteration 1500 it was 0.29-0.43, the same
  as at 2500. Rewards were already at 98% of their maximum: 2.456 of 2.5/s, with all
  penalties costing 0.03/s. There is nothing left in the objective that asks for it.

### Desiderata: which are under- or over-valued

| Desideratum | Term (weight) | What it costs / earns now | Measured | Verdict |
|---|---|---|---|---|
| Velocity tracking | track_lin 1.0, track_ang 0.5 | 1.47 of 1.5/s | 95-101% in every direction | right |
| Gait timing and swing height | foot_phase 1.0, swing_contact −0.25 | 0.99 of 1/s | on the clock, apex 6.5-8.5 cm | right |
| Steady body | lin_vel_z −2, ang_vel_xy −0.05 | −0.02/s | tilt < 2.3° | right |
| Hips not splayed | default_pose (hips) −0.5 | −0.007/s | fine | right |
| **Smooth actions** | action_rate −0.001 | −0.0015/s (0.06%) | tremor 0.37, 20-32% of action power > 6 Hz | **undervalued** |
| Low energy | torques −2e-6 | −0.0002/s (0.01%) | 77-188 W | undervalued, but tremor wastes power too: revisit after 0b |
| Soft landing | none | 0 | 1-2.3 bounces/foot-s | unrewarded: revisit after 0b (tremor at touchdown may cause it) |
| No foot slip | none | 0 | 0.14-0.28 m/s (includes the ball foot rolling) | unrewarded: measure the real contact-point slip first |
| Stand still on a zero command | none (the clock always runs) | 0 | steps in place, 77 W | by design in PGTT: handle on the robot with a stand mode, not with a reward |

Nothing is measurably over-valued: tracking and the gait clock are both satisfied, not
overdone. Compared per radian of joint-target change, PGTT's smoothing is very weak:
- PGTT: 0.004 /rad²
- legged_gym: 0.16 /rad² (40× PGTT)
- mjlab's stock Go2: 0.8 /rad² (200× PGTT)

### The fix: rung 0b, a penalty on the action's second difference

`Mjlab-Velocity-PGTT-Repro-Smooth-Unitree-Go2` is repro0 plus one term, `action_acc_l2`
(weight −0.02), on a_t − 2a_(t−1) + a_(t−2).
- **Why the second difference.** For a sine of frequency f, the first difference scales as
  (2 sin(πf·dt))² and the second as its square. Per unit of amplitude, the second difference
  costs a 12 Hz tremor about 900× more than a 2 Hz gait. The first difference
  (`action_rate`) does that only 30× more, so it also fights the steps. On repro0's actions,
  98% of this term's cost is tremor and 2% is the gait. Walk These Ways uses the same term
  (its action smoothness 2).
- **Why −0.02.** It's the largest round weight that keeps the zero clip slack early on: a
  random policy pays about 0.35/s of the ~1/s it earns, while −0.05 would pay 0.9/s and bring
  back v4's binding clip. On repro0 it costs the tremor 0.061/s and the gait 0.001/s, which is
  the biggest remaining gradient once tracking is saturated.
- **What it does not change.** Weight −0.02 can't make standing more attractive than walking:
  both step on the clock, and the gait part costs 0.001/s.

**Pass criteria for 0b (fixed before the run).** Fresh run, 1500 iterations, compared with
repro0's model_1500, which has the same training budget.
- **No regression:** `speed_ratio` ≥ 0.9, 0 falls, saturation < 0.05, swings/s within 5% of
  the clock, apex ≥ 6 cm.
- **Fixed:** `action_tremor` < 0.15 (repro0: 0.37) and action power above 6 Hz < 10% in
  every scenario (repro0: 20-32%).
- **Watched, not criteria:** bounces/s, power, slip.

What a failure would mean:
- Tremor above 0.25 means the weight is too weak; try −0.05 with the clip-binding share
  logged.
- Tracking below 0.9 means it is too strong.

### Should repro0 keep training? No.

- From 1000 to 3000 its metrics hardly moved (speed ratio 0.935 → 0.957) and the reward is at
  98% of its maximum.
- What is left to fix (tremor) isn't in its objective, so more iterations of the same run
  can't fix it.
- Repro0 is kept frozen as the parent and the control.
- Each rung starts fresh. A rung has to show that its change can be learned from scratch,
  because the final teacher must be. repro0 reached 93% tracking by iteration 300, so
  1500 iterations (about 1.5 hours) per rung is enough.

### What `suggestions.md` raised, and where it stands

| Point | Status |
|---|---|
| Don't call the port known-good until measured | Measured now: it walks, on one seed. |
| "std must fall" is not a correctness test (even correct entropy can raise it) | Agreed; kept as a diagnostic only. It fell here. |
| 0.65 is not PGTT's metric; also check overspeed and per-direction error | eval_gait reports each direction: no overspeed, cross errors ≤ 0.01. |
| 5.3 robot-minutes can't certify < 0.2 falls/min | Training log: ~0 falls over 4096 robots for 1500 iterations. |
| Feet-only collisions hide body contact; measure clearance geometrically | Lowest knee origin is 6.7 cm; base at 0.28 m. |
| Pass at several checkpoints and seeds | Passes at 1000, 1500 and 2500. Only one seed so far: run a second seed on the final rung. |
| Compare a child against its parent at the same budget | 0b at 1500 is compared with repro0's model_1500. |
| Split rungs (collisions vs termination, motor vs sensor delay) | Adopted where cheap: split if a combined rung fails. |
| Checker samples after resets; per-cause falls can double count | Doesn't affect a run with zero falls. Still to fix before rungs with falls. |

## Lessons

1. **Check the ordering of behaviours, not single weights.** Score scripted behaviours (stand,
   step, lie, vibrate, random) and check walk > stand > shortcuts. Run 1's weights looked
   reasonable one by one.
2. **A shaping term must not pay for doing nothing.** Per-foot `foot_phase` paid planted feet.
3. **Episode length only means "didn't fall" if falling ends the episode.**
4. **If living can cost more than dying, the policy will die.** Clip the step reward at 0, or
   make sure penalties can't outgrow the task reward.
5. **A termination is only a punishment if the future it throws away is positive.**
6. **Bound the actions.** With unbounded actions and clamped targets, outputs past the limits
   are free, noise grows, and penalties on actions can explode.
7. **A penalty ramp delays failures.** v2 looked healthy until the ramp ended at iteration 1500.
   Judge ramped runs after 1500.
8. **Read the reference code, not only the paper.** The tanh squash (Brax PPO's default) and the
   reward clip are in PGTT's training code; we found them there, not in the paper.
9. **A change of action distribution changes the entropy too.** A bonus on the wrong entropy
   pushes the std up for free and the mean into saturation.
10. **Reproduce a known-good result before changing it.** Then add one change at a time, with
    pass criteria fixed before the run.
11. **Once the reward saturates, what is left is only what the penalties ask for.** repro0 got
    98% of the maximum reward; its tremor survived 1500 more iterations because nothing paid
    for removing it. Training longer is not a fix for something the objective doesn't measure.
12. **Penalise what you want gone, at the frequency where it lives.** The second difference
    of the action separates a 12 Hz tremor from a 2 Hz gait about 900× per unit size; the first
    difference only 30×. Measure the spectrum before choosing the term.

## What to watch in a rung run (repro0, 0b, ...)

| When | Graph or check | Good | Bad |
|---|---|---|---|
| iteration ~300 | `Train/mean_episode_length` | climbing towards 1000 | stuck low |
| throughout | `Policy/mean_noise_std` | falls below 0.7 | rises (the entropy problem is back) |
| throughout | `Metrics/hack/action_saturation` | below 0.2 | above 0.5 (bang-bang) |
| throughout | `Metrics/hack/action_jitter` | 0.05-0.3 | above 0.8 |
| from ~1000 | `Metrics/hack/speed_ratio` | rising towards 1 | near 0 |
| throughout | `Metrics/hack/action_tremor` | falling below 0.15 (0b) | stays ~0.35 like repro0 |
| 1500 | hacks check + `eval_gait.py` on the checkpoint | that rung's pass criteria | anything HACK, or a regression against its parent |

`body_ground_frac` and `leg_ground_frac` stay 0 in the reproduction: with feet-only collisions
nothing but the feet can touch the ground.
