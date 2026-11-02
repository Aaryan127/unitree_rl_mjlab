# Reward hacking in the PGTT teacher: runs 1-4, the fixes, the first walking policy, and the ladder

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
| smooth0b | `go2_pgtt_repro/*_smooth0b` (1500) | walks (viewer); + tremor penalty | parent of rungs 1-7 |
| rungs 1-7 | `go2_pgtt_repro/*_rung<N>_<name>` | 1-3 walk (WARN: slight kneeling); 4 and 5 learn to walk, then collapse | [The ladder in detail](#the-ladder-in-detail-what-every-rung-trains-with) |
| `rung4_term100` | `go2_pgtt_repro/*_rung4_term100` (1500) | kneels in place instead of falling | [Rungs 4 and 5 collapse](#rungs-4-and-5-collapse-1-nov) |
| diagnosis | `go2_pgtt_repro/*_r[456]_*` (2 of 4 run) | both collapse: neither the fall price nor the entropy bonus is the cause | [The diagnosis batch](#the-diagnosis-batch-scriptsdiagnose_collapsesh-2-nov) |

**Where things stand (2 Nov, 23:00):**
- Rungs 0-3 walk. Rungs 4 and 5 learned to walk, then collapsed.
- **Cause found: a bug.** Rung 4's "weaker motors" randomization made the motors weaker
  again at *every* reset, so each robot's torque shrank over training (to ~11% after 30
  resets). Fixed and tested. See
  [Root cause: the motors weakened at every reset](#root-cause-the-motors-weakened-at-every-reset-2-nov).
- Next: run rung 4 again with the fix, then rungs 5-7, then the full teacher (`COMMANDS.md` 4a/4d).
- Commands for all of this: [COMMANDS.md](../COMMANDS.md).

## How the reward works here

- Every term is `weight × value × dt` (dt = 0.02 s), summed each step. Below, rewards are given
  per second (`weight × value`), which is what the scripted measurements report.
- PPO maximises the discounted sum of future rewards (γ = 0.99 in runs 1-4, 0.97 from rung 0 on). An episode ends at 20 s
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
| `base_height` | −50 below 0.265 m (0.25 m before 3 Nov) | ours: crouching costs (PGTT's robot cannot lie down) |
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

**Rungs 1-7 as built** (`unitree_go2_pgtt_rung_env_cfg`, tasks
`Mjlab-Velocity-PGTT-Rung<N>-Unitree-Go2`, all trained in a row by `scripts/train_ladder.sh`).
Each rung copies its change from the full teacher config, so rung 7 is the teacher except
for what is listed under "left out". Every rung's settings and rewards in full:
[The ladder in detail](#the-ladder-in-detail-what-every-rung-trains-with).

| Rung | Run name | Adds (on top of the rung before) | Iterations |
|---|---|---|---|
| 1 | `rung1_collisions` | full-body collisions; trunk/hip contact ends the episode | 1500 |
| 2 | `rung2_delays` | 0-20 ms actuator delay; 1-step sensor delays (3 for the student's map) | 1500 |
| 3 | `rung3_pushes` | velocity pushes every 4-8 s, ±15 N / ±2 Nm on the base every 3-6 s | 1500 |
| 4 | `rung4_randomization` | the teacher's events: friction 0.3-1.2, mass -1/+2 kg, CoM, link mass, gains, effort limits, joint friction, armature, encoder bias, tilted and moving resets | 1500 |
| 5 | `rung5_rewards` | `base_height` (-50), `leg_contact` (-0.25), `joint_limits` (-1), no ramp | 1500 |
| 6 | `rung6_commands` | all three command axes at once, 5% standing, resampled every 8-12 s | 1500 |
| 7 | `rung7_terrain` | mixed terrain (stairs, slopes, rough) with the terrain curriculum | 6000 |

- **Rung 5 is smaller than "our extra reward terms".** It adds only the terms for what full
  collisions and terrain make possible: crouching, kneeling or dragging a shin, pressing into
  the joint limits. Left out on purpose: the doubled tracking weights, the -100 termination,
  the power/torque terms and the penalty ramps. The PGTT rewards already walk, `action_acc`
  already smooths, and a ramp only delays a failure (lesson 7).
- **Left out of rung 7 compared with the teacher task:** those rewards, PGTT's passive
  joint damping (0.5) and its joint_pos noise (±0.03), and the runner (Brax settings, not
  the teacher's). All are PGTT's own choices, kept because they walk.
- **The script's gate is weaker than the pass criteria.** It stops only on a HACK verdict
  in the hacks check of the last checkpoint, so it can run overnight. Each rung still has to
  be judged on the full criteria (hacks check with no WARN, `eval_gait.py` against the rung
  before) before its teacher is trusted.
- **The checker counts each fall once now.** With full collisions a fall can trip
  `fell_over` and `body_contact` in the same step; before, both were added to `falls_per_min`.

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

**0b result (2026-11-01):** run `smooth0b`, 1500 iterations. It walks in the viewer. The
numbers (tremor, share of action power above 6 Hz) are not measured yet; the ladder (rungs
1-7) builds on it from here.

### Should repro0 keep training? No.

- From 1000 to 3000 its metrics hardly moved (speed ratio 0.935 → 0.957) and the reward is at
  98% of its maximum.
- What is left to fix (tremor) isn't in its objective, so more iterations of the same run
  can't fix it.
- Repro0 is kept frozen as the parent and the control.
- Each rung starts fresh. A rung has to show that its change can be learned from scratch,
  because the final teacher must be. repro0 reached 93% tracking by iteration 300, so
  1500 iterations (about 3 hours on the 3070, ~7 s per iteration) per rung is enough.

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

## The ladder in detail: what every rung trains with

This section describes every rung completely: the robot, the world, the commands, the
rewards and the training settings, with the code each comes from. Rung 0b is described in
full; each later rung lists only what it changes, and keeps everything else from the rung
before it.

**Why a ladder.** v1-v4 changed about 15 things at once against the published setup, and
none of them walked. A failure could not be traced to any one change. Each rung adds exactly
one change to a setup that walks and trains a new policy from scratch. If rung N fails and
rung N-1 passed, rung N's change caused it. That one difference was checked by diffing the
complete flattened configs of neighbouring rungs.

**Where it lives.**
- Configs: `unitree_go2_pgtt_repro_env_cfg` (rung 0),
  `unitree_go2_pgtt_repro_smooth_env_cfg` (0b) and `unitree_go2_pgtt_rung_env_cfg(rung)`
  (1-7), all in `mjlab/tasks/velocity/config/go2/env_cfgs.py`.
- Tasks: `Mjlab-Velocity-PGTT-Repro-Unitree-Go2`, `...-Repro-Smooth-...`,
  `...-Rung1-...` to `...-Rung7-...`.
- Runs: `logs/rsl_rl/go2_pgtt_repro/<date>_<time>_<run name>/`.
- Rungs 1-7 are trained in a row by `scripts/train_ladder.sh`, which stops at the first HACK.

### What every rung shares

**Policy and training (the same for all rungs: `unitree_go2_pgtt_repro_runner_cfg`).**
- Actor and critic MLPs 512-256-128, ELU.
- The actor sees `policy` (base angular velocity, gravity direction, the command, joint
  positions and velocities, the last action, the gait phase), `proprio_history` (the last
  3 joint positions, 2 joint velocities and 2 actions), `height` (11 × 9 clean height scan
  around the base) and `privileged` (base linear velocity, foot contacts, forces and normals,
  foot friction, leg-ground contact, the external force on the base).
- The critic sees the same plus the clean `critic` group.
- PPO with Brax's (PGTT's) settings:

  | Setting | Value |
  |---|---|
  | γ | 0.97 |
  | λ | 0.95 |
  | learning rate | 3e-4, fixed |
  | epochs × mini-batches | 8 × 4 |
  | clip | 0.3 |
  | entropy | 0.01, on the tanh-squashed action (the v4 fix) |
  | initial std | 0.7 |
  | rollout | 4096 robots × 20 steps per iteration |
  | checkpoints | every 100 iterations |

**Actions.** 12 joint position targets: `target = default pose + 0.5 × tanh(z)`, so each
target stays within ±0.5 rad of the standing pose. A PD controller tracks them with Kp 40,
Kd 0.5, plus passive joint damping 0.5 (PGTT). Policy at 50 Hz (dt 0.02 s), physics at
200 Hz.

**Gait clock.** Each robot gets a trot clock at 1-3 Hz. The diagonal feet share a phase:
the first half of a cycle is stance, the second half swing. The clock is in the observations
and drives `foot_phase` and `swing_contact`.

**Episode.** 20 s. Flipping over (tilt > 90°) ends it from rung 0 on; trunk or hip contact
ends it from rung 1 on.

### How the reward is computed (all rungs)

- Each step: `r = clip( Σ_terms weight × value × dt , min 0 ) + termination term`.
- The clip at 0 is PGTT's: a step's reward is never negative, so staying alive never costs
  more than dying (the v2 lesson).
- The termination term is added after the clip, so a fall always costs something.
- Per second, the most a step can pay is **2.5**: 1.0 linear tracking + 0.5 turn tracking +
  1.0 foot phase. Everything else is a penalty.

### Reward terms of rungs 0b-4 (PGTT's rewards + one smoothness term)

Set in `unitree_go2_pgtt_repro_env_cfg` (`cfg.rewards = {...}`) and
`unitree_go2_pgtt_repro_smooth_env_cfg`. Most functions are in
`mjlab/tasks/velocity/mdp/rewards.py`; `is_terminated`, `action_rate_l2` and
`action_acc_l2` are in `mjlab/envs/mdp/rewards.py`.

| Term | Weight | Value (per step, before × weight × dt) | What it asks for | A good gait pays /s |
|---|---|---|---|---|
| `track_lin_vel` | +1.0 | `exp(-‖v_cmd,xy − v_xy‖² / 0.25)`, body frame | walk at the commanded speed and direction | ~0.95 |
| `track_ang_vel` | +0.5 | `exp(-(ω_cmd,z − ω_z)² / 0.25)` | turn at the commanded rate | ~0.49 |
| `foot_phase` | +1.0 | `exp(-Σ_feet (h_target − h)² / 0.05)`, h = foot height below the IMU in the body frame; target −0.30 m in stance, rising smoothly to −0.20 m (+ terrain step, ≤ 0.15 m) at mid-swing | lift each foot on the clock, ~10 cm, and keep the body at walking height | ~0.9-1.0 |
| `swing_contact` | −0.25 | number of feet on the ground while their clock says swing | actually lift the swing feet | small |
| `lin_vel_z` | −2.0 | v_z² | no bouncing | small |
| `ang_vel_xy` | −0.05 | ω_x² + ω_y² | no rolling or pitching | small |
| `default_pose` | −0.5 | Σ_hips (q − q_default)² (thigh and knee weight 0) | don't splay the legs sideways | small |
| `torques` | −2e-6 | ‖τ‖₂ + ‖τ‖₁ | don't waste torque | ~0.001 |
| `action_rate` | −0.001 | Σ (a_t − a_t−1)² | smooth actions (PGTT's, weak) | ~0.002 |
| `action_acc` | −0.02 | Σ (a_t − 2a_t−1 + a_t−2)² | **no tremor**: costs a 12 Hz shake ~900× more than a 2 Hz step (rung 0b; repro0 doesn't have it) | ~0.06 on repro0's tremor |
| `termination` | −1.0 | 1 on the step an episode ends early | don't fall (−0.02 per fall; the real cost is the lost future reward) | 0 |
| `hacking_monitor` | (always 0) | logs `Metrics/hack/*` | nothing, only measures | 0 |

Measured on repro0 (rung 0) at iteration 2500: 2.456 of the possible 2.5 per second, and
all penalties together 0.03 per second.

### Rung 0 (`repro0`) and rung 0b (`smooth0b`): the base

The official PGTT flat-ground setup, ported to mjlab (section 2 of
[Why four runs failed](#why-four-runs-failed)).

| Part | Setting |
|---|---|
| Ground | flat plane |
| Collisions | **feet only** (PGTT's `scene_mjx_feetonly.xml`): the body and legs pass through the ground |
| Delays | none: actions and sensors are instant |
| Commands | each of vx, vy, ωz uniform in [−1, 1], non-zero with probability 0.9 / 0.25 / 0.5 (so mostly forward, sometimes sideways or turning); new command every 2-8 s |
| Gait frequency | 1-3 Hz |
| Sensor noise | joint position ±0.03, joint velocity ±1.5, angular velocity ±0.2, gravity ±0.05 |
| Reset | anywhere in ±0.5 m, any yaw, body level, almost still (±0.1 m/s), joints at the default pose |
| Randomization | friction 0.4-1.0, base mass ±1 kg, base CoM ±5 cm, link mass ×0.9-1.1, Kp ×0.9-1.1, armature ×1.0-1.05, joint damping ×0.9-1.1 |
| Pushes | none |
| Rewards | the table above (0b adds `action_acc`) |
| Iterations | 3000 (repro0), 1500 (0b) |

- **Result 0:** walks, 95% of commanded speed, no falls, but with a 12 Hz tremor on top of
  the gait.
- **Result 0b:** walks in the viewer; tremor numbers still to be measured.

### Rung 1 (`rung1_collisions`): the whole body can touch the ground

- **Changes:** the robot's collisions go from feet-only to the full model, so trunk, hips,
  thighs and calves collide with the ground. A new termination, `body_contact`, ends the
  episode when the trunk or a hip touches the terrain.
- **Why:** the real robot has a body. A feet-only robot can't kneel, lie down or hit a step
  with its shin, so no policy trained on it learns to avoid that. The termination is the
  v1 fix (lying on the belly and paddling).
- **Rewards:** unchanged.
- **Risk:** falls now end episodes much more often (any body contact, not only a flip).
  Thighs and calves can touch without ending anything, so kneeling is possible and still
  costs nothing (rung 5 adds that cost).
- **Watch:** `Episode_Termination/body_contact`, `Metrics/hack/leg_ground_frac`,
  `body_ground_frac`.

### Rung 2 (`rung2_delays`): late motors and sensors

- **Changes:**
  - Each episode, the motor command arrives 0-20 ms late (0-4 physics steps; the actuator
    becomes a `DelayedActuatorCfg`, event `actuator_delay`).
  - Angular velocity, gravity, joint positions and velocities can be up to 1 policy step
    (20 ms) old.
  - The student's height map can be up to 3 steps (60 ms) old. The teacher doesn't see it.
- **Why:** on the real Go2, commands and readings travel over DDS. A policy that only works
  with instant feedback shakes or falls on the robot.
- **Rewards:** unchanged.
- **Risk:** delays make high-gain reactions unstable, which can bring the tremor back.
- **Watch:** `action_tremor`, `action_jitter`, falls.

### Rung 3 (`rung3_pushes`): shoves and forces

- **Changes:**
  - `push_robot`: every 4-8 s the base's velocity is suddenly changed by up to ±0.5 m/s in
    x and y.
  - `external_wrench`: every 3-6 s a force of up to ±15 N and a torque of up to ±2 Nm act on
    the base. The teacher sees this force in its privileged observations.
- **Why:** the robot must recover from bumps, and the student learns to estimate them from
  the teacher's reactions.
- **Rewards:** unchanged.
- **Risk:** falls after pushes; tracking drops briefly after each push.
- **Watch:** falls, speed ratio.

### Rung 4 (`rung4_randomization`): our wider randomization

- **Changes:** all of the teacher's randomization events, wider than PGTT's:

  | Event | Rung 0-3 (PGTT) | Rung 4+ (ours) |
  |---|---|---|
  | foot/body friction | 0.4-1.0 | 0.3-1.2 |
  | base mass | ±1 kg | −1 to +2 kg |
  | Kp / Kd | ×0.9-1.1 / ×1.0 | ×0.85-1.15 / ×0.8-1.2 |
  | armature | ×1.0-1.05 | ×0.8-1.2 |
  | motor strength (effort limit) | 100% | 85-100% |
  | joint friction | 0 | 0-0.2 Nm |
  | encoder bias | 0 | ±0.015 rad per joint |
  | reset pose | flat, joints at default | tilted ±0.1 rad, joints ±0.1 rad |
  | reset velocity | ±0.1 m/s | ±0.3 m/s, ±0.3 rad/s yaw |

  Base CoM, link mass and joint damping randomization stay as before.
- **Why:** the real robot's friction, battery, motors and sensors differ from the model.
  A policy that works across these ranges is more likely to work on the real one.
- **Rewards:** unchanged.
- **Risk:** the hardest cases (low friction, weak motors) may cause slipping and falls.
- **Watch:** falls, slip (`eval_gait.py`), speed ratio.

### Rung 5 (`rung5_rewards`): penalties for what full collisions make possible

- **Changes:** three reward terms from the teacher config, at its weights, **with no ramp**
  (full weight from iteration 0):

  | Term | Weight | Value | What it stops | Cost /s |
  |---|---|---|---|---|
  | `base_height` | −50 | (0.265 − h)² when the base is lower than 0.265 m above the ground under it, else 0 (target 0.25 m before 3 Nov) | crouching or crawling | 0 at walking height (~0.28 m); −0.011 at 0.25 m, −0.21 at 0.20 m, −0.66 at 0.15 m |
  | `leg_contact` | −0.25 | number of thigh/calf collision geoms touching the ground (0-12) | kneeling, dragging a shin, resting a knee on a step | −0.25 per touching geom |
  | `joint_limits` | −1.0 | how far (rad) the joints are past their soft limits, summed | pressing into the joint stops | 0 inside the limits |
- **Why:** from rung 1 the robot can kneel and crawl, and on stairs (rung 7) resting a shin
  on a step edge would be an easy shortcut. These terms make that cost something.
- **Left out on purpose** (the rest of the old teacher's reward set, which never walked):
  - doubled tracking weights (2.0 / 1.0);
  - the −100 termination;
  - `joint_power` and `joint_torques`;
  - `action_rate` at −0.01;
  - `orientation`;
  - the penalty ramps (they only delayed v2's failure).
- **Risk:** `base_height` uses the height scan; on rough ground a wrong reading could
  punish normal walking. On flat ground it should be 0 for a good gait.
- **Watch:** `Episode_Reward/base_height` and `Episode_Reward/leg_contact` (should be ~0
  once it walks), speed ratio (should not drop).
- **Result (1 Nov):** HACK, falls about once a second. The cause is in rung 4; see
  [Rungs 4 and 5 collapse](#rungs-4-and-5-collapse-1-nov).
- **Result with the motor fix (3 Nov):** walks, legs off the ground, body at 0.278 m (see
  [Root cause](#root-cause-the-motors-weakened-at-every-reset-2-nov)). This run used the
  0.25 m target.
- **Target raised to 0.265 m (3 Nov), weight kept at −50.** This brings the threshold up to
  just under the walking height, so a slight sag (rung 4's ~0.25 m) starts to cost. The
  weight stays at −50 because `base_height` is inside the zero clip. A random early policy
  sits at ~0.20 m and earns ~1/s. At −50 it pays 0.21/s there, which leaves the clip slack.
  At −300 it would pay 1.3/s, and most early steps would score exactly 0, with no
  gradient (lesson 4 and the 0b reasoning).

  A stronger weight would also fight the crouch that stairs need in rung 7 (the height is
  measured against the ground within 0.2 m of the trunk). `leg_contact`, not
  `base_height`, is what stops the kneeling.

  The ladder that is running keeps 0.25 for rungs 5-7 (one change per rung). 0.265 applies
  from the full teacher on.

### Rung 6 (`rung6_commands`): harder commands

- **Changes:** the teacher's command settings.

  | | Rung 0-5 (PGTT) | Rung 6+ |
  |---|---|---|
  | which axes move | x with probability 0.9, y 0.25, turn 0.5 | all three every time, each uniform in [−1, 1] |
  | standing still | ~4% (all three axes 0) | 5% of robots told to stand |
  | new command every | 2-8 s | 8-12 s |

  In both, a command whose total size is below 0.1 is set to 0 (stand).
- **Why:** the deployed robot gets any joystick combination, including standing still.
- **Rewards:** unchanged.
- **Risk:** combined diagonal-and-turning commands are harder, so tracking may drop.
- **Watch:** speed ratio, `standing_drift`, `eval_gait.py`'s diagonal and stand rows.

### Rung 7 (`rung7_terrain`): stairs and rough ground, the full teacher

- **Changes:**
  - The flat plane becomes a 10 × 20 grid of 8 m × 8 m patches:
    - 20% flat;
    - 50% stairs (up and down, steps 30 or 40 cm deep and 1-13 cm high);
    - 10% random rough ground (up to 5 cm);
    - 15% slopes (up to 0.25);
    - 5% waves.
  - The terrain curriculum (`terrain_levels_vel`) starts robots on the easier rows (level
    ≤ 3).
    - A robot that walked more than 4 m in an episode moves up a row (higher steps).
    - One that walked less than half of its commanded distance moves down.
  - Trains 6000 iterations instead of 1500.
- **Why:** the target task is stairs (PGTT) on a robot that also has to cross everything
  else.
- **Rewards:** unchanged, but `foot_phase` now matters more: its swing target rises with
  the terrain step under the foot (up to +15 cm). `base_height` and `leg_contact` now
  measure against real steps.
- **What rung 7 still doesn't have, compared with the old teacher task:**
  - the left-out reward terms (rung 5);
  - PGTT's joint damping and joint-position noise are kept instead of ours;
  - PGTT's PPO settings are kept.

  All of these are deliberate.
- **Risk:** hitting step edges with the shins, refusing to climb (low speed ratio on
  stairs), or the curriculum stalling at low rows.
- **Watch:** `Curriculum/terrain_levels` (should rise), speed ratio (≥ 0.65 on stairs),
  `leg_ground_frac`, falls. In the viewer: `--terrain stairs`.

### How each rung is judged

| | Automatic, by `train_ladder.sh` | By hand, before trusting the rung |
|---|---|---|
| Check | hacks check on the last checkpoint, written to `hacking_check.txt` | the same check plus `eval_gait.py`, compared with the rung before at the same iteration |
| Fails when | any HACK (the chain stops) | speed ratio < 0.9 (rung 7: < 0.65 on stairs), falls ≥ 0.2/min, saturation ≥ 0.05, tremor ≥ 0.15, swings/s more than 5% off the clock, or a clear regression against the rung before |
| WARN | counts as a pass, the chain continues | read which metric warned and whether it is new in this rung |

### Status (2 Nov)

| Rung | Status |
|---|---|
| 1 collisions | finished, **WARN**: tremor 0.118; leg contact 0.001, base 0.285 m, 0 falls |
| 2 delays | finished, **WARN**: tremor 0.115; leg contact 0.099, base 0.282 m, 0 falls |
| 3 pushes | finished, **WARN**: leg contact **0.248**, tremor 0.119; base 0.277 m, 0 falls |
| 4 randomization | stopped by Ctrl+C at ~1200. The check says **WARN**, but training was collapsing: see [Rungs 4 and 5 collapse](#rungs-4-and-5-collapse-1-nov) |
| 5 rewards | finished, **HACK**: 58 falls per robot-minute, speed ratio 0.03; the chain stopped. Same collapse as rung 4, faster |
| 6-7 | not run |
| 4 again (`rung4_term100`) | finished, **HACK**: kneels in place. `leg_ground_frac` **0.645**, speed ratio **0.005**, base 0.249 m, falls 5.6/min (fell_over 27, body_contact 10) |
| diagnosis (2 of 4 runs) | both collapse (`r5_term100`, `r4_ent001`): see [The diagnosis batch](#the-diagnosis-batch-scriptsdiagnose_collapsesh-2-nov) |
| 4 again, with the motor fix (`2026-11-02_23-13-26_rung4_randomization`) | finished, **WARN**: `leg_ground_frac` **0.297** (HACK from 0.3), plus tremor. No collapse. The kneeling trend of rungs 1-3 continues (0.001 → 0.099 → 0.248 → 0.297): rung 4 has no `leg_contact` term, and its weaker motors and heavier trunk make leaning on a leg more useful. Rung 5's term is the fix: rung 5 trains with legs down 0.0005 |
| **rungs 4-7: to run again** | every rung-4+ result so far was trained with the motor bug: see [Root cause](#root-cause-the-motors-weakened-at-every-reset-2-nov) |

**Seen in the viewer (rungs 1-4):**
- In rungs 1-3 the hind legs kneel slightly at low commanded speeds, sometimes with the body
  tilted backwards.
- Rung 4 at iteration 1200 kneels outright.

**Why more training won't fix it:**
- From rung 1 on, a thigh or calf can rest on the ground. Nothing in rungs 1-4 charges for
  that: only the trunk or a hip ends an episode, and `leg_contact` and `base_height` only
  come in rung 5.
- A slight tilt is almost free too. `foot_phase` measures the feet relative to the body,
  with a loose σ (0.05 m² on the summed error). A 5° pitch moves the rear feet by about
  1.7 cm against the front ones and costs about 0.02 of the 2.5 per second.
- At low speed, tracking is easy whatever the posture, so resting on the knees saves effort
  at no cost.
- Rung 4's weaker motors (85-100% strength, Kp down to ×0.85) and up to +2 kg of mass make
  resting even more attractive.
- As with repro0's tremor, this is a flaw the objective doesn't measure, so more iterations
  of the same rung can't remove it.

**What should fix it:**
- Rung 5 adds `leg_contact` (−0.25 per touching geom, so kneeling on both hind calves costs
  ~0.5/s) and `base_height`, on top of everything from rungs 1-4. It is the test.
- If rung 5 still tilts backwards without touching, the posture needs its own term.
  `orientation` at the teacher's −0.2 is too weak: a 5° tilt costs 0.0015/s. It would need
  about −5, where 5° costs ~0.04/s.
- **Result:** rung 5 did not get to test this. Training collapsed before the posture
  mattered (next section).

**What the checks say** (rungs 1-3; the check runs without pushes or randomization):
- Every WARN includes `action_tremor` at 0.115-0.119. The WARN threshold is 0.1, but the
  0b pass criterion is 0.15 and repro0 had 0.37. So the smoothness term still works, and
  this WARN is not a problem.
- The share of time with a thigh or calf on the ground (`leg_ground_frac`) grows rung by
  rung: 0.001 → 0.099 → 0.248. Body height drops slightly with it: 0.285 → 0.282 → 0.277 m.
  Each new difficulty (delays, then pushes) makes resting on a leg more useful, and nothing
  charges for it. In rung 3 it happens even in the check, which has no pushes, so it is a
  learned habit, not a reaction to a shove.

**Ladder lesson:** a change that opens a new shortcut (full collisions → kneeling) should
come in the same rung as the term that closes it. Rung 1 and rung 5 should have been one rung.

### Rungs 4 and 5 collapse (1 Nov)

**In short:** both rungs learned to walk within 100-200 iterations, then got worse until the
robots fell about once a second. It starts in rung 4 (randomization): more and more robots
hit the ground with the trunk or a hip, a fall costs almost nothing, and once the reward
drops, PPO's entropy bonus makes the actions noisier, which causes more falls. Rung 5's
penalties only make it faster.

#### What the hack check said on rung 5 (`model_1499.pt`)

| Metric | Value | Verdict |
|---|---|---|
| `falls_per_min` | **58** (fell_over 209, body_contact 173 in 32 robots × 10 s) | HACK |
| `speed_ratio` | **0.03** | HACK |
| `standing_drift` | 0.39 m/s | HACK |
| `action_tremor` | 0.52 | HACK |
| `leg_ground_frac` | 0.15 | WARN |
| `base_height` | 0.287 m | OK |

Reward per second:
- Positive: 1.21 (foot_phase 0.75, track_lin_vel 0.30, track_ang_vel 0.15).
- Penalties: 1.71 (ang_vel_xy −0.69, lin_vel_z −0.54, then default_pose, swing_contact and
  action_acc ~−0.13 each).

The robot shakes and tumbles, ignores the command, and still earns foot_phase.

#### First guess, wrong: the zero clip

The guess: rung 5's penalties pushed most steps' sum below 0 from the start, so the clip
paid 0 everywhere and nothing could be learned.

The training curves disprove it. `Train/mean_reward` is the reward summed over an episode;
`Train/mean_episode_length` is in steps (1000 = the 20 s maximum).

| Iteration | Rung 4 reward | Rung 4 episode length | Rung 5 reward | Rung 5 episode length |
|---|---|---|---|---|
| 0 | 0.05 | 11 | 0.05 | 11 |
| 100 | 24.2 | 794 | 23.2 | 766 |
| 200 | **28.9** | 811 | 21.0 | 647 |
| 400 | 22.2 | 663 | 17.6 | 641 |
| 800 | 17.0 | 739 | 0.72 | 141 |
| 1199 | 1.6 | 173 | 0.06 | 53 |
| 1499 | — (stopped) | — | 0.07 | 50 |

Both learned to walk first (about 1.8/s at their best), so the clip did not block learning.
They collapsed later. Rung 4 collapses too, more slowly.

#### What the logs show

How to read the TensorBoard values below:
- `Episode_Reward/<term>` is the term's episode total divided by the 20 s maximum, so it
  also drops when episodes get shorter.
- `Episode_Termination/<cause>` is how many of the 4096 robots ended that way per policy
  step.

| | Rung 3 (healthy) | Rung 4 | Rung 5 |
|---|---|---|---|
| `body_contact` endings (it 100 → 400 → 800 → 1199 → 1499) | 0.65 → 0.3 → 0.5 → 0.1 → **0** | 1.25 → 2.55 → 3 → **18.4** → — | 1.1 → 3.85 → 24.3 → 61.1 → **69.8** |
| `fell_over` endings | ≤ 0.15, then 0 | ≤ 0.15, then 0 | ≤ 0.25 |
| `Policy/mean_noise_std` (it 400 → end) | 0.229 → 0.215, still falling | 0.287 → **0.387**, rising from ~600 | 0.284 → **0.575** (0.49 at 1499) |
| `Loss/entropy` (end) | −6.1 | −0.5 (from −4.2) | +3.5 (peak +5.7) |
| `Loss/value_function` | 0.0009, falling | 0.0013, falling | 2e-6 (rewards ~0, nothing to predict) |
| `Episode_Reward/foot_phase` (it 200 → end) | 0.87 → 0.96 | 0.77 → **0.06** | 0.57 → **0.02** |
| `Episode_Reward/track_lin_vel` (it 200 → end) | 0.82 → 0.93 | 0.66 → 0.08 | 0.51 → 0.02 |

What this rules in and out:

1. **The critic is fine.** Its loss keeps falling in rung 4, so it is not the cause.
2. **Falls start it.**
   - In rung 4, `body_contact` endings rise from iteration ~200.
   - The gait gets worse even per second alive: foot_phase ÷ (episode length / 1000) goes
     0.95 (it 200) → 0.86 (400) → 0.79 (600) → 0.57 (1000) → 0.37 (1199).
   - Meanwhile the noise std is still shrinking (0.294 → 0.287).
3. **The growing noise comes second and makes it worse.**
   - From ~600 the std turns up, entropy rises and the actions get noisier.
   - Noisier actions mean more falls, less reward, and an even larger entropy bonus relative
     to the reward.
   - In rung 3 the std falls all the way to 0.22.

#### Why rung 4 falls more and more

- **A fall costs almost nothing.** `termination` is −1 × dt = **−0.02 per fall** (added after
  the clip, like PGTT's −1 inside it). The rest of the cost is the future reward the robot
  loses.
- **Some robots lose little when they fall.** Rung 4 makes some robots heavy (+2 kg), with
  weak motors (85%, Kp ×0.85), on slippery ground (friction 0.3). They earn little even
  standing up, so the policy gains little from keeping them up. Rung 4 also kneels outright
  in the viewer, and from kneeling a hip is close to the ground.
- **The entropy bonus wins once the reward shrinks.** The bonus (0.01) is the same every
  iteration, while the reward that pushes against it shrinks. So the std grows, which is
  the feedback loop above.
- **Rung 5 is the same, faster.** Its penalties (`leg_contact`, `base_height`) take away
  reward from robots that kneel or sag, which are the same robots that fall.

#### Why the hack check only said WARN for rung 4

`check_reward_hacking.py` runs the policy **without** randomization (nominal robot, no pushes).
A policy that only falls apart on heavy, weak or slippery robots looks fine there. When the
check and the training curves disagree, believe the curves. The check should also run once
with the training randomization.

#### What we try next

1. **Make falling cost something** (testing now).
   - Rerun rung 4 with `termination` −100, the teacher's value, which is −2 per fall instead
     of −0.02.
   - This is safe with the zero clip: every step pays ≥ 0, so dying can never pay more than
     staying alive. That rules out run 2's "fall on purpose" hack.
   - Command line: `--env.rewards.termination.weight=-100 --agent.run-name rung4_term100`.
   - It works if, at 800 iterations, `body_contact` stays near 0 and the noise std keeps
     falling, as in rung 3.
   - **Result (2 Nov): HACK, worse.** Hack check on `model_1499.pt`:
     - The robot kneels on its legs 65% of the time (`leg_ground_frac` 0.645, base 0.249 m).
     - It does not walk (speed ratio 0.005) and still earns foot_phase (0.88/s, 55% of
       the positive reward).
     - Falls drop from rung 5's 58/min to 5.6/min, but they are not zero.
     - Why: falling now costs 2, but kneeling still costs nothing in rung 4 (no
       `leg_contact`), so the safest way to keep the reward is to kneel and stay put. This
       is the ladder lesson again: pricing the fall without pricing the kneel moves the
       robot into the kneel.
2. **If it still collapses:** a smaller entropy bonus, `--agent.algorithm.entropy-coef=0.001`.
3. **When it works:** put the fix into the rungs, rerun rung 5, then continue the chain.

#### The diagnosis batch (`scripts/diagnose_collapse.sh`, 2 Nov)

Four 1500-iteration runs, back to back. Each answers one question:

| # | Run | Setup | Question |
|---|---|---|---|
| 1 | `r5_term100` | rung 5 + `termination` −100 | Does pricing the fall (−2) **and** the kneel (`leg_contact`) together fix it? The candidate fix. |
| 2 | `r4_ent001` | rung 4 + entropy bonus 0.001 (from 0.01) | With cheap falls, is the entropy bonus the root cause? |
| 3 | `r5_term100_ent001` | run 1 + entropy 0.001 | Do we need both? |
| 4 | `r6_term100` | rung 6 + `termination` −100 | Does the fix survive the teacher's commands (the last step before terrain)? |

**A run passes** (fixed before the results):
- Hack check: no HACK, speed ratio ≥ 0.8, falls < 0.2/min, `leg_ground_frac` < 0.1.
- Curves look like rung 3's:
  - episode length stays above 900 after its peak;
  - noise std keeps falling (no turn upwards after iteration ~600);
  - `body_contact` endings fall towards 0.

**What each outcome means:**

| Result | Then |
|---|---|
| 1 passes | The fix is `termination` −100 on rungs 5+. If 4 also passes, train the full teacher (rung 7) for 10000 iterations with it. |
| 1 fails but 3 passes | Both changes are needed. Rung 4 never ran with them, so the full teacher run's first ~800 iterations are its test (watch the curves). |
| 2 passes | The entropy bonus was the root cause: Brax's 0.01 is too large for our setup. |
| 2 fails | Cheap falls plus wide randomization collapse even without the bonus. |
| 1, 3 fail | Rung 4's randomization itself is the problem. Next: split rung 4's events to find which one. |
| 1 passes, 4 fails | The teacher's commands break it. Rung 6 needs its own fix before the full run. |

**How to run it and read the result:** `COMMANDS.md` 4c (start, restart after a reboot) and 9d
(the comparison table, `scripts/summarize_runs.py`). The script writes the table to
`logs/rsl_rl/go2_pgtt_repro/diagnosis.txt` when all four are done.

**Progress:**

| When | What |
|---|---|
| 2 Nov 14:35 | started; `r5_term100` reached iteration 162 |
| 2 Nov ~15:28 | the GPU machine rebooted, which killed the batch. At 162: episode length 867, noise std 0.316, body_contact 0.7, slightly healthier than rungs 4 and 5 at iteration 100. Too early to judge: their decline began after 200-400. |
| 2 Nov 16:36 | restarted from experiment 1 (~12 h for all four) |
| 2 Nov 17:24 | the GPU machine restarted again, which killed it. Restarted at 18:20 |
| 2 Nov 18:20-~20:10 | `r5_term100` finished: 1500 iterations and the check |
| 2 Nov ~20:11-21:50 | `r4_ent001` reached iteration 997, then the machine crashed a third time. It has checkpoints but no check |
| 2 Nov 22:38 | table printed (`diagnosis.txt`). `r5_term100_ent001` and `r6_term100` not run |

**Results (2 Nov, 2 of 4 runs).** Curves at iteration 200 / 800 / last:

| Run | Hack check | Episode length | Noise std | `body_contact` endings |
|---|---|---|---|---|
| `rung4_term100` (1 Nov) | HACK: speed ratio 0.005, falls 5.6/min, legs down 0.645, base 0.249 m | 909 / 887 / 206 | 0.304 / 0.361 / 0.507 | 0.8 / 0.9 / 19.7 |
| `r5_term100` | HACK: speed ratio 0.038, falls 6.2/min, legs down 0.290, base 0.252 m | 847 / 662 / **119** | 0.297 / 0.400 / **0.639** | 0.9 / 3.1 / **32** |
| `r4_ent001` (to 997) | at `model_900` (the newer checkpoints were cut off by the crash, empty files): HACK, legs down **0.444**, speed ratio 0.569, falls **0.19/min**, base 0.289 m | 779 / 621 / 414 | 0.233 / **0.141 / 0.143** | 1.3 / 3.65 / 8.7 |

What this shows:
1. **Run 1 fails.** A fall that costs 2 instead of 0.02, plus the kneel penalty, still
   collapses, and in the same way and at the same time as plain rung 5. **Cheap falls (H1) are
   not the main cause.**
2. **Run 2 fails too, even before its end.** With a tenth of the bonus, the noise std goes
   *down* (0.23 → 0.14) and stays there, but the falls still climb (1.3 → 8.7) and the
   episodes shorten (779 → 414). The robot gets worse with almost no noise, so the
   **mean action itself degrades. The entropy bonus (H2) is not the cause**; the rising std
   in the other runs is a symptom.
3. In the nominal check (no randomization), the robots fall ~6/min. In training (with
   rung 4's randomization), the episodes end after ~2-4 s. **Most falls happen on the
   randomized robots.** This fits H5 or a specific rung 4 item. It is also why the nominal
   check rated rung 4 only a WARN.
4. **The common sign is the legs on the ground.** `r4_ent001` at 900 still walks (speed ratio
   0.57) and hardly falls in the nominal check (0.19/min), but drags or kneels on a leg 44%
   of the time. The trend began before rung 4: rungs 1-3 had legs down 0.001 → 0.099 →
   0.248. The term100 runs have stopped walking (speed ratio ≤ 0.04) and sit low (base
   ~0.25 m). A reading to test, not a finding: the policy leans on its legs more and more, and
   on the randomized robots (weaker motors, heavier trunks) the low posture tips into a hip or
   trunk contact, which ends the episode. Leg contacts also grip well (μ 1.0, see 7f below),
   so a knee is a useful prop. H4 is still open.

Per the table "What each outcome means" (row "1, 3 fail"), **rung 4's randomization itself is
the problem.** Runs 3 and 4 are dropped: each combines changes that have both failed (3), or
builds on a run that failed (4). Next: find *which* rung 4 item causes it. Cheapest first:
tests T1/T2 on the existing checkpoints (minutes), then the 4L/4S bisection (~6 h) only if
they don't point at one item. See "What we are missing" below.

**Update (2 Nov, 23:00):** the rung 4 item was found by reading the code: `effort_limits`
compounded at every reset. See [Root cause](#root-cause-the-motors-weakened-at-every-reset-2-nov).

### Root cause: the motors weakened at every reset (2 Nov)

**Found while writing test T2** (falls per robot parameter), by reading how each randomization
event stores its value.

- **The bug.** Rung 4 adds `effort_limits`: at every reset, each motor's maximum torque is
  multiplied by a random 0.85-1.0. The code (`randomize_effort_limits` in
  `mjlab/envs/mdp/events.py`, from the original repo) multiplied the *current* limit, not
  the original one. So the factors piled up:

  | Resets of a robot | Mean max torque (nominal 23.5 Nm) |
  |---|---|
  | 1 | 21.9 |
  | 5 | 16.4 |
  | 10 | 11.4 |
  | 20 | 5.3 |
  | 30 | 2.6 (11%) |

  Measured by resetting a rung 4 environment 30 times. `randomize_pd_gains`, right above it,
  already scales the stored defaults and has a comment saying why.
- **Why it fits every observation.** A robot resets at least every 20 s (a time-out), which is
  every 50 iterations, and more often when it falls.

  | Observation | Explained by |
  |---|---|
  | rung 4 walks at iteration 200, then gets worse | at 200 a robot has had ≥4 resets (~74% torque); by 1500 ≥30 (~11%) |
  | the collapse speeds up | weak robots fall, falling resets them, each reset weakens them further |
  | rungs 0-3 are fine | they don't have `effort_limits` (rung 0 removes it) |
  | a 100× fall price (`r5_term100`) and a 10× smaller entropy bonus (`r4_ent001`) don't help | neither changes the motors |
  | the noise std rises late | once most robots can't stand, nothing the policy does pays, and the entropy bonus wins |
  | more and more kneeling (`leg_ground_frac` 0.44-0.65), body ~0.25 m | weak motors can't hold the stance; a knee on the ground (μ 1.0) takes the load |
  | the nominal check looks much calmer than training | it removes `effort_limits`, so it runs with full-strength motors |

  This is a strong candidate for the whole rung 4/5 collapse. It is not yet proven: that
  needs rung 4 trained again with the fix (below).
- **The fix.** Scale the stored default limit, as `randomize_pd_gains` does (both branches:
  MuJoCo position actuators and `IdealPdActuator`). After 30 resets the torque now stays
  between 20.0 and 23.5 Nm. The regression test
  `tests/test_pgtt_audit.py::test_effort_limits_do_not_compound_over_resets` fails on the old
  code (2.7 Nm after 20 resets) and passes on the new one.
- **What it invalidates.**
  - Every run with `effort_limits`: rungs 4 and 5, `rung4_term100`, the diagnosis runs, and the
    old teacher runs v1-v4 (the teacher task has it too). Rungs 0-3 and repro0/smooth0b are
    unaffected.
  - In hindsight, v2's "walk at 500, then fall on purpose" has the same shape. The motor bug
    may have contributed to v1-v4 as well. That's a candidate, not checked.
  - The hypotheses and lessons written before this (H1 cheap falls, lesson 14) were reasoning
    about a robot whose motors were silently fading. Keep them as candidates for the fixed
    setup, not as findings.
- **What it does not explain:** the slight kneeling in rungs 1-3 (`leg_ground_frac` 0.001 →
  0.099 → 0.248), which has no `effort_limits`. That is still the reward's blind spot from
  [Status](#status-2-nov). Rung 5's `leg_contact` term is still its test.
- **Next:**
  1. Rung 4 again with the fix, 1500 iterations (`bash scripts/train_ladder.sh 4`).
     Pass: the curves stay like rung 3's past iteration 800, and
     `check_reward_hacking.py --randomized True` shows no collapse.
  2. If it passes, the ladder continues to rungs 5-7.
  3. Then seed 2 and the full teacher.
- **First result (3 Nov, ladder restarted at rung 4 with the fix):** rung 4 finished without a
  HACK, and the ladder went on to rung 5. Rung 5 at iteration ~1260, past the point where every
  earlier rung 4/5 run had collapsed: episode length 996, speed ratio 0.91, noise std 0.21,
  entropy −6.2 (rung 3: 0.22 and −6), `fell_over` 0, `body_contact` 0.1, legs on the
  ground 0.0005 (the kneeling is gone), body 0.278 m. Before the fix, rung 5 at the end had
  episode length 173, speed ratio 0.03 and 58 falls/min. This is what the motor bug predicted.
  Still to come: rung 5's hack check, and rung 4's `--randomized True` check.

#### Tests T1 and T2: `check_reward_hacking.py --randomized True`

The check can now run under the training conditions: domain randomization, pushes, delays and
sensor noise, with only the curricula off. Then it prints a table, one row per third of each
randomized robot parameter (low / middle / high value):

```
per robot parameter, in thirds (low / middle / high value):
  parameter           third              range falls/min legs down  speed
  motor_strength      low       0.902-0.924         ...
```

| Column | Meaning |
|---|---|
| `falls/min` | falls per robot-minute |
| `legs down` | share of time with a thigh or calf on the ground |
| `speed` | speed ratio |

The parameters:

| Parameter | What it is |
|---|---|
| `motor_strength` | torque limit ÷ nominal |
| `kp_scale`, `kd_scale` | Kp and Kd ÷ nominal |
| `trunk_mass_add_kg` | mass added to the trunk |
| `leg_mass_scale` | leg mass ÷ nominal |
| `trunk_com_x_m` | trunk centre of mass shift forward |
| `foot_friction` | foot friction coefficient |
| `joint_friction_nm` | joint friction |
| `armature_scale` | rotor inertia ÷ nominal |
| `encoder_bias_rad` | joint encoder error |

A parameter whose thirds differ a lot is where the falls come from. It also prints how many
falls happened within 0.5 s of a reset (T3: the reset state, not the gait). Use ≥1024 robots,
so each third has enough falls to compare.

It runs the *current* code. Old checkpoints are then checked with the fixed motors, not the
fading ones they were trained with.

### What we are missing: hypotheses, predictions and next rungs (2 Nov, before any batch result)

Written on 2 Nov 2026 while `r5_term100` was still training, before any of the four batch
results was seen, so the predictions below can't be fitted to them afterwards. Everything here
is a **candidate**: nothing is established until a test below supports it. No literature
search was done for this section. The mentions of Miki et al. 2022 are from memory and are
not verified here.

#### The observations, as measured

| # | Run | Measured | Limits of the measurement |
|---|---|---|---|
| O1 | rungs 1-3 | walk; legs on the ground 0.001 → 0.099 → 0.248 of the time; base 0.285 → 0.277 m; 0 falls | nominal check only (no randomization); one seed each |
| O2 | rung 4 | best at iteration 200 (811-step episodes); then trunk/hip endings per step 1.25 → 18.4; `foot_phase` per second alive 0.95 → 0.37; noise std turns up only after ~600 (0.287 → 0.387); value loss keeps falling; kneels in the viewer at 1200 | stopped at ~1200; nominal check WARN; one seed |
| O3 | rung 5 | same shape, faster; std 0.58; entropy +5.7; value loss → 2e-6; in the **nominal** check: 58 falls/min, speed ratio 0.03, deterministic tremor 0.52 | one seed |
| O4 | `rung4_term100` | nominal check: legs on the ground 0.645, speed ratio 0.005, base 0.249 m, falls 5.6/min, `foot_phase` 55% of the positive reward | training curves not read yet; one seed |
| O5 | all | the hack check never tests the training randomization; we never measured **which** randomized robots fall | — |

O3 matters: rung 5 falls even on the **nominal** robot, so by the end the policy itself is
broken, not only its behaviour on the hard robots.

#### What is lacking (independent of which hypothesis wins)

1. **Rung 4 is not one change.** It changes about 10 things at once:
   - friction range;
   - trunk mass;
   - Kp/Kd;
   - rotor inertia;
   - motor strength;
   - joint friction;
   - encoder bias;
   - reset tilt, reset joint offsets and reset velocity.

   That is the mistake the ladder was built to avoid, so a rung 4 failure can't be pinned on
   a cause.
2. **We can't see which robots fail.** Nothing records a robot's sampled mass, strength or
   friction next to its falls. Nothing checks a policy under the training randomization.
3. **The reward ordering was never checked for the PGTT reward with full collisions.**
   Lesson 1 says to score walking, standing, stepping in place and kneeling against each
   other. That was done for v1's reward set, never for the set the ladder trains with. O4
   suggests kneel-stepping earns a large share of what walking earns.
4. **A fall costs 0.02.** Compare that with the ~1-1.5 the robot expects to earn over its
   0.7 s horizon.
5. **One seed per configuration.** A single collapse or a single success can be luck.

#### Candidate explanations (rivals from different classes)

| ID | Candidate | Class |
|---|---|---|
| H1 | **Hard-robot abandonment.** Rung 4 creates heavy, weak, slippery robots that earn little upright. A fall costs 0.02, so the shared policy loses little by letting them sag and fall. The degradation then spreads to all robots. | mechanism (reward × randomization) |
| H2 | **The entropy bonus drives it.** Once the reward signal weakens, the constant 0.01 bonus wins and the std grows until the actions fall apart. | mechanism (optimizer) |
| H3 | **Reset states.** Rung 4's tilted, moving resets with offset joints cause falls in the first second, independent of the learned behaviour. | artifact of a specific change |
| H4 | **Wrong reward ordering.** Kneel-stepping or stepping in place pays nearly as much as walking, so under harder dynamics it becomes the better bet. Kneeling then brings the hips near the ground. | mechanism (reward design) |
| H5 | **Infeasible robots.** Some combinations (+2 kg, 85% strength, Kp ×0.85) can't hold the standing pose against Kp 40 without sagging onto the hips. | physical feasibility |
| H6 | **Seed or chance.** The collapse is one bad run per configuration. | stochastic variation |
| H7 | **Measurement artifact.** Hip-ground contacts in a low but healthy stance count as falls. | measurement |

What the data already says (from O2 and O3):
- **H2 as the sole cause** is weakened: in rung 4 the falls and the per-second `foot_phase` drop
  came before the std turned up. It remains a candidate amplifier.
- **H7** is weakened: rung 5's falls are mostly `fell_over` (tilt) in the nominal check.
- **H3** is weakened: a reset effect should be present from iteration 0, not grow over
  training.
- None of them is ruled out.

#### Predictions for the running batch (fixed before the results)

| Result pattern | Supports | Weakens |
|---|---|---|
| `r5_term100` passes (falls < 0.2/min, legs on the ground < 0.1, speed ratio ≥ 0.8, curves like rung 3) and `r4_ent001` collapses like rung 4 | H1 + H4 (cheap falls plus a free kneel) | H2 as primary |
| `r4_ent001` passes, `r5_term100` collapses with the std rising | H2 | H1 |
| Only `r5_term100_ent001` passes | H1/H4 and H2 together | either alone |
| `r5_term100` doesn't fall but **stands or steps in place** (speed ratio < 0.5, legs mostly up) | H4 in its "step in place upright" form: pricing falls and kneels moves it to the next free posture | H1 as sufficient |
| All four collapse | H5, or a specific rung 4 item not covered by these runs; also H6 | H1/H2 as the only causes |
| `r6_term100` fails while `r5_term100` passes | the teacher's commands (all three axes at once, 5% standing) are a separate problem | — |

An outcome that falls between two rows (e.g. a pass on the check but curves that start to
drift) is **indeterminate**. It calls for a second seed, not a conclusion.

#### Tests that need no training run (proposed, not yet built)

These need small additions to `check_reward_hacking.py` and would only be built after
approval.

| Test | What it measures | Separates |
|---|---|---|
| T1 `--randomized` check | the same metrics with the training randomization kept | lacking item 2; checks O2 directly |
| T2 per-robot fall attribution | each robot's sampled mass, strength and friction next to its falls, as falls per bin | H1 (falls pile up in the hard bins) vs H2/H6 (spread evenly) |
| T3 falls by episode age | share of falls in the first 0.5 s after a reset | H3 (mostly early) vs the rest |
| T4 cross-evaluation | reward per second of known behaviours in the rung 4/5 env with randomization: `rung3_pushes` (walks), `rung4_term100` (kneel-steps), a zero-action policy (stands) | H4: if kneel-stepping earns ≥ 90% of walking, the ordering is broken |
| T5 standing-sag test | zero-action base height for the extreme robots (+2 kg, 85% strength, Kp ×0.85, friction 0.3) | H5: sagging to hip contact means rung 4 contains infeasible robots |

#### Rungs to add, depending on the result

The ladder rule is one change per rung, judged against fixed criteria. Which rungs to run
follows from the prediction table:

| Rung | Change against its parent | Run it if |
|---|---|---|
| **4L** "load" (parent: rung 3) | rung 4's mass, motor strength, Kp/Kd and friction ranges only | all four batch runs fail: one half of a bisection of rung 4 |
| **4S** "state and sensing" (parent: rung 3) | rung 4's reset tilt/velocity/joint offsets, joint friction, rotor inertia and encoder bias only | the other half (4L and 4S run back to back, ~6 h). Then split whichever half fails. |
| **5c** "coupled gait reward" (parent: winning rung 5 setup) | `foot_phase × (0.5 + 0.5 × track_lin_vel)` (needs a new reward function) | the "stands or steps in place" outcome (H4) |
| **5k** "sustained kneel ends the episode" (parent: winning rung 5 setup) | a termination when any thigh/calf stays on the ground > 0.2 s | 5c still kneels |
| **4C** "randomization curriculum" (parent: rung 3) | rung 4's ranges, widened from rung 3's values over the first ~1000 iterations | H1 is supported but the fall cost alone isn't enough. A fix, not a diagnosis: it changes two things over time, so judge it only after the widening ends. |
| **seed 2** of the winner | `--agent.seed 2`, nothing else | always, before the 10000-iteration teacher (H6) |

A **7f "leg friction"** rung was planned here and dropped on 2 Nov after a check: the legs are
*not* frictionless. Their geoms have `condim=1`, but a contact between two geoms of the same
priority uses the larger `condim` and the larger friction of the two. Against the terrain
(`condim=3`, μ 1.0), a thigh or calf contact gets a full friction cone with μ **1.0**, more grip
than the feet (μ 0.6, priority 1, so the foot's own value is used). Checked by dropping a folded
Go2 in the rung 4 scene: every ground contact had `dim 3, mu 1.0`.

Bisection (4L/4S) costs about 6 h. Splitting rung 4 into its ~10 items would cost about 30 h.

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
13. **A run that learns and then gets worse: find what moved first.** Rungs 4 and 5 walked by
    iteration 200 and collapsed later. A rising noise std was visible, but the falls started
    before it; the std only amplified them. Print the curves side by side with a healthy run.
14. **If falling is nearly free, robots that earn little will be dropped.** −0.02 per fall is
    noise. With wide randomization some robots earn little upright, so losing them costs
    the policy little, and the falls spread.
15. **A check without randomization can't see a collapse under randomization.** Rung 4's
    check said WARN while its training episodes had shrunk from 811 to 173 steps.
16. **When training gets worse over time, look for something in the environment that changes
    over time.** The rung 4 collapse came from a randomization that piled up at every reset:
    the motors faded to ~11% over training. Two diagnosis runs (12 h with the crashes) tested
    the policy side (the fall price, the entropy bonus); reading the event code found it in
    minutes. Check that randomized parameters stay in range after many resets, with a test.
    (Lesson 14 was written before this was known.)

## What to watch in a rung run (repro0, 0b, ...)

| When | Graph or check | Good | Bad |
|---|---|---|---|
| iteration ~300 | `Train/mean_episode_length` | climbing towards 1000 | stuck low |
| throughout | `Policy/mean_noise_std` | falls below 0.7 | rises (the entropy problem is back) |
| throughout | `Episode_Termination/body_contact` | falls towards 0 (rung 3) | rises after the first ~200 iterations (rungs 4-5) |
| throughout | `Train/mean_episode_length` after its peak | stays near 1000 | falls again (rung 4: 811 → 173) |
| throughout | `Metrics/hack/action_saturation` | below 0.2 | above 0.5 (bang-bang) |
| throughout | `Metrics/hack/action_jitter` | 0.05-0.3 | above 0.8 |
| from ~1000 | `Metrics/hack/speed_ratio` | rising towards 1 | near 0 |
| throughout | `Metrics/hack/action_tremor` | falling below 0.15 (0b) | stays ~0.35 like repro0 |
| 1500 | hacks check + `eval_gait.py` on the checkpoint | that rung's pass criteria | anything HACK, or a regression against its parent |

`body_ground_frac` and `leg_ground_frac` stay 0 in the reproduction: with feet-only collisions
nothing but the feet can touch the ground.
