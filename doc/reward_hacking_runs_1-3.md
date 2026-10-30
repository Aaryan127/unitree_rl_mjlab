# Reward hacking in the PGTT teacher: runs 1-4 and the fixes

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

| Run | Folder (`logs/rsl_rl/go2_pgtt_teacher/`) | What it learned | Fix (commit) |
|---|---|---|---|
| v1 | `2026-10-27_20-18-35_flat` → `2026-10-28_01-14-38_flat` (to 11000) | lie on the belly and paddle | rebalanced rewards, body-contact termination, hacking checker (`5981fe3`) |
| v2 | `2026-10-29_04-12-16_flat_v2` (to 4300, machine reboot) | walk at 500, then fall on purpose | reward clip at 0, noise std cap (`6a11dc1`) |
| v3 | `*_flat_v3` (10000) | vibrate the joints between their limits | tanh-bounded actions, PGTT gains and foot-phase formula (`3164449`) |
| v4 | `2026-10-30_16-23-42_flat_v4` (6300 checked) | bang-bang vibration, a fall every 1.2 s | entropy on the squashed action, PGTT reproduction as baseline (not committed yet) |

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

### 1. The entropy bonus rewards saturated actions (proven; the cause of v2-v4 is likely, not yet confirmed)

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
`tests/test_pgtt_audit.py`). Still to confirm on run 4's logs: that `Policy/mean_noise_std`
stayed at 1.0 and that most raw outputs |μ| > 2 (see "Data still needed" below).

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
  1. full-body collisions + trunk/hip termination,
  2. motor and sensor delays,
  3. pushes and external force,
  4. our wider domain randomization,
  5. our extra reward terms,
  6. full command range,
  7. stairs and mixed terrain.

  Each run uses the same pass criteria, so a failure points at exactly one change.
- Then the student is distilled from the last teacher that passed.

**Data still needed from run 4** (to confirm section 1 on the run itself): the TensorBoard
events file and a few checkpoints. Check `Policy/mean_noise_std` (expect it pinned at 1.0) and
the share of raw outputs |μ| > 2 per checkpoint.

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

## What to watch in the reproduction run

| When | Graph or check | Good | Bad |
|---|---|---|---|
| iteration ~300 | `Train/mean_episode_length` | climbing towards 1000 | stuck low |
| throughout | `Policy/mean_noise_std` | falls below 0.7 | rises (the entropy problem is back) |
| throughout | `Metrics/hack/action_saturation` | below 0.2 | above 0.5 (bang-bang) |
| throughout | `Metrics/hack/action_jitter` | 0.05-0.3 | above 0.8 |
| from ~1000 | `Metrics/hack/speed_ratio` | rising towards 1 | near 0 |
| 2500 | hacks check on the checkpoint | the pass criteria above | anything HACK |

`body_ground_frac` and `leg_ground_frac` stay 0 in the reproduction: with feet-only collisions
nothing but the feet can touch the ground.
