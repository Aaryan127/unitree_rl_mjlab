# Research: Go2 step heights, obstacle terrains, teacher privileged state

Researched 2026-11-04 from primary sources only: unitree.com, the arXiv papers, and the
official GitHub repos pinned to commits. Source tags such as [S1] are listed at the end.
"Not verified" means I could not find the claim in a primary source.

## Summary and recommendations

- **Unitree's spec for the Go2's stock controller.** "Max Climb Drop Height" is about 15 cm
  on the Air and about 16 cm on the Pro/X/EDU. "Max Climb Angle" is 30° on the Air and 40°
  on the Pro/X/EDU [S1]. Unitree does not give a separate stair-height figure.
- **Go2 size.** The Go2 is about the size of an A1. Both are about 40 cm tall standing
  [S1][S4 fn.1]. The Go2's thigh and calf are 0.213 m each. At the home pose the hip is
  about 0.27–0.29 m above the ground and the belly is about 0.21–0.23 m up [S11][S12].
- **What a trotting Go2 policy can realistically climb:**
  - **10–15 cm** is comfortable. It matches Unitree's spec, PGTT's real Go2 setup (trained
    ≤ 13 cm, most stable 7–9 cm in sim) and Isaac Lab's Go2 preset (≤ 12 cm) [S1][S9][S15].
  - **18–20 cm** is a stretch. It needs perception and a swing apex that grows with the
    step (≥ step height plus a few cm). It is about 0.7× hip height. Scaling Miki's ANYmal
    result (30.5 cm) by body height gives about 17 cm on a Go2 [S7][S4 fn.1].
  - **25–30 cm** is roughly hip height (0.9–1.1×). The only same-size results are:
    - an A1 that climbed 24–25 cm stairs and 26 cm curbs. It used a depth camera, no gait
      prior, and an emergent wide hip-abduction climbing motion [S4];
    - separate "climb"/jump skills for 40–50 cm, which are not a trot [S5][S6].
- **What the repo uses now.** The current rung-9 cap is `PGTT_HARD_MAX_STEP = 0.18`
  (`mjlab/terrains/config.py:157`), at the top of this range. Going to 25–30 cm would need
  all of the following:
  1. Exteroception that sees the riser before contact. A blind ANYmal fails from 20 cm on
     [S7].
  2. Removing or loosening the fixed trot/phase prior and the swing-height cap. Agarwal
     credits not having gait priors for reaching 25 cm on an A1 [S4 §1].
  3. A curriculum that goes past 30 cm. Miki trains grid steps up to 0.4 m to get 30.5 cm
     on hardware [S7 §S4].
  4. Box (vertical-riser) stairs rather than heightfield stairs [S7].
  5. Tolerating knee and shank contact and large body pitch. A tread-scale staircase is
     about atan(h/0.30) ≈ 45° at 30 cm, which is above Unitree's 30–40° climb-angle spec.
- **Boulder terrains.**
  - legged_gym and Isaac Gym build "discrete obstacles" as heightfield rectangles (20 per
    tile, 1–2 m wide). The height goes up to 0.05 + 0.2·d m, so 0.23 m on the hardest
    row [S13].
  - Isaac Lab's default rough mix uses a mesh "random grid" of 0.45 m boxes with heights
    in ±(0.05→0.20) m, so neighbouring cells can differ by up to 0.4 m. Its Go2 preset
    shrinks this to ±(0.025→0.10) m [S14][S15].
  - Those defaults are ANYmal-scale. For a Go2, 0.10–0.15 m is the analogous top (the
    repo's boulders use 3–15 cm).
- **Privileged state.**
  - The ETH teachers (Lee 2020, Miki 2022) see contact states, forces and normals, foot
    friction, thigh and shank contact, the external force or wrench, and noiseless
    terrain heights (plus swing duration or airtime in Miki) [S7][S8].
  - RMA's 17-D vector is payload mass and position, motor strength ×12, friction and local
    terrain height. Kp/Kd are randomized but are not in it [S2].
  - None of these teachers is given latency/delay or PD gains as an input.
  - Several papers warn that a teacher given information the student cannot infer caps or
    breaks distillation [S4 §2][S10][S16][S17]. PGTT itself drops the teacher-student
    split. It gives only the critic one extra input, the base linear velocity [S9 §III-D].
  - The repo's 90-value `dynamics` group (mass, CoM, motor strength, gains, joint
    friction, delay, encoder bias) is broader than any of the cited teachers. Audit the
    student's ability to reconstruct each block before keeping it.

## Q1. Stair and step height for a Go2

### 1a. Unitree official spec [S1]

| Model | Max Climb Drop Height | Max Climb Angle | Standing dims (L×W×H) | Weight |
|---|---|---|---|---|
| Go2 AIR | about 15 cm | 30° | 70 × 31 × 40 cm | about 15 kg |
| Go2 PRO | about 16 cm | 40° | 70 × 31 × 40 cm | about 15 kg |
| Go2 X | about 16 cm | 40° | 70 × 31 × 40 cm | about 15 kg |
| Go2 EDU | about 16 cm | 40° | 70 × 31 × 40 cm | about 15 kg |

The page has no field named "stair height" or "step height". "Max Climb Drop Height" is the
closest figure. It is for Unitree's built-in controller, not an RL policy.

### 1b. Published RL step-height results

| Work | Robot (size) | Trained on (sim) | Achieved (real unless noted) | Special gait or skill? |
|---|---|---|---|---|
| Agarwal 2022 [S4] | A1, 40 cm tall, 28 cm hip joint (fn.1, §1) | legged_gym terrains; stair heights not given in the paper (App., Table 3) | Stairs up to 24 cm high with 28 cm treads, both directions (§4). Curbs up to 26 cm. The intro says "stairs of height upto 25 cm, 89 % relative to its height". Benchmark: 17 cm × 30 cm stairs, 13 steps at 100 % (Fig. 4) | No gait prior. An emergent large hip abduction "because the small size of the robot doesn't leave any space between the body and stair" (§4, Fig. 2c). The blind baseline fails upstairs (0 %) and dislocated a leg going down |
| Extreme Parkour, Cheng 2023 [S5] | A1, 26 cm thigh joint, 40 cm body (§4.1) | Tilted ramps, gaps, hurdles and "high step" terrain; numeric ranges not in the paper | 0.5 m gym box, "twice the height of its hip joint" (§4.2.1) | Yes. A high jump: the hind legs kick and the front legs reach the top |
| Robot Parkour Learning, Zhuang 2023 [S6] | A1 and Go1 | Climb skill on obstacles 0.20–0.45 m (Table 1); test 0.25–0.5 m | Climbs 0.40 m ("1.53× robot height") at 80 %. Outdoors, "several stairs of 0.42 m high each" (§4) | Yes. A dedicated `π_climb` skill, distilled into one vision policy |
| Miki 2022 [S7] | ANYmal-C, 70 cm tall [S4 fn.1] | Stairs h ∈ [0.01, 0.22] m, tread [0.25, 1.0] m. Grid steps h ∈ [0.05, 0.4] m. "Large steps" up to 0.4 m. Boxes ≤ 0.25 m (§S4) | Single steps: reliable up to 30.5 cm. Hesitates above 32 cm. The blind baseline drops off at 20 cm (Fig. 4) | CPG phase + residual action (§S5). The robot lifts its legs higher and leans forward |
| Lee 2020 [S8] | ANYmal-B/C, blind | Steps 0.05–0.30 m, stairs 0.02–0.20 m (Table S2) | 16.8 cm step (Fig. 3B), 18 cm SubT stairs | Foot trajectory generator (PMTG) with a foot-trapping reflex |
| PGTT, Ntagkas 2025 [S9] | Go2 (real), ANYmal-C (sim) | Stair-like tiles, 4 levels: [1,3], [1,7], [1,10], [1,13] cm. Tread 0.30–0.45 m (§IV-C) | Sim: "can traverse obstacles up to 12 cm, the most stable behaviors were observed at heights between 7 cm and 9 cm" (§IV-E). Real Go2 stairs are shown, but no height is reported. LiDAR limited it to 0.4 m/s | Phase prior through the reward only (Hermite swing spline). Apex = d_s + δH (terrain statistic) |
| Isaac Lab Go2 rough preset [S15] | Go2 | Stairs 0.025–0.12 m, boxes ±0.025–0.10 m | (sim preset, no hardware claim) | none |

Scaling note: by overall height [S1][S4 fn.1], 1 m of terrain for a 70 cm ANYmal is about
0.57 m for a 40 cm Go2. That maps Miki's 22 cm training stairs to about 12.6 cm and his
30.5 cm hardware step to about 17.4 cm. Treat this as a rough heuristic, not a law.

### 1c. Go2 leg geometry (MJCF [S11], URDF [S12])

| Quantity | Value | Source line |
|---|---|---|
| Hip (abduction) joint offsets from base | x ±0.1934, y ±0.0465 m. Front–rear hip spacing 0.387 m | go2.xml L77/107/137/167. URDF `FL_hip_joint` origin |
| Thigh-joint lateral offset | 0.0955 m | go2.xml L84. URDF `FL_thigh_joint` |
| Thigh length (thigh→calf joint) | 0.213 m | go2.xml L92 `FL_calf pos="0 0 -0.213"`. URDF `FL_calf_joint` |
| Calf length (calf joint→foot centre) | 0.213 m. Foot sphere r = 0.022 m | go2.xml L32–33 (`foot` class). URDF `FL_foot_joint`, `<sphere radius="0.022">` |
| Joint ranges | Abduction ±1.047. Front thigh [−1.571, 3.491]. Rear thigh [−0.524, 4.538]. Knee [−2.723, −0.838] rad | go2.xml L11–23. URDF `<limit>` |
| Torque and speed limits | Hip/thigh 23.7 N·m, 30.1 rad/s. Knee 45.43 N·m, 15.70 rad/s | go2.xml L10, L24. URDF `<limit>` |
| Home keyframe | Base z = 0.27 m, thigh 0.9, knee −1.8 rad | go2.xml L216 |
| Base collision box | Half-sizes 0.188 × 0.047 × 0.057 m | go2.xml L73 |
| Total mass (sum of bodies) | ≈ 15.2 kg | go2.xml `mass=` attributes |

The derived numbers below are my own forward kinematics on the values above:
- At the home pose (0.9, −1.8), the foot centre is 0.265 m below the hip. The sole is
  0.287 m below it, so the hip stands at about 0.27–0.29 m and the belly (box bottom) at
  about 0.21–0.23 m.
- The hip-to-foot distance spans 0.089 m (knee at −2.72) to 0.389 m (knee at −0.84).
- The joint limits let a front foot reach above hip level when it is 0.1–0.2 m ahead of the
  hip, so leg reach does not limit a 30 cm step. The limits are body clearance, stance-leg
  torque and pitch.

| Step h | h / hip height (0.28 m) | vs belly (≈0.22 m) | Stair pitch atan(h/0.30 m tread) | Single step under both hips, equal legs: asin(h/0.387) |
|---|---|---|---|---|
| 10 cm | 0.36 | well below | 18° | 15° |
| 20 cm | 0.71 | just below | 34° | 31° |
| 30 cm | 1.07 | above, so the body must rise or pitch | 45° | 51° |

A 30 cm stair is steeper than the Go2's 40° climb-angle spec [S1]. A 30 cm curb puts the
belly at riser height unless the robot pitches up or lengthens its rear legs.

**Conclusion.** For a trotting Go2 policy, 10–15 cm is the realistic working range and
18–20 cm is reachable with perception and a terrain-adaptive swing apex. 25–30 cm requires:
1. perception with look-ahead, so the robot sees the riser before contact;
2. no hard trot/phase prior, or a relaxed one, with swing apex ≥ h + 5 cm;
3. a training curriculum that goes past 30 cm (box risers, not heightfield);
4. large pitch and hip-abduction freedom, and tolerance of knee contact;
5. near-peak knee torque.

The A1 evidence says this is reachable on hardware only with emergent, non-trot climbing
[S4]. Above that, skills become jumping or climbing [S5][S6].

## Q2. Discrete obstacles and "boulders" in terrain generators

| Generator | Geometry type | Parameters | Difficulty scaling | Top height (default, ANYmal-scale) |
|---|---|---|---|---|
| legged_gym `discrete_obstacles_terrain` call [S13 terrain.py L117, L133–137] | Heightfield (int16, h-scale 0.1 m, v-scale 0.005 m). Converted to trimesh with `slope_treshold = 0.75`, so steep faces become vertical walls [S13 config L44–66] | `num_rectangles = 20`, size 1–2 m, central platform 3 m. 20 % of columns (`terrain_proportions[4]`) | `height = 0.05 + 0.2·d`, `d = row / num_rows` (0…0.9) | 0.23 m (row 9). For comparison, stairs are `0.05 + 0.18·d` → 0.212 m at 0.31 m treads (L116, L132) |
| Isaac Gym `terrain_utils.discrete_obstacles_terrain` body | Heightfield | Not verified. Isaac Gym's `terrain_utils.py` ships only in the Isaac Gym Preview download and is not in an official GitHub repo | – | – |
| Isaac Lab `HfDiscreteObstaclesTerrainCfg` [S14 hf_terrains.py L217–293, cfg L148–169] | Heightfield → mesh (`@height_field_to_mesh`) | `obstacle_width_range`, `obstacle_height_range`, `num_obstacles`, `platform_width` (default 1 m). `obstacle_height_mode="choice"` draws each from {−h, −h/2, +h/2, +h} (pits and pillars). `"fixed"` uses +h | `h = lo + d·(hi − lo)` | No default. It is not used in `ROUGH_TERRAINS_CFG` |
| Isaac Lab `MeshRandomGridTerrainCfg` "boxes" [S14 mesh_terrains.py L243–371, rough.py L37–42] | Triangle mesh of boxes (true vertical sides) | `grid_width = 0.45` m, `grid_height_range = (0.05, 0.2)`, platform 2 m. 20 % proportion | `g = lo + d·(hi − lo)`. Each cell is offset by U(−g, +g) (L340) | ±0.20 m, so adjacent cells can differ by up to 0.40 m |
| Isaac Lab Go2 rough preset [S15 L53–59] | Same | `boxes.grid_height_range = (0.025, 0.1)`, stairs (0.025, 0.12), rough noise (0.01, 0.06) | Same | ±0.10 m |
| Isaac Lab `MeshRepeatedBoxesTerrainCfg` / `MeshBoxTerrainCfg` [S14 mesh_terrains_cfg.py L122–136, L202–311; mesh_terrains.py L800–801] | Mesh boxes | `num_objects`, `height`, `size`, `max_yx_angle`, lerped between `object_params_start` and `object_params_end`. Box terrain is one central box of `box_height_range` | Linear in d | Config-dependent |
| Isaac Lab difficulty with curriculum [S14 terrain_generator.py L48–59] | – | `d = (row + U(0,1)) / num_rows · (upper − lower) + lower` | – | – |
| Miki 2022 "boxes" [S7 §S4] | RaiSim boxes, random yaw | "multiple boxes with maximum height 0.25 m" | Particle-filter curriculum | 0.25 m |

Notes:
- **Heightfield vs mesh.** Miki trained stairs as boxes because "stair risers modeled by a
  height map are not perfectly vertical; we observed that the policy exploited these
  non-vertical edges in simulation, resulting in poor sim-to-real transfer" [S7, Training
  environment]. legged_gym and Isaac Lab work around this by meshing the heightfield with a
  slope threshold that makes steep faces vertical.
- **Scale.** Defaults top out at 0.20–0.25 m for ANYmal. The Isaac Lab Go2 preset halves
  them (10–12 cm). The repo's `BoxBouldersTerrainCfg(height_range=(0.03, 0.15),
  size_range=(0.3, 0.8), num_boulders=30)` is in line with that Go2 scaling.

## Q3. Privileged state given to teacher policies

| Item (teacher-only unless noted) | Lee 2020 [S8 Table S4] | Miki 2022 [S7 Table S3] | RMA 2021 [S2 §IV] | Agarwal 2022 [S4 §2.1] | Robot Parkour 2023 [S6 §3.1] | PGTT 2025 [S9 §III-D] |
|---|---|---|---|---|---|---|
| Foot contact states | ✓ (4) | ✓ (4) | Student too (binary contacts in the state) | – | – | – |
| Contact forces | ✓ (4) | ✓ (12) | – | – | – | – |
| Contact or terrain normals | ✓ terrain normal per foot (12) | ✓ contact normals (12) | – | – | – | – |
| Friction | ✓ (4) | ✓ (4) | ✓ (1) | ✓ | ✓ | – |
| Thigh/shank contact | ✓ (4+4) | ✓ (8) | – | – | – | – |
| External force on base | ✓ force (3) | ✓ force + torque (6) | – | – | – | – |
| Swing duration / airtime | – | ✓ (4) | – | – | – | – |
| Terrain heights | ✓ 9 points per foot on a 10 cm circle (36). The student is blind | Teacher: noiseless samples. Student: same samples through a noise model (208 samples in all: 52 per foot on 5 rings of radius 0.08–0.48 m) | ✓ local terrain height (1) | Teacher: scandots. Student: depth camera | Obstacle distance, height, width, 4-way type | Actor and critic both use the heightmap |
| Mass / CoM | – | – (randomized, not observed) | ✓ payload mass and position (3) | ✓ CoM | ✓ (mass, CoM randomized) | – |
| Motor strength | – | – | ✓ (12) | ✓ | ✓ | – |
| PD gains | – | – | Randomized Kp [50,60], Kd [0.4,0.8] but not in e_t | Randomized Kp [35,45], Kd [0.4,0.6] (App. Table 3) | – | Randomized |
| Delays | – | – | – | Height-map latency randomized, not observed | – | – |
| Base linear velocity | Both (state estimator on hardware) | Both ("body velocity", 6) | Neither (state has no velocity) | Neither | – | Critic only (`s_t = [o_t, v_t]`) |
| Total privileged dims | 71 | 50 + clean heights | 17 | – | – | 3 |

Papers on giving the teacher too much:
- **Agarwal [S4 §2].** "the scandots might contain more information than what depth can
  infer. To get around this, we choose scandots and camera field-of-view such that phase 2
  loss is low." They back this with a bound (Thm 2.1).
- **Miki [S7].** The student gets noisy heights, with three map-failure modes in a 60/30/10
  mix and a curriculum on the noise. A gated belief encoder reconstructs both the privileged
  state and the heights.
- **Lee [S8].** "A key hypothesis here is that the latent features l̄_t can be (partially)
  recovered" from history.
- **RMA [S2].** The extrinsics need not be correct "in some 'ground truth' sense", only to
  lead to the right action.
- **Extreme Parkour [S5 §3.3].** Feeding the predicted heading directly causes
  "catastrophic distribution drift". They use a teacher/student mixture, plus DAgger [S5].
- **DreamWaQ [S10].** "behavior cloning (BC) bounds the student policy's performance with
  the teacher policy". The paper uses an asymmetric actor-critic instead: the critic gets
  velocity, disturbance force and a height map.
- **HIM [S16].** The two-phase paradigm "inevitably induces information loss in the
  mimicking learning phase".
- **PGTT [S9 §III-D].** "the resulting policy can inherit suboptimality and distribution
  mismatch from the teacher's occupancy measure". PGTT uses an asymmetric actor-critic
  with only v_t privileged.
- **SLR, deployed on a Go2 [S17 §1].** Hand-picked privileged parameters "may not enhance
  the interpretability or effectiveness" of the policy. Its ablation used a 10-D
  privileged vector: friction, restitution, foot height and foot contact.

## Sources

- [S1] Unitree Go2 product page, spec table — https://www.unitree.com/go2
- [S2] Kumar et al., "RMA: Rapid Motor Adaptation for Legged Robots", RSS 2021 — https://arxiv.org/abs/2107.04034 (Table I; §IV "Environmental Variations"; state definition)
- [S4] Agarwal et al., "Legged Locomotion in Challenging Terrains using Egocentric Vision", CoRL 2022 — https://arxiv.org/abs/2211.07638 (§1 and footnote 1; §2 and §2.1; §4 "Urban Environments"; Fig. 2 and 4; Appendix Table 3)
- [S5] Cheng et al., "Extreme Parkour with Legged Robots", 2023 — https://arxiv.org/abs/2309.14341 (Table 1; §3.2–3.3; §4.1–4.2.1)
- [S6] Zhuang et al., "Robot Parkour Learning", CoRL 2023 — https://arxiv.org/abs/2309.05665 (§3.1; Table 1; §4 real-world; Supp. B)
- [S7] Miki et al., "Learning robust perceptive locomotion for quadrupedal robots in the wild", Science Robotics 7(62), 2022 — https://arxiv.org/abs/2201.08117 ("Training environment"; "Student policy training"; Fig. 4; Supp. §S4, §S5 and Table S3)
- [S8] Lee et al., "Learning quadrupedal locomotion over challenging terrain", Science Robotics 5(47), 2020 — https://arxiv.org/abs/2010.11251 ("Teacher policy"; "Student policy"; Tables S2 and S4; Fig. 3)
- [S9] Ntagkas, Kiourt, Chatzilygeroudis, "PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion", 2025 — https://arxiv.org/abs/2510.18348 (§III-D, §III-G/H, §IV-B/C/E, §V). Code: https://github.com/NtagkasAlex/phase_guided_terrain_traversal (not inspected for this note)
- [S10] Nahrendra et al., "DreamWaQ", ICRA 2023 — https://arxiv.org/abs/2301.10602 (§II-A/B)
- [S11] MuJoCo Menagerie `unitree_go2/go2.xml` @98a70c5 — https://github.com/google-deepmind/mujoco_menagerie/blob/98a70c569c4c1b6da467a20012f766ee1f8dc521/unitree_go2/go2.xml
- [S12] Unitree `go2_description.urdf` @25746e8 — https://github.com/unitreerobotics/unitree_ros/blob/25746e8c73f097c3b32bed7b6c16d4e195dfacd8/robots/go2_description/urdf/go2_description.urdf
- [S13] legged_gym @8fa29ac — https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/utils/terrain.py#L75-L145 and https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/base/legged_robot_config.py#L43-L66
- [S14] Isaac Lab @515fdd7 (main, fetched 2026-11-04; line numbers may drift):
  - https://github.com/isaac-sim/IsaacLab/blob/515fdd711e303914d3aa53a5c9ab8e7968e7de90/source/isaaclab/isaaclab/terrains/height_field/hf_terrains.py#L216-L293
  - https://github.com/isaac-sim/IsaacLab/blob/515fdd711e303914d3aa53a5c9ab8e7968e7de90/source/isaaclab/isaaclab/terrains/height_field/hf_terrains_cfg.py#L148-L169
  - https://github.com/isaac-sim/IsaacLab/blob/515fdd711e303914d3aa53a5c9ab8e7968e7de90/source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py#L243-L371
  - https://github.com/isaac-sim/IsaacLab/blob/515fdd711e303914d3aa53a5c9ab8e7968e7de90/source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains_cfg.py
  - https://github.com/isaac-sim/IsaacLab/blob/515fdd711e303914d3aa53a5c9ab8e7968e7de90/source/isaaclab/isaaclab/terrains/config/rough.py#L11-L53
  - https://github.com/isaac-sim/IsaacLab/blob/515fdd711e303914d3aa53a5c9ab8e7968e7de90/source/isaaclab/isaaclab/terrains/terrain_generator.py#L36-L60
- [S15] Isaac Lab Go2 rough env @515fdd7 — https://github.com/isaac-sim/IsaacLab/blob/515fdd711e303914d3aa53a5c9ab8e7968e7de90/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/rough_env_cfg.py#L53-L59
- [S16] Long et al., "Hybrid Internal Model", ICLR 2024 — https://arxiv.org/abs/2312.11460 (§1)
- [S17] Chen et al., "SLR: Learning Quadruped Locomotion without Privileged Information", CoRL 2024 — https://arxiv.org/abs/2406.04835 (§1, §4.1; Go2 deployment in §4)
- Also cited for the "information gap" point: Chen et al., "Light-Loco-Parkour", 2026 (humanoid) — https://arxiv.org/abs/2608.02653. It says "This information gap between an over-informed teacher and a partially observed student is well documented" and designs the teacher's actor inputs "backward from the student". It cites Walsman et al. ICLR 2022, Warrington et al. ICML 2021 and Messikommer et al. arXiv:2412.09149.

Not verified:
- the body of Isaac Gym's `terrain_utils.discrete_obstacles_terrain` (not on official GitHub);
- Agarwal's and Extreme Parkour's simulated stair-height ranges (not stated in the papers);
- PGTT's real-world stair height (not reported);
- ANYmal hip height (not taken from a primary source here; scaling uses overall height).
