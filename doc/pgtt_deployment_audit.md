# Go2 deployment-stack audit — 27 October 2026

**Result: the available software checks pass after the fixes below. Physical deployment
readiness is still unproven.** A trained and evaluated policy, robot calibration, live
LiDAR/localization, and hardware acceptance tests remain necessary. Passing an export
comparison or compiling the controller does not establish locomotion quality.

This audit follows the Go2 PGTT path through task construction, simulation, teacher PPO,
student distillation, checkpoint/resume, play, ONNX export, LiDAR map delivery, and the C++
motor controller. Repository-wide syntax, package contents, task registration, and Python
dependency consistency were checked as supporting infrastructure. Other robots were not
physically tested or run through a training acceptance matrix.

## Findings and repairs

| Severity | Finding | Repair / evidence |
|---|---|---|
| Critical | C++ `bad_orientation()` always returned false. | Restored finite-data and tilt checks. The offline C++ contract test checks upright, inverted, and invalid gravity. |
| Critical | Go2 policy and 1 kHz command threads accessed joint targets concurrently. | Added a mutex-protected target handoff and atomic worker state; command axes are copied under the SDK low-state mutex. |
| High | Inference exceptions or a stalled worker could leave the controller using stale commands. | Exceptions and a 100 ms update watchdog request Passive. Retiring a worker does not block the FSM transition; re-entry is refused until that worker finishes. Live fault transitions still need hardware testing. |
| High | Initial processed actions were zeros, which could briefly become zero-angle targets when enabling the policy. | Reset now computes the affine zero-action target. Entry initializes targets before enabling policy gains. |
| High | Stand-up interpolation began from the last commanded pose, which could differ from the actual pose after Passive. | It now starts from measured joint angles. |
| High | Neural-network position targets had no joint-limit clamp. | New PGTT runs clamp targets to model limits; the exact same bounds are exported and applied by C++. Older checkpoints preserve their saved behavior. |
| High | Export/play rebuilt current defaults, losing training overrides and potentially changing observation/action meaning. | New checkpoints contain complete environment and runner snapshots. Export/play restore them. Legacy checkpoints require explicit opt-in. |
| High | Runner construction mutated caller configuration dictionaries while removing class names. | Runners now preserve an independent configuration snapshot before constructing algorithms. |
| High | Viewer playback did not clear the student's GRU memory at automatic or manual resets. | Added a reset-aware policy wrapper and viewer hooks; covered by playback regression tests. |
| High | ONNX verification did not reset hidden state after environment resets. | Both PyTorch and ONNX states now reset together; verification deliberately uses short episodes to exercise repeated resets. |
| High | The default PGTT distribution contained flat ground and stairs, but no rough ground. | Added a default mixed distribution and explicit flat/stairs/rough/mixed profiles shared by training, play, and evaluation. |
| High | A fresh cloud could be combined with an old base pose in the LiDAR bridge. | TF/odometry freshness is checked independently. Tests confirm stale cloud, TF, and odometry inputs do not refresh the controller map. |
| Medium | A single oversized cloud could exceed the point-buffer limit. | Each chunk is bounded as well as the rolling buffer. |
| Medium | Reading the C++ gait observation advanced the phase, including during initialization/reset. | PGTT phase now derives from the policy-step counter. Repeated reads do not change it. |
| Medium | Dummy play required a checkpoint; video wrapping was unconditional. | Zero/random play works without a checkpoint or renderer. Trained playback wraps video only when requested. |
| Medium | Resume could silently change policy inputs/actions and repeat the last iteration. | Added compatibility checks for architecture, routing, observations, actions and robot settings. Resume begins with the next iteration. |
| Medium | Model/config dimensions and invalid numbers could reach C++ indexing/inference unchecked. | Added deployment-vector, observation/action, recurrent-state, and finite-value checks, plus startup inference validation. |
| Medium | The Go2 CMake file relied on implicit C++ standard and hardcoded dependency paths/libraries. | Requires C++17 and uses installed CMake dependency targets; a complete native build succeeded. |
| Medium | A non-editable Python installation omitted subpackages/assets; MuJoCo Warp was unpinned. | Namespace-package/data discovery now includes the stack and robot assets; MuJoCo Warp is pinned to the tested commit. |
| Low | Selecting the default GPU failed on hosts with no GPU. | Empty GPU availability selects CPU; tested explicitly. |
| Low | LiDAR comparison checked only its final scenario and could fail on an empty accuracy sample. | Aggregates all scenarios, handles empty samples, and now includes terrain edges. |

## Surface coverage

The default `Mjlab-Velocity-PGTT-{Teacher,Student}-Unitree-Go2` tasks now train on one
mixed recipe:

| Component | Share | Configured range |
|---|---:|---|
| Flat | 20% | Plane-like patches |
| Ascending/descending stairs | 50% | 1–13 cm rise; 30 or 40 cm treads |
| Random roughness | 10% | Height samples from 0–5 cm |
| Ascending/descending slopes | 15% | Gradient up to 0.25 |
| Waves | 5% | Configured amplitude up to 5 cm |

These are training ranges, not certified terrain limits. Random-roughness height range
is fixed by its recipe; its generator does not scale that range with the difficulty
argument. The other procedural terrain generators use their own difficulty scaling.

Use the same mixed distribution for both teacher and student. Existing stairs-only or
short smoke checkpoints do not acquire rough-terrain capability from this code change.
The old stairs-only recipe remains available via `--terrain stairs`; flat and rough
profiles can also be selected explicitly.

## Executed checks

| Check | Result and practical limit |
|---|---|
| Python regression suite | **22 tests passed**, including geometry, stale ROS inputs with a simulated ROS interface, checkpoint configuration isolation, student gradient updates, and headless play/reset paths. |
| Teacher PPO smoke | Two updates on the mixed scene, finite losses, saved snapshots and ONNX export. Establishes execution, not learning quality. |
| Student distillation smoke | Two updates with full 24-step recurrent sequences, saved snapshots and ONNX export. Establishes execution, not learning quality. |
| Resume smoke | Loaded the teacher snapshot with compatible settings and continued at the next iteration. |
| Surface simulation | Flat, stairs, and rough profiles each ran 20 robots × 50 steps with finite observations/rewards/actions. This is only one second of motion per robot. |
| Simulator ↔ LiDAR bridge | 32 robots across the mixed scene, standing, moving, and relocated to terrain edges: maximum map disagreement below display precision (`0.00000 m`). |
| LiDAR coverage in that comparison | Mean approximately 82–85%; minimum approximately 77–78%, above the 50% acceptance threshold. Actual robot coverage is unknown. |
| PyTorch ↔ ONNX | Maximum action error **1.49e-7**, 80 sequential steps with repeated recurrent resets. |
| Python ORT ↔ real C++ runner | Maximum action error **4.17e-7**, including an explicit recurrent reset. Invalid input dimensions, NaN inputs, and invalid height-map fallbacks were also checked. |
| Full Go2 controller | Compiled and linked on **x86_64 Linux**; `go2_ctrl --help` ran. No DDS connection to a physical robot was made. |
| Offline C++ contract test | Passed observation order, relative joints, command axes, gait timing, reset targets, clipping, and fall detection using a mock articulation and the real managers/ONNX runner. |
| Packaging | Built to an isolated directory and verified inclusion of Python subpackages, scene XML, and robot meshes. |
| Supporting checks | Python compilation, dependency consistency, SVG regeneration, and whitespace checks passed. |

The short evaluation policy is deliberately a smoke model. At a requested 0.3 m/s it
made approximately **-0.10 m mean body-forward progress in one second**, rather than the
requested positive motion. Its short survival result must not be interpreted as a
successful flat-ground or stair policy. The bundled `deploy/.../pgtt/v0` model was
previously exported from `model_2.pt` of a three-iteration run and is also a smoke artifact.

The full build used the official Unitree SDK revision
[`d1abf001220d3e5f59e00a0c4aaa595e00089a4a`](https://github.com/unitreerobotics/unitree_sdk2/tree/d1abf001220d3e5f59e00a0c4aaa595e00089a4a),
whose [Go2 joint ordering](https://github.com/unitreerobotics/unitree_sdk2/blob/d1abf001220d3e5f59e00a0c4aaa595e00089a4a/include/unitree/dds_wrapper/robots/go2/defines.h)
matches the exporter. Dependencies and SDK were assembled under `/tmp`; no system
packages were installed. Tested simulation versions were MuJoCo 3.4.0, Warp 1.11.1,
and MuJoCo Warp commit `9491175b7cbea87e28d3e3e67733095317c33398`.

## Remaining deployment blockers

1. **A competent trained policy.** Train both stages on the mixed recipe and evaluate
   command tracking, stability, and terrain traversal over full episodes and multiple
   seeds. The current smoke artifacts are unsuitable for judging locomotion quality.
2. **GPU validation.** This host has CPU-only PyTorch/no CUDA driver. GPU kernels,
   throughput, and long-run learning were not validated here.
3. **Physical LiDAR and localization.** Confirm the actual cloud topic, mount transform,
   gravity frame, odometry timing, scan deskewing, height sign, map alignment, latency and
   coverage on the Go2. The simulated head-LiDAR mount/pattern is an approximation.
4. **Hardware controller acceptance.** Verify installed SDK/firmware compatibility,
   motor/joint mapping, gains, limits, state transitions, stop behavior, fall detection
   and inference timing on the intended computer/robot. The 100 ms inference watchdog
   and the 1 radian controller tilt limit are runtime guards, not proof of recovery.
5. **Target platform.** The complete native build passed on x86_64. The aarch64/Jetson
   build and real-time performance remain untested.
6. **Terrain-specific behavior.** Measure actual ascent, descent and rough-ground
   traversal, including stair approaches/edges and reduced perception. Surviving a
   short rollout on a starting platform is insufficient. Measure the intended physical
   stair rise/tread and ground roughness against the training ranges; extend the recipe
   and retrain if they fall outside that range. This controller follows an operator's
   velocity command; the operator selects the route.

Zero-map fallback is trained behavior, not proof that the ground is flat or that blind
stair traversal is safe. The UDP float packet also has no source timestamp or sequence
number; the controller's freshness timer measures receipt age. Use a local/controlled
link and measure end-to-end latency before relying on terrain perception.

## Reproducible path to a deployment candidate

Train on a GPU-capable host; the task defaults now use mixed terrain:

```bash
python scripts/train.py Mjlab-Velocity-PGTT-Teacher-Unitree-Go2 --terrain mixed --env.scene.num-envs 4096
python scripts/train.py Mjlab-Velocity-PGTT-Student-Unitree-Go2 --terrain mixed --env.scene.num-envs 4096 --teacher-checkpoint <trained-teacher.pt>
```

Evaluate the resulting student separately on each surface. Repeat difficulty values
0, 0.5 and 1, and inspect the motion as well as the JSON metrics. The evaluator reports
measurements; it does not declare an untrained policy deployable.

```bash
python scripts/evaluate_pgtt.py --checkpoint-file <trained-student.pt> --terrains flat stairs rough --num-envs 64 --steps 1000 --seeds 42 43 44 --difficulty 0.5 --device cuda:0 --output logs/evaluation_nominal.json
python scripts/evaluate_pgtt.py --checkpoint-file <trained-student.pt> --terrains flat stairs rough --randomized True --device cuda:0 --output logs/evaluation_randomized.json
python scripts/evaluate_pgtt.py --checkpoint-file <trained-student.pt> --terrains flat stairs rough --blind True --device cuda:0 --output logs/evaluation_blind.json
python scripts/check_lidar_map.py
```

Export to a **new version directory while the controller is stopped**, then run the
actual C++ inference comparison. Export verification checks numerical agreement and
records provenance; it does not certify the policy's behavior.

```bash
python scripts/export_deploy.py --checkpoint-file <trained-student.pt> --output-dir deploy/robots/go2/config/policy/pgtt/v1
python scripts/check_cpp_policy.py deploy/robots/go2/config/policy/pgtt/v1/exported/policy.onnx
cmake -S deploy/robots/go2 -B deploy/robots/go2/build -DPGTT_BUILD_TESTS=ON -DPGTT_TEST_POLICY_DIR=<absolute-path-to-v1>
cmake --build deploy/robots/go2/build -j2
ctest --test-dir deploy/robots/go2/build --output-on-failure
```

New checkpoints preserve configuration snapshots. Legacy export/play/resume requires
`--allow-legacy-config True` and manual comparison with the original training settings.
For deployment candidates, prefer a new trained checkpoint with a snapshot and keep
its export manifest and evaluation reports with the model.
