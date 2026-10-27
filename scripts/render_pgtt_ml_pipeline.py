"""Render the PGTT Go2 machine-learning pipeline as a standalone SVG poster.

Run from the repository root: python scripts/render_pgtt_ml_pipeline.py
The diagram documents the settings in the current Go2 PGTT task and runner configs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape


WIDTH, HEIGHT = 2100, 3890
OUT = Path(__file__).resolve().parents[1] / "doc" / "pgtt_ml_pipeline.svg"
PNG = OUT.with_suffix(".png")
parts: list[str] = []


def add(markup: str) -> None:
  parts.append(markup)


def txt(x: int, y: int, value: str, *, size: int = 25, color: str = "#e8efff",
        weight: int = 400, anchor: str = "start", mono: bool = False) -> None:
  family = "monospace" if mono else "Inter, DejaVu Sans, Arial, sans-serif"
  add(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
      f'font-weight="{weight}" font-family="{family}" text-anchor="{anchor}">'
      f'{escape(value)}</text>')


def panel(y: int, h: int, number: str, title: str, subtitle: str) -> None:
  add(f'<rect x="45" y="{y}" width="2010" height="{h}" rx="30" '
      'fill="#111d31" stroke="#32445f" stroke-width="2"/>')
  add(f'<rect x="75" y="{y+25}" width="55" height="55" rx="15" fill="#30476c"/>')
  txt(102, y+62, number, size=26, weight=800, anchor="middle")
  txt(150, y+60, title, size=35, weight=750)
  txt(151, y+94, subtitle, size=21, color="#a8bbd5")


def box(x: int, y: int, w: int, h: int, title: str, lines: list[str],
        color: str = "#1b2a42", edge: str = "#4a6381", *, size: int = 22,
        line_gap: int = 29) -> None:
  add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" '
      f'fill="{color}" stroke="{edge}" stroke-width="2"/>')
  txt(x+24, y+41, title, size=26, weight=730)
  for index, line in enumerate(lines):
    txt(x+24, y+76+line_gap*index, line, size=size, color="#cbd8eb")


def arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#74d6cb",
          width: int = 4) -> None:
  add(f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" '
      f'stroke-width="{width}" marker-end="url(#arrow)"/>')


def pill(x: int, y: int, w: int, label: str, color: str) -> None:
  add(f'<rect x="{x}" y="{y}" width="{w}" height="38" rx="19" fill="{color}"/>')
  txt(x+w//2, y+27, label, size=19, weight=650, anchor="middle")


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
    f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
    'aria-label="PGTT Go2 complete machine learning pipeline">')
add('''<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9"
          markerHeight="9" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#74d6cb"/>
  </marker>
  <linearGradient id="bg" x2="0" y2="1">
    <stop offset="0" stop-color="#0a1221"/>
    <stop offset="1" stop-color="#0e1b2c"/>
  </linearGradient>
</defs>''')
add(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="url(#bg)"/>')
txt(75, 95, "PGTT  /  GO2", size=58, weight=850)
txt(75, 141, "Complete task, teacher PPO, student GRU, distillation, and deployment", size=28,
    color="#a9bed8")
pill(1640, 65, 355, "ACTUAL CODE CONFIG", "#155b62")
txt(76, 185, "Read top to bottom • blue = simulator/teacher • teal = student/deployment • orange = loss/decision",
    size=21, color="#9aabc2")

# 1 — Task and data.
panel(215, 710, "1", "TASK AND SIMULATION LOOP",
      "What the robot is trying to do, what it observes, and how an action changes the world")
box(80, 335, 600, 300, "Go2 on mixed terrain", [
  "10 × 20 terrain patches, each 8 × 8 m",
  "Flat + stairs 1–13 cm; 0.30/0.40 m treads",
  "Roughness/waves ≤5 cm; slope ≤0.25",
  "20 s episode; fall if tilt > 90°",
  "Physics 0.005 s × 4 = policy 50 Hz",
  "Difficulty rises when distance improves",
], "#1a3148", "#467fa2")
box(750, 335, 600, 300, "Command and gait clock", [
  "Target (vx, vy, yaw rate): each in [-1, 1]",
  "Velocity command resampled every 8–12 s",
  "Trot phase offsets: (0, π, π, 0)",
  "Frequency f sampled 1–3 Hz per episode",
  "Stance: 0..π; swing: π..2π",
  "Policy sees [cos φ ×4, sin φ ×4, f]",
], "#21314a", "#697eb0")
box(1420, 335, 600, 300, "Action → physics", [
  "Network outputs 12 joint values a",
  "Target q = default pose + 0.25 × a",
  "Joint limits clamp targets; PD → torque",
  "MuJoCo/Warp steps 4 physics frames",
  "Then rewards, done flags, new sensors",
  "Observation/action loop repeats",
], "#193a42", "#3a938e")
arrow(680, 485, 745, 485)
arrow(1350, 485, 1415, 485)
box(80, 675, 940, 210, "Observation routing", [
  "policy 54 = IMU 3 + gravity 3 + command 3 + joints 24 + last action 12 + gait 9",
  "history 84 = joint position ×3 + velocity ×2 + action ×2",
  "height 99 = clean vertical rays; student map 99 = simulated head LiDAR",
  "privileged 53 = body velocity, contacts, friction, force; critic base 57",
], "#202c48", "#667bb0", size=19, line_gap=27)
box(1050, 675, 970, 210, "Reward and episode decisions", [
  "+1.0 velocity XY; +0.5 yaw tracking; +1.0 foot-phase tracking",
  "-0.25 swing contact; penalties for tilt, vertical motion, power, limits...",
  "r = step_dt × weighted sum; timeout truncates, fall terminates",
  "Curriculum changes terrain level and ramps regularization penalties",
], "#3a2d30", "#ae795f", size=19, line_gap=27)

# 2 — Teacher architecture.
panel(955, 600, "2", "STAGE 1 — TEACHER ACTOR–CRITIC",
      "Clean terrain and simulator-only state let PPO discover a good gait")
box(80, 1080, 480, 110, "Body + history 138", ["policy 54 + history 84 → normalize"],
    "#1b314b", "#567db2", size=19)
box(80, 1210, 480, 110, "Clean heights 99", ["99 → 128 → 64 → 32 encoder"],
    "#1b314b", "#567db2", size=19)
box(80, 1340, 480, 110, "Privileged state 53", ["normalize → 53 → 64 → 32 → 16"],
    "#1b314b", "#567db2", size=19)
box(660, 1110, 680, 315, "Teacher actor / action mean", [
  "Concatenate 138 + 32 + 16 = 186",
  "MLP: 186 → 512 → 256 → 128 → 12",
  "ELU hidden layers; linear 12-output head",
  "During rollout: sample Gaussian action",
  "Initial action std = 1.0 (learned)",
  "At evaluation: use mean action",
  "Actor ≈ 291k learned parameters",
], "#173650", "#4f94c0", size=20, line_gap=29)
arrow(560, 1133, 652, 1200)
arrow(560, 1265, 652, 1265)
arrow(560, 1395, 652, 1330)
box(1420, 1110, 600, 315, "Critic / value estimate", [
  "Clean critic 57 + height 99 + priv 53",
  "Concatenate = 209 inputs",
  "MLP: 209 → 512 → 256 → 128 → 1",
  "Output V(s): expected future return",
  "Used to compute PPO advantages",
  "Critic is discarded for deployment",
  "Critic ≈ 272k learned parameters",
], "#27314f", "#7791c9", size=20, line_gap=29)
arrow(1340, 1267, 1412, 1267)
txt(80, 1500, "Teacher checkpoint stores the actor, critic, learned action noise, and observation-normalizer state.",
    size=21, color="#adbfdb")

# 3 — PPO decisions.
panel(1585, 555, "3", "TEACHER LEARNING — PPO",
      "The teacher is the only PGTT policy updated directly from environment reward")
box(80, 1715, 450, 285, "A • collect rollout", [
  "24 steps per parallel robot",
  "Store obs, actions, reward, done",
  "Also V(s), log π(a|s), mean, std",
  "Sampled teacher actions move Go2",
  "One rollout spans 0.48 s of sim time",
], "#19364e", "#4d8cae", size=20)
box(580, 1715, 450, 285, "B • estimate advantage", [
  "GAE: γ = 0.99, λ = 0.95",
  "δ = r + γV(next) - V(now)",
  "A = δ + γλ × later advantage",
  "Normalize advantages across rollout",
  "Timeout bootstraps value estimate",
], "#19364e", "#4d8cae", size=20)
box(1080, 1715, 450, 285, "C • clipped update", [
  "ratio ρ = πnew(a|s) / πold(a|s)",
  "Actor minimizes max(-Aρ, -A clip ρ)",
  "Critic: clipped value MSE vs return",
  "Loss = actor + value - 0.01 entropy",
  "Clip range ρ ∈ [0.8, 1.2]",
], "#3c3035", "#ad765e", size=20)
box(1580, 1715, 440, 285, "D • optimizer", [
  "5 epochs × 4 minibatches",
  "Adam, initial LR = 0.001",
  "Adaptive LR, desired KL = 0.01",
  "Gradient norm limited to 1.0",
  "Save teacher checkpoint",
], "#3c3035", "#ad765e", size=20)
arrow(530, 1858, 572, 1858)
arrow(1030, 1858, 1072, 1858)
arrow(1530, 1858, 1572, 1858)
txt(80, 2083, "Reward shaping: gait phase targets stance contact and a terrain-adaptive swing apex (8 cm + nearby height change, capped at 15 cm).",
    size=19, color="#aebbd0")

# 4 — Student architecture.
panel(2170, 795, "4", "STAGE 2 — DEPLOYABLE STUDENT + GRU BELIEF",
      "The student sees its own sensors and must infer what the privileged teacher knew")
box(80, 2295, 455, 130, "Robot values 54", [
  "Normalize IMU, joints, command,",
  "last action, gait phase",
], "#164148", "#4caea4", size=19)
box(80, 2450, 455, 175, "Noisy LiDAR map 99", [
  "16 rings, -80°..-10°, 4° azimuth",
  "Returns → 11 × 9 grid (≥50% coverage)",
  "Noise: drift, holes, outliers, shifts, blind",
  "Encoder 99 → 128 → 64 → 32",
], "#164148", "#4caea4", size=19)
box(640, 2300, 650, 325, "GRU + gated belief", [
  "Input: normalized body 54 + map latent 32 = 86",
  "Single-layer GRU; hidden memory h = 256",
  "Current h and previous h carry recent context",
  "Height belief 32 = state(h) + sigmoid(gate(h)) × e",
  "Privileged belief 16 = privileged_state(h)",
  "Combined inferred belief = 48 values",
  "Reset h for ended episodes; detach each 24 steps",
], "#1b3d45", "#5ac4b4", size=20, line_gap=32)
arrow(535, 2360, 632, 2415)
arrow(535, 2540, 632, 2505)
box(1390, 2300, 630, 325, "Student policy", [
  "Input: normalized body 54 + belief 48 = 102",
  "MLP: 102 → 512 → 256 → 128 → 12",
  "Outputs 12 joint-action means",
  "Rollout: Gaussian sample, fixed std 0.1",
  "Robot/inference: deterministic mean",
  "GRU state flows to the next policy step",
  "Deployable network ≈ 573k parameters",
], "#1b3d45", "#5ac4b4", size=20, line_gap=32)
arrow(1290, 2460, 1382, 2460)
box(80, 2670, 930, 245, "Training-only belief decoder", [
  "GRU output 256 → reconstructed clean height 99",
  "GRU output 256 → reconstructed privileged state 53",
  "The clean height target is the teacher's vertical-ray map",
  "The privileged target uses the teacher's normalizer",
  "Decoder is removed from the deployed ONNX graph",
], "#314047", "#8eaa9b", size=20, line_gap=29)
box(1090, 2670, 930, 245, "Why the GRU matters", [
  "Each step is a 20 ms snapshot; terrain can be occluded or noisy.",
  "Memory estimates hidden terrain and pushes from recent motion.",
  "The gate can reduce the influence of an unreliable current map.",
  "Input h shape: [1, robots, 256]; output becomes next h.",
  "Student training parameters incl. decoder ≈ 651k.",
], "#314047", "#8eaa9b", size=20, line_gap=29)

# 5 — Distillation.
panel(2995, 430, "5", "STUDENT LEARNING — DISTILLATION",
      "Actions are learned from the frozen teacher; reconstruction trains the belief state")
box(80, 3120, 450, 245, "A • initialize", [
  "Load latest or specified teacher",
  "Freeze teacher weights",
  "Copy teacher height encoder",
  "into student's height encoder",
], "#263648", "#6c8caa", size=20)
box(580, 3120, 450, 245, "B • student rollout", [
  "Student actions drive simulation",
  "At each visited state, teacher",
  "labels the action it would take",
  "Store 24 ordered steps per robot",
], "#1b3d45", "#5ac4b4", size=20)
box(1080, 3120, 450, 245, "C • supervised loss", [
  "MSE(student a, teacher a)",
  "+ 0.5 × [MSE(clean height)",
  "       + MSE(privileged state)]",
  "Rewards are logged, not optimized",
], "#3c3035", "#ad765e", size=20)
box(1580, 3120, 440, 245, "D • train memory", [
  "Backprop through 24 GRU steps",
  "1 epoch; Adam LR 0.0005",
  "Clip gradient norm to 1.0",
  "Save student checkpoint",
], "#3c3035", "#ad765e", size=20)
arrow(530, 3240, 572, 3240)
arrow(1030, 3240, 1072, 3240)
arrow(1530, 3240, 1572, 3240)

# 6 — Deployment.
panel(3455, 355, "6", "DEPLOYMENT / INFERENCE",
      "Only the student policy and its GRU memory run on the Go2")
box(80, 3565, 450, 185, "Export", [
  "Student checkpoint → ONNX",
  "Export gains, joint order, grid config",
  "Verify ONNX vs PyTorch actions",
], "#164148", "#4caea4", size=19)
box(580, 3565, 450, 185, "Robot inputs", [
  "54 body/command + 99 heights = 153",
  "LiDAR → 0.4 s buffer → UDP map",
  "Stale/missing map → zeros",
], "#164148", "#4caea4", size=19)
box(1080, 3565, 450, 185, "ONNX step", [
  "(obs [1,153], h [1,1,256])",
  "→ (actions [1,12], next h)",
  "Next h feeds the following step",
], "#164148", "#4caea4", size=19)
box(1580, 3565, 440, 185, "Go2 controller", [
  "Joint target = q_default + 0.25a",
  "Motor PD applies torque",
  "No teacher, critic, or decoder",
], "#164148", "#4caea4", size=19)
arrow(530, 3657, 572, 3657)
arrow(1030, 3657, 1072, 3657)
arrow(1530, 3657, 1572, 3657)
txt(80, 3848, "Sources: mjlab/tasks/velocity/config/go2 · velocity_env_cfg.py · rewards.py · rsl_rl/{modules,networks,algorithms,runners} · scripts/export_deploy.py",
    size=19, color="#879ab6")
add("</svg>")


def main() -> None:
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
  print(OUT)
  if shutil.which("ffmpeg"):
    subprocess.run(
      ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(OUT),
       "-frames:v", "1", str(PNG)],
      check=True,
    )
    print(PNG)


if __name__ == "__main__":
  main()
