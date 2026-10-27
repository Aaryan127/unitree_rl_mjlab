"""Refresh the atlas's curated source snapshot from a local mjlab checkout.

Usage: .venv/bin/python doc/research-atlas-site/snapshot.py
No simulator, credentials, checkpoints, or logs are included.
"""
import dataclasses
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "dist"
from mjlab.tasks.velocity.config.go2.env_cfgs import unitree_go2_pgtt_env_cfg
from mjlab.tasks.velocity.config.go2.rl_cfg import (
    unitree_go2_pgtt_teacher_runner_cfg, unitree_go2_pgtt_student_runner_cfg,
)

def serial(value):
    if dataclasses.is_dataclass(value):
        return {f.name: serial(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)

FILES = [
    "instruction.md", "CODE_GUIDE.md", "DEBUG_AND_EXPERIMENT.md",
    "mjlab/tasks/velocity/velocity_env_cfg.py",
    "mjlab/tasks/velocity/config/go2/env_cfgs.py",
    "mjlab/tasks/velocity/config/go2/rl_cfg.py",
    "mjlab/tasks/velocity/config/go2/__init__.py",
    "mjlab/tasks/velocity/terrain_profiles.py",
    "mjlab/tasks/velocity/mdp/observations.py", "mjlab/tasks/velocity/mdp/rewards.py",
    "mjlab/tasks/velocity/mdp/gait_command.py", "mjlab/tasks/velocity/mdp/curriculums.py",
    "mjlab/tasks/velocity/mdp/velocity_command.py", "mjlab/tasks/registry.py",
    "mjlab/terrains/config.py", "mjlab/terrains/terrain_generator.py",
    "mjlab/terrains/heightfield_terrains.py", "mjlab/terrains/primitive_terrains.py",
    "mjlab/rsl_rl/networks/teacher_mlp.py", "mjlab/rsl_rl/networks/belief_encoder.py",
    "mjlab/rsl_rl/networks/mlp.py", "mjlab/rsl_rl/networks/normalization.py",
    "mjlab/rsl_rl/modules/actor_critic_wild.py", "mjlab/rsl_rl/modules/student_teacher_wild.py",
    "mjlab/rsl_rl/algorithms/ppo.py", "mjlab/rsl_rl/algorithms/distillation.py",
    "mjlab/rsl_rl/algorithms/distillation_wild.py",
    "mjlab/rsl_rl/runners/on_policy_runner.py", "mjlab/rsl_rl/runners/distillation_runner.py",
    "mjlab/rsl_rl/storage/rollout_storage.py", "mjlab/rl/config.py", "mjlab/rl/vecenv_wrapper.py",
    "mjlab/envs/manager_based_rl_env.py", "mjlab/managers/observation_manager.py",
    "mjlab/managers/reward_manager.py", "mjlab/managers/action_manager.py",
    "mjlab/managers/event_manager.py", "mjlab/managers/command_manager.py",
    "mjlab/managers/curriculum_manager.py", "mjlab/managers/termination_manager.py",
    "mjlab/sensor/raycast_sensor.py", "mjlab/sensor/contact_sensor.py",
    "mjlab/utils/noise/height_scan_noise.py", "mjlab/utils/checkpoint.py", "mjlab/utils/debug.py",
    "mjlab/asset_zoo/robots/unitree_go2/go2_constants.py",
    "scripts/train.py", "scripts/play.py", "scripts/debug_env.py", "scripts/evaluate_pgtt.py",
    "scripts/export_deploy.py", "scripts/lidar_height_map.py", "scripts/check_lidar_map.py",
    "scripts/check_cpp_policy.py", "scripts/send_height_map.py", "scripts/pgtt.sh",
    "deploy/include/FSM/State_RLBase.h", "deploy/include/FSM/State_FixStand.h",
    "deploy/include/isaaclab/algorithms/algorithms.h",
    "deploy/include/isaaclab/envs/mdp/observations/observations.h",
    "deploy/include/isaaclab/devices/height_map/height_map.h",
    "deploy/robots/go2/src/State_RLBase.cpp", "doc/pgtt_deployment_audit.md",
]
cfg = unitree_go2_pgtt_env_cfg()
data = {
    "snapshot": "2026-10-27 · working tree",
    "sources": {p: (ROOT/p).read_text() for p in FILES if (ROOT/p).is_file()},
    "observations": serial(cfg.observations),
    "rewards": serial(cfg.rewards), "events": serial(cfg.events),
    "curriculum": serial(cfg.curriculum),
    "terrain": serial(cfg.scene.terrain.terrain_generator),
    "teacher": serial(unitree_go2_pgtt_teacher_runner_cfg()),
    "student": serial(unitree_go2_pgtt_student_runner_cfg()),
}
OUT.mkdir(exist_ok=True)
(OUT / "data.js").write_text("window.ATLAS_DATA=" + json.dumps(data, ensure_ascii=False) + ";\n")
for name in ["pgtt_ml_pipeline.svg", "pgtt_ml_pipeline.png", "pgtt_ml_pipeline.md", "pgtt_deployment_audit.md"]:
    path = ROOT / "doc" / name
    if path.exists():
        (OUT/name).write_bytes(path.read_bytes())
print(f"Atlas: {len(data['sources'])} source files, {len(cfg.rewards)} configured rewards")
