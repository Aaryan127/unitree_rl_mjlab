"""Check that the robot's LiDAR bridge builds the same height map as training.

Runs the PGTT environment, takes the simulated LiDAR's points, feeds them through the
robot bridge (scripts/lidar_height_map.py::sample_grid, with the settings training
uses) and compares the result with the student's training map (mdp.lidar_height_map).
Also reports how much of the grid the LiDAR sees and where its blind spots are.

Run it after changing the LiDAR mount/pattern or any map setting:
  python scripts/check_lidar_map.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import lidar_height_map as bridge
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Velocity-PGTT-Student-Unitree-Go2"
SHARED = ("min_z", "max_z", "body_x", "body_y", "body_min_z", "cell_radius", "min_points",
          "quantile", "min_coverage")


def _tf(pos, quat_wxyz):
  return SimpleNamespace(
    translation=SimpleNamespace(x=pos[0], y=pos[1], z=pos[2]),
    rotation=SimpleNamespace(x=quat_wxyz[1], y=quat_wxyz[2], z=quat_wxyz[3], w=quat_wxyz[0]),
  )


def main(num_envs: int = 32) -> None:
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  term = cfg.observations["height_noisy"].terms["height_scan"]
  term.params["point_noise_std"] = 0.0  # compare geometry, not random noise
  term.delay_max_lag = 0  # compare the same frame (training delays the map 0-3 steps)
  lp = term.params
  params = {
    "num_points": 99, "size_x": lp["size_x"], "size_y": lp["size_y"],
    "resolution": lp["resolution"], "base_height": lp["base_height"],
    "lidar_map": {k: lp[k] for k in SHARED},
  }
  bcfg = bridge.apply_training_settings(bridge.BridgeConfig(), params)

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  for _ in range(20):
    env.step(torch.zeros(num_envs, env.action_manager.total_action_dim))
  for label in ("standing", "moving randomly"):
    if label == "moving randomly":
      for _ in range(30):
        env.step(2 * torch.rand(num_envs, env.action_manager.total_action_dim) - 1)
    obs = env.observation_manager.compute()
    sim, ideal = obs["height_noisy"].numpy(), obs["height"].numpy()
    lidar, robot = env.scene["lidar"].data, env.scene["robot"].data
    errors, coverage = [], []
    for i in range(num_envs):
      pts = lidar.hit_pos_w[i][lidar.distances[i] >= 0].numpy().astype(float)
      pose = _tf(robot.root_link_pos_w[i].numpy(), robot.root_link_quat_w[i].numpy())
      values, cov = bridge.sample_grid(pts, pose, params, bcfg)
      if cov < bcfg.min_coverage:
        values = np.zeros_like(values)  # bridge sends nothing -> controller uses zeros
      errors.append(np.abs(values - sim[i]).max())
      coverage.append(cov)
    errors = np.array(errors)
    seen = sim != 0
    acc = np.abs(sim - ideal)[seen]
    print(f"\n[{label}]")
    print(f"  bridge vs training map: max diff {errors.max():.5f} m "
          f"({(errors > 1e-3).sum()}/{num_envs} robots differ by > 1 mm)")
    print(f"  LiDAR grid coverage: mean {np.mean(coverage):.2f}, min {np.min(coverage):.2f}")
    print(f"  LiDAR map vs ideal ray-cast map on seen cells: mean {acc.mean()*100:.2f} cm, "
          f"max {acc.max()*100:.2f} cm")
  missing = (sim == 0).mean(0).reshape(9, 11)
  print("\n  fraction of robots with no data per cell (rows y left->right, cols x back->front):")
  for row in missing:
    print("   " + " ".join(f"{v:.1f}" for v in row))
  ok = errors.max() <= 1e-3
  print("\nRESULT:", "bridge matches training" if ok else "MISMATCH between bridge and training")
  env.close()
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
