"""Send synthetic height maps to the Go2 controller (for bring-up and testing).

The controller's `height_scan` observation listens on UDP (deploy.yaml:
observations.height_scan.params.udp_port) for one datagram of float32 values per map.
This script sends maps in exactly the training layout (see
mjlab/tasks/velocity/mdp/observations.py::height_scan):

    grid size_x x size_y, `resolution` spacing, centered on the base, yaw-aligned,
    y outer loop, x inner loop, value = (base_z - terrain_z) - base_height, in [-1, 1]

A real deployment replaces this with an elevation-mapping node that fills the same
grid from the LiDAR.

Examples:
  python scripts/send_height_map.py --mode flat
  python scripts/send_height_map.py --mode step --step-height 0.1 --host 192.168.123.18
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
import yaml


@dataclass
class SenderConfig:
  deploy_yaml: str = "deploy/robots/go2/config/policy/pgtt/v0/params/deploy.yaml"
  host: str = "127.0.0.1"
  """Machine running go2_ctrl."""
  mode: Literal["flat", "step", "stairs"] = "flat"
  step_height: float = 0.1
  """Height of the step (m) for step/stairs modes; terrain starts step_distance ahead."""
  step_distance: float = 0.25
  rate: float = 20.0
  duration: float = 0.0
  """Seconds to send (0 = forever)."""


def make_grid(params: dict) -> tuple[np.ndarray, np.ndarray]:
  """x, y coordinates of the rays in the same order as GridPatternCfg.generate_rays."""
  res = params["resolution"]
  x = np.arange(-params["size_x"] / 2, params["size_x"] / 2 + res * 0.5, res)
  y = np.arange(-params["size_y"] / 2, params["size_y"] / 2 + res * 0.5, res)
  grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
  return grid_x.flatten(), grid_y.flatten()


def make_map(cfg: SenderConfig, x: np.ndarray) -> np.ndarray:
  terrain_z = np.zeros_like(x)
  if cfg.mode == "step":
    terrain_z[x > cfg.step_distance] = cfg.step_height
  elif cfg.mode == "stairs":
    steps = np.floor(np.clip(x - cfg.step_distance, 0.0, None) / 0.3) + (x > cfg.step_distance)
    terrain_z = steps * cfg.step_height
  # Higher terrain -> smaller (base_z - terrain_z) -> negative values.
  return np.clip(-terrain_z, -1.0, 1.0).astype(np.float32)


def main(cfg: SenderConfig) -> None:
  with open(cfg.deploy_yaml) as f:
    params = yaml.safe_load(f)["observations"]["height_scan"]["params"]
  x, _ = make_grid(params)
  assert len(x) == params["num_points"], (len(x), params["num_points"])
  height_map = make_map(cfg, x)
  port = int(params["udp_port"])

  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  print(f"Sending '{cfg.mode}' height map ({len(height_map)} points) to {cfg.host}:{port} at {cfg.rate} Hz")
  start = time.time()
  while cfg.duration <= 0 or time.time() - start < cfg.duration:
    sock.sendto(height_map.tobytes(), (cfg.host, port))
    time.sleep(1.0 / cfg.rate)


if __name__ == "__main__":
  main(tyro.cli(SenderConfig))
