"""PGTT height grid coordinates shared by test sender and LiDAR bridge."""

import numpy as np


def make_grid(params: dict) -> tuple[np.ndarray, np.ndarray]:
  """Sample positions in the same order as GridPatternCfg.generate_rays."""
  res = float(params["resolution"])
  x = np.arange(-params["size_x"] / 2, params["size_x"] / 2 + res * 0.5, res)
  y = np.arange(-params["size_y"] / 2, params["size_y"] / 2 + res * 0.5, res)
  grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
  return grid_x.flatten(), grid_y.flatten()
