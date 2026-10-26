"""Check the live LiDAR bridge against PGTT's simulated height observation."""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lidar_height_map import (  # noqa: E402
  IDENTITY, BridgeConfig, PointBuffer, points_to_world, sample_grid, terrain_observation,
)
from send_height_map import make_grid  # noqa: E402


PARAMS = {
  "num_points": 99,
  "size_x": 1.0,
  "size_y": 0.8,
  "resolution": 0.1,
  "base_height": 0.25,
}


def transform(x=0.0, y=0.0, z=0.0, yaw=0.0):
  return SimpleNamespace(
    translation=SimpleNamespace(x=x, y=y, z=z),
    rotation=SimpleNamespace(x=0.0, y=0.0,
                             z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
  )


class HeightMapTests(unittest.TestCase):
  def setUp(self):
    self.cfg = BridgeConfig()
    self.grid_x, self.grid_y = make_grid(PARAMS)

  def cloud(self, terrain_z):
    # Two returns near each sample so every cell meets the minimum count.
    x = np.repeat(self.grid_x, 2) + np.tile([-0.005, 0.005], 99)
    y = np.repeat(self.grid_y, 2)
    z = np.repeat(terrain_z, 2)
    return np.column_stack((x, y, z))

  def test_flat_and_stair_sign_and_packet(self):
    heights = np.where(self.grid_x >= 0.2, 0.1, 0.0)
    points = self.cloud(heights)
    values, coverage = terrain_observation(
      points, transform(), transform(z=0.25), PARAMS, self.cfg)
    expected = np.where(self.grid_x >= 0.2, -0.1, 0.0)
    np.testing.assert_allclose(values, expected, atol=1e-6)
    self.assertEqual(coverage, 1.0)
    self.assertEqual(values.dtype, np.float32)
    self.assertEqual(len(values.astype("<f4").tobytes()), 99 * 4)

  def test_yaw_aligned_grid_and_base_height(self):
    yaw = math.pi / 2
    x = np.repeat(self.grid_x, 2)
    y = np.repeat(self.grid_y, 2)
    points = np.column_stack((-y, x, np.full(198, 0.1)))
    values, coverage = terrain_observation(
      points, transform(), transform(z=0.35, yaw=yaw), PARAMS, self.cfg)
    np.testing.assert_allclose(values, 0.0, atol=1e-6)
    self.assertEqual(coverage, 1.0)

  def test_cloud_frame_translation(self):
    values, coverage = terrain_observation(
      self.cloud(np.zeros(99)), transform(x=1.0, y=-2.0),
      transform(x=1.0, y=-2.0, z=0.25), PARAMS, self.cfg)
    np.testing.assert_allclose(values, 0.0, atol=1e-6)
    self.assertEqual(coverage, 1.0)

  def test_rising_stairs_above_base(self):
    # Two 13 cm steps ahead of a robot whose base is 0.27 m above the floor: the second
    # step top is level with the base. The simulator reports these (rays start 0.5 m up).
    heights = np.where(self.grid_x >= 0.15, 0.13, 0.0) + np.where(self.grid_x >= 0.45, 0.13, 0.0)
    values, coverage = terrain_observation(
      self.cloud(heights), transform(), transform(z=0.27), PARAMS, self.cfg)
    expected = np.clip(0.27 - heights - 0.25, -1, 1)
    np.testing.assert_allclose(values, expected, atol=1e-6)
    self.assertEqual(coverage, 1.0)

  def test_self_returns_ignored(self):
    # Ground at 0, plus a return from the robot's own body right above the centre cell.
    points = np.vstack([self.cloud(np.zeros(99)), [[0.0, 0.0, 0.20], [0.0, 0.0, 0.20]]])
    values, _ = terrain_observation(points, transform(), transform(z=0.25), PARAMS, self.cfg)
    np.testing.assert_allclose(values, 0.0, atol=1e-6)

  def test_accumulated_scans_fill_grid(self):
    # Two sparse scans (half of the cells each) together cover the whole grid.
    cloud = self.cloud(np.zeros(99))
    buf = PointBuffer(horizon=0.4, max_points=10_000)
    buf.add(0.0, points_to_world(cloud[:80], IDENTITY))       # 40 of 99 cells
    _, cov_one = sample_grid(buf.points(), transform(z=0.25), PARAMS, self.cfg)
    buf.add(0.1, points_to_world(cloud[80:], IDENTITY))       # the other 59
    values, cov_two = sample_grid(buf.points(), transform(z=0.25), PARAMS, self.cfg)
    self.assertLess(cov_one, self.cfg.min_coverage)
    self.assertEqual(cov_two, 1.0)
    np.testing.assert_allclose(values, 0.0, atol=1e-6)

  def test_buffer_drops_old_points(self):
    buf = PointBuffer(horizon=0.4, max_points=10_000)
    buf.add(0.0, np.zeros((5, 3)))
    buf.add(0.5, np.ones((3, 3)))
    self.assertEqual(len(buf.points()), 3)

  def test_latest_pose_after_robot_moved(self):
    # Scan taken at x=0 with a 10 cm step starting at x=0.35 (between grid lines).
    # The robot has since walked 0.2 m forward: sampling with the new pose must show
    # the step 0.2 m closer, exactly like the simulator would.
    world = points_to_world(self.cloud(np.where(self.grid_x >= 0.35, 0.1, 0.0)), IDENTITY)
    values, _ = sample_grid(world, transform(x=0.2, z=0.25), PARAMS, self.cfg)
    moved_x = self.grid_x + 0.2              # where each cell now looks in the world
    on_step = (moved_x >= 0.35) & (moved_x <= 0.5 + 1e-6)   # inside the scanned area
    np.testing.assert_allclose(values[on_step], -0.1, atol=1e-6)
    np.testing.assert_allclose(values[moved_x < 0.35], 0.0, atol=1e-6)

  def test_missing_and_invalid_points(self):
    points = self.cloud(np.zeros(99))[:40]
    points[0, 2] = np.nan
    points[2, 2] = 0.8  # above the base; a robot or other nonground return
    values, coverage = terrain_observation(
      points, transform(), transform(z=0.25), PARAMS, self.cfg)
    self.assertLess(coverage, self.cfg.min_coverage)
    self.assertTrue(np.all(values == 0.0))


if __name__ == "__main__":
  unittest.main()
