"""ROS 2 PointCloud2 to PGTT's robot-centric height grid over UDP.

Builds the same 11 x 9 height map the policy was trained with (see
mjlab/tasks/velocity/mdp/observations.py::height_scan) from the Go2 LiDAR and sends it
to the controller (deploy/include/isaaclab/devices/height_map/height_map.h).

Pipeline per received cloud:
  1. transform the cloud into a gravity-aligned world frame (normally `odom`)
  2. add it to a short rolling buffer (`accumulate_time`), so a sparse LiDAR still
     covers the grid; old points are dropped
  3. sample the grid around the robot's LATEST pose (yaw-aligned, base-Z relative),
     so latency between the scan and now does not shift the map
  4. send it if enough cells are covered; otherwise send nothing, and after the
     controller's 0.2 s timeout the student gets the all-zero map it was trained for

Robot pose comes from TF (`--pose-source tf`) or from a nav_msgs/Odometry topic
(`--pose-source odom`, useful when the robot publishes odometry but no TF tree).
The cloud is transformed with TF, or used as-is when its frame already is the world
frame (e.g. a deskewed cloud published in `odom`).

The ROS imports are kept in main() so the geometry can be tested without ROS.
"""

from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import yaml

from height_map_grid import make_grid


# Map-building settings that training also uses: exported by scripts/export_deploy.py into
# deploy.yaml (observations.height_scan.params.lidar_map) and loaded from there, so the robot
# builds the map exactly like the student saw it in simulation. Command-line values that
# differ from the defaults below override the file.
_SHARED = {
  "min_z": "min_z", "max_z": "max_z", "body_x": "body_x", "body_y": "body_y",
  "body_min_z": "body_min_z", "cell_radius": "cell_radius",
  "min_points": "min_points_per_cell", "quantile": "quantile", "min_coverage": "min_coverage",
}


@dataclass
class BridgeConfig:
  deploy_yaml: str = "deploy/robots/go2/config/policy/pgtt/v0/params/deploy.yaml"
  cloud_topic: str = "/points"
  world_frame: str = "odom"
  base_frame: str = "base_link"
  host: str = "127.0.0.1"
  pose_source: str = "tf"
  """'tf': base pose from TF world_frame -> base_frame. 'odom': from odom_topic."""
  odom_topic: str = "/odom"
  """nav_msgs/Odometry with the base pose in world_frame (pose_source='odom')."""
  min_z: float = -0.9
  """Lowest accepted point relative to base_link, in metres."""
  max_z: float = 0.45
  """Highest accepted point relative to base_link. Must stay above the base: on stairs
  going up the next step tops are level with (or above) the base, and the simulator's rays
  (z_offset 0.5) see them. A low cut here turns rising stairs into "flat" cells."""
  body_x: float = 0.30
  body_y: float = 0.15
  body_min_z: float = -0.10
  """Self-filter box in the base-yaw frame: points with |x| <= body_x, |y| <= body_y and
  z >= body_min_z are the robot's own trunk/hips and are dropped. The Go2 trunk is about
  +-0.19 x +-0.05 m with its bottom ~6 cm below the base, so real terrain can't be inside
  this box (the body would be resting on it); step tops 10+ cm below the base are kept.
  Lower legs/feet are at ground level and the low quantile below prefers the ground."""
  cell_radius: float = 0.075
  """Only use points this close to a grid sample; avoid smearing stair edges."""
  min_points_per_cell: int = 2
  quantile: float = 0.2
  """Ground height of a cell = this quantile (lower index) of its points' heights."""
  min_coverage: float = 0.5
  """Minimum fraction of observed grid cells before sending a map. The student was
  trained with ~5 % missing cells (or whole maps missing), not with half-empty maps."""
  accumulate_time: float = 0.4
  """Seconds of past clouds kept (in the world frame) to fill the grid. 0 = last cloud only."""
  max_buffer_points: int = 200_000
  max_cloud_age: float = 0.15
  """Clouds older than this are discarded; controller timeout is 0.2 s by default."""
  use_receive_time: bool = False
  """Ignore cloud timestamps (LiDAR and this computer not time-synced): treat each cloud
  as captured on arrival and use the latest pose/TF."""
  max_rate: float = 20.0
  max_pose_age: float = 0.15
  """Reject a stale base pose even while fresh point clouds continue to arrive."""


def quaternion_matrix(rotation) -> np.ndarray:
  """3x3 rotation matrix from an object with ROS quaternion fields."""
  x, y, z, w = (float(getattr(rotation, key)) for key in ("x", "y", "z", "w"))
  norm = x*x + y*y + z*z + w*w
  if norm < 1e-12:
    raise ValueError("Invalid zero quaternion in TF")
  s = 2.0 / norm
  return np.array([
    [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
    [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
    [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
  ])


def _translation(transform) -> np.ndarray:
  t = transform.translation
  return np.array([t.x, t.y, t.z], dtype=np.float64)


def points_to_world(points: np.ndarray, cloud_transform) -> np.ndarray:
  """[N, 3] cloud-frame points -> [N, 3] world-frame points (finite rows only)."""
  if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError("Point cloud must have shape [N, 3]")
  world = points.astype(np.float64) @ quaternion_matrix(cloud_transform.rotation).T
  world += _translation(cloud_transform)
  return world[np.all(np.isfinite(world), axis=1)]


def sample_grid(
  world_points: np.ndarray,
  base_transform,
  params: dict,
  cfg: BridgeConfig,
) -> tuple[np.ndarray, float]:
  """Return (float32 observations, observed-cell fraction) around the base pose.

  Identical to training (mjlab/tasks/velocity/mdp/observations.py::
  grid_height_map_from_points): each point goes to its nearest grid cell (grid
  yaw-aligned with the base, y outer / x inner), a cell's ground is the `quantile`
  lowest point (lower index), cells with < min_points_per_cell points are 0. The
  min-coverage rule ("send nothing") is applied by the caller.
  """
  base_rot = quaternion_matrix(base_transform.rotation)
  relative = world_points - _translation(base_transform)
  yaw = np.arctan2(base_rot[1, 0], base_rot[0, 0])
  c, s = np.cos(yaw), np.sin(yaw)
  lx = c * relative[:, 0] + s * relative[:, 1]
  ly = -s * relative[:, 0] + c * relative[:, 1]
  z = relative[:, 2]
  size_x, size_y, res = float(params["size_x"]), float(params["size_y"]), float(params["resolution"])
  nx, ny = int(round(size_x / res)) + 1, int(round(size_y / res)) + 1
  m = nx * ny
  if m != int(params["num_points"]):
    raise ValueError("Grid dimensions disagree with exported num_points")

  ix = np.round((lx + size_x / 2) / res)
  iy = np.round((ly + size_y / 2) / res)
  d2 = (lx - (ix * res - size_x / 2)) ** 2 + (ly - (iy * res - size_y / 2)) ** 2
  on_body = (np.abs(lx) <= cfg.body_x) & (np.abs(ly) <= cfg.body_y) & (z >= cfg.body_min_z)
  ok = (
    (z >= cfg.min_z) & (z <= cfg.max_z)
    & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    & (d2 <= cfg.cell_radius ** 2)
    & ~on_body
  )
  cell = (iy[ok] * nx + ix[ok]).astype(np.int64)
  z_ok = z[ok]
  order = np.lexsort((z_ok, cell))  # sort by cell, then height
  cell, z_ok = cell[order], z_ok[order]
  counts = np.bincount(cell, minlength=m)[:m]
  starts = np.cumsum(counts) - counts
  values = np.zeros(m, dtype=np.float32)
  observed = counts >= cfg.min_points_per_cell
  k = np.floor(cfg.quantile * np.maximum(counts - 1, 0)).astype(np.int64)
  for i in np.nonzero(observed)[0]:
    ground_z = z_ok[starts[i] + k[i]]
    values[i] = np.clip(-ground_z - float(params["base_height"]), -1.0, 1.0)
  return values, float(observed.mean())


def terrain_observation(
  points: np.ndarray,
  cloud_transform,
  base_transform,
  params: dict,
  cfg: BridgeConfig,
) -> tuple[np.ndarray, float]:
  """One cloud -> (observations, coverage). Single-scan version of the pipeline."""
  return sample_grid(points_to_world(points, cloud_transform), base_transform, params, cfg)


class PointBuffer:
  """Rolling buffer of world-frame points from the last `horizon` seconds."""

  def __init__(self, horizon: float, max_points: int):
    if horizon < 0 or max_points < 1:
      raise ValueError("Point-buffer horizon must be nonnegative and capacity positive")
    self.horizon = horizon
    self.max_points = max_points
    self._chunks: list[tuple[float, np.ndarray]] = []

  def add(self, stamp: float, world_points: np.ndarray) -> None:
    self._chunks.append((stamp, world_points[-self.max_points:]))
    self.prune(stamp)

  def prune(self, now: float) -> None:
    self._chunks = [(t, p) for t, p in self._chunks if now - t <= self.horizon]
    total = sum(len(p) for _, p in self._chunks)
    while len(self._chunks) > 1 and total > self.max_points:  # drop oldest first
      total -= len(self._chunks.pop(0)[1])

  def points(self) -> np.ndarray:
    if not self._chunks:
      return np.zeros((0, 3))
    return np.concatenate([p for _, p in self._chunks], axis=0)


IDENTITY = SimpleNamespace(
  translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
  rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
)


def _odom_to_transform(msg):
  """nav_msgs/Odometry -> object with .translation and .rotation like a TF transform."""
  p, q = msg.pose.pose.position, msg.pose.pose.orientation
  return SimpleNamespace(translation=SimpleNamespace(x=p.x, y=p.y, z=p.z), rotation=q)


def apply_training_settings(cfg: BridgeConfig, params: dict) -> BridgeConfig:
  """Use the map settings training used (deploy.yaml) unless overridden on the command line."""
  from dataclasses import replace

  trained = params.get("lidar_map", {})
  defaults = BridgeConfig()
  updates = {}
  for yaml_key, field_name in _SHARED.items():
    if yaml_key in trained and getattr(cfg, field_name) == getattr(defaults, field_name):
      updates[field_name] = type(getattr(defaults, field_name))(trained[yaml_key])
  if not trained:
    print("[lidar_height_map] deploy.yaml has no lidar_map settings (old export?): using defaults")
  return replace(cfg, **updates)


def main(cfg: BridgeConfig) -> None:
  with open(cfg.deploy_yaml, encoding="utf-8") as f:
    params = yaml.safe_load(f)["observations"]["height_scan"]["params"]
  cfg = apply_training_settings(cfg, params)
  if (cfg.min_points_per_cell < 1 or cfg.cell_radius <= 0 or cfg.max_rate <= 0
      or cfg.max_cloud_age <= 0 or cfg.max_pose_age <= 0):
    raise ValueError("Cell count, radius and rate must be positive")
  if not 0 <= cfg.min_coverage <= 1 or cfg.min_z >= cfg.max_z:
    raise ValueError("Invalid coverage or ground-height range")
  if not 0 <= cfg.quantile <= 1:
    raise ValueError("Ground quantile must be between 0 and 1")
  if not 1 <= int(params["udp_port"]) <= 65535:
    raise ValueError("Height-map UDP receiver is disabled or has an invalid port")
  if cfg.pose_source not in ("tf", "odom"):
    raise ValueError("pose_source must be 'tf' or 'odom'")
  if cfg.max_cloud_age >= float(params["timeout"]):
    raise ValueError("max_cloud_age must be below the controller map timeout")

  try:
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from tf2_ros import Buffer, TransformException, TransformListener
  except ImportError as exc:
    raise SystemExit("Source ROS 2 and install rclpy, sensor_msgs_py and tf2_ros") from exc

  class HeightBridge(Node):
    def __init__(self) -> None:
      super().__init__("pgtt_lidar_height_map")
      self.tf_buffer = Buffer()
      self.tf_listener = TransformListener(self.tf_buffer, self)
      self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      self.points = PointBuffer(cfg.accumulate_time, cfg.max_buffer_points)
      self.base_pose = None  # latest base pose (odom mode)
      self.base_pose_received = 0.0
      self.base_pose_source_age = 0.0
      self.last_sent = 0.0
      self.last_warning = 0.0
      self.create_subscription(PointCloud2, cfg.cloud_topic, self.on_cloud, qos_profile_sensor_data)
      if cfg.pose_source == "odom":
        from nav_msgs.msg import Odometry
        self.create_subscription(Odometry, cfg.odom_topic, self.on_odom, qos_profile_sensor_data)
      self.get_logger().info(
        f"Listening to {cfg.cloud_topic} (pose: {cfg.pose_source}); sending "
        f"{params['num_points']} cells to {cfg.host}:{params['udp_port']}")

    def warn_limited(self, reason: str) -> None:
      now = time.monotonic()
      if now - self.last_warning > 5.0:
        self.get_logger().warn(reason)
        self.last_warning = now

    def on_odom(self, msg) -> None:
      if msg.header.frame_id and msg.header.frame_id != cfg.world_frame:
        self.warn_limited(f"Odometry frame '{msg.header.frame_id}' != world_frame '{cfg.world_frame}'")
        return
      self.base_pose = _odom_to_transform(msg)
      self.base_pose_received = time.monotonic()
      self.base_pose_source_age = (
        self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9

    def lookup(self, target: str, source: str, stamp):
      return self.tf_buffer.lookup_transform(
        target, source, stamp, timeout=Duration(seconds=0.02)).transform

    def on_cloud(self, msg: PointCloud2) -> None:
      now = time.monotonic()
      if cfg.use_receive_time:
        stamp = Time()  # "latest available" for TF lookups
      else:
        stamp = Time.from_msg(msg.header.stamp)
        age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
        if age < -0.05 or age > cfg.max_cloud_age:
          self.warn_limited(f"Ignoring cloud with age {age:.3f} s (unsynced clocks? try --use-receive-time True)")
          return
      try:
        if msg.header.frame_id == cfg.world_frame:
          cloud_tf = IDENTITY  # cloud already in the world frame (e.g. a deskewed odom cloud)
        else:
          cloud_tf = self.lookup(cfg.world_frame, msg.header.frame_id, stamp)
        if cfg.pose_source == "tf":
          stamped_pose = self.tf_buffer.lookup_transform(
            cfg.world_frame, cfg.base_frame, Time(), timeout=Duration(seconds=0.02))
          pose_age = (self.get_clock().now() - Time.from_msg(stamped_pose.header.stamp)).nanoseconds * 1e-9
          if pose_age < -0.05 or pose_age > cfg.max_pose_age:
            self.warn_limited(f"Ignoring stale base TF: age {pose_age:.3f} s")
            return
          base_tf = stamped_pose.transform
        else:
          base_tf = self.base_pose
          if base_tf is None:
            self.warn_limited(f"No odometry received on {cfg.odom_topic} yet")
            return
          pose_age = now - self.base_pose_received
          if not cfg.use_receive_time:
            pose_age += self.base_pose_source_age
          if pose_age < -0.05 or pose_age > cfg.max_pose_age:
            self.warn_limited(f"Ignoring stale odometry: age {pose_age:.3f} s")
            return
      except TransformException as exc:
        self.warn_limited(f"Missing TF: {exc}")
        return
      try:
        cloud = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.points.add(now, points_to_world(np.asarray(cloud, dtype=np.float64), cloud_tf))
      except (ValueError, KeyError, TypeError) as exc:
        self.warn_limited(f"Invalid point cloud: {exc}")
        return
      if now - self.last_sent < 1.0 / cfg.max_rate:
        return
      values, coverage = sample_grid(self.points.points(), base_tf, params, cfg)
      if coverage < cfg.min_coverage:
        # Sending nothing lets the controller's timeout switch to its trained
        # all-zero fallback; a sparse map must not keep an old map alive.
        self.warn_limited(f"Sparse terrain: {coverage:.0%} grid coverage (need {cfg.min_coverage:.0%})")
        return
      self.sock.sendto(values.astype("<f4", copy=False).tobytes(),
                       (cfg.host, int(params["udp_port"])))
      self.last_sent = now

  rclpy.init()
  node = HeightBridge()
  try:
    rclpy.spin(node)
  finally:
    node.sock.close()
    node.destroy_node()
    rclpy.shutdown()


def _parse_bool(text: str) -> bool:
  if text.lower() in ("1", "true", "yes", "on"):
    return True
  if text.lower() in ("0", "false", "no", "off"):
    return False
  raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  defaults = BridgeConfig()
  for name in defaults.__dataclass_fields__:
    value = getattr(defaults, name)
    kind = _parse_bool if isinstance(value, bool) else type(value)
    parser.add_argument("--" + name.replace("_", "-"), default=value, type=kind)
  main(BridgeConfig(**vars(parser.parse_args())))
