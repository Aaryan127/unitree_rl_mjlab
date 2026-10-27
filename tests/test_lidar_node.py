"""Exercise the ROS callback logic without requiring a ROS installation."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as S
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lidar_height_map import BridgeConfig, PointBuffer, main
from height_map_grid import make_grid


class Time:
  def __init__(self, seconds=0):
    self.nanoseconds = int(seconds * 1e9)

  @classmethod
  def from_msg(cls, msg):
    return cls(msg.sec + msg.nanosec / 1e9)

  def __sub__(self, other):
    return S(nanoseconds=self.nanoseconds - other.nanoseconds)


def module(**attributes):
  result = ModuleType("fake_ros")
  result.__dict__.update(attributes)
  return result


class LidarNodeTests(unittest.TestCase):
  def run_node(self, *, pose_source="tf", pose_stamp=10, cloud_stamp=10):
    params = dict(num_points=99, size_x=1.0, size_y=0.8, resolution=0.1,
                  base_height=0.25, udp_port=9870, timeout=0.2)
    x, y = make_grid(params)
    points = np.column_stack((np.repeat(x, 2), np.repeat(y, 2), np.zeros(198)))
    transform = S(translation=S(x=0., y=0., z=0.25), rotation=S(x=0., y=0., z=0., w=1.))
    stamp = lambda seconds: S(sec=seconds, nanosec=0)
    packets, warnings = [], []

    class Node:
      def __init__(self, name):
        self.callbacks = {}

      def create_subscription(self, msg_type, topic, callback, qos):
        self.callbacks[topic] = callback

      def get_clock(self):
        return S(now=lambda: Time(10))

      def get_logger(self):
        return S(info=lambda message: None, warn=warnings.append)

      def destroy_node(self):
        pass

    class Buffer:
      def lookup_transform(self, *args, **kwargs):
        return S(transform=transform, header=S(stamp=stamp(pose_stamp)))

    def spin(node):
      if pose_source == "odom":
        node.on_odom(S(header=S(frame_id="odom", stamp=stamp(pose_stamp)),
                       pose=S(pose=S(position=transform.translation, orientation=transform.rotation))))
      node.on_cloud(S(header=S(frame_id="odom", stamp=stamp(cloud_stamp)), points=points))

    modules = {
      "rclpy": module(init=lambda: None, spin=spin, shutdown=lambda: None),
      "rclpy.node": module(Node=Node), "rclpy.time": module(Time=Time),
      "rclpy.duration": module(Duration=lambda **kwargs: None),
      "rclpy.qos": module(qos_profile_sensor_data=None),
      "sensor_msgs": module(), "sensor_msgs.msg": module(PointCloud2=object),
      "sensor_msgs_py": module(point_cloud2=S(read_points_numpy=lambda msg, **kwargs: msg.points)),
      "nav_msgs": module(), "nav_msgs.msg": module(Odometry=object),
      "tf2_ros": module(Buffer=Buffer, TransformListener=lambda *args: None,
                         TransformException=RuntimeError),
    }
    sock = S(sendto=lambda payload, address: packets.append(payload), close=lambda: None)
    with tempfile.TemporaryDirectory() as temp:
      config_path = Path(temp) / "deploy.yaml"
      config_path.write_text(yaml.safe_dump({"observations": {"height_scan": {"params": params}}}))
      with patch.dict(sys.modules, modules), patch("lidar_height_map.socket.socket", return_value=sock):
        main(BridgeConfig(deploy_yaml=str(config_path), pose_source=pose_source))
    return packets, warnings

  def test_fresh_cloud_and_tf_produce_correct_packet(self):
    packets, _ = self.run_node()
    self.assertEqual(len(packets), 1)
    values = np.frombuffer(packets[0], dtype="<f4")
    self.assertEqual(values.size, 99)
    np.testing.assert_array_equal(values, 0.)

  def test_stale_tf_does_not_refresh_controller_map(self):
    packets, warnings = self.run_node(pose_stamp=9)
    self.assertFalse(packets)
    self.assertTrue(any("stale base TF" in warning for warning in warnings))

  def test_stale_odom_does_not_refresh_controller_map(self):
    packets, warnings = self.run_node(pose_source="odom", pose_stamp=9)
    self.assertFalse(packets)
    self.assertTrue(any("stale odometry" in warning for warning in warnings))

  def test_stale_cloud_is_discarded(self):
    packets, warnings = self.run_node(cloud_stamp=9)
    self.assertFalse(packets)
    self.assertTrue(any("Ignoring cloud" in warning for warning in warnings))

  def test_single_large_cloud_is_bounded(self):
    buffer = PointBuffer(0.4, 10)
    buffer.add(0., np.zeros((100, 3)))
    self.assertEqual(len(buffer.points()), 10)


if __name__ == "__main__":
  unittest.main()
