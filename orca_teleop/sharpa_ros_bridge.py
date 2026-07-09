#!/usr/bin/env python3
"""Joints-only SharpaWave SDK <-> ROS 2 bridge — trimmed from the SDK sample
/opt/sharpa-wave-sdk/sample/ROS/wave_ros_server.py (v5.0.4).

Why this exists: the SDK sample also publishes the tactile topics, which pull
in cv_bridge — built against numpy 1.x on ROS 2 Humble and broken next to a
system numpy 2.x. Teleop only needs joints, so this bridge drops tactile (and
with it cv_bridge/numpy) entirely. Everything else follows the SDK sample:
connection flow, control-source setup, per-hand topics.

Subscribed:  wave/{left,right}/joint_commands   (sensor_msgs/JointState,
             positions BY INDEX in the SDK 22-joint order, radians —
             hand_teleop_node.py --hand sharpa publishes exactly this)
Published:   wave/{left,right}/joint_states     (feedback, --feedback_hz)

Run (system python + ROS 2 sourced, hand on USB, Sharpa Pilot CLOSED):
  python3 sharpa_ros_bridge.py
"""
import argparse
import ctypes
import sys
import threading
import time

sys.path.insert(0, '/opt/sharpa-wave-sdk/python')

# Preload system libfmt to prevent conflict between the SDK's libfmt.so.10
# and ROS 2 Humble's libfmt.so.8 (same workaround as the SDK sample).
try:
    ctypes.CDLL("libfmt.so.8", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

from sharpa import SharpaWaveManager, HandSide, DeviceType, ControlSource

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# SDK joint order, indices 0..21 (same names as the SDK ROS sample)
JOINT_NAMES = [
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_DIP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC_FE", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
]


def _hand_side(info, wave):
    if hasattr(info, 'hand_side'):
        return "left" if info.hand_side == HandSide.LEFT else "right"
    if hasattr(wave, 'get_hand_side'):
        return "left" if wave.get_hand_side() == HandSide.LEFT else "right"
    return "right"


class SharpaJointBridge(Node):
    def __init__(self, feedback_hz):
        super().__init__('sharpa_joint_bridge')
        self._lock = threading.Lock()
        self._wave_by_hand = {}
        self._warned = {}

        self.pub_js = {h: self.create_publisher(
            JointState, f'wave/{h}/joint_states', 10) for h in ('left', 'right')}
        self.sub_cmd = {h: self.create_subscription(
            JointState, f'wave/{h}/joint_commands',
            lambda msg, hh=h: self.on_cmd(msg, hh), 10) for h in ('left', 'right')}
        if feedback_hz > 0:
            self.create_timer(1.0 / feedback_hz, self.publish_states)
        self._n_cmd = 0

    def connect(self):
        self.get_logger().info('Waiting for SharpaWave devices...')
        self.manager = SharpaWaveManager.get_instance()
        time.sleep(1)
        while True:
            infos = [i for i in (self.manager.get_all_devices() or [])
                     if i.device_type == DeviceType.HAND]
            if infos:
                break
            self.get_logger().info('Waiting for device connection...')
            time.sleep(1)
        for info in infos:
            wave = self.manager.connect(info.sn)
            hand = _hand_side(info, wave)
            err = wave.set_control_source(ControlSource.SDK)
            if err.code != 0:
                self.get_logger().warn(f'{info.sn} set_control_source: {err.message}')
            if hasattr(wave, 'set_enable_state'):
                err = wave.set_enable_state(True)
                if err.code != 0:
                    self.get_logger().warn(f'{info.sn} set_enable_state: {err.message}')
            if not wave.start():
                self.get_logger().error(f'{info.sn} failed to start')
                continue
            self._wave_by_hand[hand] = wave
            self.get_logger().info(f'{info.sn} connected + started as {hand} hand')
        if not self._wave_by_hand:
            raise RuntimeError('no SharpaWave hand started')

    def disconnect(self):
        with self._lock:
            for wave in self._wave_by_hand.values():
                try:
                    if hasattr(wave, 'set_enable_state'):
                        wave.set_enable_state(False)
                    wave.stop()
                except Exception as e:
                    self.get_logger().warn(f'stop failed: {e}')
            try:
                self.manager.disconnect_all()
            except Exception:
                pass
            self._wave_by_hand.clear()
        self.get_logger().info('devices disconnected')

    def on_cmd(self, msg, hand):
        wave = self._wave_by_hand.get(hand)
        if wave is None:
            n = self._warned.get(hand, 0) + 1
            self._warned[hand] = n
            if n == 1 or n % 100 == 0:
                self.get_logger().warn(f'no {hand} hand connected ({n} cmds dropped)')
            return
        with self._lock:
            try:
                wave.set_joint_position([float(p) for p in msg.position])
            except Exception as e:
                self.get_logger().error(f'set_joint_position({hand}): {e}')
        self._n_cmd += 1
        if self._n_cmd <= 3 or self._n_cmd % 250 == 0:
            self.get_logger().info(f'[{hand}] cmd #{self._n_cmd}  '
                                   f'q[:5]={[round(p, 2) for p in msg.position[:5]]}')

    def publish_states(self):
        with self._lock:
            for hand, wave in self._wave_by_hand.items():
                try:
                    state = wave.get_states()
                    angles = list(state.angles) if hasattr(state, 'angles') else []
                except Exception:
                    continue
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = JOINT_NAMES[:len(angles)]
                msg.position = [float(a) for a in angles]
                self.pub_js[hand].publish(msg)


def main():
    p = argparse.ArgumentParser(description='SharpaWave joints-only ROS 2 bridge')
    p.add_argument('--feedback_hz', type=float, default=10.0,
                   help='joint-state feedback rate (0 = off)')
    args = p.parse_args()

    rclpy.init(args=sys.argv)
    node = SharpaJointBridge(args.feedback_hz)
    try:
        node.connect()
        node.get_logger().info('SharpaJointBridge running — Ctrl+C to exit')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.disconnect()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
