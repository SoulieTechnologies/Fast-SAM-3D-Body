#!/usr/bin/env python3
"""ROS 2 driver node for the REAL Orca hand (orca_core over USB dynamixel).

    JointState on --topic (from hand_teleop_node.py: URDF names, radians)
        → joint_map yaml (URDF name → orca_core name, sign, offset, rad→deg)
        → per-tick velocity clamp (deg/s)
        → OrcaHand.set_joint_pos({orca_name: deg})

This node deliberately knows NOTHING about the MPC — it only needs rclpy +
orca_core (+ pyyaml/numpy), so it runs in the lab's orca_core env while
hand_teleop_node runs in the acados env. The two only meet on the ROS topic.

Safety layers (in order):
  - first command starts from the hand's MEASURED pose (get_joint_pos), so
    the velocity clamp ramps smoothly from wherever the hand actually is
  - per-tick joint velocity clamp (--vmax-deg, deg/s) on every command
  - targets clipped to the calibrated joint ROMs (orca_core clips again)
  - NaN / unmapped joints skipped (the locked wrist is never commanded)
  - no target yet → no writes (torque-on hold at the current pose)
  - targets stale > --idle s (teleop node died) → ramp to config neutral
  - shutdown / crash → disable torque + disconnect (finally block)

Run (orca_core env, ROS 2 sourced; hand must be calibrated once beforehand
with orca_core/scripts/calibrate.py):
  python orca_hand_driver_node.py --model ~/code/orca_core/orca_core/models/orcahand_v1_right
Dry-run without the hand (prints the deg commands, validates topic+mapping):
  python orca_hand_driver_node.py --dry
"""
import argparse
import math
import threading
import time
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent


class FakeHand:
    """--dry stand-in for OrcaHand: same surface, prints instead of moving."""
    joint_ids = ["thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip",
                 "index_abd", "index_mcp", "index_pip",
                 "middle_abd", "middle_mcp", "middle_pip",
                 "ring_abd", "ring_mcp", "ring_pip",
                 "pinky_abd", "pinky_mcp", "pinky_pip", "wrist"]
    joint_roms_dict = {j: [-60.0, 110.0] for j in joint_ids}
    neutral_position = {j: 0.0 for j in joint_ids}

    def connect(self): return True, "dry run (no hardware)"
    def enable_torque(self): pass
    def disconnect(self): return True, "dry"
    def get_joint_pos(self, as_list=True): return {j: 0.0 for j in self.joint_ids}

    def set_joint_pos(self, pos, **kw):
        self._last = pos  # printed by the driver heartbeat


class OrcaHandDriver:
    """Mapping + safety core; ROS subscription is pluggable (see run_ros)."""

    def __init__(self, args):
        self.args = args
        raw = yaml.safe_load(Path(args.map).read_text())
        # urdf_name -> (orca_name, sign, offset_deg)
        self.jmap = {u: (m["joint"], float(m.get("sign", 1.0)),
                         float(m.get("offset_deg", 0.0))) for u, m in raw.items()}

        if args.dry:
            self.hand = FakeHand()
        else:
            from orca_core import OrcaHand
            self.hand = OrcaHand(args.model or None)
        ok, msg = self.hand.connect()
        if not ok:
            raise RuntimeError(f"OrcaHand connect failed: {msg}")
        print(f"connected: {msg}")
        self.hand.enable_torque()

        mapped = {m[0] for m in self.jmap.values()}
        unknown = mapped - set(self.hand.joint_ids)
        if unknown:
            raise RuntimeError(f"joint map targets unknown orca joints: {unknown}")
        self.roms = {j: self.hand.joint_roms_dict[j] for j in mapped}
        self.neutral = {j: float(self.hand.neutral_position.get(j, 0.0)) for j in mapped}

        # start the ramp from the MEASURED pose so the first command never jumps
        meas = self.hand.get_joint_pos(as_list=False)
        self.cmd = {j: float(meas[j]) for j in mapped}      # last commanded, deg
        print(f"driving {len(mapped)} joints from measured pose "
              f"(wrist untouched); vmax {args.vmax_deg:.0f} deg/s")

        self._lock = threading.Lock()
        self._target = None          # {urdf_name: rad}, latest received
        self._target_t = 0.0

    def on_joint_state(self, names, positions):
        with self._lock:
            self._target = dict(zip(names, positions))
            self._target_t = time.time()

    def tick(self, now):
        """One write tick → status_str. Never raises past the serial write."""
        a = self.args
        with self._lock:
            target, t = self._target, self._target_t

        if target is None:
            return "waiting(no target yet)"       # torque-on hold, no writes
        if now - t > a.idle:
            des, status = dict(self.neutral), "idle→neutral"
        else:
            des, status = {}, "track"
            for u, rad in target.items():
                m = self.jmap.get(u)
                if m is None or not math.isfinite(rad):
                    continue                       # wrist / unknown / NaN
                name, sign, off = m
                lo, hi = self.roms[name]
                des[name] = min(max(sign * math.degrees(rad) + off, lo), hi)
            if not des:
                return "hold(no mapped joints in msg)"

        # per-tick velocity clamp on every commanded joint — also the ramp
        step = a.vmax_deg * a.dt
        out = {}
        for name, d in des.items():
            c = self.cmd[name]
            out[name] = c + min(max(d - c, -step), step)
        self.hand.set_joint_pos(out)
        self.cmd.update(out)
        return status

    def shutdown(self):
        try:
            self.hand.disconnect()                 # orca_core disables torque
            print("torque disabled, disconnected")
        except Exception as e:
            print(f"disconnect failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# ROS 2 wrapper (same shape as hand_teleop_node.py)
# ═══════════════════════════════════════════════════════════════════════════

def run_ros(core, args):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class OrcaHandDriverNode(Node):
        def __init__(self):
            super().__init__("orca_hand_driver")
            self.create_subscription(JointState, args.topic, self.on_msg, 10)
            self.create_timer(args.dt, self.on_timer)
            self._last_status = ""

        def on_msg(self, msg):
            core.on_joint_state(msg.name, msg.position)

        def on_timer(self):
            status = core.tick(time.time())
            if status != self._last_status:
                self.get_logger().info(f"state: {status}")
                self._last_status = status

    rclpy.init()
    print(f"listening for JointState on {args.topic}, writing at {1/args.dt:.0f} Hz")
    try:
        rclpy.spin(OrcaHandDriverNode())
    finally:
        core.shutdown()


def run_dry_loop(core, args):
    """--dry --no-ros: feed a slow synthetic flexion wave through the full
    mapping+clamp path and print what would be written (no rclpy needed)."""
    urdf_names = list(core.jmap.keys())
    print("DRY LOOP — synthetic wave through the mapping (Ctrl-C to stop)")
    n = 0
    try:
        while True:
            t = n * args.dt
            rad = 0.5 * (1 - math.cos(0.8 * t))    # 0 → ~1 rad flexion wave
            core.on_joint_state(urdf_names, [rad] * len(urdf_names))
            status = core.tick(time.time())
            if n % 25 == 0:
                sent = getattr(core.hand, "_last", {})
                head = {k: round(v, 1) for k, v in list(sent.items())[:4]}
                print(f"  [{status}] {head} ...", flush=True)
            n += 1
            time.sleep(args.dt)
    finally:
        core.shutdown()


def main():
    p = argparse.ArgumentParser(description="Orca hand ROS 2 driver (orca_core)")
    p.add_argument("--model", default="",
                   help="orcahand model folder for OrcaHand() (default: orca_core's default)")
    p.add_argument("--map", default=str(_HERE / "joint_map_v1_right.yaml"),
                   help="yaml {urdf_joint: {joint, sign, offset_deg}}")
    p.add_argument("--topic", default="/orca/joint_states_target")
    p.add_argument("--dt", type=float, default=0.04, help="serial write period (s)")
    p.add_argument("--vmax-deg", type=float, default=200.0,
                   help="max commanded joint velocity (deg/s) — also the ramp")
    p.add_argument("--idle", type=float, default=3.0,
                   help="ramp to config neutral when targets are older than this (s)")
    p.add_argument("--dry", action="store_true", help="no hardware, print commands")
    p.add_argument("--no-ros", action="store_true",
                   help="with --dry: synthetic wave instead of a ROS subscription")
    args = p.parse_args()

    if args.no_ros and not args.dry:
        # check BEFORE constructing the driver — it connects + enables torque
        raise SystemExit("--no-ros requires --dry (never wave the real hand blind)")

    core = OrcaHandDriver(args)
    if args.no_ros:
        run_dry_loop(core, args)
    else:
        run_ros(core, args)


if __name__ == "__main__":
    main()
