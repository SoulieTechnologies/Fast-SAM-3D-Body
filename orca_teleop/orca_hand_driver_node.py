#!/usr/bin/env python3
"""Driver node for the REAL Orca hand (orca_core over USB dynamixel).

    JointState on --topic (ROS 2)  OR  q stream on --listen-q (plain TCP,
    from hand_teleop_node.py --no-ros --emit-q: radians, joint_map key order)
        → joint_map yaml (URDF name → orca_core name, sign, offset, rad→deg)
        → per-tick velocity clamp (deg/s)
        → OrcaHand.set_joint_pos({orca_name: deg})

This node deliberately knows NOTHING about the MPC — it only needs orca_core
(+ pyyaml, and rclpy only for the ROS mode), so it runs wherever the hand is
plugged in (the orca_core venv) while hand_teleop_node runs in the acados
env, possibly on another machine. The TCP mode exists because rclpy is not
importable from the acados env (py3.11) nor on the Mac — same hop as
sharpa_tcp_driver.py. Index alignment of the TCP stream is by construction:
hand_teleop_node publishes in _orca_publish_order() = the key order of the
SAME joint_map yaml this driver maps with.

Safety layers (in order):
  - first command starts from the hand's MEASURED pose (get_joint_pos), so
    the velocity clamp ramps smoothly from wherever the hand actually is
  - per-tick joint velocity clamp (--vmax-deg, deg/s) on every command
  - targets clipped to the calibrated joint ROMs (orca_core clips again)
  - NaN / unmapped / out-of-range (|q|>π) joints skipped (the locked wrist
    is never commanded)
  - no target yet → no writes (torque-on hold at the current pose)
  - targets stale > --idle s (teleop node died) → ramp to config neutral
  - shutdown / crash → disable torque + disconnect (finally block)

Run — TCP mode, hand on this machine's USB (orca_core venv; hand calibrated
once beforehand with orca_core/scripts/calibrate.py), teleop node elsewhere:
  python orca_hand_driver_node.py --listen-q crslab:8093
Full offline rehearsal (simulated motors, REAL config+calibration mapping):
  python orca_hand_driver_node.py --mock --listen-q localhost:8093
  python orca_hand_driver_node.py --mock --no-ros      # synthetic wave
ROS mode (orca_core env WITH rclpy, e.g. hand plugged into the lab PC):
  python orca_hand_driver_node.py
Dry-run without orca_core at all (prints commands, validates the mapping):
  python orca_hand_driver_node.py --dry --no-ros
"""
import argparse
import math
import socket
import struct
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
        elif args.mock:
            from orca_core import MockOrcaHand
            self.hand = MockOrcaHand(args.model or None)
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
        uncal = sorted(j for j in mapped if meas.get(j) is None)
        if uncal:
            raise RuntimeError(
                f"joints without a calibrated reading: {uncal} — run "
                "orca_core/scripts/calibrate.py before teleop")
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
                if m is None or not math.isfinite(rad) or abs(rad) > 3.2:
                    continue      # wrist / unknown / NaN / garbage (>π rad)
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
# TCP wrapper — consumes hand_teleop_node --emit-q, no ROS anywhere
# (same wire format + reconnect loop as sharpa_tcp_driver.py)
# ═══════════════════════════════════════════════════════════════════════════

def _rx_q_thread(host, port, names, core):
    """[>I frame][len(names)×'<f4' radians] → core.on_joint_state.
    Positions are index-aligned with `names` (joint_map key order — the node
    publishes in _orca_publish_order(), read from the same yaml)."""
    nmsg = 4 + len(names) * 4
    fmt = f"<{len(names)}f"
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            print(f"connected to {host}:{port}")
            buf = b""
            while True:
                d = s.recv(65536)
                if not d:
                    raise ConnectionError
                buf += d
                msg = None
                while len(buf) >= nmsg:            # keep only the newest frame
                    msg = buf[:nmsg]
                    buf = buf[nmsg:]
                if msg is not None:
                    core.on_joint_state(names, struct.unpack(fmt, msg[4:]))
        except OSError:
            time.sleep(1.0)


def run_tcp(core, args):
    host, port = args.listen_q.rsplit(":", 1)
    names = list(core.jmap.keys())                 # yaml insertion order
    threading.Thread(target=_rx_q_thread, args=(host, int(port), names, core),
                     daemon=True).start()
    print(f"listening for q on tcp {host}:{port} ({len(names)} joints), "
          f"writing at {1/args.dt:.0f} Hz")
    last_status, n = "", 0
    try:
        while True:
            t0 = time.time()
            status = core.tick(t0)
            if status != last_status:
                print(f"state: {status}", flush=True)
                last_status = status
            n += 1
            if n % 250 == 0:
                head = {k: round(v, 1) for k, v in list(core.cmd.items())[:4]}
                print(f"  [{status}] {head} ...", flush=True)
            time.sleep(max(0.0, args.dt - (time.time() - t0)))
    finally:
        core.shutdown()


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
                head = {k: round(v, 1) for k, v in list(core.cmd.items())[:4]}
                print(f"  [{status}] {head} ...", flush=True)
            n += 1
            time.sleep(args.dt)
    finally:
        core.shutdown()


def main():
    p = argparse.ArgumentParser(description="Orca hand ROS 2 driver (orca_core)")
    p.add_argument("--model", default="orcahand_v1_right",
                   help="orcahand model name or folder for OrcaHand() — NB "
                        "orca_core's own default sorts orcahand_v1_LEFT first")
    p.add_argument("--map", default=str(_HERE / "joint_map_v1_right.yaml"),
                   help="yaml {urdf_joint: {joint, sign, offset_deg}}")
    p.add_argument("--topic", default="/orca/joint_states_target")
    p.add_argument("--listen-q", default="",
                   help="host:port of hand_teleop_node --emit-q → consume the "
                        "command stream over TCP instead of a ROS topic")
    p.add_argument("--mock", action="store_true",
                   help="orca_core MockOrcaHand: simulated motors, REAL "
                        "config/calibration mapping (full rehearsal, no hand)")
    p.add_argument("--dt", type=float, default=0.04, help="serial write period (s)")
    p.add_argument("--vmax-deg", type=float, default=200.0,
                   help="max commanded joint velocity (deg/s) — also the ramp")
    p.add_argument("--idle", type=float, default=3.0,
                   help="ramp to config neutral when targets are older than this (s)")
    p.add_argument("--dry", action="store_true", help="no hardware, print commands")
    p.add_argument("--no-ros", action="store_true",
                   help="with --dry: synthetic wave instead of a ROS subscription")
    args = p.parse_args()

    # checks BEFORE constructing the driver — it connects + enables torque
    if args.no_ros and not (args.dry or args.mock):
        raise SystemExit("--no-ros requires --dry or --mock "
                         "(never wave the real hand blind)")
    if args.dry and args.mock:
        raise SystemExit("--dry and --mock are mutually exclusive")

    core = OrcaHandDriver(args)
    try:
        if args.no_ros:
            run_dry_loop(core, args)
        elif args.listen_q:
            run_tcp(core, args)
        else:
            run_ros(core, args)
    except KeyboardInterrupt:
        pass                       # shutdown ran in the loop's finally block


if __name__ == "__main__":
    main()
