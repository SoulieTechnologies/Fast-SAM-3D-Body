#!/usr/bin/env python3
"""ROS 2 hand-teleop node for a REAL robot hand (--hand orca|sharpa) — adapted
from the nero_touch safety-filter node design (virtual MPC state, hold-last on
solver failure, staleness watchdog), with the retarget_mpc.py MPC inside.

    SAM3D hand keypoints (TCP, from cosmik_hand_demo --emit-hand-port 8092)
        → palm alignment + scale → fingertip MPC (acados)
        → velocity-clamped JointState on --topic (default: per-hand driver topic
          — orca: /orca/joint_states_target → orca_hand_driver_node.py;
            sharpa: wave/right/joint_commands → the SDK's wave_ros_server.py,
            positions reordered into the SDK's 22-joint index order)

Safety layers (in order):
  - startup ramp: output velocity-clamps from NEUTRAL toward the first target,
    so the hand never jumps on connect (~1 s to converge)
  - per-tick joint velocity clamp (--vmax rad/s) on the PUBLISHED command
  - hard joint limits inside the MPC (never violated by construction)
  - keypoints stale > --stale s  → hold the last command
  - keypoints stale > --release s → drive slowly back to neutral (open hand)
  - solver failure → hold the last command (throttled warning)

The MPC integrates its own virtual state (the solver's internal trajectory);
measured hand feedback is not required — the Orca's dynamixels are position
controlled and low-inertia. Publishes RADIANS with the URDF joint names —
for the real hand, run orca_hand_driver_node.py downstream (it owns the
URDF→orca_core name/unit/sign mapping via joint_map_v1_right.yaml); only
use --joint-map here for some other driver that wants renamed radians.

Run (needs rclpy visible + the acados env — source ROS 2 first, see
nero_touch/launch for the PYTHONPATH pattern):
  export ACADOS_SOURCE_DIR=~/code/comfi-examples-hands/acados
  export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
  export ACADOS_EXT_FUN_COMPILE_FLAGS=-O1
  python hand_teleop_node.py --listen localhost:8092
Dry-run without ROS (prints instead of publishing):
  python hand_teleop_node.py --listen localhost:8092 --no-ros
"""
import argparse
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # retarget_mpc lives next to this file — make it importable from any cwd (ros2 run, launch files)

# Reuse the proven retargeting blocks — nothing re-implemented here.
from retarget_mpc import (FINGERS, HANDS, _RX,  # noqa: E402
                          HandMPC, PalmMapper,
                          build_static_tracking, _FINGER_BASE)


class HandTeleop:
    """MPC retargeting core + safety layers; ROS publishing is pluggable."""

    def __init__(self, args):
        cfg = HANDS[args.hand]
        urdf = args.urdf or cfg.urdf
        print(f"[1/2] Model ({cfg.name}) + mapper...")
        model, _, _ = pin.buildModelsFromUrdf(urdf, str(Path(urdf).parent))
        if cfg.wrist_joint_hint:
            wrist = [model.getJointId(n) for n in model.names
                     if cfg.wrist_joint_hint in n]
            if wrist:
                model = pin.buildReducedModel(model, wrist, pin.neutral(model))
        self.model = model
        self.offsets = {f: cfg.tip_offsets[f].copy() for f in FINGERS}
        self.offsets_flat = np.concatenate([self.offsets[f] for f in FINGERS])
        self.static_track = build_static_tracking(cfg, args.w_mid, args.w_mcp)
        self.mapper = PalmMapper(cfg, model,
                                 float(np.linalg.norm(self.offsets["middle"])))

        print("[2/2] Building acados MPC (~1 min first time)...")
        self.mpc = HandMPC(cfg, model, self.static_track, args.w_tip,
                           N=args.N, dt=args.dt, w_dq=args.w_dq, w_u=args.w_u,
                           self_collision=not args.no_self_collision,
                           col_margin=args.col_margin)
        self.q_neutral = pin.neutral(model)
        self.mpc.warm_start(self.q_neutral)

        self.args = args
        self.q_pub = self.q_neutral.copy()      # last PUBLISHED (vel-clamped) command
        self.last_kp_t = 0.0
        self._was_tracking = False              # were we tracking on the previous tick?
        # publish format: some drivers (sharpa's wave_ros_server) index by
        # POSITION, so reorder q into the driver's joint order when configured
        if cfg.publish_order:
            self.pub_idx = [model.joints[model.getJointId(n)].idx_q
                            for n in cfg.publish_order]
            self.joint_names = list(cfg.publish_names or cfg.publish_order)
        else:
            self.pub_idx = None                 # publish in pinocchio/URDF order
            self.joint_names = list(model.names[1:])
        if args.joint_map:
            import yaml
            m = yaml.safe_load(Path(args.joint_map).read_text())
            self.joint_names = [m.get(n, n) for n in self.joint_names]

    def q_out(self, q):
        """Command vector in the published joint order."""
        return q[self.pub_idx] if self.pub_idx is not None else q

    def emit(self, q):
        """Mirror the command over TCP (--emit-q) — write buf BEFORE n so a
        fresh n is always paired with its matching q (same rule as _rx).
        Explicit little-endian floats (the driver decodes '<f')."""
        _EMITQ["buf"] = self.q_out(q).astype("<f4").tobytes()
        _EMITQ["n"] += 1

    def step(self, now):
        """One control tick → (q_command, status_str). Never raises."""
        a = self.args
        kp = _RX["data"]
        stale = now - self.last_kp_t

        if kp is None or stale > a.release:
            # no tracking (or lost for a while): drift slowly back to neutral
            q_des, status = self.q_neutral, "neutral"
            vmax = a.vmax / 4.0
            self._was_tracking = False
        elif stale > a.stale:
            self._was_tracking = False
            return self.q_pub, "hold(stale)"       # brief dropout: freeze
        else:
            targets = self.mapper(kp)
            if targets is None:
                self._was_tracking = False
                return self.q_pub, "hold(bad kp)"
            tip_t = np.array([targets[_FINGER_BASE[f]] for f in FINGERS])
            static_t = np.array([targets[i] for _, i, _ in self.static_track])
            # Resync the MPC's virtual state to the actually-published pose when
            # tracking (re)starts: while neutral/holding, q_pub drifts but the
            # solver's internal trajectory stays frozen at the last tracked
            # solution — resuming from that stale state would kick the first step.
            if not self._was_tracking:
                self.mpc.warm_start(self.q_pub)
            try:
                q_des = self.mpc.solve(tip_t, static_t, self.offsets_flat)
                status = "track"
                self._was_tracking = True
            except Exception as e:                  # solver failure → hold last
                self._was_tracking = False
                return self.q_pub, f"hold(solver: {e})"
            vmax = a.vmax

        # velocity clamp on the published command — also the startup ramp
        step = np.clip(q_des - self.q_pub, -vmax * a.dt, vmax * a.dt)
        self.q_pub = np.clip(self.q_pub + step,
                             self.model.lowerPositionLimit,
                             self.model.upperPositionLimit)
        return self.q_pub, status


# ── optional TCP re-emit of the published command (--emit-q) ─────────────────
# Same [>I frame][n×float32] framing as the keypoint stream. Lets a driver in
# a DIFFERENT python env (e.g. sharpa_tcp_driver.py on the system python with
# the Wave SDK) consume the commands without sharing rclpy with the acados env.

_EMITQ = {"buf": b"", "n": 0}


def _emitq_server(port):
    srv = socket.create_server(("0.0.0.0", port))
    print(f"emitting q on tcp :{port}")
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=_emitq_client, args=(conn,), daemon=True).start()


def _emitq_client(conn):
    last = -1
    try:
        while True:
            if _EMITQ["n"] != last and _EMITQ["buf"]:
                last = _EMITQ["n"]
                conn.sendall(struct.pack(">I", last) + _EMITQ["buf"])
            time.sleep(0.004)
    except OSError:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# ROS 2 wrapper (same shape as nero_touch/nero/safety/node.py)
# ═══════════════════════════════════════════════════════════════════════════

def run_ros(core, args):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class OrcaHandTeleopNode(Node):
        def __init__(self):
            super().__init__("orca_hand_teleop")
            self.pub = self.create_publisher(JointState, args.topic, 10)
            self.create_timer(args.dt, self.tick)
            self._last_status = ""

        def tick(self):
            q, status = core.step(time.time())
            core.emit(q)
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = core.joint_names
            msg.position = core.q_out(q).tolist()
            self.pub.publish(msg)
            if status != self._last_status:
                self.get_logger().info(f"state: {status}")
                self._last_status = status

    rclpy.init()
    print(f"publishing JointState on {args.topic} at {1/args.dt:.0f} Hz")
    rclpy.spin(OrcaHandTeleopNode())


def run_dry(core, args):
    """No-ROS loop: prints a heartbeat; still emits over --emit-q if set."""
    print(f"NO-ROS — {'emitting q over tcp' if args.emit_q else 'dry run'} "
          f"at {1/args.dt:.0f} Hz (topic {args.topic} unused)")
    n = 0
    while True:
        t0 = time.time()
        q, status = core.step(t0)
        core.emit(q)
        n += 1
        if n % 25 == 0:
            print(f"  [{status}] q[:4]={np.round(q[:4], 2).tolist()}", flush=True)
        time.sleep(max(0.0, args.dt - (time.time() - t0)))


def main():
    p = argparse.ArgumentParser(description="Orca hand teleop ROS 2 node (MPC retargeting)")
    p.add_argument("--listen", default="localhost:8092",
                   help="host:port of cosmik_hand_demo --emit-hand-port")
    p.add_argument("--hand", choices=sorted(HANDS), default="orca")
    p.add_argument("--urdf", default="", help="override the hand config's URDF")
    p.add_argument("--topic", default="",
                   help="publish topic (default: the hand config's driver topic)")
    p.add_argument("--joint-map", default="",
                   help="yaml {urdf_joint_name: driver_joint_name} for the real hand")
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--w-tip", type=float, default=50.0)
    p.add_argument("--w-mid", type=float, default=2.0)
    p.add_argument("--w-mcp", type=float, default=0.1)
    p.add_argument("--w-dq", type=float, default=1e-3)
    p.add_argument("--w-u", type=float, default=1e-4)
    p.add_argument("--no-self-collision", action="store_true",
                   help="drop the self-collision constraints from the MPC")
    p.add_argument("--col-margin", type=float, default=0.0,
                   help="extra safety margin (m) added to every sphere pair")
    p.add_argument("--vmax", type=float, default=3.0,
                   help="max published joint velocity (rad/s) — also the startup ramp")
    p.add_argument("--stale", type=float, default=0.5,
                   help="hold the last command when keypoints are older than this (s)")
    p.add_argument("--release", type=float, default=3.0,
                   help="drift back to neutral when keypoints are older than this (s)")
    p.add_argument("--no-ros", action="store_true", help="run without rclpy")
    p.add_argument("--emit-q", type=int, default=0,
                   help="if >0: also stream the published command over TCP on "
                        "this port ([>I frame][n×float32], publish joint order) "
                        "— lets sharpa_tcp_driver.py run without ROS")
    args = p.parse_args()
    if not args.topic:
        args.topic = HANDS[args.hand].default_topic

    core = HandTeleop(args)
    if args.emit_q:
        threading.Thread(target=_emitq_server, args=(args.emit_q,),
                         daemon=True).start()

    host, port = args.listen.rsplit(":", 1)
    rx = threading.Thread(target=_rx_wrapper, args=(host, int(port), core), daemon=True)
    rx.start()

    if args.no_ros:
        run_dry(core, args)
    else:
        run_ros(core, args)


def _rx_wrapper(host, port, core):
    """Wrap retarget_mpc's receiver to stamp arrival times for the watchdog."""
    import retarget_mpc as R
    last_n = -1

    def watcher():
        nonlocal last_n
        while True:
            if R._RX["n"] != last_n:
                last_n = R._RX["n"]
                core.last_kp_t = time.time()
            time.sleep(0.005)

    threading.Thread(target=watcher, daemon=True).start()
    R._rx_thread(host, port)


if __name__ == "__main__":
    main()
