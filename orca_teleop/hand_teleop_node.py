#!/usr/bin/env python3
"""ROS 2 hand-teleop node for the REAL Orca hand — adapted from the nero_touch
safety-filter node design (virtual MPC state, hold-last on solver failure,
staleness watchdog), with the hand-retargeting MPC of retarget_mpc.py inside.

    SAM3D hand keypoints (TCP, from cosmik_hand_demo --emit-hand-port 8092)
        → palm alignment + scale → fingertip MPC (acados)
        → velocity-clamped JointState on --topic (default /orca/joint_states_target)

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
controlled and low-inertia. Joint names default to the URDF names; pass
--joint-map map.yaml ({urdf_name: driver_name}) to rename for your driver.

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
import threading
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

# Reuse the proven retargeting blocks — nothing re-implemented here.
from retarget_mpc import (FINGERS, TIP_OFFSETS_CALIB, WRIST_JOINT_HINT, _RX,
                          HandMPC, PalmMapper,
                          build_static_tracking, _FINGER_BASE)

_HERE = Path(__file__).resolve().parent


class HandTeleop:
    """MPC retargeting core + safety layers; ROS publishing is pluggable."""

    def __init__(self, args):
        print("[1/2] Model + mapper...")
        model, _, _ = pin.buildModelsFromUrdf(args.urdf, str(Path(args.urdf).parent))
        wrist = [model.getJointId(n) for n in model.names if WRIST_JOINT_HINT in n]
        if wrist:
            model = pin.buildReducedModel(model, wrist, pin.neutral(model))
        self.model = model
        self.offsets = {f: TIP_OFFSETS_CALIB[f].copy() for f in FINGERS}
        self.offsets_flat = np.concatenate([self.offsets[f] for f in FINGERS])
        self.static_track = build_static_tracking(args.w_mid, args.w_mcp)
        self.mapper = PalmMapper(model, float(np.linalg.norm(self.offsets["middle"])))

        print("[2/2] Building acados MPC (~1 min first time)...")
        self.mpc = HandMPC(model, self.static_track, args.w_tip,
                           N=args.N, dt=args.dt, w_dq=args.w_dq, w_u=args.w_u)
        self.q_neutral = pin.neutral(model)
        self.mpc.warm_start(self.q_neutral)

        self.args = args
        self.q_pub = self.q_neutral.copy()      # last PUBLISHED (vel-clamped) command
        self.last_kp_t = 0.0
        self.joint_names = list(model.names[1:])
        if args.joint_map:
            import yaml
            m = yaml.safe_load(Path(args.joint_map).read_text())
            self.joint_names = [m.get(n, n) for n in self.joint_names]

    def step(self, now):
        """One control tick → (q_command, status_str). Never raises."""
        a = self.args
        kp = _RX["kp"]
        stale = now - self.last_kp_t

        if kp is None or stale > a.release:
            # no tracking (or lost for a while): drift slowly back to neutral
            q_des, status = self.q_neutral, "neutral"
            vmax = a.vmax / 4.0
        elif stale > a.stale:
            return self.q_pub, "hold(stale)"       # brief dropout: freeze
        else:
            targets = self.mapper(kp)
            if targets is None:
                return self.q_pub, "hold(bad kp)"
            tip_t = np.array([targets[_FINGER_BASE[f]] for f in FINGERS])
            static_t = np.array([targets[i] for _, i, _ in self.static_track])
            try:
                q_des = self.mpc.solve(tip_t, static_t, self.offsets_flat)
                status = "track"
            except Exception as e:                  # solver failure → hold last
                return self.q_pub, f"hold(solver: {e})"
            vmax = a.vmax

        # velocity clamp on the published command — also the startup ramp
        step = np.clip(q_des - self.q_pub, -vmax * a.dt, vmax * a.dt)
        self.q_pub = np.clip(self.q_pub + step,
                             self.model.lowerPositionLimit,
                             self.model.upperPositionLimit)
        return self.q_pub, status


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
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = core.joint_names
            msg.position = q.tolist()
            self.pub.publish(msg)
            if status != self._last_status:
                self.get_logger().info(f"state: {status}")
                self._last_status = status

    rclpy.init()
    print(f"publishing JointState on {args.topic} at {1/args.dt:.0f} Hz")
    rclpy.spin(OrcaHandTeleopNode())


def run_dry(core, args):
    """No-ROS dry run: same loop, prints a heartbeat instead of publishing."""
    print(f"DRY RUN — would publish on {args.topic} at {1/args.dt:.0f} Hz")
    n = 0
    while True:
        t0 = time.time()
        q, status = core.step(t0)
        n += 1
        if n % 25 == 0:
            print(f"  [{status}] q[:4]={np.round(q[:4], 2).tolist()}", flush=True)
        time.sleep(max(0.0, args.dt - (time.time() - t0)))


def main():
    p = argparse.ArgumentParser(description="Orca hand teleop ROS 2 node (MPC retargeting)")
    p.add_argument("--listen", default="localhost:8092",
                   help="host:port of cosmik_hand_demo --emit-hand-port")
    p.add_argument("--urdf", default=str(_HERE / "orcahand" / "orcahand_right.urdf"))
    p.add_argument("--topic", default="/orca/joint_states_target")
    p.add_argument("--joint-map", default="",
                   help="yaml {urdf_joint_name: driver_joint_name} for the real hand")
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--w-tip", type=float, default=50.0)
    p.add_argument("--w-mid", type=float, default=2.0)
    p.add_argument("--w-mcp", type=float, default=0.1)
    p.add_argument("--w-dq", type=float, default=1e-3)
    p.add_argument("--w-u", type=float, default=1e-4)
    p.add_argument("--vmax", type=float, default=3.0,
                   help="max published joint velocity (rad/s) — also the startup ramp")
    p.add_argument("--stale", type=float, default=0.5,
                   help="hold the last command when keypoints are older than this (s)")
    p.add_argument("--release", type=float, default=3.0,
                   help="drift back to neutral when keypoints are older than this (s)")
    p.add_argument("--no-ros", action="store_true", help="dry run without rclpy")
    args = p.parse_args()

    core = HandTeleop(args)

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
