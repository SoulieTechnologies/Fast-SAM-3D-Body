#!/usr/bin/env python3
"""Orca-hand teleoperation retargeting — SAM3D hand keypoints → Orca joint angles
via an acados MPC (sliding-horizon IK), visualised in meshcat.

Input: 21 right-hand keypoints (SAM3D hand-decoder order, wrist-relative 3D):
    local idx  0..3   thumb  [tip, DIP, PIP, MCP]
               4..7   index  [tip, DIP, PIP, MCP]
               8..11  middle          "
               12..15 ring            "
               16..19 pinky           "
               20     wrist
The palm orientation is normalised每 frame (human palm basis → Orca palm basis),
so only finger ARTICULATION is retargeted — rotating your whole hand does not
rotate the robot fingers. Targets are globally scaled to the Orca's finger
lengths (the robot cannot be resized — week-1 lesson).

MPC (same design as the comfi ACADOS IK): state x=[q,dq], control u=ddq,
cost = Σ w_i·||FK_i(q)−p_i||² + w_dq·||dq||² + w_u·||u||², hard joint limits.
Weights follow the week-1 tuning: fingertips 50, mid-phalanges 2, MCP 0.1.

Run (acados env):
  export ACADOS_SOURCE_DIR=~/code/comfi-examples-hands/acados
  export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
  export ACADOS_EXT_FUN_COMPILE_FLAGS=-O1
  # replay a cosmik_hand_demo recording:
  python retarget_mpc.py --replay ../output_cosmik_demo/<ts>/goliath70_3d.npy
  # live from cosmik_hand_demo --emit-hand-port 8092:
  python retarget_mpc.py --listen localhost:8092
"""

import argparse
import os
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

_HERE = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════
# HUMAN(21) ↔ ORCA FRAME MAPPING  (right hand)
# ═══════════════════════════════════════════════════════════════════════════
# Finger order across the Orca palm (from joint origins): thumb T-, index I-,
# middle M-*e04a96f2/08efa608/34afb748, ring M-*6ec59111/8660a1eb/424a8e75,
# pinky P-.  Chains: AP (MCP) → PP (PIP) → FingerTipAssembly (distal).

_F = {
    "thumb":  {"mcp": "R-T-AP_a9723101", "pip": "T-PP_68395e98",
               "dist": "T-DP_b7429e50"},
    "index":  {"mcp": "I-AP-R_d95d02d1", "pip": "I-PP_bacbd481",
               "dist": "I-FingerTipAssembly_ec49c16c"},
    "middle": {"mcp": "M-AP_e04a96f2", "pip": "M-PP_08efa608",
               "dist": "M-FingerTipAssembly_34afb748"},
    "ring":   {"mcp": "M-AP_6ec59111", "pip": "M-PP_8660a1eb",
               "dist": "M-FingerTipAssembly_424a8e75"},
    "pinky":  {"mcp": "P-AP_f5e42b61", "pip": "P-PP_1d411b9b",
               "dist": "P-FingerTipAssembly_cd219176"},
}
_FINGER_BASE = {"thumb": 0, "index": 4, "middle": 8, "ring": 12, "pinky": 16}
PALM_LINK = "R-Carpals_8d1f1041"
WRIST_JOINT_HINT = "to_TopTower"          # the wrist joint name contains this

# (tracked frame name, human local idx, weight) — week-1 weight scheme
def build_tracking(tip_weight, mid_weight, mcp_weight):
    track = []
    for f, base in _FINGER_BASE.items():
        track.append((f"{f}_tip", base + 0, tip_weight))       # OP frame (added)
        track.append((_F[f]["dist"], base + 1, mid_weight))    # distal link origin
        track.append((_F[f]["pip"], base + 2, mid_weight))
        track.append((_F[f]["mcp"], base + 3, mcp_weight))
    return track


def add_tip_frames(model, tip_offset):
    """Add an OP frame '<finger>_tip' beyond each distal link, along the
    PIP→distal direction at neutral pose (automatic, no per-link axis guessing)."""
    data = model.createData()
    q0 = pin.neutral(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    for f in _F:
        fid_d = model.getFrameId(_F[f]["dist"])
        fid_p = model.getFrameId(_F[f]["pip"])
        p_d = data.oMf[fid_d].translation
        p_p = data.oMf[fid_p].translation
        u = p_d - p_p
        u = u / (np.linalg.norm(u) + 1e-12)
        tip_world = p_d + tip_offset * u
        p_local = data.oMf[fid_d].rotation.T @ (tip_world - p_d)
        frame_d = model.frames[fid_d]
        placement = frame_d.placement * pin.SE3(np.eye(3), p_local)
        model.addFrame(pin.Frame(f"{f}_tip", frame_d.parentJoint, fid_d,
                                 placement, pin.FrameType.OP_FRAME,
                                 pin.Inertia.Zero()), False)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# PALM ALIGNMENT + SCALE  (human wrist-relative kp → Orca palm frame)
# ═══════════════════════════════════════════════════════════════════════════

def _basis(fwd, lat):
    f = fwd / (np.linalg.norm(fwd) + 1e-12)
    l = lat - f * (lat @ f)
    l = l / (np.linalg.norm(l) + 1e-12)
    return np.stack([f, l, np.cross(f, l)], axis=1)          # columns


class PalmMapper:
    """Maps wrist-relative human keypoints into Orca palm-frame 3D targets."""

    def __init__(self, model, tip_offset):
        data = model.createData()
        pin.forwardKinematics(model, data, pin.neutral(model))
        pin.updateFramePlacements(model, data)
        P = lambda n: data.oMf[model.getFrameId(n)].translation.copy()
        self.palm = P(PALM_LINK)
        self.R_orca = _basis(P(_F["middle"]["mcp"]) - self.palm,
                             P(_F["pinky"]["mcp"]) - P(_F["index"]["mcp"]))
        # Orca middle-finger chain length (for the global scale)
        self.orca_len = (np.linalg.norm(P(_F["middle"]["mcp"]) - self.palm)
                         + np.linalg.norm(P(_F["middle"]["pip"]) - P(_F["middle"]["mcp"]))
                         + np.linalg.norm(P(_F["middle"]["dist"]) - P(_F["middle"]["pip"]))
                         + tip_offset)
        self.scale = None

    def __call__(self, kp21):
        """kp21: (21,3) wrist-relative human keypoints → (21,3) Orca-frame targets."""
        k = kp21 - kp21[20]
        if not np.isfinite(k[[8, 9, 10, 11]]).all():
            return None
        if self.scale is None:
            human_len = (np.linalg.norm(k[11]) + np.linalg.norm(k[10] - k[11])
                         + np.linalg.norm(k[9] - k[10]) + np.linalg.norm(k[8] - k[9]))
            if human_len < 1e-6:
                return None
            self.scale = self.orca_len / human_len
            print(f"  human→orca scale locked: {self.scale:.3f} "
                  f"(orca chain {self.orca_len*100:.1f} cm)")
        if not np.isfinite(k[[7, 19]]).all():
            return None
        R_h = _basis(k[11], k[19] - k[7])                    # fwd: wrist→middle MCP
        R = self.R_orca @ R_h.T
        return self.palm + (self.scale * (R @ k.T)).T


# ═══════════════════════════════════════════════════════════════════════════
# ACADOS MPC  (double integrator on q, marker tracking — comfi design)
# ═══════════════════════════════════════════════════════════════════════════

class HandMPC:
    def __init__(self, model, track, N=8, dt=0.04, w_dq=1e-3, w_u=1e-4):
        import casadi
        import pinocchio.casadi as cpin
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

        self.model = model
        self.track = [(n, i, w) for n, i, w in track if model.existFrame(n)]
        assert len(self.track) == len(track), "missing frames in the model!"
        self.N, self.dt = N, dt
        nq, nv = model.nq, model.nv
        self.nq, self.nv = nq, nv
        nmc = 3 * len(self.track)

        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        cx = casadi.SX.sym("x", nq + nv)
        cu = casadi.SX.sym("u", nv)
        cq, cdq = cx[:nq], cx[nq:]
        x_next = casadi.vertcat(cpin.integrate(cmodel, cq, cdq * dt), cdq + cu * dt)
        cpin.framesForwardKinematics(cmodel, cdata, cq)
        markers = casadi.vertcat(*[cdata.oMf[cmodel.getFrameId(n)].translation
                                   for n, _, _ in self.track])

        am = AcadosModel()
        am.name = "orca_hand_mpc"
        am.x, am.u = cx, cu
        am.disc_dyn_expr = x_next
        am.cost_y_expr = casadi.vertcat(markers, cdq, cu)
        am.cost_y_expr_e = cdq
        am.p = casadi.SX.sym("p", nmc)

        ocp = AcadosOcp()
        ocp.model = am
        ocp.solver_options.N_horizon = N
        ocp.solver_options.tf = N * dt
        ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
        ny = nmc + nv + nv
        ocp.cost.yref = np.zeros(ny)
        W = np.zeros((ny, ny))
        for i, (_, _, w) in enumerate(self.track):
            W[3 * i:3 * i + 3, 3 * i:3 * i + 3] = w * np.eye(3)
        W[nmc:nmc + nv, nmc:nmc + nv] = w_dq * np.eye(nv)
        W[nmc + nv:, nmc + nv:] = w_u * np.eye(nv)
        ocp.cost.W = W
        self._W = W
        ocp.cost.yref_e = np.zeros(nv)
        ocp.cost.W_e = w_dq * np.eye(nv)

        am.con_h_expr = cq
        ocp.constraints.lh = np.array(model.lowerPositionLimit)
        ocp.constraints.uh = np.array(model.upperPositionLimit)
        ocp.constraints.x0 = np.zeros(nq + nv)
        ocp.parameter_values = np.zeros(nmc)

        so = ocp.solver_options
        so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        so.hessian_approx = "GAUSS_NEWTON"
        so.integrator_type = "DISCRETE"
        so.nlp_solver_type = "SQP"
        so.nlp_solver_max_iter = 10
        so.tol = 1e-4
        so.ext_fun_compile_flags = os.environ.get("ACADOS_EXT_FUN_COMPILE_FLAGS", "-O1")
        self.solver = AcadosOcpSolver(ocp)
        self.nmc = nmc

    def warm_start(self, q):
        x0 = np.zeros(self.nq + self.nv)
        x0[:self.nq] = q
        for k in range(self.N + 1):
            self.solver.set(k, "x", x0)

    def solve(self, targets):
        """targets: (n_track, 3) — the CURRENT reference, held over the horizon."""
        x0 = self.solver.get(0, "x")
        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)
        p = np.zeros(self.nmc)
        Wk = self._W.copy()
        for i in range(len(self.track)):
            t = targets[i]
            if np.isfinite(t).all():
                p[3 * i:3 * i + 3] = t
            else:
                Wk[3 * i:3 * i + 3, 3 * i:3 * i + 3] = 0.0
        yref = np.concatenate([p, np.zeros(self.nv), np.zeros(self.nv)])
        for k in range(self.N):
            self.solver.set(k, "yref", yref)
            self.solver.set(k, "p", p)
            self.solver.cost_set(k, "W", Wk)
        self.solver.set(self.N, "yref", np.zeros(self.nv))
        self.solver.solve()
        # apply the FIRST step (true receding horizon), shift for warm start
        q = self.solver.get(1, "x")[:self.nq]
        for k in range(self.N):
            self.solver.set(k, "x", self.solver.get(k + 1, "x"))
        return q


# ═══════════════════════════════════════════════════════════════════════════
# INPUT SOURCES
# ═══════════════════════════════════════════════════════════════════════════

_RX = {"kp": None, "n": 0}
_MSG = 4 + 21 * 3 * 4


def _rx_thread(host, port):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            print(f"  connected to {host}:{port}")
            buf = b""
            while True:
                d = s.recv(65536)
                if not d:
                    raise ConnectionError
                buf += d
                msg = None
                while len(buf) >= _MSG:
                    msg = buf[:_MSG]
                    buf = buf[_MSG:]
                if msg is not None:
                    _RX["n"] = struct.unpack(">I", msg[:4])[0]
                    _RX["kp"] = np.frombuffer(msg[4:], np.float32).reshape(21, 3).copy()
        except OSError:
            time.sleep(1.0)


def replay_frames(path, hand="right"):
    """Yield (21,3) wrist-relative keypoints from a goliath70_3d.npy recording."""
    g = np.load(path)
    sl = slice(21, 42) if hand == "right" else slice(42, 63)
    for f in range(len(g)):
        yield g[f, sl] - g[f, sl][20]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="SAM3D hand → Orca hand MPC retargeting")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", help="goliath70_3d.npy from cosmik_hand_demo")
    src.add_argument("--listen", help="host:port of cosmik_hand_demo --emit-hand-port")
    p.add_argument("--urdf", default=str(_HERE / "orcahand" / "orcahand_right.urdf"))
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--w-tip", type=float, default=50.0)
    p.add_argument("--w-mid", type=float, default=2.0)
    p.add_argument("--w-mcp", type=float, default=0.1)
    p.add_argument("--w-dq", type=float, default=1e-3)
    p.add_argument("--w-u", type=float, default=1e-4)
    p.add_argument("--tip-offset", type=float, default=0.028,
                   help="fingertip OP-frame offset beyond the distal link origin (m)")
    p.add_argument("--free-wrist", action="store_true",
                   help="keep the wrist joint free (default: locked)")
    p.add_argument("--replay-fps", type=float, default=25.0)
    p.add_argument("--save-q", default="", help="save the joint trajectory to CSV")
    args = p.parse_args()

    print("[1/4] Model...")
    model, coll, vis = pin.buildModelsFromUrdf(args.urdf, str(Path(args.urdf).parent))
    if not args.free_wrist:
        wrist = [model.getJointId(n) for n in model.names if WRIST_JOINT_HINT in n]
        if wrist:
            model, (coll, vis) = pin.buildReducedModel(
                model, [coll, vis], wrist, pin.neutral(model))
            print(f"  wrist locked ({len(wrist)} joint) → nq={model.nq}")
    model = add_tip_frames(model, args.tip_offset)
    track = build_tracking(args.w_tip, args.w_mid, args.w_mcp)
    mapper = PalmMapper(model, args.tip_offset)

    print("[2/4] Meshcat...")
    import meshcat.geometry as mg
    import meshcat.transformations as mtf
    from pinocchio.visualize import MeshcatVisualizer
    viz = MeshcatVisualizer(model, coll, vis)
    viz.initViewer(open=False)
    viz.loadViewerModel("orca")
    viewer = viz.viewer
    for name, _, w in track:
        if w >= 10:            # show target spheres for the fingertips only
            viewer[f"target/{name}"].set_object(
                mg.Sphere(0.006), mg.MeshLambertMaterial(color=0xFF3333, opacity=0.8))
    viz.display(pin.neutral(model))
    print(f"  meshcat: {viewer.url()}")

    print("[3/4] Building acados MPC (~1 min first time)...")
    mpc = HandMPC(model, track, N=args.N, dt=args.dt, w_dq=args.w_dq, w_u=args.w_u)
    mpc.warm_start(pin.neutral(model))

    print("[4/4] Running — Ctrl+C to stop.")
    if args.listen:
        host, port = args.listen.rsplit(":", 1)
        threading.Thread(target=_rx_thread, args=(host, int(port)), daemon=True).start()

    data = model.createData()
    q_log, t_solve = [], []
    src_iter = replay_frames(args.replay) if args.replay else None
    last_n = -1
    try:
        while True:
            if args.replay:
                try:
                    kp = next(src_iter)
                except StopIteration:
                    break
                time.sleep(1.0 / args.replay_fps)
            else:
                if _RX["kp"] is None or _RX["n"] == last_n:
                    time.sleep(0.002)
                    continue
                last_n = _RX["n"]
                kp = _RX["kp"]

            targets_full = mapper(kp)
            if targets_full is None:
                continue
            # order targets to match the tracked frames
            targets = np.array([targets_full[i] for _, i, _ in track])

            t0 = time.perf_counter()
            q = mpc.solve(targets)
            t_solve.append(time.perf_counter() - t0)
            q_log.append(q)

            viz.display(q)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            for j, (name, i, w) in enumerate(track):
                if w >= 10 and np.isfinite(targets[j]).all():
                    viewer[f"target/{name}"].set_transform(
                        mtf.translation_matrix(targets[j]))
            if len(q_log) % 50 == 0:
                st = np.array(t_solve[-50:])
                errs = [np.linalg.norm(
                    data.oMf[model.getFrameId(n)].translation - targets[j])
                    for j, (n, i, w) in enumerate(track)
                    if w >= 10 and np.isfinite(targets[j]).all()]
                print(f"  {len(q_log)} solves | {1e3*st.mean():.1f} ms/solve | "
                      f"tip err mean {1e3*np.mean(errs):.1f} mm", flush=True)
    except KeyboardInterrupt:
        pass

    if args.save_q and q_log:
        np.savetxt(args.save_q, np.asarray(q_log), delimiter=",",
                   header=",".join(model.names[1:]), comments="")
        print(f"  saved {len(q_log)} configurations → {args.save_q}")


if __name__ == "__main__":
    main()
