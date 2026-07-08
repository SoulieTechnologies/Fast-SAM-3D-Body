#!/usr/bin/env python3
"""Orca-hand teleoperation retargeting — SAM3D hand keypoints → Orca joint angles
via an acados MPC (sliding-horizon IK), visualised in VISER (web UI + sliders).

Input: 21 right-hand keypoints (SAM3D hand-decoder order, wrist-relative 3D):
    local idx  0..3   thumb  [tip, DIP, PIP, MCP]
               4..7   index  [tip, DIP, PIP, MCP]
               8..11  middle          "
               12..15 ring            "
               16..19 pinky           "
               20     wrist
The palm orientation is normalised per frame (human palm basis → Orca palm
basis), so only finger ARTICULATION is retargeted. Targets are globally scaled
to the Orca's finger lengths.

Fingertip frames: each Orca fingertip is the distal link origin + a LOCAL
offset. The offsets are acados *parameters* — the viser sliders move them live
(green spheres), no solver rebuild. Use the "print tip offsets" button to get
CLI values once the green spheres sit exactly on the mesh fingertips.

MPC: state x=[q,dq], control u=ddq, cost = Σ w·||FK−p||² + w_dq·||dq||² +
w_u·||u||², hard joint limits. Weights: tips 50, mid-phalanges 2, MCP 0.1.

Run (acados env; pip install viser yourdfpy):
  export ACADOS_SOURCE_DIR=~/code/comfi-examples-hands/acados
  export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
  export ACADOS_EXT_FUN_COMPILE_FLAGS=-O1
  python retarget_mpc.py --replay ../output_cosmik_demo/<ts>/goliath70_3d.npy
  python retarget_mpc.py --listen localhost:8092     # live
Viser UI: http://localhost:8080
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
# Finger order across the Orca palm (from joint origins): chains are
# AP (MCP) → PP (PIP) → FingerTipAssembly / T-DP (distal).

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
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
_FINGER_BASE = {"thumb": 0, "index": 4, "middle": 8, "ring": 12, "pinky": 16}
PALM_LINK = "R-Carpals_8d1f1041"
WRIST_JOINT_HINT = "to_TopTower"


def tip_directions(model):
    """Unit PIP→distal direction expressed in each DISTAL LOCAL frame, at
    neutral pose — the axis the default tip offset extends along."""
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    dirs = {}
    for f in FINGERS:
        M_d = data.oMf[model.getFrameId(_F[f]["dist"])]
        p_p = data.oMf[model.getFrameId(_F[f]["pip"])].translation
        u = M_d.translation - p_p
        u = u / (np.linalg.norm(u) + 1e-12)
        dirs[f] = M_d.rotation.T @ u
    return dirs


# static (non-tip) tracked frames: (frame name, human local idx, weight)
def build_static_tracking(mid_weight, mcp_weight):
    track = []
    for f in FINGERS:
        b = _FINGER_BASE[f]
        track.append((_F[f]["dist"], b + 1, mid_weight))
        track.append((_F[f]["pip"], b + 2, mid_weight))
        track.append((_F[f]["mcp"], b + 3, mcp_weight))
    return track


# ═══════════════════════════════════════════════════════════════════════════
# PALM ALIGNMENT + SCALE
# ═══════════════════════════════════════════════════════════════════════════

def _basis(fwd, lat):
    f = fwd / (np.linalg.norm(fwd) + 1e-12)
    l = lat - f * (lat @ f)
    l = l / (np.linalg.norm(l) + 1e-12)
    return np.stack([f, l, np.cross(f, l)], axis=1)


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
        self.orca_len = (np.linalg.norm(P(_F["middle"]["mcp"]) - self.palm)
                         + np.linalg.norm(P(_F["middle"]["pip"]) - P(_F["middle"]["mcp"]))
                         + np.linalg.norm(P(_F["middle"]["dist"]) - P(_F["middle"]["pip"]))
                         + tip_offset)
        self.scale = None

    def __call__(self, kp21):
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
        R_h = _basis(k[11], k[19] - k[7])
        R = self.R_orca @ R_h.T
        return self.palm + (self.scale * (R @ k.T)).T


# ═══════════════════════════════════════════════════════════════════════════
# ACADOS MPC — tip offsets are PARAMETERS (live-tunable, no rebuild)
# ═══════════════════════════════════════════════════════════════════════════

class HandMPC:
    def __init__(self, model, static_track, w_tip, N=8, dt=0.04,
                 w_dq=1e-3, w_u=1e-4):
        import casadi
        import pinocchio.casadi as cpin
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

        self.model = model
        self.static_track = static_track
        self.N, self.dt = N, dt
        nq, nv = model.nq, model.nv
        self.nq, self.nv = nq, nv
        self.n_tips = len(FINGERS)
        self.nmc = 3 * (self.n_tips + len(static_track))

        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        cx = casadi.SX.sym("x", nq + nv)
        cu = casadi.SX.sym("u", nv)
        cq, cdq = cx[:nq], cx[nq:]
        x_next = casadi.vertcat(cpin.integrate(cmodel, cq, cdq * dt), cdq + cu * dt)
        cpin.framesForwardKinematics(cmodel, cdata, cq)

        # parameters: per-finger LOCAL tip offset in the distal frame (15)
        p_off = casadi.SX.sym("p_off", 3 * self.n_tips)
        exprs = []
        for i, f in enumerate(FINGERS):
            M = cdata.oMf[cmodel.getFrameId(_F[f]["dist"])]
            exprs.append(M.translation + M.rotation @ p_off[3 * i:3 * i + 3])
        for n, _, _ in static_track:
            exprs.append(cdata.oMf[cmodel.getFrameId(n)].translation)
        markers = casadi.vertcat(*exprs)

        am = AcadosModel()
        am.name = "orca_hand_mpc"
        am.x, am.u = cx, cu
        am.disc_dyn_expr = x_next
        am.cost_y_expr = casadi.vertcat(markers, cdq, cu)
        am.cost_y_expr_e = cdq
        am.p = p_off

        ocp = AcadosOcp()
        ocp.model = am
        ocp.solver_options.N_horizon = N
        ocp.solver_options.tf = N * dt
        ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
        ny = self.nmc + nv + nv
        ocp.cost.yref = np.zeros(ny)
        W = np.zeros((ny, ny))
        for i in range(self.n_tips):
            W[3 * i:3 * i + 3, 3 * i:3 * i + 3] = w_tip * np.eye(3)
        for j, (_, _, w) in enumerate(static_track):
            k = 3 * (self.n_tips + j)
            W[k:k + 3, k:k + 3] = w * np.eye(3)
        W[self.nmc:self.nmc + nv, self.nmc:self.nmc + nv] = w_dq * np.eye(nv)
        W[self.nmc + nv:, self.nmc + nv:] = w_u * np.eye(nv)
        ocp.cost.W = W
        self._W = W
        ocp.cost.yref_e = np.zeros(nv)
        ocp.cost.W_e = w_dq * np.eye(nv)

        am.con_h_expr = cq
        ocp.constraints.lh = np.array(model.lowerPositionLimit)
        ocp.constraints.uh = np.array(model.upperPositionLimit)
        ocp.constraints.x0 = np.zeros(nq + nv)
        ocp.parameter_values = np.zeros(3 * self.n_tips)

        so = ocp.solver_options
        so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        so.hessian_approx = "GAUSS_NEWTON"
        so.integrator_type = "DISCRETE"
        so.nlp_solver_type = "SQP"
        so.nlp_solver_max_iter = 10
        so.tol = 1e-4
        so.ext_fun_compile_flags = os.environ.get("ACADOS_EXT_FUN_COMPILE_FLAGS", "-O1")
        self.solver = AcadosOcpSolver(ocp)

    def warm_start(self, q):
        x0 = np.zeros(self.nq + self.nv)
        x0[:self.nq] = q
        for k in range(self.N + 1):
            self.solver.set(k, "x", x0)

    def solve(self, tip_targets, static_targets, offsets_flat):
        """tip_targets (5,3), static_targets (n,3), offsets_flat (15,)."""
        x0 = self.solver.get(0, "x")
        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)
        p_ref = np.zeros(self.nmc)
        Wk = self._W.copy()
        allt = np.vstack([tip_targets, static_targets])
        for i, t in enumerate(allt):
            if np.isfinite(t).all():
                p_ref[3 * i:3 * i + 3] = t
            else:
                Wk[3 * i:3 * i + 3, 3 * i:3 * i + 3] = 0.0
        yref = np.concatenate([p_ref, np.zeros(self.nv), np.zeros(self.nv)])
        for k in range(self.N):
            self.solver.set(k, "yref", yref)
            self.solver.set(k, "p", offsets_flat)
            self.solver.cost_set(k, "W", Wk)
        self.solver.set(self.N, "yref", np.zeros(self.nv))
        self.solver.solve()
        q = self.solver.get(1, "x")[:self.nq]
        for k in range(self.N):
            self.solver.set(k, "x", self.solver.get(k + 1, "x"))
        return q


# ═══════════════════════════════════════════════════════════════════════════
# VISER UI  (robot + red target / green URDF-tip spheres + offset sliders)
# ═══════════════════════════════════════════════════════════════════════════

class ViserViz:
    def __init__(self, urdf_path, model, offsets, port=8080):
        import viser
        from viser.extras import ViserUrdf
        self.model = model
        self.offsets = offsets                 # dict finger → (3,) LOCAL offset (shared, live)
        self.server = viser.ViserServer(port=port)
        self.vurdf = ViserUrdf(self.server, Path(urdf_path), root_node_name="/orca")

        # pinocchio q ↔ yourdfpy configuration mapping (by joint name; the
        # locked wrist is absent from the reduced model → stays at 0)
        self._cfg_names = self.vurdf.get_actuated_joint_names()
        self._cfg_idx = []
        for n in self._cfg_names:
            if model.existJointName(n):
                self._cfg_idx.append(model.joints[model.getJointId(n)].idx_q)
            else:
                self._cfg_idx.append(-1)

        self.red = {f: self.server.scene.add_icosphere(
            f"/targets/{f}", radius=0.006, color=(255, 60, 60)) for f in FINGERS}
        self.green = {f: self.server.scene.add_icosphere(
            f"/urdf_tips/{f}", radius=0.005, color=(60, 255, 60)) for f in FINGERS}

        # per-finger sliders: LOCAL x/y/z offset in the distal frame
        self._sliders = {}
        for f in FINGERS:
            with self.server.gui.add_folder(f"{f} tip offset (mm, local)"):
                for ax in range(3):
                    s = self.server.gui.add_slider(
                        "xyz"[ax], min=-60.0, max=60.0, step=0.5,
                        initial_value=float(self.offsets[f][ax] * 1e3))
                    s.on_update(self._make_cb(f, ax))
                    self._sliders[(f, ax)] = s
        btn = self.server.gui.add_button("print tip offsets")

        @btn.on_click
        def _(_):
            print("  current tip offsets (m, local distal frame):")
            for f in FINGERS:
                o = self.offsets[f]
                print(f"    {f}: [{o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f}]")

        print(f"  viser UI: http://localhost:{port}")

    def _make_cb(self, f, ax):
        def cb(_):
            self.offsets[f][ax] = self._sliders[(f, ax)].value * 1e-3
        return cb

    def set_q(self, q):
        cfg = np.zeros(len(self._cfg_idx))
        for i, idx in enumerate(self._cfg_idx):
            if idx >= 0:
                cfg[i] = q[idx]
        self.vurdf.update_cfg(cfg)

    def update_spheres(self, data, tip_targets):
        for i, f in enumerate(FINGERS):
            M = data.oMf[self.model.getFrameId(_F[f]["dist"])]
            self.green[f].position = M.translation + M.rotation @ self.offsets[f]
            if np.isfinite(tip_targets[i]).all():
                self.red[f].position = tip_targets[i]


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
    g = np.load(path)
    sl = slice(21, 42) if hand == "right" else slice(42, 63)
    for f in range(len(g)):
        yield g[f, sl] - g[f, sl][20]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="SAM3D hand → Orca hand MPC retargeting (viser)")
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
    p.add_argument("--tip-offset", type=float, default=0.033,
                   help="initial tip offset for the four fingers (m), along PIP→distal")
    p.add_argument("--tip-offset-thumb", type=float, default=0.028)
    p.add_argument("--free-wrist", action="store_true")
    p.add_argument("--viser-port", type=int, default=8080)
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
            print(f"  wrist locked → nq={model.nq}")
    dirs = tip_directions(model)
    offsets = {f: (args.tip_offset_thumb if f == "thumb" else args.tip_offset)
               * dirs[f] for f in FINGERS}
    static_track = build_static_tracking(args.w_mid, args.w_mcp)
    mapper = PalmMapper(model, args.tip_offset)

    print("[2/4] Viser...")
    viz = ViserViz(args.urdf, model, offsets, port=args.viser_port)
    data = model.createData()
    q = pin.neutral(model)
    viz.set_q(q)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    viz.update_spheres(data, np.full((5, 3), np.nan))

    print("[3/4] Building acados MPC (~1 min first time)...")
    mpc = HandMPC(model, static_track, args.w_tip, N=args.N, dt=args.dt,
                  w_dq=args.w_dq, w_u=args.w_u)
    mpc.warm_start(q)

    print("[4/4] Running — Ctrl+C to stop.")
    if args.listen:
        host, port = args.listen.rsplit(":", 1)
        threading.Thread(target=_rx_thread, args=(host, int(port)), daemon=True).start()

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
            tip_targets = np.array([targets_full[_FINGER_BASE[f]] for f in FINGERS])
            static_targets = np.array([targets_full[i] for _, i, _ in static_track])
            offsets_flat = np.concatenate([offsets[f] for f in FINGERS])

            t0 = time.perf_counter()
            q = mpc.solve(tip_targets, static_targets, offsets_flat)
            t_solve.append(time.perf_counter() - t0)
            q_log.append(q)

            viz.set_q(q)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            viz.update_spheres(data, tip_targets)

            if len(q_log) % 50 == 0:
                st = np.array(t_solve[-50:])
                per = []
                for i, f in enumerate(FINGERS):
                    if np.isfinite(tip_targets[i]).all():
                        M = data.oMf[model.getFrameId(_F[f]["dist"])]
                        tip = M.translation + M.rotation @ offsets[f]
                        per.append((f, np.linalg.norm(tip - tip_targets[i])))
                detail = "  ".join(f"{f} {1e3*e:.0f}" for f, e in per)
                print(f"  {len(q_log)} solves | {1e3*st.mean():.1f} ms/solve | "
                      f"tip err mm: mean {1e3*np.mean([e for _, e in per]):.1f} "
                      f"[{detail}]", flush=True)
    except KeyboardInterrupt:
        pass

    if args.save_q and q_log:
        np.savetxt(args.save_q, np.asarray(q_log), delimiter=",",
                   header=",".join(model.names[1:]), comments="")
        print(f"  saved {len(q_log)} configurations → {args.save_q}")
    print("  final tip offsets (m):")
    for f in FINGERS:
        o = offsets[f]
        print(f"    {f}: [{o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f}]")


if __name__ == "__main__":
    main()
