#!/usr/bin/env python3
"""Robot-hand teleoperation retargeting — SAM3D hand keypoints → joint angles
via an acados MPC (sliding-horizon IK), visualised in VISER (web UI + sliders).
Supports several hands via --hand (see HANDS: orca, sharpa).

Input: 21 right-hand keypoints (SAM3D hand-decoder order, wrist-relative 3D):
    local idx  0..3   thumb  [tip, DIP, PIP, MCP]
               4..7   index  [tip, DIP, PIP, MCP]
               8..11  middle          "
               12..15 ring            "
               16..19 pinky           "
               20     wrist
The palm orientation is normalised per frame (human palm basis → robot palm
basis), so only finger ARTICULATION is retargeted. Targets are globally scaled
to the robot's finger lengths.

Fingertip frames: each fingertip is a tip frame origin + a LOCAL offset. The
offsets are acados *parameters* — the viser sliders move them live (green
spheres), no solver rebuild. Use the "print tip offsets" button to get CLI
values once the green spheres sit exactly on the mesh fingertips. (The Sharpa
URDF has real fingertip frames → offsets default to zero.)

MPC: state x=[q,dq], control u=ddq, cost = Σ w·||FK−p||² + w_dq·||dq||² +
w_u·||u||², hard joint limits. Weights: tips 50, mid-phalanges 2, MCP 0.1.

Run (acados env; pip install viser yourdfpy):
  export ACADOS_SOURCE_DIR=~/code/comfi-examples-hands/acados
  export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
  export ACADOS_EXT_FUN_COMPILE_FLAGS=-O1
  python retarget_mpc.py --replay ../output_cosmik_demo/<ts>/goliath70_3d.npy
  python retarget_mpc.py --listen localhost:8092     # live
Viser UI: http://localhost:8080

Manus accuracy comparison (Task): overlay a transparent-red GHOST hand
retargeted from a Manus glove on top of the SAM3D one, and dump a per-joint
angle-comparison graph on exit. The glove feed comes from manus_bridge.py
(reads sharpa-manus-sdk, emits the same TCP format):
  python retarget_mpc.py --hand sharpa --listen localhost:8092 \
      --manus-listen localhost:8095            # glove 21 kp → SAME MPC (fair)
  python retarget_mpc.py --hand sharpa --listen localhost:8092 \
      --manus-listen localhost:8095 --manus-mode angles   # SDK angles, no MPC
"""

import argparse
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

_HERE = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════
# HAND CONFIGS — human(21) ↔ robot frame mapping, per supported hand
# ═══════════════════════════════════════════════════════════════════════════

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]      # human-side order
_FINGER_BASE = {"thumb": 0, "index": 4, "middle": 8, "ring": 12, "pinky": 16}


@dataclass(frozen=True)
class HandConfig:
    """Everything hand-specific: URDF, frame names, offsets, publish format.

    frames[finger] = {"tip", "dist", "pip", "mcp"}: tip carries the local tip
    offset (acados parameter); dist/pip/mcp are the static-tracking frames
    matched to the human DIP/PIP/MCP keypoints. For hands without a dedicated
    fingertip frame, tip == dist and the offset is slider-calibrated.
    """
    name: str
    urdf: str
    frames: dict
    palm_link: str
    wrist_joint_hint: str          # joint-name substring to lock ("" = none)
    tip_offsets: dict              # finger → (3,) local offset in the TIP frame
    default_topic: str
    publish_order: tuple = ()      # URDF joint names in the driver's index order
    publish_names: tuple = ()      # JointState names (defaults to publish_order)
    collision_spheres: dict = None # name → (frameA, frameB, radius_m); sphere at the frames' midpoint
    collision_pairs: tuple = ()    # (sphere_a, sphere_b) self-collision constraints in the MPC


# ── Orca (v1 right): CAD-hash names, chains AP (MCP) → PP (PIP) → distal ────
# Tip offsets calibrated on the meshes with the viser sliders (2026-07-08).
TIP_OFFSETS_CALIB = {
    "thumb":  np.array([0.0009, 0.0000, 0.0270]),
    "index":  np.array([-0.0085, -0.0000, 0.0400]),
    "middle": np.array([-0.0085, 0.0000, 0.0400]),
    "ring":   np.array([-0.0075, 0.0000, 0.0400]),
    "pinky":  np.array([-0.0085, 0.0000, 0.0330]),
}

def _orca_publish_order():
    """URDF joint names in joint_map_v1_right.yaml key order — the SAME file
    orca_hand_driver_node maps with, so the --emit-q TCP stream (positions
    only, no names) is index-aligned with the driver by construction."""
    import yaml
    return tuple(yaml.safe_load(
        (_HERE / "joint_map_v1_right.yaml").read_text()))


def _orca_frames():
    f = {
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
    for d in f.values():
        d["tip"] = d["dist"]                       # no fingertip frame in URDF
    return f


# ── Sharpa Wave (right): human-readable names, real fingertip frames ────────
# Link chain per finger: PP (origin=MCP) → MP (PIP) → DP (DIP) → fingertip;
# thumb: MC (CMC) → PP (MCP) → DP (IP) → fingertip.
def _sharpa_frames():
    f = {}
    for fg in ("index", "middle", "ring", "pinky"):
        f[fg] = {"mcp": f"right_{fg}_PP", "pip": f"right_{fg}_MP",
                 "dist": f"right_{fg}_DP", "tip": f"right_{fg}_fingertip"}
    f["thumb"] = {"mcp": "right_thumb_MC", "pip": "right_thumb_PP",
                  "dist": "right_thumb_DP", "tip": "right_thumb_fingertip"}
    return f


# SDK joint order (indices 0..21 of SharpaWave.set_joint_position) expressed
# in URDF joint names; names below are the SDK ROS bridge's JOINT_NAMES.
_SHARPA_ORDER = tuple(
    [f"right_thumb_{j}" for j in ("CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP")]
    + [f"right_{fg}_{j}" for fg in ("index", "middle", "ring")
       for j in ("MCP_FE", "MCP_AA", "PIP", "DIP")]
    + [f"right_pinky_{j}" for j in ("CMC", "MCP_FE", "MCP_AA", "PIP", "DIP")])
_SHARPA_NAMES = tuple(
    ["thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_DIP"]
    + [f"{fg}_{j}" for fg in ("index", "middle", "ring")
       for j in ("MCP_FE", "MCP_AA", "PIP", "DIP")]
    + ["pinky_CMC_FE", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP"])

# ── Sharpa self-collision spheres ───────────────────────────────────────────
# One sphere per proximal/middle phalanx (at the midpoint of its two joint
# frames), radii from the collision-STL half-widths minus ~1 mm (PP 9.3 mm,
# MP 8.6 mm, thumb PP 9.9 mm, thumb DP 8.3 mm; adjacent MCPs are only
# 20.5-21.7 mm apart → real lateral gap ~2 mm, so full radii would bind at
# neutral). The DISTAL segments of the four fingers carry NO sphere:
# fingertip contact (pinching, fingers held together) is intentional, and
# since abduction lives at the MCP, constraining PP/MP already prevents the
# tips from actually crossing. The thumb keeps a distal sphere against the
# index/middle phalanges (sweeping under flexed fingers), but there is no
# thumb-vs-finger-distal pair, so tip-to-tip opposition stays free.
def _sharpa_collision():
    sph = {}
    for fg in ("index", "middle", "ring", "pinky"):
        sph[f"{fg}_prox"] = (f"right_{fg}_PP", f"right_{fg}_MP", 0.0085)
        sph[f"{fg}_mid"] = (f"right_{fg}_MP", f"right_{fg}_DP", 0.0080)
    sph["thumb_prox"] = ("right_thumb_PP", "right_thumb_DP", 0.0090)
    sph["thumb_dist"] = ("right_thumb_DP", "right_thumb_fingertip", 0.0080)
    pairs = [(f"{a}_{s}", f"{b}_{s}")
             for a, b in (("index", "middle"), ("middle", "ring"),
                          ("ring", "pinky"))
             for s in ("prox", "mid")]
    pairs += [(ts, f"{fg}_{s}") for ts in ("thumb_prox", "thumb_dist")
              for fg in ("index", "middle") for s in ("prox", "mid")]
    return sph, tuple(pairs)


_SHARPA_COL_SPHERES, _SHARPA_COL_PAIRS = _sharpa_collision()


def collision_gaps(cfg, model, data, margin=0.0):
    """Per-pair (a, b, gap_m) given FK already computed in `data`;
    gap = ||c_a − c_b|| − (r_a + r_b + margin), negative = violated."""
    c = {n: 0.5 * (data.oMf[model.getFrameId(fa)].translation
                   + data.oMf[model.getFrameId(fb)].translation)
         for n, (fa, fb, _) in cfg.collision_spheres.items()}
    return [(a, b, float(np.linalg.norm(c[a] - c[b]))
             - (cfg.collision_spheres[a][2] + cfg.collision_spheres[b][2] + margin))
            for a, b in cfg.collision_pairs]


HANDS = {
    "orca": HandConfig(
        name="orca",
        urdf=str(_HERE / "orcahand" / "orcahand_right.urdf"),
        frames=_orca_frames(),
        palm_link="R-Carpals_8d1f1041",
        wrist_joint_hint="to_TopTower",
        tip_offsets=TIP_OFFSETS_CALIB,
        default_topic="/orca/joint_states_target",
        publish_order=_orca_publish_order(),
    ),
    "sharpa": HandConfig(
        name="sharpa",
        urdf=str(_HERE / "sharpawave" / "right_sharpa_wave.urdf"),
        frames=_sharpa_frames(),
        palm_link="right_hand_C_MC",
        wrist_joint_hint="",                       # hand-only URDF, no wrist
        tip_offsets={f: np.zeros(3) for f in FINGERS},
        default_topic="wave/right/joint_commands",  # SDK's wave_ros_server.py
        publish_order=_SHARPA_ORDER,
        publish_names=_SHARPA_NAMES,
        collision_spheres=_SHARPA_COL_SPHERES,
        collision_pairs=_SHARPA_COL_PAIRS,
    ),
}


def tip_directions(cfg, model):
    """Unit PIP→distal direction expressed in each TIP LOCAL frame, at
    neutral pose — the axis the default tip offset extends along."""
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    dirs = {}
    for f in FINGERS:
        M_t = data.oMf[model.getFrameId(cfg.frames[f]["tip"])]
        p_d = data.oMf[model.getFrameId(cfg.frames[f]["dist"])].translation
        p_p = data.oMf[model.getFrameId(cfg.frames[f]["pip"])].translation
        u = p_d - p_p
        u = u / (np.linalg.norm(u) + 1e-12)
        dirs[f] = M_t.rotation.T @ u
    return dirs


# static (non-tip) tracked frames: (frame name, human local idx, weight)
def build_static_tracking(cfg, mid_weight, mcp_weight):
    track = []
    for f in FINGERS:
        b = _FINGER_BASE[f]
        track.append((cfg.frames[f]["dist"], b + 1, mid_weight))
        track.append((cfg.frames[f]["pip"], b + 2, mid_weight))
        track.append((cfg.frames[f]["mcp"], b + 3, mcp_weight))
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
    """Maps wrist-relative human keypoints into robot palm-frame 3D targets."""

    def __init__(self, cfg, model, tip_offset, scale_frames=15):
        data = model.createData()
        pin.forwardKinematics(model, data, pin.neutral(model))
        pin.updateFramePlacements(model, data)
        P = lambda n: data.oMf[model.getFrameId(n)].translation.copy()
        F = cfg.frames
        self.palm = P(cfg.palm_link)
        self.R_orca = _basis(P(F["middle"]["mcp"]) - self.palm,
                             P(F["pinky"]["mcp"]) - P(F["index"]["mcp"]))
        self.orca_len = (np.linalg.norm(P(F["middle"]["mcp"]) - self.palm)
                         + np.linalg.norm(P(F["middle"]["pip"]) - P(F["middle"]["mcp"]))
                         + np.linalg.norm(P(F["middle"]["dist"]) - P(F["middle"]["pip"]))
                         + np.linalg.norm(P(F["middle"]["tip"]) - P(F["middle"]["dist"]))
                         + tip_offset)
        self.scale = None
        self._scale_buf = []
        self._scale_frames = scale_frames

    def __call__(self, kp21):
        # defensive re-anchor: live emit and replay already send wrist-relative
        # keypoints (kp21[20]≈0), but this keeps the mapper correct if a caller
        # ever passes raw keypoints.
        k = kp21 - kp21[20]
        if not np.isfinite(k[[8, 9, 10, 11]]).all():
            return None
        if self.scale is None:
            human_len = (np.linalg.norm(k[11]) + np.linalg.norm(k[10] - k[11])
                         + np.linalg.norm(k[9] - k[10]) + np.linalg.norm(k[8] - k[9]))
            if human_len < 1e-6:
                return None
            # Lock the scale on the MEDIAN of the first N valid frames — a single
            # bad first frame (curled/occluded fingers) must not mis-scale the
            # whole session. Returns None (caller holds) until enough frames seen.
            self._scale_buf.append(human_len)
            if len(self._scale_buf) < self._scale_frames:
                return None
            self.scale = self.orca_len / float(np.median(self._scale_buf))
            print(f"  human→orca scale locked: {self.scale:.3f} (orca chain "
                  f"{self.orca_len*100:.1f} cm, median of {self._scale_frames} frames)")
        if not np.isfinite(k[[7, 19]]).all():
            return None
        R_h = _basis(k[11], k[19] - k[7])
        R = self.R_orca @ R_h.T
        return self.palm + (self.scale * (R @ k.T)).T


# ═══════════════════════════════════════════════════════════════════════════
# ACADOS MPC — tip offsets are PARAMETERS (live-tunable, no rebuild)
# ═══════════════════════════════════════════════════════════════════════════

class HandMPC:
    def __init__(self, cfg, model, static_track, w_tip, N=8, dt=0.04,
                 w_dq=1e-3, w_u=1e-4, self_collision=True, col_margin=0.0,
                 name_suffix=""):
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
            M = cdata.oMf[cmodel.getFrameId(cfg.frames[f]["tip"])]
            exprs.append(M.translation + M.rotation @ p_off[3 * i:3 * i + 3])
        for n, _, _ in static_track:
            exprs.append(cdata.oMf[cmodel.getFrameId(n)].translation)
        markers = casadi.vertcat(*exprs)

        am = AcadosModel()
        am.name = f"{cfg.name}_hand_mpc{name_suffix}"   # per-hand codegen dir
                                              # (ghost gets its own to coexist)
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

        # h constraints: joint limits (hard) + self-collision spheres (slacked).
        # Same pattern as the lab's collision-avoidance MPC: for each pair,
        # ||c_a − c_b||² − r_safe² ≥ 0 with the sphere centres at phalanx
        # midpoints (FK). Pinch-relevant pairs are simply absent from the list
        # (see _sharpa_collision).
        self.col_pairs = (tuple(cfg.collision_pairs)
                          if (self_collision and cfg.collision_spheres) else ())
        self.col_margin = col_margin
        h_exprs = [cq]
        lh = [np.array(model.lowerPositionLimit)]
        uh = [np.array(model.upperPositionLimit)]
        if self.col_pairs:
            cen = {n: 0.5 * (cdata.oMf[cmodel.getFrameId(fa)].translation
                             + cdata.oMf[cmodel.getFrameId(fb)].translation)
                   for n, (fa, fb, _) in cfg.collision_spheres.items()}
            for a, b in self.col_pairs:
                r_safe = (cfg.collision_spheres[a][2]
                          + cfg.collision_spheres[b][2] + col_margin)
                h_exprs.append(casadi.sumsqr(cen[a] - cen[b]) - r_safe ** 2)
            lh.append(np.zeros(len(self.col_pairs)))
            uh.append(np.full(len(self.col_pairs), 1e9))
        am.con_h_expr = casadi.vertcat(*h_exprs)
        ocp.constraints.lh = np.concatenate(lh)
        ocp.constraints.uh = np.concatenate(uh)
        if self.col_pairs:
            # slack ONLY the collision rows (joint limits stay hard): the QP can
            # never go infeasible — a violated start just pays a steep penalty
            # and gets pushed out. h is in m² so gradients are ~2·d·∂d; 1e2/1e5
            # dwarf the tip cost (w_tip=50, mm-scale errors) near contact.
            # Stage 0 (pinned x0) is safe on both acados generations: new ones
            # only apply con_h_expr at nodes 1..N-1 (node 0 needs con_h_expr_0,
            # unset here), old ones apply idxsh slacks at node 0 too.
            ns = len(self.col_pairs)
            ocp.constraints.idxsh = np.arange(nq, nq + ns)
            ocp.cost.zl = 1e2 * np.ones(ns)
            ocp.cost.zu = 1e2 * np.ones(ns)
            ocp.cost.Zl = 1e5 * np.ones(ns)
            ocp.cost.Zu = 1e5 * np.ones(ns)
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
        # name-derived json so a second solver (the Manus ghost) doesn't share
        # the default acados_ocp_nlp.json with the primary
        self.solver = AcadosOcpSolver(ocp, json_file=f"acados_ocp_{am.name}.json")

        if self.col_pairs:                     # sanity: must be feasible at rest
            data = model.createData()
            pin.forwardKinematics(model, data, pin.neutral(model))
            pin.updateFramePlacements(model, data)
            gaps = collision_gaps(cfg, model, data, col_margin)
            worst = min(gaps, key=lambda g: g[2])
            print(f"  self-collision: {len(gaps)} sphere pairs (slacked), "
                  f"neutral worst gap {worst[0]}–{worst[1]} "
                  f"{1e3 * worst[2]:.1f} mm")
            for a, b, g in gaps:
                if g <= 0:
                    print(f"  ⚠ {a}–{b} violated at NEUTRAL ({1e3 * g:.1f} mm)"
                          f" — shrink the radii or --col-margin")

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
    def __init__(self, cfg, urdf_path, model, offsets, port=8080,
                 show_collision=False):
        import viser
        from viser.extras import ViserUrdf
        self.cfg = cfg
        self.model = model
        self.offsets = offsets                 # dict finger → (3,) LOCAL offset (shared, live)
        self.server = viser.ViserServer(port=port)
        self.vurdf = ViserUrdf(self.server, Path(urdf_path), root_node_name="/hand")

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

        self.col_spheres = {}
        if show_collision and cfg.collision_spheres:
            for n, (_, _, r) in cfg.collision_spheres.items():
                try:
                    self.col_spheres[n] = self.server.scene.add_icosphere(
                        f"/collision/{n}", radius=r, color=(150, 150, 180),
                        opacity=0.35)
                except TypeError:              # older viser: no opacity kwarg
                    self.col_spheres[n] = self.server.scene.add_icosphere(
                        f"/collision/{n}", radius=r, color=(150, 150, 180))

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
            M = data.oMf[self.model.getFrameId(self.cfg.frames[f]["tip"])]
            self.green[f].position = M.translation + M.rotation @ self.offsets[f]
            if np.isfinite(tip_targets[i]).all():
                self.red[f].position = tip_targets[i]
        for n, h in self.col_spheres.items():
            fa, fb, _ = self.cfg.collision_spheres[n]
            h.position = 0.5 * (data.oMf[self.model.getFrameId(fa)].translation
                                + data.oMf[self.model.getFrameId(fb)].translation)

    def add_ghost(self, urdf_path, color=(0.9, 0.15, 0.15), opacity=0.45):
        """A second copy of the hand (transparent red) overlaid on the SAM3D
        one — the Manus-glove retarget, for side-by-side comparison. Recolour/
        opacity APIs differ across viser versions, so this is all best-effort:
        a viz nicety must never crash the comparison."""
        from viser.extras import ViserUrdf
        try:
            self.ghost = ViserUrdf(self.server, Path(urdf_path),
                                   root_node_name="/ghost",
                                   mesh_color_override=color)
        except TypeError:
            self.ghost = ViserUrdf(self.server, Path(urdf_path),
                                   root_node_name="/ghost")
        try:
            meshes = (getattr(self.ghost, "_meshes", None)
                      or getattr(self.ghost, "_mesh_handles", []))
            for h in (meshes.values() if isinstance(meshes, dict) else meshes):
                for attr, val in (("opacity", opacity), ("color", color)):
                    if hasattr(h, attr):
                        try:
                            setattr(h, attr, val)
                        except Exception:
                            pass
        except Exception:
            pass
        self._ghost_idx = [
            (self.model.joints[self.model.getJointId(n)].idx_q
             if self.model.existJointName(n) else -1)
            for n in self.ghost.get_actuated_joint_names()]
        print("  ghost (Manus) hand added — transparent red")
        return self.ghost

    def set_ghost_q(self, q):
        if getattr(self, "ghost", None) is None:
            return
        cfg = np.zeros(len(self._ghost_idx))
        for i, idx in enumerate(self._ghost_idx):
            if idx >= 0:
                cfg[i] = q[idx]
        self.ghost.update_cfg(cfg)


# ═══════════════════════════════════════════════════════════════════════════
# INPUT SOURCES
# ═══════════════════════════════════════════════════════════════════════════

_MSG = 4 + 21 * 3 * 4                        # SAM3D 21x3 float32 payload
_RX = {"data": None, "n": 0}                 # primary (SAM3D) keypoint stream
_MANUS = {"data": None, "n": 0}              # optional Manus-glove ghost stream


def _rx_loop(host, port, slot, msg_size, parse, tag=""):
    """Generic latest-value TCP receiver. Writes slot['data'] BEFORE slot['n']
    so a reader keying off 'n' always gets the matching payload (no 1-frame
    skew) — same discipline the primary stream always used."""
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            print(f"  connected to {host}:{port}{tag}")
            buf = b""
            while True:
                d = s.recv(65536)
                if not d:
                    raise ConnectionError
                buf += d
                msg = None
                while len(buf) >= msg_size:
                    msg = buf[:msg_size]
                    buf = buf[msg_size:]
                if msg is not None:
                    slot["data"] = parse(msg)
                    slot["n"] = struct.unpack(">I", msg[:4])[0]
        except OSError:
            time.sleep(1.0)


def _parse_kp(msg):
    return np.frombuffer(msg[4:], np.float32).reshape(21, 3).copy()


def _rx_thread(host, port):                  # primary SAM3D stream
    _rx_loop(host, port, _RX, _MSG, _parse_kp)


def manus_rx_thread(host, port, mode, n_sdk):
    """Manus ghost stream. mode='kp' → 21x3 keypoints (same format as SAM3D,
    fed to the SAME MPC); mode='angles' → n_sdk float32 Sharpa-SDK joint
    angles (what sharpa-manus-sdk already outputs), applied to the ghost URDF
    directly (no MPC — the fast path)."""
    if mode == "kp":
        _rx_loop(host, port, _MANUS, _MSG, _parse_kp, " (manus kp)")
    else:
        msg_q = 4 + n_sdk * 4
        _rx_loop(host, port, _MANUS, msg_q,
                 lambda m: np.frombuffer(m[4:], "<f4").copy(), " (manus angles)")


def replay_frames(path, hand="right"):
    g = np.load(path)
    sl = slice(21, 42) if hand == "right" else slice(42, 63)
    for f in range(len(g)):
        yield g[f, sl] - g[f, sl][20]


def manus_replay_frames(path, mode):
    """Offline Manus stream for testing: (T,21,3) keypoints (mode='kp', wrist
    re-anchored) or (T,n_sdk) SDK joint angles (mode='angles')."""
    a = np.load(path)
    for f in range(len(a)):
        yield (a[f] - a[f][20]) if mode == "kp" else a[f]


# ═══════════════════════════════════════════════════════════════════════════
# SAM3D vs MANUS COMPARISON — joint mapping, live stats, offline graph
# ═══════════════════════════════════════════════════════════════════════════

def build_sdk_to_q(cfg, model):
    """Map the Sharpa-SDK joint vector (cfg.publish_order = URDF joint names in
    SDK index order) to pinocchio configuration indices. Returns a list idx_q
    where idx_q[i] is the q-index for SDK joint i, or -1 if absent from the
    (possibly wrist-reduced) model."""
    if not cfg.publish_order:
        raise SystemExit(f"--manus-mode angles needs a hand with a known SDK "
                         f"joint order; {cfg.name} has none (use --manus-mode kp)")
    idx_q = []
    for name in cfg.publish_order:
        if model.existJointName(name):
            idx_q.append(model.joints[model.getJointId(name)].idx_q)
        else:
            idx_q.append(-1)
    return idx_q


def sdk_angles_to_q(model, idx_q, angles):
    """Sharpa-SDK joint angles → pinocchio q (neutral for any absent joint)."""
    q = pin.neutral(model)
    for i, iq in enumerate(idx_q):
        if iq >= 0 and i < len(angles) and np.isfinite(angles[i]):
            q[iq] = angles[i]
    return q


def angle_diff_stats(q_a, q_b):
    """(mean, max) absolute per-joint difference in DEGREES."""
    d = np.abs(np.asarray(q_a) - np.asarray(q_b)) * 180.0 / np.pi
    return float(d.mean()), float(d.max())


def plot_angle_comparison(joint_names, log_sam, log_manus, out_png, csv_path=""):
    """Offline figure: per-joint SAM3D vs Manus retargeted angle over time, plus
    a per-joint RMS-difference summary. log_* are (T, nq) arrays (deg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = np.asarray(log_sam, float) * 180.0 / np.pi
    B = np.asarray(log_manus, float) * 180.0 / np.pi
    T, nq = A.shape
    both = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    rms = np.full(nq, np.nan)
    if both.any():
        rms = np.sqrt(np.nanmean((A[both] - B[both]) ** 2, axis=0))

    if csv_path:
        hdr = ",".join([f"{n}_sam" for n in joint_names]
                       + [f"{n}_manus" for n in joint_names])
        np.savetxt(csv_path, np.hstack([A, B]), delimiter=",", header=hdr,
                   comments="")
        print(f"  saved angle trajectories → {csv_path}")

    cols = 4
    rows = int(np.ceil((nq + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.2 * rows),
                             squeeze=False)
    t = np.arange(T)
    for j in range(nq):
        ax = axes[j // cols][j % cols]
        ax.plot(t, A[:, j], lw=1.0, color="#1f77b4", label="SAM3D")
        ax.plot(t, B[:, j], lw=1.0, color="#d62728", alpha=0.8, label="Manus")
        ax.set_title(f"{joint_names[j]}  (rms {rms[j]:.1f}°)", fontsize=8)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6)
    # summary bar of per-joint RMS difference
    ax = axes[nq // cols][nq % cols]
    order = np.argsort(np.nan_to_num(rms))[::-1]
    ax.barh([joint_names[k] for k in order], rms[order], color="#555")
    ax.set_title(f"per-joint RMS diff (deg), mean {np.nanmean(rms):.1f}", fontsize=8)
    ax.tick_params(labelsize=6)
    for k in range(nq + 1, rows * cols):                 # hide unused axes
        axes[k // cols][k % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"  saved angle comparison → {out_png}  (mean RMS "
          f"{np.nanmean(rms):.1f} deg over {int(both.sum())} paired frames)")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="SAM3D hand → robot hand MPC retargeting (viser)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", help="goliath70_3d.npy from cosmik_hand_demo")
    src.add_argument("--listen", help="host:port of cosmik_hand_demo --emit-hand-port")
    p.add_argument("--hand", choices=sorted(HANDS), default="orca")
    p.add_argument("--urdf", default="", help="override the hand config's URDF")
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--w-tip", type=float, default=50.0)
    p.add_argument("--w-mid", type=float, default=2.0)
    p.add_argument("--w-mcp", type=float, default=0.1)
    p.add_argument("--w-dq", type=float, default=1e-3)
    p.add_argument("--w-u", type=float, default=1e-4)
    p.add_argument("--tip-offset", type=float, default=None,
                   help="override: tip offset magnitude for the four fingers (m) "
                        "along PIP→distal (default: use the slider-calibrated "
                        "TIP_OFFSETS_CALIB values)")
    p.add_argument("--tip-offset-thumb", type=float, default=None)
    p.add_argument("--no-self-collision", action="store_true",
                   help="drop the self-collision constraints from the MPC")
    p.add_argument("--col-margin", type=float, default=0.0,
                   help="extra safety margin (m) added to every sphere pair")
    p.add_argument("--show-collision", action="store_true",
                   help="draw the self-collision spheres in viser")
    p.add_argument("--free-wrist", action="store_true")
    p.add_argument("--viser-port", type=int, default=8080)
    p.add_argument("--replay-fps", type=float, default=25.0)
    p.add_argument("--save-q", default="", help="save the joint trajectory to CSV")
    # ── Manus-glove comparison ghost ──
    p.add_argument("--manus-listen", default="",
                   help="host:port of the Manus bridge (manus_bridge.py). Adds a "
                        "transparent-red GHOST hand retargeted from the glove, "
                        "overlaid on the SAM3D hand, for accuracy comparison")
    p.add_argument("--manus-replay", default="",
                   help="offline Manus stream (.npy): (T,21,3) keypoints for "
                        "--manus-mode kp, or (T,n_sdk) SDK joint angles for "
                        "--manus-mode angles")
    p.add_argument("--manus-mode", choices=["kp", "angles"], default="kp",
                   help="kp: Manus 21 keypoints through the SAME MPC as SAM3D "
                        "(fair pipeline comparison, default); angles: Sharpa-SDK "
                        "joint angles applied straight to the ghost (fast, no MPC)")
    p.add_argument("--compare-out", default="",
                   help="output dir for the offline angle-comparison graph + CSV "
                        "(default: next to --save-q, else ./compare_out)")
    args = p.parse_args()
    cfg = HANDS[args.hand]
    urdf = args.urdf or cfg.urdf

    print(f"[1/4] Model ({cfg.name})...")
    model, coll, vis = pin.buildModelsFromUrdf(urdf, str(Path(urdf).parent))
    if not args.free_wrist and cfg.wrist_joint_hint:
        wrist = [model.getJointId(n) for n in model.names if cfg.wrist_joint_hint in n]
        if wrist:
            model, (coll, vis) = pin.buildReducedModel(
                model, [coll, vis], wrist, pin.neutral(model))
            print(f"  wrist locked → nq={model.nq}")
    if args.tip_offset is None and args.tip_offset_thumb is None:
        offsets = {f: cfg.tip_offsets[f].copy() for f in FINGERS}
    else:
        dirs = tip_directions(cfg, model)
        mag_f = args.tip_offset if args.tip_offset is not None else 0.033
        mag_t = args.tip_offset_thumb if args.tip_offset_thumb is not None else 0.028
        offsets = {f: (mag_t if f == "thumb" else mag_f) * dirs[f] for f in FINGERS}
    static_track = build_static_tracking(cfg, args.w_mid, args.w_mcp)
    mapper = PalmMapper(cfg, model, float(np.linalg.norm(offsets["middle"])))

    print("[2/4] Viser...")
    viz = ViserViz(cfg, urdf, model, offsets, port=args.viser_port,
                   show_collision=args.show_collision)
    data = model.createData()
    q = pin.neutral(model)
    viz.set_q(q)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    viz.update_spheres(data, np.full((5, 3), np.nan))

    print("[3/4] Building acados MPC (~1 min first time)...")
    mpc = HandMPC(cfg, model, static_track, args.w_tip, N=args.N, dt=args.dt,
                  w_dq=args.w_dq, w_u=args.w_u,
                  self_collision=not args.no_self_collision,
                  col_margin=args.col_margin)
    mpc.warm_start(q)

    # ── Manus comparison ghost: a second retarget of the same hand from the
    # glove, overlaid transparent-red. kp mode reuses the SAME MPC (fair
    # pipeline comparison); angles mode drives the URDF straight from the
    # Sharpa-SDK joint vector (fast). ──
    compare = bool(args.manus_listen or args.manus_replay)
    ghost_mpc = mapper_ghost = sdk_idx = manus_iter = diff_gui = None
    sam_log, manus_log = [], []
    if compare:
        print(f"[+] Manus ghost ({args.manus_mode})...")
        viz.add_ghost(urdf)
        if args.manus_mode == "kp":
            mapper_ghost = PalmMapper(cfg, model,
                                      float(np.linalg.norm(offsets["middle"])))
            ghost_mpc = HandMPC(cfg, model, static_track, args.w_tip, N=args.N,
                                dt=args.dt, w_dq=args.w_dq, w_u=args.w_u,
                                self_collision=not args.no_self_collision,
                                col_margin=args.col_margin, name_suffix="_ghost")
            ghost_mpc.warm_start(q)
        else:
            sdk_idx = build_sdk_to_q(cfg, model)
        if args.manus_listen:
            mh, mp = args.manus_listen.rsplit(":", 1)
            threading.Thread(target=manus_rx_thread,
                             args=(mh, int(mp), args.manus_mode,
                                   len(cfg.publish_order or ())),
                             daemon=True).start()
        else:
            manus_iter = manus_replay_frames(args.manus_replay, args.manus_mode)
        try:
            diff_gui = viz.server.gui.add_text("SAM3D vs Manus |dq| (deg)", "waiting")
        except Exception:
            diff_gui = None

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
                if _RX["data"] is None or _RX["n"] == last_n:
                    time.sleep(0.002)
                    continue
                last_n = _RX["n"]
                kp = _RX["data"]

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

            # ── Manus ghost: retarget the glove input and overlay it ──
            if compare:
                if manus_iter is not None:
                    md = next(manus_iter, None)
                else:
                    md = _MANUS["data"]
                q_m = None
                if md is not None:
                    if args.manus_mode == "kp":
                        tf = mapper_ghost(md)
                        if tf is not None:
                            q_m = ghost_mpc.solve(
                                np.array([tf[_FINGER_BASE[f]] for f in FINGERS]),
                                np.array([tf[i] for _, i, _ in static_track]),
                                offsets_flat)
                    else:
                        q_m = sdk_angles_to_q(model, sdk_idx, md)
                    if q_m is not None:
                        viz.set_ghost_q(q_m)
                sam_log.append(np.asarray(q).copy())
                manus_log.append(np.asarray(q_m).copy() if q_m is not None
                                 else np.full(model.nq, np.nan))
                if diff_gui is not None and q_m is not None and len(sam_log) % 5 == 0:
                    mean_d, max_d = angle_diff_stats(q, q_m)
                    try:
                        diff_gui.value = f"mean {mean_d:.1f}  max {max_d:.1f}"
                    except Exception:
                        pass

            if len(q_log) % 50 == 0:
                st = np.array(t_solve[-50:])
                per = []
                for i, f in enumerate(FINGERS):
                    if np.isfinite(tip_targets[i]).all():
                        M = data.oMf[model.getFrameId(cfg.frames[f]["tip"])]
                        tip = M.translation + M.rotation @ offsets[f]
                        per.append((f, np.linalg.norm(tip - tip_targets[i])))
                detail = "  ".join(f"{f} {1e3*e:.0f}" for f, e in per)
                col = ""
                if mpc.col_pairs:
                    g = min(collision_gaps(cfg, model, data, mpc.col_margin),
                            key=lambda t: t[2])
                    col = f" | col gap {1e3*g[2]:.0f} mm ({g[0]}·{g[1]})"
                print(f"  {len(q_log)} solves | {1e3*st.mean():.1f} ms/solve | "
                      f"tip err mm: mean {1e3*np.mean([e for _, e in per]):.1f} "
                      f"[{detail}]{col}", flush=True)
                if compare and manus_log:
                    sl = np.array(sam_log[-50:])
                    ml = np.array(manus_log[-50:])
                    m = np.isfinite(sl).all(1) & np.isfinite(ml).all(1)
                    if m.any():
                        dd = np.abs(sl[m] - ml[m]).mean() * 180 / np.pi
                        print(f"    Manus ghost: mean |dq| {dd:.1f} deg "
                              f"({int(m.sum())}/{len(m)} frames paired)", flush=True)
    except KeyboardInterrupt:
        pass

    if args.save_q and q_log:
        np.savetxt(args.save_q, np.asarray(q_log), delimiter=",",
                   header=",".join(model.names[1:]), comments="")
        print(f"  saved {len(q_log)} configurations → {args.save_q}")

    if compare and sam_log:
        outd = (args.compare_out
                or (os.path.dirname(args.save_q) if args.save_q else "")
                or "compare_out")
        os.makedirs(outd, exist_ok=True)
        plot_angle_comparison(
            list(model.names[1:]), sam_log, manus_log,
            os.path.join(outd, "angle_comparison.png"),
            os.path.join(outd, "angle_comparison.csv"))
    print("  final tip offsets (m):")
    for f in FINGERS:
        o = offsets[f]
        print(f"    {f}: [{o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f}]")


if __name__ == "__main__":
    main()
