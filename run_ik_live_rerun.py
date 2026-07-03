#!/usr/bin/env python3
"""LIVE ACADOS IK with Rerun mirroring — receive gravity-aligned Goliath-70 3D over
TCP from rerun_demo.py (or stream_demo.py --emit-port), retarget onto the MANO body
in meshcat, and mirror the retargeted markers into the SAME Rerun viewer as the
extractor (panel "Retargeted (ACADOS IK)").

Run (comfi env, server):
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate acados
  cd <comfi-examples_new>
  export ACADOS_SOURCE_DIR=$PWD/acados LD_LIBRARY_PATH=$PWD/acados/lib:$LD_LIBRARY_PATH
  export ACADOS_EXT_FUN_COMPILE_FLAGS=-O1
  python scripts/run_ik_live_rerun.py --emit-port 8090 \
      --rerun-url rerun+http://127.0.0.1:9876/proxy \
      --N 10 --w-dq 1e-3 --w-u 1e-4 --subject-height 1.75 --subject-weight 70

Viewers:
  meshcat (full URDF human): http://localhost:7000/static/   (ssh -L 7000:localhost:7000)
  Rerun   (shared with the extractor): http://localhost:9090 (ssh -L 9090:localhost:9090)

Pass --rerun-url "" to disable Rerun (behaves like run_ik_live.py).
The 3D received is ALREADY gravity-aligned (ROS frame) → no cv_to_ros here.
Requires (for Rerun): pip install "rerun-sdk>=0.28" in the acados env.
"""

import argparse
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pinocchio as pin

# Reuse everything from the modified offline script.
from run_ik_acados_mpc_sam3d_live import (
    GOLIATH_NAMES, HAND_KP_TO_MANO_LINK, HAND_SOLVER_KEYS, SOLVER_KEYS,
    JOINTS_TO_LOCK, AcadosMPCIKSolver, register_tracking_frames,
    scale_model_from_goliath,
)

_MSG = 4 + 70 * 3 * 4          # 4-byte counter + 70*3 float32
_RX = {"kp": None, "n": 0}


def _rx_thread(sock):
    """Continuously read (70,3) frames; always keep only the LATEST (drop-old)."""
    buf = b""
    while True:
        try:
            d = sock.recv(65536)
        except OSError:
            return
        if not d:
            return
        buf += d
        msg = None
        while len(buf) >= _MSG:          # drain to the most recent complete message
            msg = buf[:_MSG]
            buf = buf[_MSG:]
        if msg is not None:
            _RX["n"] = struct.unpack(">I", msg[:4])[0]
            _RX["kp"] = np.frombuffer(msg[4:], dtype=np.float32).reshape(70, 3).copy()


def _connect_with_retry(host, port, timeout=120.0):
    """Keep trying to connect until the extractor's emit server is up."""
    t0 = time.time()
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            return sock
        except OSError:
            sock.close()
            if time.time() - t0 > timeout:
                raise TimeoutError(f"emit server {host}:{port} not up after {timeout}s")
            print(f"  waiting for extractor on {host}:{port} ...", flush=True)
            time.sleep(2.0)


def _frame_dict(kp):
    return {name: kp[i] for i, name in enumerate(GOLIATH_NAMES)}


# ═══════════════════════════════════════════════════════════════════════════
# RERUN MIRROR  (retargeted skeleton in the extractor's viewer)
# ═══════════════════════════════════════════════════════════════════════════

def _skeleton_pairs():
    """(nameA, nameB) bone pairs over the tracked marker names."""
    pairs = [
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("left_ankle", "left_heel"), ("left_ankle", "left_big_toe"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
        ("right_ankle", "right_heel"), ("right_ankle", "right_big_toe"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("nose", "neck"),
        ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ]
    for side in ("left", "right"):
        for finger in ("thumb", "forefinger", "middle_finger",
                       "ring_finger", "pinky_finger"):
            chain = [f"{side}_wrist", f"{side}_{finger}_third_joint",
                     f"{side}_{finger}2", f"{side}_{finger}3", f"{side}_{finger}4"]
            pairs += list(zip(chain[:-1], chain[1:]))
    return pairs


class RerunMirror:
    def __init__(self, url, valid_keys):
        import rerun as rr
        self.rr = rr
        self.keys = valid_keys
        self.pairs = [(a, b) for a, b in _skeleton_pairs()
                      if a in valid_keys and b in valid_keys]
        rr.init("fastsam3d_live")        # same app id → same viewer
        rr.connect_grpc(url)
        # retarget space is the gravity-aligned ROS frame: X fwd, Y left, Z up.
        rr.log("retarget", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        print(f"  Rerun mirror connected: {url}")

    def log(self, frame_n, est, gt, ik_ms):
        rr = self.rr
        rr.set_time("frame", sequence=frame_n)
        rr.log("timing/ik_ms", rr.Scalars(float(ik_ms)))

        pts = np.array([est[k] for k in self.keys if k in est])
        if len(pts):
            rr.log("retarget/est/joints",
                   rr.Points3D(positions=pts, radii=0.012, colors=[80, 170, 255]))
        strips = [[est[a].tolist(), est[b].tolist()] for a, b in self.pairs
                  if a in est and b in est]
        if strips:
            rr.log("retarget/est/bones",
                   rr.LineStrips3D(strips, colors=[80, 170, 255], radii=0.005))

        gt_pts = np.array([gt[k] for k in self.keys
                           if k in gt and np.isfinite(gt[k]).all()])
        if len(gt_pts):
            rr.log("retarget/target",
                   rr.Points3D(positions=gt_pts, radii=0.006, colors=[255, 60, 60]))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Live ACADOS IK + Rerun from streamed SAM3D 3D")
    p.add_argument("--host", default="localhost", help="extractor host")
    p.add_argument("--emit-port", type=int, default=8090)
    p.add_argument("--rerun-url", default="rerun+http://127.0.0.1:9876/proxy",
                   help="extractor's Rerun gRPC url ('' disables the mirror)")
    p.add_argument("--model-dir", default=str(Path(__file__).resolve().parent.parent / "model"))
    p.add_argument("--urdf-file", default="urdf/human_with_mano.urdf")
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--dt", type=float, default=0.033)
    p.add_argument("--w-markers", type=float, default=1.0)
    p.add_argument("--w-hand", type=float, default=3.0)
    p.add_argument("--w-fingertip-mult", type=float, default=2.0)
    p.add_argument("--w-dq", type=float, default=1e-3)
    p.add_argument("--w-dq-hand", type=float, default=None)
    p.add_argument("--w-u", type=float, default=1e-4)
    p.add_argument("--subject-height", type=float, default=1.75)
    p.add_argument("--subject-weight", type=float, default=70.0)
    p.add_argument("--gender", default="male", choices=["male", "female"])
    p.add_argument("--augmenter-path",
                   default=str(Path(__file__).resolve().parent.parent / "augmentation_model"))
    p.add_argument("--warmup", type=int, default=30, help="frames to collect before starting")
    args = p.parse_args()

    urdf_path = str(Path(args.model_dir) / args.urdf_file)

    # ── connect (with retry: the extractor may still be loading) + warmup ────
    print(f"[1/5] Connecting to {args.host}:{args.emit_port} ...")
    sock = _connect_with_retry(args.host, args.emit_port)
    threading.Thread(target=_rx_thread, args=(sock,), daemon=True).start()

    print(f"[2/5] Collecting {args.warmup} warmup frames...")
    warm = []
    last = -1
    while len(warm) < args.warmup:
        if _RX["kp"] is not None and _RX["n"] != last:
            last = _RX["n"]
            warm.append(_RX["kp"].copy())
        else:
            time.sleep(0.01)
    warm_arr = np.stack(warm)

    # ── model: load, scale (COSMIK+LSTM), register frames, lock thorax ────────
    print("[3/5] Building + scaling model...")
    human_model, human_coll, human_vis = pin.buildModelsFromUrdf(
        urdf_path, str(Path(urdf_path).parent), root_joint=pin.JointModelFreeFlyer())
    human_model = scale_model_from_goliath(
        human_model, warm_arr, augmenter_path=args.augmenter_path,
        subject_height=args.subject_height, subject_weight=args.subject_weight,
        gender=args.gender)
    human_model, registered = register_tracking_frames(human_model)
    lock = [human_model.getJointId(j) for j in JOINTS_TO_LOCK if human_model.existJointName(j)]
    q0 = pin.neutral(human_model)
    if lock:
        human_model, (human_coll, human_vis) = pin.buildReducedModel(
            human_model, [human_coll, human_vis], lock, q0)
    human_data = pin.Data(human_model)
    valid_keys = [k for k in SOLVER_KEYS if human_model.existFrame(k)]

    # ── meshcat (full URDF human) ─────────────────────────────────────────────
    print("[4/5] Meshcat...")
    import meshcat.geometry as mg
    import meshcat.transformations as mtf
    from pinocchio.visualize import MeshcatVisualizer
    viz = MeshcatVisualizer(human_model, human_coll, human_vis)
    viz.initViewer(open=False)
    viz.loadViewerModel("human")
    viewer = viz.viewer
    for mk in valid_keys:
        viewer[f"gt/{mk}"].set_object(mg.Sphere(0.012),
                                      mg.MeshLambertMaterial(color=0xFF0000, opacity=0.6))
    viz.display(pin.neutral(human_model))
    print(f"      open meshcat: {viewer.url()}  (ssh -L 7000:localhost:7000)")

    # ── Rerun mirror (optional) ───────────────────────────────────────────────
    mirror = None
    if args.rerun_url:
        try:
            mirror = RerunMirror(args.rerun_url, valid_keys)
        except Exception as e:
            print(f"[WARN] Rerun mirror disabled ({e}) — meshcat only")

    # ── IPOPT warm-start + acados solver ─────────────────────────────────────
    print("[5/5] IPOPT warm-start + acados build (~1 min first time)...")
    from comfi_examples.ik_utils import RT_IK
    wkeys = [k for k in registered if human_model.existFrame(k)]
    omega = {k: (0.5 if k in HAND_KP_TO_MANO_LINK else 1.0) for k in wkeys}
    fd0 = _frame_dict(warm_arr[-1])
    q_init = RT_IK(human_model, fd0, pin.neutral(human_model),
                   wkeys, args.dt, omega).solve_ik_sample_casadi()
    ik = AcadosMPCIKSolver(
        pin_model=human_model, keys_to_track=valid_keys, N=args.N, dt=args.dt,
        w_markers=args.w_markers, w_hand=args.w_hand, w_dq=args.w_dq,
        w_dq_hand=args.w_dq_hand, w_u=args.w_u, hand_keys=HAND_SOLVER_KEYS,
        w_fingertip_mult=args.w_fingertip_mult)
    ik.reset_warm_start(q_init)

    window = deque([_frame_dict(f) for f in warm_arr[-args.N:]], maxlen=args.N)
    while len(window) < args.N:
        window.appendleft(window[0])

    print("LIVE — retargeting... (Ctrl+C to stop)")
    last = -1
    solved = 0
    t_report = time.time()
    while True:
        if _RX["kp"] is None or _RX["n"] == last:
            time.sleep(0.002)
            continue
        last = _RX["n"]
        window.append(_frame_dict(_RX["kp"]))
        t0 = time.perf_counter()
        q = ik.solve(list(window))
        ik_ms = (time.perf_counter() - t0) * 1e3
        viz.display(q)
        pin.forwardKinematics(human_model, human_data, q)
        pin.updateFramePlacements(human_model, human_data)
        cur = window[-1]
        est = {}
        for mk in valid_keys:
            est[mk] = np.array(human_data.oMf[human_model.getFrameId(mk)].translation)
            gt = np.asarray(cur[mk]).flatten()[:3]
            if np.isfinite(gt).all():
                viewer[f"gt/{mk}"].set_transform(mtf.translation_matrix(gt))
        if mirror is not None:
            mirror.log(last, est, cur, ik_ms)
        solved += 1
        if time.time() - t_report > 2.0:
            print(f"  IK {solved / (time.time() - t_report):.1f} solves/s "
                  f"({ik_ms:.0f} ms last)", flush=True)
            solved = 0
            t_report = time.time()


if __name__ == "__main__":
    main()
