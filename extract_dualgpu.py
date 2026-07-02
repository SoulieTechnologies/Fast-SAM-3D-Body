#!/usr/bin/env python3
"""Dual-GPU two-camera extraction — one view per GPU in TRUE parallel, then triangulate.

The dual-camera bottleneck is running SAM3D on both views sequentially (~2x62 ms →
~8 FPS). Here each view runs in its OWN process pinned to its OWN GPU, so the two
SAM3D passes overlap → ~62 ms per stereo pair → ~16 FPS. Triangulation is a cheap
CPU post-step. Avoids in-process multi-GPU / TRT-singleton issues (subprocesses are
fully isolated).

Architecture:
  orchestrator → spawns 2 workers (this file --worker) → waits → triangulates → overlay
  worker(gpu)  → SAM3D on one video, selects the main subject, saves (T,70,2) 2D

Usage:
  python extract_dualgpu.py \
      --cam0 test_input/take_01/cam0.mp4 --cam1 test_input/take_01/cam1.mp4 \
      --stereo test_input/cam_params/stereo_params.npz \
      --output_dir ./output_dualgpu --gpu0 7 --gpu1 1
"""

import os
import sys

_WORKER = "--worker" in sys.argv

# ── Worker: pin GPU + TRT flags BEFORE importing torch ───────────────────────
if _WORKER:
    import argparse as _ap
    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument("--gpu", type=int, default=0)
    _gpu = _pre.parse_known_args()[0].gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)
    for k, v in {
        "USE_TRT_BACKBONE": "1", "MHR_NO_CORRECTIVES": "1", "SKIP_KEYPOINT_PROMPT": "1",
        "USE_COMPILE": "0", "LAYER_DTYPE": "fp32", "GPU_HAND_PREP": "1",
        "INTERM_PRED_INTERVAL": "999", "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
        "COMPILE_MODE": "default", "MHR_USE_CUDA_GRAPH": "0",
        "TRT_BACKBONE_PATH":
            "/home/users/theo/code/checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine",
    }.items():
        os.environ.setdefault(k, v)

import argparse
import subprocess
import time

import cv2
import numpy as np

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)


def _select_person(outputs, prev_centroid, img_diag, w_pen=5.0):
    """Largest body bbox + temporal continuity → (kp2d (70,2), centroid) or None."""
    cands = []
    for p in outputs:
        k2 = np.asarray(p["pred_keypoints_2d"], dtype=np.float32)
        body = k2[:21]
        v = np.isfinite(body).all(1)
        if v.sum() < 4:
            continue
        bb = body[v]
        size = (bb[:, 0].max() - bb[:, 0].min()) * (bb[:, 1].max() - bb[:, 1].min())
        cands.append((size, bb.mean(0), k2))
    if not cands:
        return None
    if prev_centroid is None:
        _, c, k2 = max(cands, key=lambda x: x[0])
    else:
        _, c, k2 = max(cands, key=lambda x: x[0] / (1.0 + w_pen * np.linalg.norm(x[1] - prev_centroid) / img_diag))
    return k2, c


def run_worker(args):
    """Extract one view on one GPU; save (T,70,2) SAM3D 2D keypoints to --out2d."""
    import contextlib, io
    import torch
    from notebook.utils import setup_sam_3d_body

    K = np.load(args.intrinsics)["K"] if args.intrinsics else None
    cam_int = torch.tensor(K[None], dtype=torch.float32) if K is not None else None

    det = args.detector_model
    if det.endswith(".pt") and os.path.exists(det.replace(".pt", ".engine")):
        det = det.replace(".pt", ".engine")
    est = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(args.checkpoint_dir, "assets", "mhr_model.pt"),
        detector_name="yolo", detector_model=det, fov_name="", device="cuda",
    )

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_diag = float(np.hypot(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    quiet = (contextlib.redirect_stdout(io.StringIO())
             if os.environ.get("SAM3D_PROFILE", "0") != "1" else contextlib.nullcontext())

    kp, cen, t_proc = [], None, 0.0
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        with torch.no_grad(), quiet:
            out = est.process_one_image(rgb, inference_type="body",
                                        **({"cam_int": cam_int} if cam_int is not None else {}))
        t_proc += time.perf_counter() - t0
        sel = _select_person(out, cen, img_diag) if out else None
        if sel is not None:
            kp.append(sel[0]); cen = sel[1]
        else:
            kp.append(np.full((70, 2), np.nan, np.float32))
    cap.release()
    np.save(args.out2d, np.stack(kp))
    print(f"[worker gpu{args.gpu}] {len(kp)} frames, SAM3D compute {1.0/(t_proc/max(len(kp),1)):.1f} FPS")


def run_orchestrator(args):
    stereo_dir = os.path.dirname(os.path.abspath(args.stereo))
    sys.path.insert(0, stereo_dir)
    from triangulate import StereoTriangulator
    tri = StereoTriangulator(args.stereo)
    os.makedirs(args.output_dir, exist_ok=True)

    v0 = os.path.join(args.output_dir, "_view0_2d.npy")
    v1 = os.path.join(args.output_dir, "_view1_2d.npy")
    # save per-camera intrinsics for the workers (better than default FOV)
    k0 = os.path.join(args.output_dir, "_K0.npz"); np.savez(k0, K=tri.K1)
    k1 = os.path.join(args.output_dir, "_K1.npz"); np.savez(k1, K=tri.K2)

    def worker_cmd(video, gpu, out2d, intr):
        return [sys.executable, os.path.abspath(__file__), "--worker",
                "--video", video, "--gpu", str(gpu), "--out2d", out2d, "--intrinsics", intr,
                "--checkpoint_dir", args.checkpoint_dir, "--detector_model", args.detector_model]

    print(f"[orchestrator] launching 2 workers — cam0→gpu{args.gpu0}, cam1→gpu{args.gpu1} (parallel)")
    t0 = time.perf_counter()
    p0 = subprocess.Popen(worker_cmd(args.cam0, args.gpu0, v0, k0))
    p1 = subprocess.Popen(worker_cmd(args.cam1, args.gpu1, v1, k1))
    r0, r1 = p0.wait(), p1.wait()
    wall = time.perf_counter() - t0
    if r0 != 0 or r1 != 0:
        raise RuntimeError(f"worker failed (cam0={r0}, cam1={r1})")

    p2d0, p2d1 = np.load(v0), np.load(v1)
    T = min(len(p2d0), len(p2d1))
    print(f"[orchestrator] parallel extraction wall-clock: {wall:.1f}s → {T/wall:.1f} FPS for the stereo pair")

    print("[orchestrator] triangulating Goliath-70...")
    z3 = np.zeros(3)
    tri3d = np.full((T, 70, 3), np.nan, np.float32)
    reproj = np.full((T, 70, 2), np.nan, np.float32)
    dropped = 0
    for f in range(T):
        a, b = p2d0[f], p2d1[f]
        v = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        d3 = np.full((70, 3), np.nan, np.float32)
        if v.sum():
            X = tri.triangulate(a[v].astype(np.float32), b[v].astype(np.float32))
            d3[v] = (tri.R1.T @ X.T).T
        bad = ~np.isfinite(d3).all(1) | (d3[:, 2] < args.depth_min) | (d3[:, 2] > args.depth_max)
        d3[bad] = np.nan
        vv = np.isfinite(d3).all(1)
        if not vv.any():
            continue
        rp = np.full((70, 2), np.nan, np.float32)
        rp[vv] = cv2.projectPoints(d3[vv].astype(np.float64), z3, z3, tri.K1, tri.D1)[0].reshape(-1, 2)
        bodyv = vv[:21]
        if bodyv.any() and np.median(np.linalg.norm(rp[:21][bodyv] - a[:21][bodyv], axis=1)) > args.reproj_gate:
            dropped += 1
            continue
        tri3d[f] = d3
        reproj[f] = rp

    np.save(os.path.join(args.output_dir, "joints_3d_tri.npy"), tri3d)
    np.save(os.path.join(args.output_dir, "joints_reproj_cam0.npy"), reproj)
    # Gravity-aligned world frame (nose->ankle=+Z, shoulders->+Y) — feed THIS to the
    # ACADOS IK with --no-cv-to-ros (cam0 is a tilted stereo cam, not upright/head-on).
    def _med(i):
        v = tri3d[:, i, :]; v = v[np.isfinite(v).all(1)]
        return np.median(v, 0) if len(v) else np.full(3, np.nan)
    up = (_med(5) + _med(6)) / 2 - (_med(13) + _med(14)) / 2
    world = tri3d
    if np.isfinite(up).all() and np.linalg.norm(up) > 1e-6:
        up = up / np.linalg.norm(up)
        left = _med(5) - _med(6); left = left - up * (left @ up); left /= np.linalg.norm(left)
        M = np.stack([np.cross(left, up), left, up]).astype(np.float32)
        world = (M @ tri3d.reshape(-1, 3).T).T.reshape(tri3d.shape).astype(np.float32)
    np.save(os.path.join(args.output_dir, "joints_3d_world.npy"), world)
    for tmp in (v0, v1, k0, k1):
        os.remove(tmp)

    kept = int(np.isfinite(tri3d).all(2).any(1).sum())
    rerr = np.linalg.norm(reproj - p2d0[:T], axis=2)
    print("=" * 60)
    print(f"Frames {T}  kept {kept}  dropped {dropped}")
    print(f"Reproj (cam0): median={np.nanmedian(rerr):.1f}px  p90={np.nanpercentile(rerr,90):.1f}px")
    print(f"Output: {args.output_dir}/joints_3d_tri.npy  ({T},70,3) Goliath metric")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dual-GPU two-camera extraction + triangulation")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    # worker args
    p.add_argument("--video"); p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out2d"); p.add_argument("--intrinsics", default="")
    # orchestrator args
    p.add_argument("--cam0"); p.add_argument("--cam1")
    p.add_argument("--stereo"); p.add_argument("--output_dir", default="./output_dualgpu")
    p.add_argument("--gpu0", type=int, default=7); p.add_argument("--gpu1", type=int, default=1)
    p.add_argument("--checkpoint_dir", default="/home/users/theo/code/checkpoints/sam-3d-body-dinov3")
    p.add_argument("--detector_model", default="./checkpoints/yolo/yolo11m-pose.pt")
    p.add_argument("--depth-min", type=float, default=0.2)
    p.add_argument("--depth-max", type=float, default=4.0)
    p.add_argument("--reproj-gate", type=float, default=15.0)
    a = p.parse_args()
    if a.worker:
        run_worker(a)
    else:
        run_orchestrator(a)
