#!/usr/bin/env python3
"""Two-camera SAM-3D-Body extraction + stereo triangulation.

Runs SAM-3D-Body on two synchronised, calibrated camera views, then
triangulates the 70 Goliath 2D keypoints into **metric** 3D — removing the
monocular scale ambiguity at the source (no average-human prior).

Hands: by default the 42 finger joints are triangulated like the rest.
With ``--hands b2`` they instead use cam0's mono SAM3D hand, metric-scaled
by the triangulated forearm and re-anchored at the triangulated wrist
(smoother than triangulating small/noisy finger detections).

Outputs (in --output_dir):
  joints_3d_tri.npy        (T, 70, 3)  metric 3D in cam0 frame (triangulated)
  joints_2d_cam0.npy       (T, 70, 2)  SAM3D 2D keypoints, cam0
  joints_2d_cam1.npy       (T, 70, 2)  SAM3D 2D keypoints, cam1
  joints_reproj_cam0.npy   (T, 70, 2)  triangulated 3D reprojected onto cam0 (overlay)
  joints_3d_mono0.npy      (T, 70, 3)  cam0 mono SAM3D 3D (prior-scaled, for comparison)

Usage:
  python extract_two_cameras.py \
      --cam0 test_input/take_01/cam0.mp4 \
      --cam1 test_input/take_01/cam1.mp4 \
      --stereo test_input/cam_params/stereo_params.npz \
      --output_dir ./output_twocam --gpu 7

Visualise the triangulated skeleton reprojected on cam0:
  python utils/visualize_skeleton_video.py \
      --npy ./output_twocam/joints_reproj_cam0.npy \
      --video test_input/take_01/cam0.mp4 \
      --output ./output_twocam/overlay_tri_cam0.mp4
"""

import os
import sys

# ── TensorRT / speed flags — must be set before importing sam_3d_body ────────
# Identical to realtime_extractor.py (body-only ~16 FPS per view).
os.environ.setdefault("USE_TRT_BACKBONE", "1")
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")
os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")
os.environ.setdefault("USE_COMPILE", "0")
os.environ.setdefault("LAYER_DTYPE", "fp32")
os.environ.setdefault("GPU_HAND_PREP", "1")
os.environ.setdefault("INTERM_PRED_INTERVAL", "999")
os.environ.setdefault("KEYPOINT_PROMPT_INTERM_INTERVAL", "999")
os.environ.setdefault("COMPILE_MODE", "default")
os.environ.setdefault("MHR_USE_CUDA_GRAPH", "0")
os.environ.setdefault(
    "TRT_BACKBONE_PATH",
    "/home/users/theo/code/checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine",
)

# Pin GPU before importing torch (CUDA_VISIBLE_DEVICES is ignored after init).
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    import argparse as _ap

    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument("--gpu", type=int, default=0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre.parse_known_args()[0].gpu)

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

import argparse
import contextlib
import io
import time

import cv2
import numpy as np
import torch
from notebook.utils import setup_sam_3d_body

# Goliath-70 keypoint indices used for metric sanity checks / B2 hands.
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_HIP, R_HIP = 9, 10
L_KNE, R_KNE = 11, 12
R_WRI, L_WRI = 41, 62  # SAM3D Goliath wrist indices
R_HAND = range(21, 42)  # right-hand finger joints (incl. wrist 41)
L_HAND = range(42, 63)  # left-hand finger joints (incl. wrist 62)


def _quiet():
    """Suppress per-frame model stdout unless SAM3D_PROFILE=1."""
    if os.environ.get("SAM3D_PROFILE", "0") == "1":
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def _cam_int(K):
    """(3,3) numpy intrinsics → (1,3,3) float32 torch tensor for SAM3D."""
    return torch.tensor(K[None], dtype=torch.float32)


def _select_person(outputs, prev_centroid, img_diag, w_pen=5.0):
    """Pick the main subject among detections (largest body bbox + temporal continuity).

    The subject is the closest/biggest person to the converging cameras; bystanders
    are smaller. A penalty on distance from the previous frame's centroid keeps the
    same identity across frames and avoids the detector's order swapping people.

    Returns (kp2d (70,2), kp3d (70,3), centroid (2,)) or None if no valid person.
    """
    cands = []
    for p in outputs:
        k2 = np.asarray(p["pred_keypoints_2d"], dtype=np.float32)
        k3 = np.asarray(p["pred_keypoints_3d"], dtype=np.float32)
        body = k2[:21]
        v = np.isfinite(body).all(1)
        if v.sum() < 4:
            continue
        bb = body[v]
        size = (bb[:, 0].max() - bb[:, 0].min()) * (
            bb[:, 1].max() - bb[:, 1].min()
        )
        centroid = bb.mean(0)
        cands.append((size, centroid, k2, k3))
    if not cands:
        return None
    if prev_centroid is None:
        size, centroid, k2, k3 = max(cands, key=lambda c: c[0])
    else:
        # largest, but penalised for jumping away from the tracked subject
        def score(c):
            return c[0] / (
                1.0 + w_pen * np.linalg.norm(c[1] - prev_centroid) / img_diag
            )

        size, centroid, k2, k3 = max(cands, key=score)
    return k2, k3, centroid


def _run_view(
    estimator, frame_bgr, cam_int, body_only, prev_centroid, img_diag
):
    """Run SAM3D on one BGR frame, select the main subject.

    Returns (kp2d (70,2), kp3d (70,3), centroid or prev_centroid). On no detection,
    returns NaNs and keeps prev_centroid so tracking resumes near the last position.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad(), _quiet():
        out = estimator.process_one_image(
            rgb,
            inference_type="body" if body_only else "full",
            cam_int=cam_int,
        )
    sel = _select_person(out, prev_centroid, img_diag) if out else None
    if sel is not None:
        return sel[0], sel[1], sel[2]
    nan2 = np.full((70, 2), np.nan, dtype=np.float32)
    nan3 = np.full((70, 3), np.nan, dtype=np.float32)
    return nan2, nan3, prev_centroid


def _triangulate_frame(tri, p0, p1):
    """Triangulate one frame's 70 keypoints, NaN-safe. Returns (70,3) metric."""
    out = np.full((70, 3), np.nan, dtype=np.float64)
    valid = np.isfinite(p0).all(1) & np.isfinite(p1).all(1)
    if valid.sum() == 0:
        return out
    pts3d = tri.triangulate(
        p0[valid].astype(np.float32), p1[valid].astype(np.float32)
    )
    out[valid] = pts3d
    return out


def _global_forearm_scales(tri_seq, mono_seq):
    """Median forearm-length ratio (triangulated/mono) per hand over the whole clip.

    A single global scalar per hand is far more stable than a per-frame ratio
    (which explodes when the triangulated forearm is momentarily tiny/noisy).
    """
    scales = {}
    for wri, elb, key in [(R_WRI, R_ELB, "R"), (L_WRI, L_ELB, "L")]:
        ratios = []
        for t, m in zip(tri_seq, mono_seq):
            if np.isfinite([t[wri], t[elb], m[wri], m[elb]]).all():
                fa_m = np.linalg.norm(m[wri] - m[elb])
                if fa_m > 1e-3:
                    ratios.append(np.linalg.norm(t[wri] - t[elb]) / fa_m)
        scales[key] = float(np.median(ratios)) if ratios else 1.0
    return scales


def _rot_from_vectors(a, b):
    """Rotation matrix mapping unit(a) onto unit(b) (shortest arc)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:  # parallel (or anti-parallel)
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _b2_hands_global(tri_3d, mono_3d, scales):
    """Rigidly attach the cam0 mono hand to the triangulated forearm.

    For each hand: rotate the mono hand so its forearm (elbow→wrist) aligns with the
    triangulated forearm, scale by the GLOBAL ratio, and anchor at the triangulated
    wrist. Keeps the mono finger articulation but places it correctly in metric 3D.
    """
    out = tri_3d.copy()
    for wri, elb, rng, key in [
        (R_WRI, R_ELB, R_HAND, "R"),
        (L_WRI, L_ELB, L_HAND, "L"),
    ]:
        w_t, e_t = tri_3d[wri], tri_3d[elb]
        w_m, e_m = mono_3d[wri], mono_3d[elb]
        if not np.isfinite([w_t, e_t, w_m, e_m]).all():
            continue
        s = scales[key]
        Rrot = _rot_from_vectors(
            w_m - e_m, w_t - e_t
        )  # align forearm direction
        for i in rng:
            if np.isfinite(mono_3d[i]).all():
                out[i] = w_t + s * (Rrot @ (mono_3d[i] - w_m))
    return out


def _seg(a, i, j):
    return np.linalg.norm(a[:, i] - a[:, j], axis=1)


def _report_metric(name, j3d):
    """Print median metric limb lengths (a human torso ≈ 0.45-0.55 m)."""
    mid = lambda a, i, k: (a[:, i] + a[:, k]) / 2
    torso = np.linalg.norm(
        mid(j3d, L_SHO, R_SHO) - mid(j3d, L_HIP, R_HIP), axis=1
    )
    luarm = _seg(j3d, L_SHO, L_ELB)
    lthigh = _seg(j3d, L_HIP, L_KNE)
    f = lambda x: np.nanmedian(x)
    print(
        f"  {name:14s} torso={f(torso)*100:5.1f}cm  L-upperarm={f(luarm)*100:5.1f}cm  "
        f"L-thigh={f(lthigh)*100:5.1f}cm"
    )


def canonical_align(tri3d):
    """Rotate the triangulated 3D (cam0 frame) into a gravity-aligned ROS frame.

    The stereo cam0 is a tilted, converging camera, so its frame is neither upright
    nor head-on — feeding it straight to the IK gives a leaning/sideways person.
    This computes ONE rotation from the median pose so the body is upright
    (nose→ankle = +Z), facing forward (shoulders → +Y), right-handed (no mirror).
    Returns (world_3d (T,70,3), M (3,3)); ready for the IK with --no-cv-to-ros.
    """

    def med(idx):
        v = tri3d[:, idx, :]
        v = v[np.isfinite(v).all(1)]
        return np.median(v, 0) if len(v) else np.full(3, np.nan)

    lsho, rsho = med(L_SHO), med(R_SHO)
    mank = (med(13) + med(14)) / 2
    msho = (lsho + rsho) / 2
    up = msho - mank
    if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
        return tri3d.copy(), np.eye(
            3, dtype=np.float32
        )  # can't align → passthrough
    up /= np.linalg.norm(up)
    left = lsho - rsho
    left = left - up * (left @ up)
    left /= np.linalg.norm(left)
    fwd = np.cross(left, up)
    M = np.stack([fwd, left, up]).astype(np.float32)  # world = M @ x
    world = (
        (M @ tri3d.reshape(-1, 3).T).T.reshape(tri3d.shape).astype(np.float32)
    )
    return world, M


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Reuse the user's existing StereoTriangulator (cam_params/triangulate.py).
    stereo_dir = os.path.dirname(os.path.abspath(args.stereo))
    sys.path.insert(0, stereo_dir)
    from triangulate import StereoTriangulator

    tri = StereoTriangulator(args.stereo)
    cam_int0, cam_int1 = _cam_int(tri.K1), _cam_int(tri.K2)
    print(
        f"[1/4] Loaded stereo calib: img_size={tri.img_size}  baseline≈{np.linalg.norm(tri.T)*100:.1f}cm"
    )

    print("[2/4] Loading SAM-3D-Body...")
    det = args.detector_model
    if det.endswith(".pt") and os.path.exists(det.replace(".pt", ".engine")):
        det = det.replace(".pt", ".engine")
    estimator = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(
            args.checkpoint_dir, "assets", "mhr_model.pt"
        ),
        detector_name="yolo",
        detector_model=det,
        fov_name="",  # fixed per-camera intrinsics, no MoGe2
        device="cuda",
    )

    cap0, cap1 = cv2.VideoCapture(args.cam0), cv2.VideoCapture(args.cam1)
    if not (cap0.isOpened() and cap1.isOpened()):
        raise ValueError("Cannot open one of the input videos.")
    n0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(n0, n1)
    print(
        f"[3/4] cam0={n0} frames, cam1={n1} frames → processing {total} synced frames"
    )
    if n0 != n1:
        print(
            f"  WARNING: frame counts differ ({n0} vs {n1}); assuming aligned from frame 0."
        )

    img_diag = float(
        np.hypot(*tri.img_size)
    )  # for the tracking distance penalty
    cen0 = cen1 = None  # tracked subject centroid per view

    kp2d_0, kp2d_1, kp3d_mono0, times = [], [], [], []
    for idx in range(total):
        r0, f0 = cap0.read()
        r1, f1 = cap1.read()
        if not (r0 and r1):
            break
        t0 = time.perf_counter()
        p0, m0, cen0 = _run_view(
            estimator, f0, cam_int0, args.body_only, cen0, img_diag
        )
        p1, _, cen1 = _run_view(
            estimator, f1, cam_int1, args.body_only, cen1, img_diag
        )
        times.append(time.perf_counter() - t0)
        kp2d_0.append(p0)
        kp2d_1.append(p1)
        kp3d_mono0.append(m0)
        if idx % 20 == 0:
            avg = 1.0 / np.mean(times[-20:]) if times else 0
            print(
                f"  frame {idx}/{total}  {1.0/times[-1]:.1f} FPS (avg {avg:.1f})"
            )
    cap0.release()
    cap1.release()

    kp2d_0 = np.stack(kp2d_0)
    kp2d_1 = np.stack(kp2d_1)
    kp3d_mono0 = np.stack(kp3d_mono0)

    print("[4/4] Triangulating...")
    T = len(kp2d_0)
    z3 = np.zeros(3)

    # Pass 1 — triangulate to the ORIGINAL cam0 frame, reject implausible depths.
    raw = np.full((T, 70, 3), np.nan, dtype=np.float32)
    for f in range(T):
        d3 = _triangulate_frame(
            tri, kp2d_0[f], kp2d_1[f]
        )  # rectified cam0 frame
        v = np.isfinite(d3).all(1)
        if v.any():
            d3[v] = (tri.R1.T @ d3[v].T).T  # -> original cam0 frame
        # depth-outlier rejection (z in metres, in front of the camera)
        bad = (
            ~np.isfinite(d3).all(1)
            | (d3[:, 2] < args.depth_min)
            | (d3[:, 2] > args.depth_max)
        )
        d3[bad] = np.nan
        raw[f] = d3

    # Global per-hand forearm scale (stable), then B2 hands.
    scales = _global_forearm_scales(raw, kp3d_mono0)
    if args.hands == "b2":
        print(
            f"  global forearm scale: right={scales['R']:.3f}  left={scales['L']:.3f}"
        )

    # Pass 2 — B2 hands + reproject + per-frame reprojection gate (drops cam0/cam1
    # person mismatches: if the body reprojects far, the two views saw different people).
    tri_3d = np.full((T, 70, 3), np.nan, dtype=np.float32)
    reproj = np.full((T, 70, 2), np.nan, dtype=np.float32)
    dropped = 0
    for f in range(T):
        d3 = (
            _b2_hands_global(raw[f], kp3d_mono0[f], scales)
            if args.hands == "b2"
            else raw[f].copy()
        )
        v = np.isfinite(d3).all(1)
        if not v.any():
            continue
        pts = cv2.projectPoints(
            d3[v].astype(np.float64), z3, z3, tri.K1, tri.D1
        )[0].reshape(-1, 2)
        rp = np.full((70, 2), np.nan, dtype=np.float32)
        rp[v] = pts.astype(np.float32)
        # gate on the BODY (idx 0-20) reprojection error
        body = np.arange(21)
        bv = v[body]
        if bv.any():
            err = np.linalg.norm(rp[body][bv] - kp2d_0[f][body][bv], axis=1)
            if np.median(err) > args.reproj_gate:
                dropped += 1
                continue  # mismatch → drop frame
        tri_3d[f] = d3
        reproj[f] = rp

    # ── Save ────────────────────────────────────────────────────────────────
    out = args.output_dir
    np.save(
        os.path.join(out, "joints_3d_tri.npy"), tri_3d
    )  # cam0 frame (for overlay)
    np.save(os.path.join(out, "joints_2d_cam0.npy"), kp2d_0)
    np.save(os.path.join(out, "joints_2d_cam1.npy"), kp2d_1)
    np.save(os.path.join(out, "joints_reproj_cam0.npy"), reproj)
    np.save(os.path.join(out, "joints_3d_mono0.npy"), kp3d_mono0)
    # Gravity-aligned world frame — feed THIS to the ACADOS IK (with --no-cv-to-ros).
    world_3d, _ = canonical_align(tri_3d)
    np.save(os.path.join(out, "joints_3d_world.npy"), world_3d)

    # ── Verification ──────────────────────────────────────────────────────────
    rerr = np.linalg.norm(reproj - kp2d_0, axis=2)
    kept = int(np.isfinite(tri_3d).all(2).any(1).sum())
    print("\n" + "=" * 64)
    print(
        f"Frames: {T}   per-frame FPS (both views): {1.0/np.mean(times):.1f} "
        f"(median {np.median(times)*1000:.0f} ms)"
    )
    print(
        f"Kept {kept}/{T} frames   dropped {dropped} (reproj gate > {args.reproj_gate}px → person mismatch)"
    )
    print(
        f"Triangulation reprojection error (cam0, kept): "
        f"median={np.nanmedian(rerr):.1f}px  p90={np.nanpercentile(rerr, 90):.1f}px"
    )
    print("Metric limb lengths (triangulated should be plausible & stable):")
    _report_metric("triangulated", tri_3d)
    _report_metric("mono cam0", kp3d_mono0)
    print("=" * 64)
    print(f"Saved to {out}/  (joints_3d_tri.npy, joints_reproj_cam0.npy, ...)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Two-camera SAM3D extraction + triangulation"
    )
    p.add_argument("--cam0", required=True, help="cam0 video")
    p.add_argument("--cam1", required=True, help="cam1 video")
    p.add_argument(
        "--stereo",
        required=True,
        help="stereo_params.npz (with K1,D1,K2,D2,R,T,P1,P2,R1,R2)",
    )
    p.add_argument("--output_dir", default="./output_twocam")
    p.add_argument(
        "--checkpoint_dir",
        default="/home/users/theo/code/checkpoints/sam-3d-body-dinov3",
    )
    p.add_argument(
        "--detector_model", default="./checkpoints/yolo/yolo11m-pose.pt"
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--hands",
        choices=["tri", "b2"],
        default="b2",
        help="tri: triangulate fingers; b2: cam0 mono hand scaled+anchored to triangulated wrist",
    )
    p.add_argument(
        "--depth-min",
        type=float,
        default=0.2,
        help="reject triangulated points closer than this (m)",
    )
    p.add_argument(
        "--depth-max",
        type=float,
        default=4.0,
        help="reject triangulated points farther than this (m)",
    )
    p.add_argument(
        "--reproj-gate",
        type=float,
        default=15.0,
        help="drop a frame if median body reprojection error exceeds this (px) — catches cam0/cam1 person mismatch",
    )
    p.add_argument("--body-only", action="store_true", default=True)
    p.add_argument("--no-body-only", action="store_false", dest="body_only")
    main(p.parse_args())
