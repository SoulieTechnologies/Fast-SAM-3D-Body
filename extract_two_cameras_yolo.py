#!/usr/bin/env python3
"""Two-camera triangulation — YOLO body + SAM3D hands (variant for comparison).

Same rig/calibration as extract_two_cameras.py, but the BODY is triangulated from
the **YOLO-Pose COCO-17** keypoints (pixel-accurate, no scale prior) instead of the
SAM3D Goliath body. The HANDS still come from SAM3D (cam0 mono), rigidly aligned to
the YOLO-triangulated forearm and anchored at the YOLO-triangulated wrist.

Rationale: YOLO localises body joints in 2D more accurately than SAM3D (measured),
so triangulating YOLO should give a cleaner metric body. SAM3D remains the only
source of fingers. We do NOT average YOLO and SAM3D (different joint definitions).

Outputs (in --output_dir):
  body3d_coco.npy       (T, 17, 3)  metric COCO-17 body, cam0 frame
  hands3d.npy           (T, 70, 3)  SAM3D hands placed in metric 3D (idx 21-62 + wrists)
  reproj_coco.npy       (T, 17, 2)  body reprojected on cam0
  reproj_hands.npy      (T, 70, 2)  hands reprojected on cam0
  overlay_yolo_cam0.mp4              overlay (COCO body + SAM3D hands) on cam0

Usage:
  python extract_two_cameras_yolo.py \
      --cam0 test_input/take_01/cam0.mp4 --cam1 test_input/take_01/cam1.mp4 \
      --stereo test_input/cam_params/stereo_params.npz \
      --output_dir ./output_twocam_yolo --gpu 7
"""

import os
import sys

# ── TensorRT / speed flags (identical to realtime_extractor.py) ──────────────
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
from visualize_skeleton_video import COCO_BONES, HAND_BONES

# ── COCO-17 (YOLO-Pose) indices ──
C_L_SHO, C_R_SHO = 5, 6
C_L_ELB, C_R_ELB = 7, 8
C_L_WRI, C_R_WRI = 9, 10
C_L_HIP, C_R_HIP = 11, 12
C_L_KNE, C_R_KNE = 13, 14

# ── SAM3D Goliath indices (for the mono hands we graft on) ──
G_R_WRI, G_L_WRI = 41, 62
G_R_ELB, G_L_ELB = 8, 7
R_HAND = list(range(21, 42))    # right fingers + wrist(41)
L_HAND = list(range(42, 63))    # left  fingers + wrist(62)

# SAM3D arm joints we ALSO triangulate (from both views) so the hand attaches at the
# real SAM3D wrist in SAM3D's own definition — not the (different) YOLO COCO wrist.
SAM_ARM = [G_R_ELB, G_L_ELB, G_R_WRI, G_L_WRI]   # 8, 7, 41, 62
HAND_ALL = list(range(21, 63))                    # all finger joints + both wrists

# Hand specs: (sam_wrist, sam_elbow, sam_finger_idxs, coco_wrist_for_display, key)
HANDS = [
    (G_R_WRI, G_R_ELB, R_HAND, C_R_WRI, "R"),
    (G_L_WRI, G_L_ELB, L_HAND, C_L_WRI, "L"),
]


def _quiet():
    if os.environ.get("SAM3D_PROFILE", "0") == "1":
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def _cam_int(K):
    return torch.tensor(K[None], dtype=torch.float32)


def _rot_from_vectors(a, b):
    """Rotation matrix mapping unit(a) onto unit(b) (shortest arc)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _select_person(outputs, prev_centroid, img_diag, conf_thr=0.3, w_pen=5.0):
    """Pick the main subject (largest body bbox + temporal continuity).

    Returns (yolo_kp2d (17,2), sam_kp2d (70,2), sam_kp3d (70,3), centroid) or None.
    YOLO keypoints below conf_thr are set to NaN.
    """
    cands = []
    for p in outputs:
        yk = p.get("yolo_keypoints", None)
        if yk is None:
            continue
        yk = np.asarray(yk, dtype=np.float32)          # (17, 3) x,y,conf
        xy = yk[:, :2].copy()
        xy[yk[:, 2] < conf_thr] = np.nan               # drop low-confidence
        v = np.isfinite(xy).all(1)
        if v.sum() < 4:
            continue
        bb = xy[v]
        size = (bb[:, 0].max() - bb[:, 0].min()) * (bb[:, 1].max() - bb[:, 1].min())
        centroid = bb.mean(0)
        sam2d = np.asarray(p["pred_keypoints_2d"], dtype=np.float32)
        sam3d = np.asarray(p["pred_keypoints_3d"], dtype=np.float32)
        cands.append((size, centroid, xy, sam2d, sam3d))
    if not cands:
        return None
    if prev_centroid is None:
        _, centroid, xy, sam2d, sam3d = max(cands, key=lambda c: c[0])
    else:
        def score(c):
            return c[0] / (1.0 + w_pen * np.linalg.norm(c[1] - prev_centroid) / img_diag)
        _, centroid, xy, sam2d, sam3d = max(cands, key=score)
    return xy, sam2d, sam3d, centroid


def _run_view(estimator, frame_bgr, cam_int, prev_centroid, img_diag):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad(), _quiet():
        out = estimator.process_one_image(rgb, inference_type="body", cam_int=cam_int)
    sel = _select_person(out, prev_centroid, img_diag) if out else None
    if sel is not None:
        return sel[0], sel[1], sel[2], sel[3]
    return (np.full((17, 2), np.nan, np.float32),
            np.full((70, 2), np.nan, np.float32),
            np.full((70, 3), np.nan, np.float32), prev_centroid)


def _triangulate(tri, p0, p1, n):
    """Triangulate n matched 2D points (NaN-safe) → (n,3) in ORIGINAL cam0 frame."""
    out = np.full((n, 3), np.nan, dtype=np.float32)
    v = np.isfinite(p0).all(1) & np.isfinite(p1).all(1)
    if v.sum() == 0:
        return out, v
    X = tri.triangulate(p0[v].astype(np.float32), p1[v].astype(np.float32))
    X = (tri.R1.T @ X.T).T                                   # rectified → original cam0
    out[v] = X.astype(np.float32)
    return out, v


def _global_hand_scales(arm_seq, hands_mono_seq):
    """Median (SAM3D-triangulated forearm / SAM3D-mono forearm) per hand over the clip.

    Same joint definition on both sides → a clean metric scale (~1), no inflation.
    """
    scales = {}
    for sw, se, _, _, key in HANDS:
        ratios = []
        for arm, m in zip(arm_seq, hands_mono_seq):
            if np.isfinite([arm[sw], arm[se], m[sw], m[se]]).all():
                fa_m = np.linalg.norm(m[sw] - m[se])
                if fa_m > 1e-3:
                    ratios.append(np.linalg.norm(arm[sw] - arm[se]) / fa_m)
        scales[key] = float(np.median(ratios)) if ratios else 1.0
    return scales


def _place_hands(arm3d, mono3d, scales):
    """Place SAM3D mono hands in metric 3D, anchored at the TRIANGULATED SAM3D wrist.

    arm3d holds the triangulated SAM3D wrist+elbow (SAM3D definition). The mono hand
    is aligned to that forearm, scaled by the global ratio, and anchored at the
    triangulated wrist — so it attaches where SAM3D's wrist actually is.
    Returns a (70,3) array with finger joints 21-62 filled (+ wrists at 41/62).
    """
    hand = np.full((70, 3), np.nan, dtype=np.float32)
    for sw, se, idxs, _, key in HANDS:
        w_t, e_t = arm3d[sw], arm3d[se]            # triangulated SAM3D wrist, elbow
        w_m, e_m = mono3d[sw], mono3d[se]          # SAM3D mono wrist, elbow
        if not np.isfinite([w_t, e_t, w_m, e_m]).all():
            continue
        s = scales[key]
        Rrot = _rot_from_vectors(w_m - e_m, w_t - e_t)
        for i in idxs:
            if np.isfinite(mono3d[i]).all():
                hand[i] = w_t + s * (Rrot @ (mono3d[i] - w_m))
    return hand


def _draw(frame, coco2d, hand2d, w, h):
    """Overlay COCO-17 body + SAM3D hands (already reprojected) on a frame."""
    ov = frame.copy()
    ok = lambda p: np.isfinite(p).all() and 0 <= p[0] < w and 0 <= p[1] < h
    for a, b, col in COCO_BONES:
        pa, pb = coco2d[a], coco2d[b]
        if ok(pa) and ok(pb):
            cv2.line(ov, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), col, 3, cv2.LINE_AA)
    for a, b, col in HAND_BONES:
        pa, pb = hand2d[a], hand2d[b]
        if ok(pa) and ok(pb):
            cv2.line(ov, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), col, 2, cv2.LINE_AA)
    for i in range(17):
        if ok(coco2d[i]):
            cv2.circle(ov, (int(coco2d[i][0]), int(coco2d[i][1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(ov, (int(coco2d[i][0]), int(coco2d[i][1])), 5, (0, 0, 0), 1, cv2.LINE_AA)
    for i in range(21, 63):
        if ok(hand2d[i]):
            cv2.circle(ov, (int(hand2d[i][0]), int(hand2d[i][1])), 3, (255, 200, 100), -1, cv2.LINE_AA)
    return ov


def _report_metric(name, body3d):
    mid = lambda a, i, k: (a[:, i] + a[:, k]) / 2
    torso = np.linalg.norm(mid(body3d, C_L_SHO, C_R_SHO) - mid(body3d, C_L_HIP, C_R_HIP), axis=1)
    luarm = np.linalg.norm(body3d[:, C_L_SHO] - body3d[:, C_L_ELB], axis=1)
    lthigh = np.linalg.norm(body3d[:, C_L_HIP] - body3d[:, C_L_KNE], axis=1)
    f = lambda x: np.nanmedian(x)
    print(f"  {name:16s} torso={f(torso)*100:5.1f}cm  L-upperarm={f(luarm)*100:5.1f}cm  "
          f"L-thigh={f(lthigh)*100:5.1f}cm")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    stereo_dir = os.path.dirname(os.path.abspath(args.stereo))
    sys.path.insert(0, stereo_dir)
    from triangulate import StereoTriangulator
    tri = StereoTriangulator(args.stereo)
    cam_int0, cam_int1 = _cam_int(tri.K1), _cam_int(tri.K2)
    print(f"[1/4] Stereo calib: img_size={tri.img_size}  baseline≈{np.linalg.norm(tri.T)*100:.1f}cm")

    print("[2/4] Loading SAM-3D-Body (detector=yolo_pose for COCO-17 keypoints)...")
    det = args.detector_model
    if det.endswith(".pt") and os.path.exists(det.replace(".pt", ".engine")):
        det = det.replace(".pt", ".engine")
    estimator = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(args.checkpoint_dir, "assets", "mhr_model.pt"),
        detector_name="yolo_pose",                  # exposes COCO-17 keypoints per person
        detector_model=det,
        fov_name="",
        device="cuda",
    )

    cap0, cap1 = cv2.VideoCapture(args.cam0), cv2.VideoCapture(args.cam1)
    if not (cap0.isOpened() and cap1.isOpened()):
        raise ValueError("Cannot open one of the input videos.")
    n0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = min(n0, n1)
    print(f"[3/4] processing {total} synced frames ({w}x{h})")

    img_diag = float(np.hypot(*tri.img_size))
    cen0 = cen1 = None
    yolo0, yolo1, sam0, sam1, hands_mono0, times = [], [], [], [], [], []
    for idx in range(total):
        r0, f0 = cap0.read()
        r1, f1 = cap1.read()
        if not (r0 and r1):
            break
        t0 = time.perf_counter()
        y0, s2d0, m0, cen0 = _run_view(estimator, f0, cam_int0, cen0, img_diag)
        y1, s2d1, _, cen1 = _run_view(estimator, f1, cam_int1, cen1, img_diag)
        times.append(time.perf_counter() - t0)
        yolo0.append(y0); yolo1.append(y1); sam0.append(s2d0); sam1.append(s2d1); hands_mono0.append(m0)
        if idx % 20 == 0:
            avg = 1.0 / np.mean(times[-20:]) if times else 0
            print(f"  frame {idx}/{total}  {1.0/times[-1]:.1f} FPS (avg {avg:.1f})")
    cap0.release(); cap1.release()

    yolo0 = np.stack(yolo0); yolo1 = np.stack(yolo1)
    sam0 = np.stack(sam0); sam1 = np.stack(sam1); hands_mono0 = np.stack(hands_mono0)
    T = len(yolo0)

    print("[4/4] Triangulating YOLO body + SAM3D wrists, placing hands...")
    body3d = np.full((T, 17, 3), np.nan, np.float32)
    arm3d = np.full((T, 70, 3), np.nan, np.float32)    # triangulated SAM3D elbows+wrists
    for f in range(T):
        b, _ = _triangulate(tri, yolo0[f], yolo1[f], 17)
        bad = ~np.isfinite(b).all(1) | (b[:, 2] < args.depth_min) | (b[:, 2] > args.depth_max)
        b[bad] = np.nan
        body3d[f] = b
        # triangulate the SAM3D arm joints (same stereo geometry) for hand attachment
        a, _ = _triangulate(tri, sam0[f][SAM_ARM], sam1[f][SAM_ARM], len(SAM_ARM))
        arm3d[f, SAM_ARM] = a

    scales = _global_hand_scales(arm3d, hands_mono0)
    if args.hands == "b2":
        print(f"  hands=b2  global hand scale: right={scales['R']:.3f}  left={scales['L']:.3f}")
    else:
        print(f"  hands=tri  (triangulating fingers directly from both views)")

    hands3d = np.full((T, 70, 3), np.nan, np.float32)
    reproj_c = np.full((T, 17, 2), np.nan, np.float32)
    reproj_h = np.full((T, 70, 2), np.nan, np.float32)
    z3 = np.zeros(3)
    dropped = 0
    for f in range(T):
        b = body3d[f]
        vb = np.isfinite(b).all(1)
        if vb.any():
            rc = cv2.projectPoints(b[vb].astype(np.float64), z3, z3, tri.K1, tri.D1)[0].reshape(-1, 2)
            cand = np.full((17, 2), np.nan, np.float32); cand[vb] = rc
            err = np.linalg.norm(cand[vb] - yolo0[f][vb], axis=1)
            if np.median(err) > args.reproj_gate:        # cam0/cam1 person mismatch
                dropped += 1
                body3d[f] = np.nan
                continue
            reproj_c[f] = cand
        if args.hands == "tri":
            # triangulate the SAM3D fingers directly from both views
            hand = np.full((70, 3), np.nan, np.float32)
            ht, _ = _triangulate(tri, sam0[f][HAND_ALL], sam1[f][HAND_ALL], len(HAND_ALL))
            bad = ~np.isfinite(ht).all(1) | (ht[:, 2] < args.depth_min) | (ht[:, 2] > args.depth_max)
            ht[bad] = np.nan
            hand[HAND_ALL] = ht
        else:
            hand = _place_hands(arm3d[f], hands_mono0[f], scales)
        hands3d[f] = hand
        vh = np.isfinite(hand).all(1)
        if vh.any():
            reproj_h[f, vh] = cv2.projectPoints(hand[vh].astype(np.float64), z3, z3, tri.K1, tri.D1)[0].reshape(-1, 2)
        # connect the YOLO forearm to the SAM3D wrist so arm and hand join continuously
        for sw, _, _, cw, _ in HANDS:
            if np.isfinite(reproj_h[f, sw]).all():
                reproj_c[f, cw] = reproj_h[f, sw]

    out = args.output_dir
    np.save(os.path.join(out, "body3d_coco.npy"), body3d)
    np.save(os.path.join(out, "hands3d.npy"), hands3d)
    np.save(os.path.join(out, "reproj_coco.npy"), reproj_c)
    np.save(os.path.join(out, "reproj_hands.npy"), reproj_h)

    # ── Overlay on cam0 ──
    print("  writing overlay...")
    cap = cv2.VideoCapture(args.cam0)
    vw = cv2.VideoWriter(os.path.join(out, "overlay_yolo_cam0.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in range(T):
        ret, frame = cap.read()
        if not ret:
            break
        vw.write(_draw(frame, reproj_c[f], reproj_h[f], w, h))
    cap.release(); vw.release()

    rerr = np.linalg.norm(reproj_c - yolo0, axis=2)
    kept = int(np.isfinite(body3d).all(2).any(1).sum())
    print("\n" + "=" * 64)
    print(f"Frames: {T}   per-frame FPS (both views): {1.0/np.mean(times):.1f}")
    print(f"Kept {kept}/{T}   dropped {dropped} (reproj gate > {args.reproj_gate}px)")
    print(f"Body reprojection error (cam0, kept): median={np.nanmedian(rerr):.1f}px  "
          f"p90={np.nanpercentile(rerr, 90):.1f}px")
    _report_metric("YOLO-triangulated", body3d)
    print("=" * 64)
    print(f"Saved to {out}/  (overlay_yolo_cam0.mp4, body3d_coco.npy, ...)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Two-camera triangulation: YOLO body + SAM3D hands")
    p.add_argument("--cam0", required=True)
    p.add_argument("--cam1", required=True)
    p.add_argument("--stereo", required=True)
    p.add_argument("--output_dir", default="./output_twocam_yolo")
    p.add_argument("--checkpoint_dir", default="/home/users/theo/code/checkpoints/sam-3d-body-dinov3")
    p.add_argument("--detector_model", default="./checkpoints/yolo/yolo11m-pose.pt")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--hands", choices=["b2", "tri"], default="b2",
                   help="b2: mono hand anchored at triangulated SAM3D wrist; tri: triangulate fingers from both views")
    p.add_argument("--depth-min", type=float, default=0.2)
    p.add_argument("--depth-max", type=float, default=4.0)
    p.add_argument("--reproj-gate", type=float, default=15.0)
    main(p.parse_args())
