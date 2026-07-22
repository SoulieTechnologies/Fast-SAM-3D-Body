#!/usr/bin/env python3
"""COSMIK/NLF body + SAM hand-decoder fingers — multi-camera metric 3D keypoints.

Body backbone = RT-COSMIK's NLF path (branch nlf_humble): YOLO person
detection + the NLF torchscript predict, per calibrated view, the 2D pixel
positions of 43 ANATOMICAL MARKERS (SMPL-X canonical vertices — mocap-style
RASI/C7/RELB+RMELB/RWRI+RMWRI...). The 2D markers are triangulated into
METRIC 3D (removing the monocular scale ambiguity), and the dedicated SAM
hand decoder adds faithful fingers from one view. No IK here — this produces
and records the keypoints (Rerun UI + .npy), ready to feed the ACADOS
pipeline later.

Architecture (decoupled rates — the body never waits for the hands):
  capture thread per camera  → latest frame + timestamp (soft sync)
  body worker                → YOLO + NLF per view (batched) → triangulate 43×3 metric
  hand worker                → SAM hand decoder on each hand's BEST K views
                               (3+ cams, --hand-topk: crop size x palm-plane
                               visibility x NLF conf, hysteresis; 2 cams =
                               every view), all crops in ONE batched forward
                               → hand keypoints STEREO-triangulated across
                               views (metric scale, occlusion-robust;
                               per-joint epipolar check, mono fallback;
                               --mono-hands = old hand-cam-only behaviour)
  main loop      (body rate) → fuse (stereo hands, else fingers re-anchored at
                               the metric wrists), Rerun logging, recording

Outputs (in --output_dir):
  markers_3d.npy      (T, 43, 3)       metric 3D, cam0/world frame (MARKER_NAMES order)
  markers_2d.npy      (T, ncam, 43, 2)
  hands_2d.npy        (T, 42, 2)       best-view pixels (right 0-20, left 21-41)
  hands_src.npy       (T, 2)           which camera hands_2d came from (R, L; -1 none)
  hands_2d_views.npy  (T, ncam, 42, 2) decoder pixels per view (stereo input)
  goliath70_3d.npy    (T, 70, 3)       markers mapped to Goliath + fused fingers
  timestamps.npy      (T,)
  timing.log          per-hand-iteration profiling (always written; body ms,
                      hand prep/fwd/post/tri ms, stereo joint counts)
  overlay_cam0.mp4    (with --save-video)

Calibration (--calib) accepts:
  - stereo npz  : keys K1,D1,K2,D2,R,T          (2 cameras, cam0 = reference)
  - multi npz   : keys K0..K{n},D0..,R0..,T0..  (R0=I, T0=0 for the reference)

Setup (sam3d env): pip install ultralytics meshcat; clone RT-COSMIK branch
nlf_humble (default location ~/code/RT-COSMIK, override with --rtcosmik) and
drop the NLF weights in <rtcosmik>/weights/ (see --nlf-weights/--cano).

Run:
  # multi-camera (metric 3D):
  python cosmik_hand_demo.py --cams 0,1 --calib stereo_params.npz \
      --checkpoint_dir ./checkpoints/sam-3d-body-dinov3 --rerun-mode native
  # 4 cameras (2 front + 2 side): calibrate with stereo_calibration/
  # capture_calibration_multi.py + calibrate_multi.py, then — per-hand
  # best-view selection kicks in automatically (top-2, --hand-topk):
  python cosmik_hand_demo.py --cams 0,1,2,3 --calib multi_params.npz \
      --cap-width 1920 --cap-height 1080 --rerun-mode web
  # MONO mode (no calib yet): markers 2D + hand-decoder fingers, no metric body 3D
  python cosmik_hand_demo.py --cams 0 --fx 540 --rerun-mode native

OFFLINE (record now, infer later) — same inference, every frame, no drop-old:
  # 1. record all 4 cams synchronized (stereo_calibration/record_multi.py)
  # 2. replay the SAME recording for the 4-cam body and the 2-cam hands:
  python cosmik_hand_demo.py --cams 0,1,2,3 --calib multi_params.npz \
      --cap-width 1920 --cap-height 1080 --replay recordings/take_01
  python cosmik_hand_demo.py --cams 0,1 --calib stereo_params.npz \
      --cap-width 1920 --cap-height 1080 --replay recordings/take_01
"""

import os
import sys

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# Import first: stream_demo sets the TensorRT/speed env flags before torch is
# imported. Also reused directly for its TCP keypoint emitter (_emit_server/_EMIT).
import stream_demo

import argparse
import threading
import time
import warnings

# a dependency spams a "half is deprecated, use quantize" deprecation each
# frame — harmless, silence it (python-warnings based emitters only)
warnings.filterwarnings("ignore", message=r".*[Hh]alf.*deprecat.*")
warnings.filterwarnings("ignore", message=r".*deprecat.*[Hh]alf.*")

import cv2
import numpy as np
import torch
from notebook.utils import setup_sam_3d_body

# Reuse the proven hand-decoder building blocks (elbow→wrist crops, batched,
# un-flipped) from the Rerun demo — nothing is modified there.
from rerun_demo import (
    _H_WRIST,
    _hand_box_v2,
    _quiet,
    HAND_SRC,
    L_ELBOW,
    L_WRIST,
    R_ELBOW,
    R_WRIST,
)
from utils.hand_view_select import (
    in_frame_fraction,
    palm_normal,
    select_views,
    view_visibility,
)
from sam_3d_body.models.meta_arch.sam3d_body import _prepare_hand_batches_gpu

# ═══════════════════════════════════════════════════════════════════════════
# NLF ANATOMICAL MARKERS — names, SMPL-X vertex ids, skeleton, Goliath-70 map
# (markers + vertex ids come from RT-COSMIK settings.py, branch nlf_humble)
# ═══════════════════════════════════════════════════════════════════════════

MARKER_NAMES = [
    "RASI",
    "LASI",
    "RPSI",
    "LPSI",  # pelvis
    "C7",
    "T11",
    "T6",
    "RSHO",
    "LSHO",
    "RELB",
    "LELB",
    "RMELB",  # upper
    "LMELB",
    "RWRI",
    "LWRI",
    "RMWRI",
    "LMWRI",
    "RTHU",
    "LTHU",
    "RMID",
    "LMID",
    "RPIN",
    "LPIN",  # hands
    "RKNE",
    "LKNE",
    "RMKNE",
    "LMKNE",
    "RANK",
    "LANK",
    "RMANK",
    "LMANK",  # legs
    "R5MHD",
    "L5MHD",
    "RTOE",
    "LTOE",
    "RHEE",
    "LHEE",  # feet
    "Nose",
    "Head",
    "REar",
    "LEar",
    "REye",
    "LEye",  # face
]
NMK = len(MARKER_NAMES)  # 43
_M = {n: i for i, n in enumerate(MARKER_NAMES)}

# SMPL-X canonical vertex ids, one per marker (same order as MARKER_NAMES)
NLF_INDICES = [
    8421,
    5727,
    8371,
    5677,
    5484,
    5489,
    5500,
    6629,
    3878,
    7040,
    4302,
    7105,
    4369,
    7584,
    4848,
    7457,
    4721,
    8079,
    5361,
    7794,
    5058,
    8022,
    5286,
    6401,
    3640,
    6407,
    3646,
    8576,
    5882,
    8680,
    8892,
    8474,
    5780,
    8463,
    5770,
    8635,
    8846,
    9120,
    9002,
    616,
    6,
    9929,
    9448,
]

# display skeleton over the LATERAL marker chain (medial markers drawn as dots)
MARKER_EDGES = [
    (_M[a], _M[b])
    for a, b in [
        ("Nose", "REar"),
        ("Nose", "LEar"),
        ("Head", "Nose"),
        ("C7", "Head"),
        ("C7", "RSHO"),
        ("C7", "LSHO"),
        ("C7", "T6"),
        ("T6", "T11"),
        ("T11", "RPSI"),
        ("T11", "LPSI"),
        ("RPSI", "RASI"),
        ("LPSI", "LASI"),
        ("RASI", "LASI"),
        ("RSHO", "RELB"),
        ("RELB", "RWRI"),
        ("LSHO", "LELB"),
        ("LELB", "LWRI"),
        ("RWRI", "RTHU"),
        ("RWRI", "RMID"),
        ("RWRI", "RPIN"),
        ("LWRI", "LTHU"),
        ("LWRI", "LMID"),
        ("LWRI", "LPIN"),
        ("RASI", "RKNE"),
        ("RKNE", "RANK"),
        ("RANK", "RTOE"),
        ("RANK", "RHEE"),
        ("LASI", "LKNE"),
        ("LKNE", "LANK"),
        ("LANK", "LTOE"),
        ("LANK", "LHEE"),
    ]
]

# marker → Goliath-70 slot (single markers; elbow/wrist joint centres are the
# lateral+medial midpoints and are filled separately in fuse_goliath70).
MARKER2GOLIATH = {
    _M[m]: g
    for m, g in {
        "Nose": 0,
        "LEye": 1,
        "REye": 2,
        "LEar": 3,
        "REar": 4,
        "LSHO": 5,
        "RSHO": 6,
        "LASI": 9,
        "RASI": 10,
        "LKNE": 11,
        "RKNE": 12,
        "LANK": 13,
        "RANK": 14,
        "LTOE": 15,
        "L5MHD": 16,
        "LHEE": 17,
        "RTOE": 18,
        "R5MHD": 19,
        "RHEE": 20,
        "C7": 69,
    }.items()
}
# lateral/medial pairs → Goliath joint-centre slots (elbows 7/8, wrists 62/41)
MARKER_PAIRS2GOLIATH = [
    (_M["LELB"], _M["LMELB"], 7),
    (_M["RELB"], _M["RMELB"], 8),
    (_M["LWRI"], _M["LMWRI"], 62),
    (_M["RWRI"], _M["RMWRI"], 41),
]

# wrist joint centres (lateral, medial) used for hand crops and finger anchors
R_WRIST_PAIR = (_M["RWRI"], _M["RMWRI"])
L_WRIST_PAIR = (_M["LWRI"], _M["LMWRI"])
R_ELBOW_PAIR = (_M["RELB"], _M["RMELB"])
L_ELBOW_PAIR = (_M["LELB"], _M["LMELB"])
# palm-plane markers (thumb, pinky — with the wrist they span the palm) and
# the per-view NLF scores that proxy hand occlusion, for best-view selection
R_PALM = (_M["RTHU"], _M["RPIN"])
L_PALM = (_M["LTHU"], _M["LPIN"])
R_HAND_MARKERS = [_M[m] for m in ("RWRI", "RMWRI", "RTHU", "RMID", "RPIN")]
L_HAND_MARKERS = [_M[m] for m in ("LWRI", "LMWRI", "LTHU", "LMID", "LPIN")]


def _pair_mid(kp, a, b):
    """Midpoint of a lateral/medial marker pair; falls back to whichever is
    finite; NaN if neither."""
    fa, fb = np.isfinite(kp[a]).all(), np.isfinite(kp[b]).all()
    if fa and fb:
        return 0.5 * (kp[a] + kp[b])
    return (
        kp[a]
        if fa
        else (kp[b] if fb else np.full(kp.shape[-1], np.nan, kp.dtype))
    )


# COCO-17 synthesis for the hand-crop guide (_hand_decoder_step_views) —
# (coco_idx, marker or (lateral, medial) pair)
_COCO17_FROM_MARKERS = [
    (0, "Nose"),
    (1, "LEye"),
    (2, "REye"),
    (3, "LEar"),
    (4, "REar"),
    (5, "LSHO"),
    (6, "RSHO"),
    (7, ("LELB", "LMELB")),
    (8, ("RELB", "RMELB")),
    (9, ("LWRI", "LMWRI")),
    (10, ("RWRI", "RMWRI")),
    (11, "LASI"),
    (12, "RASI"),
    (13, "LKNE"),
    (14, "RKNE"),
    (15, "LANK"),
    (16, "RANK"),
]


def markers_to_coco17(kp2d, sc, thr):
    """(43,2) markers + scores → (17,2) COCO pixels (NaN where below thr)."""
    kp = kp2d.copy()
    kp[sc < thr] = np.nan
    k17 = np.full((17, 2), np.nan, np.float32)
    for ci, src in _COCO17_FROM_MARKERS:
        if isinstance(src, tuple):
            k17[ci] = _pair_mid(kp, _M[src[0]], _M[src[1]])
        else:
            k17[ci] = kp[_M[src]]
    return k17


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION + TRIANGULATION (self-contained, NaN-safe)
# ═══════════════════════════════════════════════════════════════════════════


def load_calibration(path, ncam):
    """Return (Ks, Ds, Rs, Ts) lists, cam0 = reference frame (R0=I, T0=0)."""
    z = np.load(path)
    keys = set(z.keys())
    if {"K1", "K2", "R", "T"} <= keys:  # stereo npz format
        if ncam != 2:
            raise ValueError(
                f"stereo calib is for 2 cameras, got --cams with {ncam}"
            )
        Ks = [z["K1"].astype(np.float64), z["K2"].astype(np.float64)]
        Ds = [
            z.get("D1", np.zeros(5)).astype(np.float64),
            z.get("D2", np.zeros(5)).astype(np.float64),
        ]
        Rs = [np.eye(3), z["R"].astype(np.float64)]
        Ts = [np.zeros(3), z["T"].astype(np.float64).reshape(3)]
        return Ks, Ds, Rs, Ts
    if f"K{ncam - 1}" in keys:  # multi-cam npz format
        Ks = [z[f"K{i}"].astype(np.float64) for i in range(ncam)]
        Ds = [
            z.get(f"D{i}", np.zeros(5)).astype(np.float64) for i in range(ncam)
        ]
        Rs = [
            z[f"R{i}"].astype(np.float64) if f"R{i}" in keys else np.eye(3)
            for i in range(ncam)
        ]
        Ts = [
            (
                z[f"T{i}"].astype(np.float64).reshape(3)
                if f"T{i}" in keys
                else np.zeros(3)
            )
            for i in range(ncam)
        ]
        return Ks, Ds, Rs, Ts
    raise ValueError(
        f"Unrecognized calibration file {path} — found keys {sorted(keys)}; "
        "expected K1,D1,K2,D2,R,T (stereo) or K0..,D0..,R0..,T0.. (multi)"
    )


def triangulate_multiview(pts2d, scores, Ks, Ds, Rs, Ts, thr=0.3):
    """DLT triangulation of (ncam, J, 2) pixels → (J, 3) in cam0 frame.

    Per joint, uses every view whose score > thr (needs >= 2). Points are
    undistorted to normalized coordinates, so P = [R|T] per view.
    """
    ncam, J, _ = pts2d.shape
    Ps = [np.hstack([Rs[i], Ts[i].reshape(3, 1)]) for i in range(ncam)]
    norm = np.full((ncam, J, 2), np.nan)
    for i in range(ncam):
        v = np.isfinite(pts2d[i]).all(1)
        if v.any():
            und = cv2.undistortPoints(
                pts2d[i][v].reshape(-1, 1, 2).astype(np.float64), Ks[i], Ds[i]
            )
            norm[i][v] = und.reshape(-1, 2)
    out = np.full((J, 3), np.nan, np.float32)
    for j in range(J):
        rows = []
        for i in range(ncam):
            if scores[i, j] > thr and np.isfinite(norm[i, j]).all():
                x, y = norm[i, j]
                rows.append(x * Ps[i][2] - Ps[i][0])
                rows.append(y * Ps[i][2] - Ps[i][1])
        if len(rows) >= 4:  # >= 2 views
            _, _, vt = np.linalg.svd(np.asarray(rows))
            X = vt[-1]
            if abs(X[3]) > 1e-12:
                out[j] = (X[:3] / X[3]).astype(np.float32)
    return out


def triangulate_with_reproj(pts2d, Ks, Ds, Rs, Ts, reproj_thr):
    """Triangulate (ncam, J, 2) pixels → (J, 3) world, rejecting bad joints.

    Unlike the NLF markers, the hand decoder has no confidence output — an
    occluded hand still yields plausible-LOOKING 2D from the crop, and DLT
    would happily blend it into garbage. So after triangulation each joint is
    reprojected into every contributing view and dropped (NaN) when any view
    disagrees by more than reproj_thr pixels: inconsistent views violate the
    epipolar constraint, consistent hallucination across views is unlikely.
    """
    sc = np.isfinite(pts2d).all(2).astype(np.float32)
    X = triangulate_multiview(pts2d, sc, Ks, Ds, Rs, Ts, thr=0.5)
    for i in range(len(Ks)):
        v = np.isfinite(X).all(1) & np.isfinite(pts2d[i]).all(1)
        if not v.any():
            continue
        rvec, _ = cv2.Rodrigues(Rs[i])
        proj, _ = cv2.projectPoints(
            X[v].astype(np.float64).reshape(-1, 1, 3),
            rvec,
            Ts[i].reshape(3, 1),
            Ks[i],
            Ds[i],
        )
        err = np.linalg.norm(proj.reshape(-1, 2) - pts2d[i][v], axis=1)
        X[np.flatnonzero(v)[err > reproj_thr]] = np.nan
    return X


# ═══════════════════════════════════════════════════════════════════════════
# WORKERS (shared latest-value slots, drop-old everywhere)
# ═══════════════════════════════════════════════════════════════════════════

_STOP = threading.Event()


class CamThread(threading.Thread):
    """Grab continuously; keep only the latest frame (+ wall-clock timestamp)."""

    def __init__(
        self,
        index,
        width,
        height,
        rotate180=False,
        lock_focus=False,
        focus=None,
    ):
        super().__init__(daemon=True)
        self.rotate180 = (
            rotate180  # must match how the calibration was captured
        )
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open camera {index}")
        # MJPG so several 720p webcams fit on the USB bus
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        foc = ""
        if lock_focus or focus is not None:
            # autofocus shifts the effective focal length (focus breathing):
            # the calibrated K drifts with every refocus. Lock it — at the
            # SAME value the calibration was captured with (like --rotate180).
            # No-op on fixed-focus cameras (driver rejects the property).
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            if focus is not None:
                self.cap.set(cv2.CAP_PROP_FOCUS, focus)
            foc = (
                f" focus={self.cap.get(cv2.CAP_PROP_FOCUS):g}"
                f" af={self.cap.get(cv2.CAP_PROP_AUTOFOCUS):g}"
            )
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fcc_s = "".join(chr((fcc >> (8 * i)) & 0xFF) for i in range(4))
        print(
            f"  cam {index}: {w}x{h} fourcc={fcc_s} "
            f"(nominal {self.cap.get(cv2.CAP_PROP_FPS):.0f} fps){foc}"
        )
        self.frame, self.ts, self.n = None, 0.0, 0
        self.fps = 0.0  # MEASURED capture rate (EMA)
        self._lock = threading.Lock()

    def run(self):
        t_prev = None
        while not _STOP.is_set():
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.rotate180:
                f = cv2.rotate(f, cv2.ROTATE_180)
            now = time.time()
            if t_prev is not None and now > t_prev:
                inst = 1.0 / (now - t_prev)
                self.fps = (
                    inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
                )
            t_prev = now
            with self._lock:
                self.frame, self.ts, self.n = f, now, self.n + 1
        self.cap.release()

    def latest(self):
        with self._lock:
            return (
                (None, 0.0, -1)
                if self.frame is None
                else (self.frame.copy(), self.ts, self.n)
            )


class BodyWorker(threading.Thread):
    """YOLO + NLF markers on every view (batched) → triangulated 43×3 metric.

    Wraps RT-COSMIK's NLFEstimator (branch nlf_humble): one batched YOLO
    person detection + one batched NLF call over all views, with per-camera
    person box locking handled inside the estimator.
    """

    def __init__(self, cams, Ks, Ds, Rs, Ts, det_thr, args):
        super().__init__(daemon=True)
        rtcosmik = os.path.expanduser(args.rtcosmik)
        src = os.path.join(rtcosmik, "src")
        if not os.path.isdir(src):
            raise SystemExit(
                f"RT-COSMIK not found at {rtcosmik} — clone branch "
                "nlf_humble there or pass --rtcosmik"
            )
        sys.path.insert(0, src)
        from rtcosmik.nlf.nlf import (
            NLFEstimator,
        )  # needs: ultralytics, meshcat

        nlf_w = args.nlf_weights or os.path.join(
            rtcosmik, "weights", "nlf", "nlf_s_multi_0.2.2.torchscript"
        )
        cano = args.cano or os.path.join(
            rtcosmik, "weights", "canonical_verts", "smplx.npy"
        )
        for path in (nlf_w, cano):
            if not os.path.isfile(path):
                raise SystemExit(f"missing NLF asset: {path} (see docstring)")
        self.est = NLFEstimator(
            yolo_path=args.yolo_weights,
            nlf_path=nlf_w,
            cano_path=cano,
            image_size=(args.cap_width, args.cap_height),
            cam_Ks=[K.astype(np.float32) for K in Ks],
            indices=NLF_INDICES,
            conf=args.yolo_conf,
            imgsz=args.yolo_imgsz,
            device="cuda:0",
        )
        self.size = (args.cap_width, args.cap_height)
        self.cams = cams
        self.calib = (Ks, Ds, Rs, Ts)
        self.det_thr = det_thr
        self.result = None  # dict: kp2d (ncam,43,2), scores, kp3d, ts, ms
        self.n = 0
        self._lock = threading.Lock()

    def run(self):
        ncam = len(self.cams)
        last_ns = [-1] * ncam
        while not _STOP.is_set():
            # wait for at least one NEW frame — otherwise we recompute the same
            # image and the reported body rate is inflated beyond the camera rate
            frames, tss, ns = [], [], []
            for c in self.cams:
                f, ts, n = c.latest()
                frames.append(f)
                tss.append(ts)
                ns.append(n)
            if any(f is None for f in frames) or all(
                n == l for n, l in zip(ns, last_ns)
            ):
                time.sleep(0.002)
                continue
            last_ns = ns
            result = self.compute(frames, tss)
            with self._lock:
                self.result = result
                self.n += 1

    def compute(self, frames, tss):
        """One body inference over a synchronized frame set. The live run()
        loop AND the offline replay driver both call this — identical math, so
        offline results equal live exactly (only the frame SOURCE differs)."""
        Ks, Ds, Rs, Ts = self.calib
        ncam = len(self.cams)
        W, H = self.size
        for f in frames:
            if f.shape[1] != W or f.shape[0] != H:
                # hard error: silently resizing would break the calibration
                # (K is for the capture resolution) → garbage triangulation
                raise SystemExit(
                    f"camera frame is {f.shape[1]}x{f.shape[0]}, expected "
                    f"{W}x{H} (--cap-width/height must match the calib)"
                )
        t0 = time.perf_counter()
        out, tms, _, boxes = self.est.estimate_from_frames(frames)
        # locked person box per cam, xywh pixels (NaN when no detection)
        pboxes = np.full((ncam, 4), np.nan, np.float32)
        for i, b in enumerate(boxes):
            if b is not None and len(b):
                pboxes[i] = b[0].detach().float().cpu().numpy()
        kp2d = np.full((ncam, NMK, 2), np.nan, np.float32)
        sc = np.zeros((ncam, NMK), np.float32)
        p2d_all = out["poses2d"]
        unc_all = out.get("uncertainties") if hasattr(out, "get") else None
        for i in range(ncam):
            p = p2d_all[i]
            if p is None or len(p) == 0 or p[0] is None:
                continue  # no (locked) person in this view
            kp2d[i] = p[0].detach().float().cpu().numpy()
            if (
                unc_all is not None
                and unc_all[i] is not None
                and len(unc_all[i])
            ):
                # NLF uncertainty (higher = worse) → score in [0,1]
                u = unc_all[i][0].detach().float().cpu().numpy()
                sc[i] = np.clip(1.0 - u, 0.0, 1.0)
            else:
                sc[i] = np.isfinite(kp2d[i]).all(1).astype(np.float32)
        if ncam >= 2:
            kp2d_masked = kp2d.copy()
            kp2d_masked[sc < self.det_thr] = np.nan
            kp3d = triangulate_multiview(
                kp2d_masked, sc, Ks, Ds, Rs, Ts, thr=self.det_thr
            )
        else:
            # MONO mode: no metric body 3D possible with a single view
            kp3d = np.full((NMK, 3), np.nan, np.float32)
        ms = (time.perf_counter() - t0) * 1e3
        return {
            "kp2d": kp2d,
            "scores": sc,
            "kp3d": kp3d,
            "boxes": pboxes,
            "frames_ts": tss,
            "ms": ms,
            "yolo_ms": tms.get("yolo_ms", 0.0),
            "nlf_ms": tms.get("nlf_ms", 0.0),
            "sync_ms": (max(tss) - min(tss)) * 1e3,
        }

    def latest(self):
        with self._lock:
            return self.result, self.n


def _metric_side_px(wrist_w, K, R, T, hand_size_m):
    """Pixel side of a hand_size_m box at the 3D wrist's depth in one view.

    wrist_w: (3,) triangulated wrist, world/cam0 frame. Returns None when the
    wrist is missing/behind the camera → caller falls back to the 2D
    heuristic. This makes the crop size exact at any distance, instead of
    the projected-forearm proxy that shrinks under foreshortening.
    """
    if (
        hand_size_m is None
        or wrist_w is None
        or not np.isfinite(wrist_w).all()
    ):
        return None
    z = float(R[2] @ wrist_w + T[2])
    if z < 0.2:
        return None
    return float(K[0, 0]) * hand_size_m / z


def _hand_box_view(k17, wrist_i, elbow_i, args, side_px=None):
    """Hand box for one view: center from the 2D forearm direction
    (_hand_box_v2: wrist pushed toward the hand), SIZE from the metric 3D
    depth when available, else the original 2D heuristic entirely."""
    box = _hand_box_v2(k17, wrist_i, elbow_i, args)
    if side_px is None or not np.isfinite(side_px):
        return box
    if box is not None:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    elif np.isfinite(k17[wrist_i]).all():
        # extreme foreshortening (no forearm direction): fingers project
        # close to the wrist anyway → wrist-centered metric box
        cx, cy = k17[wrist_i]
    else:
        return None
    h = side_px / 2
    return np.array([cx - h, cy - h, cx + h, cy + h], np.float32)


def _hand_decoder_step_views(
    model, frames, k17s, cam_ints, args, sides=None, want=None
):
    """rerun_demo._hand_decoder_step batched over views: ONE decoder forward.

    frames / k17s / cam_ints: dicts view → frame_bgr / (17,2) COCO pixels /
    (1,3,3) torch K. want: {view: (run_right, run_left)} — which hands to
    decode in each view (None = both, everywhere). Each selected (view, hand)
    is ONE entry on the BATCH dim — flat, NOT persons-per-view like
    model._merge_hand_batches — so the two hands can use DIFFERENT view
    subsets (per-hand top-K selection) and each entry keeps its own view
    intrinsics (cam_int is per batch entry, expanded to persons inside the
    hand path). A view now also runs with a single available hand box
    (previously both were required).

    Returns ({view: (kp_r21_2d, kp_l21_2d, k3_r21, k3_l21, rbox, lbox)}, tms,
    crops) — per view identical to _hand_decoder_step (3D in that view's
    camera coords, left hand un-mirrored, NOT anchored); non-decoded hands
    are None. With a single view and both hands this computes exactly what
    _hand_decoder_step does. tms = profiling breakdown in ms: prep (cvtColor
    + GPU upload/crops), fwd (decoder forward, CUDA-synced), post (GPU→CPU
    copies). crops = {view: [right_rgb | None, left_rgb | None]} — the
    decoder's ACTUAL input tiles (left un-flipped back for display), for
    judging crop tracking and --hand-res quality.
    """
    out_hw = (
        (args.hand_res, args.hand_res)
        if args.hand_res > 0
        else (model.cfg.MODEL.IMAGE_SIZE[1], model.cfg.MODEL.IMAGE_SIZE[0])
    )
    per, entries = {}, []  # entries: (view, "r"/"l", batch)
    tms = {"prep_ms": 0.0, "fwd_ms": 0.0, "post_ms": 0.0}
    t0 = time.perf_counter()
    with torch.no_grad(), _quiet():
        for v, frame in frames.items():
            run_r, run_l = (want or {}).get(v, (True, True))
            sr, sl = (sides or {}).get(v, (None, None))
            rbox = (
                _hand_box_view(k17s[v], R_WRIST, R_ELBOW, args, sr)
                if run_r
                else None
            )
            lbox = (
                _hand_box_view(k17s[v], L_WRIST, L_ELBOW, args, sl)
                if run_l
                else None
            )
            per[v] = (None, None, None, None, rbox, lbox)
            if rbox is None and lbox is None:
                continue
            # the GPU prep wants both boxes — mirror the present one into the
            # missing slot and drop that output (crops are cheap, the fwd isn't)
            bl, br, _ = _prepare_hand_batches_gpu(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                (lbox if lbox is not None else rbox)[None],
                (rbox if rbox is not None else lbox)[None],
                cam_ints[v],
                output_size=out_hw,
                padding=0.9,
                device="cuda",
            )
            if rbox is not None:
                entries.append((v, "r", br))
            if lbox is not None:
                entries.append((v, "l", bl))
        if not entries:
            return per, tms, {}
        tms["batch"] = len(entries)  # decoder crops this forward
        bh = {
            k: torch.cat([b[k] for _, _, b in entries], dim=0)
            for k in (
                "img",
                "img_size",
                "ori_img_size",
                "bbox_center",
                "bbox_scale",
                "bbox",
                "affine_trans",
                "mask",
                "mask_score",
                "person_valid",
                "cam_int",
            )
        }
        model._initialize_batch(bh)
        torch.cuda.synchronize()
        tms["prep_ms"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        merged = model.forward_step(bh, decoder_type="hand")
        torch.cuda.synchronize()
        tms["fwd_ms"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    mhr = merged["mhr_hand"]
    p2d = mhr["pred_keypoints_2d"].detach().float().cpu().numpy()
    p3d = mhr.get("pred_keypoints_3d")
    if p3d is not None:
        p3d = p3d.detach().float().cpu().numpy()
    out2d, out3d, crops = {}, {}, {}
    for i, (v, h, b) in enumerate(entries):
        kp = p2d[i][HAND_SRC].copy()
        k3 = p3d[i][HAND_SRC].copy() if p3d is not None else None
        # decoder input tile, back to uint8 RGB (left was decoded on the
        # mirrored image → un-flip tile and pixels so they read naturally)
        tile = (b["img"][0, 0].permute(1, 2, 0) * 255).byte().cpu().numpy()
        if h == "l":
            kp[:, 0] = frames[v].shape[1] - kp[:, 0] - 1
            if k3 is not None:
                k3[:, 0] *= -1
            tile = np.ascontiguousarray(tile[:, ::-1])
        out2d[(v, h)], out3d[(v, h)] = kp, k3
        crops.setdefault(v, [None, None])[0 if h == "r" else 1] = tile
    for v, (_, _, _, _, rbox, lbox) in per.items():
        per[v] = (
            out2d.get((v, "r")),
            out2d.get((v, "l")),
            out3d.get((v, "r")),
            out3d.get((v, "l")),
            rbox,
            lbox,
        )
    tms["post_ms"] = (time.perf_counter() - t0) * 1e3
    return per, tms, crops


class HandWorker(threading.Thread):
    """SAM hand decoder guided by the NLF wrists — stereo across the views.

    Runs the decoder on every view in `views` — all views in a SINGLE
    batched forward (_hand_decoder_step_views) — and triangulates the 21
    keypoints of each hand across the views, exactly like the body markers:
    metric scale and robustness when one view loses the hand (per-joint
    epipolar rejection in triangulate_with_reproj; the mono best-view
    prediction is kept as fallback for fuse_goliath70).
    views=[hand_cam] = original mono behaviour.

    With 3+ cameras (--hand-topk), each hand independently picks its K best
    views every frame (hand_view_select: crop size × palm visibility from
    the triangulated wrist/thumb/pinky markers × per-view NLF confidence ×
    in-frame fraction, with switch hysteresis) — only those crops enter the
    decoder batch, so 4 cams cost the same forward as the old 2-cam stereo
    while always decoding the closest, most face-on, least occluded views.
    """

    def __init__(self, cams, body, model, calib, args, views, hand_cam):
        super().__init__(daemon=True)
        self.cams, self.body, self.model = cams, body, model
        self.calib, self.args = calib, args
        self.views, self.hand_cam = views, hand_cam
        self.cam_ints = {
            v: torch.tensor([calib[0][v]], dtype=torch.float32) for v in views
        }
        k = (
            args.hand_topk
            if args.hand_topk >= 0
            else (2 if len(views) >= 3 else 0)
        )  # auto: select on 3+ cams
        self.topk = k if 0 < k < len(views) else 0  # 0 = decode all views
        self._sel_prev = {"r": set(), "l": set()}  # hysteresis state
        self._forearm = {"r": None, "l": None}  # EMA 3D forearm length (m)
        self.result = (
            None  # dict: best-view kp_r/kp_l/k3r/k3l + stereo X_r/X_l
        )
        self.n = 0
        self._lock = threading.Lock()

    def _select(self, res, views, sides):
        """Per-hand top-K view choice → ({view: (run_r, run_l)},
        {"r"/"l": ranked view list}). See hand_view_select for the score."""
        _, _, Rs, Ts = self.calib
        a = self.args
        picked, order = {}, {}
        for hi, (hand, pair, elb, palm, mks) in enumerate(
            (
                ("r", R_WRIST_PAIR, R_ELBOW_PAIR, R_PALM, R_HAND_MARKERS),
                ("l", L_WRIST_PAIR, L_ELBOW_PAIR, L_PALM, L_HAND_MARKERS),
            )
        ):
            wri = _pair_mid(res["kp3d"], *pair)
            n = palm_normal(wri, res["kp3d"][palm[0]], res["kp3d"][palm[1]])
            # camera->wrist rays (world frame) for the triangulation-diversity
            # pass: don't pair two near-parallel views (poor depth)
            rays = None
            if np.isfinite(wri).all():
                rays = {v: wri - (-Rs[v].T @ Ts[v]) for v in views}
            size_h = {v: (sides or {}).get(v, (None, None))[hi] for v in views}
            if any(s is None or not np.isfinite(s) for s in size_h.values()):
                # no metric side everywhere → UNIFORM 2D proxy so "biggest
                # crop" still ranks: projected forearm length per view
                # (mixing metric px with 2D px across views would mis-rank)
                for v in views:
                    w2 = _pair_mid(res["kp2d"][v], *pair)
                    e2 = _pair_mid(res["kp2d"][v], *elb)
                    ok = np.isfinite(w2).all() and np.isfinite(e2).all()
                    size_h[v] = float(np.linalg.norm(w2 - e2)) if ok else None
            cands = {}
            for v in views:
                cands[v] = {
                    "size": size_h[v],
                    "vis": view_visibility(n, wri, Rs[v], Ts[v]),
                    "conf": float(np.mean(res["scores"][v][mks])),
                    "in_frame": in_frame_fraction(
                        _pair_mid(res["kp2d"][v], *pair),
                        size_h[v],
                        a.cap_width,
                        a.cap_height,
                    ),
                }
            sel, rank, _ = select_views(
                cands,
                self.topk,
                self._sel_prev[hand],
                a.hand_switch_bonus,
                rays=rays,
            )
            self._sel_prev[hand] = set(sel)
            picked[hand], order[hand] = set(sel), rank
        want = {v: (v in picked["r"], v in picked["l"]) for v in views}
        return want, order

    def _hand_size(self, key, wrist_w, elbow_w):
        """Metric crop side (m) for one hand: hand_size_frac x the subject's
        3D forearm length — a body constant, EMA'd over valid frames, so the
        crop adapts to different body sizes instead of a manual constant.
        Falls back to the fixed --hand-size-m until/unless estimated."""
        a = self.args
        if (
            a.hand_size_frac > 0
            and np.isfinite(wrist_w).all()
            and np.isfinite(elbow_w).all()
        ):
            L = float(np.linalg.norm(wrist_w - elbow_w))
            if 0.15 < L < 0.45:  # plausible human forearm
                prev = self._forearm[key]
                self._forearm[key] = (
                    L if prev is None else 0.95 * prev + 0.05 * L
                )
        if a.hand_size_frac > 0 and self._forearm[key] is not None:
            return a.hand_size_frac * self._forearm[key]
        return a.hand_size_m if a.hand_size_m > 0 else None

    def run(self):
        last_body = -1
        while not _STOP.is_set():
            res, nb = self.body.latest()
            if res is None or nb == last_body:
                time.sleep(0.003)
                continue
            frames = {}
            for v in self.views:
                frame, _, _ = self.cams[v].latest()
                if frame is not None:
                    frames[v] = frame
            if not frames:
                time.sleep(0.003)
                continue
            result = self.compute(res, frames)
            last_body = nb
            with self._lock:
                self.result = result
                self.n += 1

    def compute(self, res, frames):
        """Hand decoder + stereo triangulation for one body result. frames =
        {view: image}. The live run() loop AND the offline replay driver both
        call this — identical math, so offline hand quality equals live."""
        Ks, Ds, Rs, Ts = self.calib
        ncam = len(Ks)
        t0 = time.perf_counter()
        k17s = {}
        for v in frames:
            # COCO-17 layout expected by the decoder step, synthesized from
            # THIS view's NLF markers (wrist/elbow = lat+med midpoints)
            k17s[v] = markers_to_coco17(
                res["kp2d"][v], res["scores"][v], self.args.det_thr
            )
        # metric crop size from the triangulated 3D wrists: exact at any
        # distance, sized to THIS subject's 3D forearm length
        # (2D-heuristic fallback per hand/view when 3D missing)
        sides = None
        if self.args.hand_size_m > 0 or self.args.hand_size_frac > 0:
            wr = _pair_mid(res["kp3d"], *R_WRIST_PAIR)
            wl = _pair_mid(res["kp3d"], *L_WRIST_PAIR)
            size_r = self._hand_size(
                "r", wr, _pair_mid(res["kp3d"], *R_ELBOW_PAIR)
            )
            size_l = self._hand_size(
                "l", wl, _pair_mid(res["kp3d"], *L_ELBOW_PAIR)
            )
            sides = {
                v: (
                    _metric_side_px(wr, Ks[v], Rs[v], Ts[v], size_r),
                    _metric_side_px(wl, Ks[v], Rs[v], Ts[v], size_l),
                )
                for v in frames
            }
        # per-hand top-K view selection (skipped when topk == 0)
        want = rank = None
        if self.topk:
            want, rank = self._select(res, list(frames), sides)
        # one batched forward for all selected (view, hand) crops
        per, tms, crops = _hand_decoder_step_views(
            self.model, frames, k17s, self.cam_ints, self.args, sides, want
        )

        # per-hand MONO source = best-ranked view with an output (fixed
        # hand-cam first when not selecting); fuse_goliath70 must rotate
        # its offsets with THIS view's extrinsics → src_r/src_l exported
        def _first(order, idx):
            for v in order:
                out = per.get(v)
                if out is not None and out[idx] is not None:
                    return v
            return None

        fallback = [self.hand_cam] + [
            v for v in self.views if v != self.hand_cam
        ]
        src_r = _first((rank or {}).get("r", fallback), 0)
        src_l = _first((rank or {}).get("l", fallback), 1)
        kp_r, k3r, rbox = (
            (per[src_r][0], per[src_r][2], per[src_r][4])
            if src_r is not None
            else (None, None, None)
        )
        kp_l, k3l, lbox = (
            (per[src_l][1], per[src_l][3], per[src_l][5])
            if src_l is not None
            else (None, None, None)
        )
        # stack each hand's 2D across views and stereo-triangulate
        kp2d_views = np.full((ncam, 42, 2), np.nan, np.float32)
        for v, out in per.items():
            if out[0] is not None:
                kp2d_views[v, :21] = out[0]
            if out[1] is not None:
                kp2d_views[v, 21:] = out[1]
        X_r = X_l = None
        t1 = time.perf_counter()
        if len(self.views) >= 2:
            X_r = triangulate_with_reproj(
                kp2d_views[:, :21], Ks, Ds, Rs, Ts, self.args.hand_reproj_thr
            )
            X_l = triangulate_with_reproj(
                kp2d_views[:, 21:], Ks, Ds, Rs, Ts, self.args.hand_reproj_thr
            )
        tms["tri_ms"] = (time.perf_counter() - t1) * 1e3
        ms = (time.perf_counter() - t0) * 1e3
        return {
            "kp_r": kp_r,
            "kp_l": kp_l,
            "k3r": k3r,
            "k3l": k3l,
            "ms": ms,
            "tms": tms,
            "crops": crops,
            "forearm": dict(self._forearm),
            "rbox": rbox,
            "lbox": lbox,
            "src_r": src_r,
            "src_l": src_l,
            "sel": (
                {h: sorted(self._sel_prev[h]) for h in ("r", "l")}
                if want
                else None
            ),
            "X_r": X_r,
            "X_l": X_l,
            "kp2d_views": kp2d_views,
            "views": {
                v: (out[0], out[1], out[4], out[5]) for v, out in per.items()
            },
        }

    def latest(self):
        with self._lock:
            return self.result


# ═══════════════════════════════════════════════════════════════════════════
# FUSION — fingers re-anchored at the metric wrists
# ═══════════════════════════════════════════════════════════════════════════

# a stereo hand block is trusted only when the wrist AND most of the 21
# joints pass the epipolar check — below that, the coherent mono hand wins
MIN_STEREO_JOINTS = 12


def fuse_goliath70(mk3d, hands, Rw_r, Rw_l):
    """(43,3) metric markers + decoder hands → (70,3) Goliath, world/cam0 frame.

    Rw_r/Rw_l: cam→world rotation of each hand's mono SOURCE view (they can
    differ — with per-hand view selection each hand has its own best view).
    Elbow/wrist Goliath slots get the lateral+medial marker midpoints (true
    joint centres). Hand blocks, best source first:
      1. STEREO (X_r/X_l): keypoints triangulated across the views — true
         metric scale AND absolute position; joints that failed the epipolar
         check are filled with the mono offsets re-anchored at the stereo
         wrist, so the hand stays internally consistent.
      2. MONO: decoder offsets rotated from the source-camera frame into the
         world frame, anchored at the triangulated (metric) wrist centre,
         with the decoder's own hand size (MHR average-hand scale).
    """
    g = np.full((70, 3), np.nan, np.float32)
    for m, gi in MARKER2GOLIATH.items():
        g[gi] = mk3d[m]
    for a, b, gi in MARKER_PAIRS2GOLIATH:
        g[gi] = _pair_mid(mk3d, a, b)
    if hands is None:
        return g
    for dec, Rw, Xst, sl, pair, disp in (
        (
            hands["k3r"],
            Rw_r,
            hands.get("X_r"),
            slice(21, 42),
            R_WRIST_PAIR,
            (+0.15, 0, 0.5),
        ),
        (
            hands["k3l"],
            Rw_l,
            hands.get("X_l"),
            slice(42, 63),
            L_WRIST_PAIR,
            (-0.15, 0, 0.5),
        ),
    ):
        off = None
        if dec is not None and Rw is not None:
            off = (dec - dec[_H_WRIST]) @ Rw.T
        if Xst is not None:
            vst = np.isfinite(Xst).all(1)
            if vst[_H_WRIST] and vst.sum() >= MIN_STEREO_JOINTS:
                h = Xst.astype(np.float32).copy()
                if off is not None:
                    h[~vst] = (
                        Xst[_H_WRIST] + off[~vst]
                    )  # occluded joints: mono fill
                g[sl] = h
                continue
        if off is None:
            continue
        wrist = _pair_mid(mk3d, *pair)
        if np.isfinite(wrist).all():
            g[sl] = wrist + off  # metric wrist anchor
        else:
            # MONO mode: no metric wrist — anchor at a fixed display position
            g[sl] = np.asarray(disp, np.float32) + off
    return g


# ═══════════════════════════════════════════════════════════════════════════
# RERUN UI
# ═══════════════════════════════════════════════════════════════════════════


class Viz:
    def __init__(self, mode, grpc_port, web_port, ncam, output_dir):
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr = rr
        rr.init("cosmik_hand_demo")
        if mode == "web":
            url = rr.serve_grpc(grpc_port=grpc_port)
            rr.serve_web_viewer(
                web_port=web_port, open_browser=False, connect_to=url
            )
            print(
                f"  Rerun UI: http://localhost:{web_port}  (tunnel {web_port} AND {grpc_port})"
            )
        elif mode == "native":
            rr.spawn(port=grpc_port)
        else:
            rr.save(os.path.join(output_dir, "session.rrd"))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        cam_views = [
            rrb.Spatial2DView(origin=f"cams/cam{i}", name=f"Camera {i}")
            for i in range(ncam)
        ]
        if ncam > 2:  # 3-4 cams: 2-wide grid instead of a tall stack
            cam_panel = rrb.Grid(*cam_views, grid_columns=2)
            shares = [3, 1, 1]
        else:
            cam_panel = rrb.Vertical(*cam_views)
            shares = [2 * max(ncam, 1), 1, 1]
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Vertical(
                        cam_panel,
                        rrb.Spatial2DView(
                            origin="cams/hand_crops", name="Hand crops"
                        ),
                        rrb.TimeSeriesView(
                            origin="timing", name="Latency (ms)"
                        ),
                        row_shares=shares,
                    ),
                    rrb.Spatial3DView(
                        origin="world",
                        name="3D (metric)",
                        background=rrb.Background(color=[25, 25, 25]),
                    ),
                    column_shares=[2, 3],
                )
            )
        )

    def log(
        self,
        seq,
        t_wall,
        overlays,
        mk3d,
        g70,
        body_ms,
        hand_ms,
        sync_ms,
        fps,
        cam_fps=None,
        jpeg_quality=75,
        hand_crops=None,
        hand_batch=None,
    ):
        rr = self.rr
        rr.set_time("frame", sequence=seq)
        rr.set_time("time", timestamp=t_wall)
        rr.log("timing/body_ms", rr.Scalars(float(body_ms)))
        if hand_ms is not None:
            rr.log("timing/hand_ms", rr.Scalars(float(hand_ms)))
        if hand_batch is not None:  # decoder crops (selection check)
            rr.log("timing/hand_batch", rr.Scalars(float(hand_batch)))
        rr.log("timing/sync_ms", rr.Scalars(float(sync_ms)))
        rr.log("timing/body_fps", rr.Scalars(float(fps)))
        for i, cf in enumerate(cam_fps or []):
            rr.log(f"timing/cam{i}_capture_fps", rr.Scalars(float(cf)))
        for i, img in enumerate(overlays):
            rr.log(
                f"cams/cam{i}",
                rr.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).compress(
                    jpeg_quality=jpeg_quality
                ),
            )
        if hand_crops is not None:  # already RGB
            rr.log(
                "cams/hand_crops",
                rr.Image(hand_crops).compress(jpeg_quality=jpeg_quality),
            )
        # 3D: NLF marker skeleton (metric) + Goliath fingers
        v = np.isfinite(mk3d).all(1)
        if v.any():
            rr.log(
                "world/body/joints",
                rr.Points3D(mk3d[v], radii=0.015, colors=[0, 230, 0]),
            )
            strips = [
                [mk3d[a].tolist(), mk3d[b].tolist()]
                for a, b in MARKER_EDGES
                if v[a] and v[b]
            ]
            if strips:
                rr.log(
                    "world/body/bones",
                    rr.LineStrips3D(strips, colors=[255, 230, 0], radii=0.006),
                )
        hv = np.isfinite(g70[21:63]).all(1)
        if hv.any():
            rr.log(
                "world/hands/joints",
                rr.Points3D(
                    g70[21:63][hv], radii=0.006, colors=[80, 170, 255]
                ),
            )
            strips = []
            for base in (21, 42):
                wrist = base + _H_WRIST
                for f0 in range(5):
                    chain = [
                        wrist,
                        base + 4 * f0 + 3,
                        base + 4 * f0 + 2,
                        base + 4 * f0 + 1,
                        base + 4 * f0,
                    ]
                    strips += [
                        [g70[a].tolist(), g70[b].tolist()]
                        for a, b in zip(chain[:-1], chain[1:])
                        if np.isfinite(g70[[a, b]]).all()
                    ]
            if strips:
                rr.log(
                    "world/hands/bones",
                    rr.LineStrips3D(
                        strips, colors=[80, 170, 255], radii=0.003
                    ),
                )


def draw_markers(img, kp, sc, thr):
    ok = (sc > thr) & np.isfinite(kp).all(1)
    for a, b in MARKER_EDGES:
        if ok[a] and ok[b]:
            cv2.line(
                img,
                tuple(kp[a].astype(int)),
                tuple(kp[b].astype(int)),
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
    for j in range(NMK):
        if ok[j]:
            cv2.circle(
                img,
                tuple(kp[j].astype(int)),
                3,
                (0, 140, 255),
                -1,
                cv2.LINE_AA,
            )
    return img


# ── hand skeleton (SAM3D 21-kp order: per finger [tip, DIP, PIP, MCP], wrist=20) ──
_HAND_FINGERS = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
    (16, 17, 18, 19),
]
_HAND_WRIST = 20
# per-finger colour (BGR): thumb, index, middle, ring, pinky
_HAND_FCOL = [
    (0, 128, 255),
    (255, 153, 255),
    (255, 178, 102),
    (51, 51, 255),
    (0, 255, 0),
]


def draw_hand_skeleton(
    img, pts21, colors=_HAND_FCOL, line_w=2, joint_r=3, tip_r=5, hollow=False
):
    """Draw a 21-keypoint hand skeleton — per-finger coloured bones + joints,
    fingertip and wrist emphasised. hollow=True draws ring markers and thin
    bones, used to overlay the reprojected metric-3D hand distinctly from the
    solid raw 2D detection (their divergence = triangulation error)."""
    for (tip, j3, j2, mcp), col in zip(_HAND_FINGERS, colors):
        chain = [_HAND_WRIST, mcp, j2, j3, tip]
        for a, b in zip(chain[:-1], chain[1:]):
            pa, pb = pts21[a], pts21[b]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                cv2.line(
                    img,
                    tuple(pa.astype(int)),
                    tuple(pb.astype(int)),
                    col,
                    line_w,
                    cv2.LINE_AA,
                )
    for (tip, j3, j2, mcp), col in zip(_HAND_FINGERS, colors):
        for j in (mcp, j2, j3):
            p = pts21[j]
            if np.isfinite(p).all():
                cv2.circle(
                    img,
                    tuple(p.astype(int)),
                    joint_r,
                    col if hollow else (255, 255, 255),
                    1 if hollow else -1,
                    cv2.LINE_AA,
                )
        p = pts21[tip]  # fingertip emphasised
        if np.isfinite(p).all():
            cv2.circle(
                img,
                tuple(p.astype(int)),
                tip_r,
                col,
                1 if hollow else -1,
                cv2.LINE_AA,
            )
    w = pts21[_HAND_WRIST]
    if np.isfinite(w).all():
        cv2.circle(
            img,
            tuple(w.astype(int)),
            joint_r + 1,
            (0, 255, 255),
            1 if hollow else -1,
            cv2.LINE_AA,
        )
    return img


def reproject_points(pts3d, K, D, R, T):
    """Project (N,3) world/cam0 points into one view → (N,2) pixels, NaN-safe
    (NaN inputs stay NaN, so a partially-occluded hand still reprojects)."""
    out = np.full((len(pts3d), 2), np.nan, np.float32)
    v = np.isfinite(pts3d).all(1)
    if v.any():
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(
            pts3d[v].astype(np.float64).reshape(-1, 1, 3),
            rvec,
            T.reshape(3, 1),
            K,
            D,
        )
        out[v] = proj.reshape(-1, 2).astype(np.float32)
    return out


def _full_to_tile(pts, box, out_hw, padding=0.9, aspect=0.75):
    """Map full-image (N,2) pixels into the decoder's hand-crop TILE pixels,
    inverting the crop transform of _prepare_hand_batches_gpu (padding 0.9,
    aspect fixed to a square of side crop_size around the box centre). Used to
    draw the detected hand skeleton onto the 'Hand crops' panel tiles. NaN-safe.
    For the LEFT hand this also gives the DISPLAYED (un-flipped) tile coords:
    the flip cancels out when the box centre is the original one (verified)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * padding, (y2 - y1) * padding
    sw, sh = (w, w / aspect) if w > h * aspect else (h * aspect, h)
    cs = max(sw, sh)
    out_h, out_w = out_hw
    tile = np.full_like(np.asarray(pts, np.float32), np.nan)
    tile[:, 0] = (pts[:, 0] - cx + cs / 2) * out_w / cs
    tile[:, 1] = (pts[:, 1] - cy + cs / 2) * out_h / cs
    return tile


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def run_offline(args, Ks, Ds, Rs, Ts, ncam, est, out_dir, R_world_handcam):
    """OFFLINE replay: run the SAME body+hand inference on recorded video.

    Reads cam{i}.mp4 from args.replay (i = position in --cams), processes every
    frame in LOCKSTEP — for frame k it feeds all cameras' frame k to
    BodyWorker.compute / HandWorker.compute (the exact methods the live loop
    calls), so results are bit-for-bit the live pipeline minus the drop-old
    real-time scheduling. No threads, no rerun, no TCP: pure batch. Writes the
    same .npy files as the live path so downstream tooling is unchanged.
    """
    # open one video per camera position; index k is aligned across files
    # (record_multi.py writes a frame to every camera in lockstep). --replay-cams
    # remaps which recorded file feeds each calib position (default 0..ncam-1).
    if args.replay_cams:
        file_idx = [int(x) for x in args.replay_cams.split(",")]
        if len(file_idx) != ncam:
            raise SystemExit(
                f"--replay-cams has {len(file_idx)} entries but "
                f"--cams has {ncam} — they must match"
            )
    else:
        file_idx = list(range(ncam))
    paths = [os.path.join(args.replay, f"cam{i}.mp4") for i in file_idx]
    caps = []
    for pth in paths:
        if not os.path.isfile(pth):
            raise SystemExit(
                f"replay file missing: {pth} (record with "
                "stereo_calibration/record_multi.py; one cam{i}.mp4 "
                "per --cams position)"
            )
        cap = cv2.VideoCapture(pth)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {pth}")
        caps.append(cap)
    counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    w0 = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = min(counts)
    if args.replay_max_frames > 0:
        nframes = min(nframes, args.replay_max_frames)
    print(
        f"  replay: {ncam} files, {counts} frames each -> {nframes} usable, "
        f"{w0}x{h0}"
    )
    if (w0, h0) != (args.cap_width, args.cap_height):
        raise SystemExit(
            f"recording is {w0}x{h0} but --cap-width/height is "
            f"{args.cap_width}x{args.cap_height} — they MUST match the "
            f"calibration resolution (K is resolution-dependent)"
        )

    # per-frame timestamps: use the recorder's if present, else synthesize
    ts_path = os.path.join(args.replay, "timestamps.npy")
    if os.path.isfile(ts_path):
        ts_all = np.load(ts_path)
        if ts_all.ndim == 1:
            ts_all = np.repeat(ts_all[:, None], ncam, axis=1)
        print(f"  using recorded timestamps ({ts_path})")
    else:
        fps = 30.0
        ts_all = (np.arange(nframes)[:, None] / fps).repeat(ncam, axis=1)
        print("  no timestamps.npy — synthesizing at 30 fps (sync spread = 0)")

    # workers WITHOUT camera threads: placeholder cam list gives them ncam +
    # nothing else (compute() takes frames explicitly; run() is never called)
    placeholder = list(range(ncam))
    body = BodyWorker(placeholder, Ks, Ds, Rs, Ts, args.det_thr, args)
    hand_views = (
        [args.hand_cam] if (args.mono_hands or ncam < 2) else list(range(ncam))
    )
    hands = HandWorker(
        placeholder,
        body,
        est.model,
        (Ks, Ds, Rs, Ts),
        args,
        hand_views,
        args.hand_cam,
    )
    if len(hand_views) >= 2:
        print(f"  STEREO hands offline: decoder on views {hand_views}")

    rec = None
    if not args.no_record:
        rec = {
            "b3d": [],
            "b2d": [],
            "h2d": [],
            "h2dv": [],
            "hsrc": [],
            "g70": [],
            "ts": [],
        }
    vw = None

    print(
        f"[offline] processing {nframes} frames (Ctrl+C to stop early "
        "and keep what's done)..."
    )
    t_start = time.time()
    done = 0
    try:
        for k in range(nframes):
            frames = []
            ok = True
            for c in caps:
                r, f = c.read()
                if not r or f is None:
                    ok = False
                    break
                if args.rotate180:
                    f = cv2.rotate(f, cv2.ROTATE_180)
                frames.append(f)
            if not ok:
                print(f"  short read at frame {k} — stopping")
                break
            row = ts_all[min(k, len(ts_all) - 1)]
            # pick the timestamp columns for the files actually loaded
            tss = [float(row[fi if fi < len(row) else 0]) for fi in file_idx]
            res = body.compute(frames, tss)
            hres = hands.compute(res, {v: frames[v] for v in hand_views})

            Rw_r = Rw_l = R_world_handcam
            if hres.get("src_r") is not None:
                Rw_r = Rs[hres["src_r"]].T
            if hres.get("src_l") is not None:
                Rw_l = Rs[hres["src_l"]].T
            g70 = fuse_goliath70(res["kp3d"], hres, Rw_r, Rw_l)

            if rec is not None:
                rec["b3d"].append(res["kp3d"])
                rec["b2d"].append(res["kp2d"])
                h2d = np.full((42, 2), np.nan, np.float32)
                if hres["kp_r"] is not None:
                    h2d[:21] = hres["kp_r"]
                if hres["kp_l"] is not None:
                    h2d[21:] = hres["kp_l"]
                rec["h2d"].append(h2d)
                rec["h2dv"].append(hres["kp2d_views"])
                rec["hsrc"].append(
                    [
                        -1 if hres.get(kk) is None else hres[kk]
                        for kk in ("src_r", "src_l")
                    ]
                )
                rec["g70"].append(g70)
                rec["ts"].append(tss[0])

            if args.save_video:
                if vw is None:
                    vw = cv2.VideoWriter(
                        os.path.join(out_dir, "overlay_cam0.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        25,
                        (w0, h0),
                    )
                img = draw_markers(
                    frames[0].copy(),
                    res["kp2d"][0],
                    res["scores"][0],
                    args.det_thr,
                )
                if args.reproj_hands and ncam >= 2:
                    for sl in (slice(21, 42), slice(42, 63)):
                        rp = reproject_points(
                            g70[sl], Ks[0], Ds[0], Rs[0], Ts[0]
                        )
                        img = draw_hand_skeleton(
                            img, rp, hollow=True, line_w=1
                        )
                vw.write(img)

            done = k + 1
            if done % 60 == 0 or done == nframes:
                el = time.time() - t_start
                nr = (
                    int(np.isfinite(hres["X_r"]).all(1).sum())
                    if hres.get("X_r") is not None
                    else -1
                )
                nl = (
                    int(np.isfinite(hres["X_l"]).all(1).sum())
                    if hres.get("X_l") is not None
                    else -1
                )
                print(
                    f"  {done}/{nframes}  {done / max(el, 1e-6):.1f} fps  "
                    f"body {res['ms']:.0f}ms hands {hres['ms']:.0f}ms  "
                    f"stereo jts R{nr} L{nl}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n  interrupted — saving processed frames")
    finally:
        for c in caps:
            c.release()
        if vw is not None:
            vw.release()
        if rec is not None and rec["ts"]:
            np.save(
                os.path.join(out_dir, "markers_3d.npy"), np.stack(rec["b3d"])
            )
            np.save(
                os.path.join(out_dir, "markers_2d.npy"), np.stack(rec["b2d"])
            )
            np.save(
                os.path.join(out_dir, "hands_2d.npy"), np.stack(rec["h2d"])
            )
            np.save(
                os.path.join(out_dir, "hands_2d_views.npy"),
                np.stack(rec["h2dv"]),
            )
            np.save(
                os.path.join(out_dir, "hands_src.npy"),
                np.asarray(rec["hsrc"], np.int16),
            )
            np.save(
                os.path.join(out_dir, "goliath70_3d.npy"), np.stack(rec["g70"])
            )
            np.save(
                os.path.join(out_dir, "timestamps.npy"), np.asarray(rec["ts"])
            )
            print(f"  saved {len(rec['ts'])} frames to {out_dir}/")


def main():
    p = argparse.ArgumentParser(
        description="NLF markers (multi-cam metric) + SAM hand decoder"
    )
    p.add_argument(
        "--cams",
        default="0",
        help="comma-separated camera indices or /dev/v4l/by-id "
        "paths (stable across reboots; order must match the "
        "calibration npz K0..K{n})",
    )
    p.add_argument(
        "--calib",
        default="",
        help="calibration npz (see docstring); "
        "required for >=2 cameras, optional in mono mode",
    )
    p.add_argument(
        "--fx",
        type=float,
        default=540.0,
        help="[mono] focal length in px (default 540 ≈ 720p webcam)",
    )
    p.add_argument("--fy", type=float, default=0)
    p.add_argument("--cx", type=float, default=0)
    p.add_argument("--cy", type=float, default=0)
    p.add_argument(
        "--hand-cam",
        type=int,
        default=0,
        help="which view (position in --cams) runs the hand decoder",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--checkpoint_dir", default="./checkpoints/sam-3d-body-dinov3"
    )
    p.add_argument("--cap-width", type=int, default=1280)
    p.add_argument("--cap-height", type=int, default=720)
    p.add_argument(
        "--rotate180",
        action="store_true",
        help="Rotate every camera frame 180° (upside-down mounted "
        "cameras). Lossless. The calibration MUST have been "
        "captured with the same flag.",
    )
    p.add_argument(
        "--lock-focus",
        action="store_true",
        help="disable autofocus on every camera (no-op on fixed-"
        "focus models). REQUIRED with any autofocus camera: "
        "refocusing shifts the effective focal length, so "
        "the calibrated K drifts. The calibration must be "
        "captured with the same lock (capture_calibration_"
        "multi --lock-focus/--focus)",
    )
    p.add_argument(
        "--focus",
        type=float,
        default=None,
        help="fixed manual focus value (V4L2 units, typically "
        "0-255 — probe with v4l2-ctl). Implies --lock-focus; "
        "use the SAME value as during calibration",
    )
    p.add_argument(
        "--det-thr",
        type=float,
        default=0.3,
        help="per-marker score threshold for triangulation/drawing",
    )
    p.add_argument(
        "--mono-hands",
        action="store_true",
        help="run the hand decoder on the hand-cam only (original "
        "behaviour, half the decoder batch) instead of all "
        "views batched + stereo triangulation",
    )
    p.add_argument(
        "--hand-topk",
        type=int,
        default=-1,
        help="decode each hand in only its K best views — score = "
        "crop px size x palm-plane visibility (from the "
        "triangulated wrist/thumb/pinky markers) x per-view "
        "NLF hand-marker confidence x in-frame fraction "
        "(hand_view_select.py). -1 = auto: 2 with 3+ cams "
        "(4 cams cost the same decoder batch as 2-cam "
        "stereo), all views with <=2 cams. 0 = all views",
    )
    p.add_argument(
        "--hand-switch-bonus",
        type=float,
        default=1.15,
        help="view-selection hysteresis: score multiplier for the "
        "views a hand used last frame — stops two near-equal "
        "views from flapping every frame (1.0 = off)",
    )
    p.add_argument(
        "--hand-reproj-thr",
        type=float,
        default=15.0,
        help="max reprojection error (px) for a stereo hand joint; "
        "above → epipolar-inconsistent (occluded/hallucinated "
        "in a view) → mono fallback for that joint",
    )
    p.add_argument(
        "--rtcosmik",
        default="~/code/RT-COSMIK",
        help="RT-COSMIK clone (branch nlf_humble) — provides the "
        "NLFEstimator code and the default weight locations",
    )
    p.add_argument(
        "--nlf-weights",
        default="",
        help="NLF torchscript (default <rtcosmik>/weights/nlf/"
        "nlf_s_multi_0.2.2.torchscript)",
    )
    p.add_argument(
        "--cano",
        default="",
        help="SMPL-X canonical vertices npy (default <rtcosmik>/"
        "weights/canonical_verts/smplx.npy)",
    )
    p.add_argument(
        "--yolo-weights",
        default="yolov10n.pt",
        help="ultralytics person detector (.pt auto-downloads; point "
        "to a .engine for TensorRT)",
    )
    p.add_argument("--yolo-conf", type=float, default=0.2)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument(
        "--hand-size-frac",
        type=float,
        default=1.05,
        help="metric hand-crop side = this x the subject's 3D "
        "forearm length (triangulated elbow->wrist, EMA'd — "
        "adapts to different body sizes; ~26.5 cm forearm -> "
        "28 cm box, delivering ~25 cm after the x0.9 GPU "
        "prep shrink). 0 = use the fixed --hand-size-m",
    )
    p.add_argument(
        "--hand-size-m",
        type=float,
        default=0.28,
        help="fixed metric hand-crop size (m): box side = fx * "
        "this / triangulated-wrist depth per view. Used "
        "until the forearm estimate exists (or always if "
        "--hand-size-frac 0). 0 = fully disable metric "
        "sizing (2D forearm heuristic only, also the "
        "per-hand fallback when the 3D wrist is missing)",
    )
    p.add_argument("--box-offset", type=float, default=0.35)
    p.add_argument(
        "--box-size",
        type=float,
        default=1.4,
        help="hand box side, in projected forearm lengths. The old "
        "default (1.0) left ~zero margin past the fingertips "
        "(the GPU crop is 0.9x the box on top): spread fingers "
        "got clipped — check the 'Hand crops' Rerun panel",
    )
    p.add_argument(
        "--box-scale-mode",
        choices=["stable", "forearm"],
        default="stable",
        help="stable: shoulder-width floor on the hand box (no foreshortening "
        "shrink); forearm: original behaviour",
    )
    p.add_argument(
        "--box-shoulder-frac",
        type=float,
        default=0.65,
        help="box floor = this x projected shoulder width (matters "
        "when the forearm points at the camera)",
    )
    p.add_argument("--hand-res", type=int, default=0)
    p.add_argument(
        "--no-reproj-hands",
        dest="reproj_hands",
        action="store_false",
        help="don't overlay the fused metric-3D hand reprojected "
        "into each view (hollow skeleton; on by default when "
        "calibrated — a retargeting-quality check: the hollow "
        "3D skeleton should sit on top of the solid detection)",
    )
    p.set_defaults(reproj_hands=True)
    p.add_argument(
        "--rerun-mode", choices=["web", "native", "save"], default="web"
    )
    p.add_argument("--rerun-grpc-port", type=int, default=9876)
    p.add_argument("--rerun-web-port", type=int, default=9090)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument("--output_dir", default="")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument(
        "--replay",
        default="",
        help="OFFLINE mode: read frames from a recording dir "
        "(cam{i}.mp4 for each --cams position, produced by "
        "stereo_calibration/record_multi.py) instead of live "
        "cameras. Processes EVERY frame in lockstep (no "
        "drop-old), so 3D quality equals the live path; runs "
        "as fast as the GPU allows, not real time. Writes the "
        "same .npy outputs. --cams then indexes recorded files, "
        "not devices; frames are used as stored (do NOT pass "
        "--rotate180 if the flip was baked at record time).",
    )
    p.add_argument(
        "--replay-max-frames",
        type=int,
        default=0,
        help="[replay] stop after N frames (0 = all) — quick checks",
    )
    p.add_argument(
        "--replay-cams",
        default="",
        help="[replay] which recorded cam{n}.mp4 files map to calib "
        "positions 0,1,... (comma-separated, len = --cams). "
        "Default = 0,1,..,ncam-1. Use it to run the 2-cam hand "
        "pipeline on a non-{0,1} pair, e.g. --cams 0,1 "
        "--calib <multi_to_stereo --pair 2,3 output> "
        "--replay-cams 2,3",
    )
    p.add_argument(
        "--emit-hand-port",
        type=int,
        default=0,
        help="if >0: stream the RIGHT hand 21x3 wrist-relative 3D over "
        "TCP (for orca_teleop/retarget_mpc.py --listen)",
    )
    args = p.parse_args()

    # same speed flags as RT-COSMIK's run_pipeline.py
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cam_idx = [
        int(x) if x.strip().isdigit() else x.strip()
        for x in args.cams.split(",")
    ]
    ncam = len(cam_idx)
    if args.calib:
        Ks, Ds, Rs, Ts = load_calibration(args.calib, ncam)
        base = np.linalg.norm(Ts[1]) if ncam > 1 else 0.0
        print(
            f"  calib: {ncam} cameras, baseline cam0-cam1 ≈ {base * 100:.1f} cm"
        )
    elif ncam == 1:
        # MONO mode: intrinsics from --fx (used by NLF + the hand decoder)
        cx = args.cx if args.cx > 0 else args.cap_width / 2.0
        cy = args.cy if args.cy > 0 else args.cap_height / 2.0
        K = np.array(
            [[args.fx, 0, cx], [0, args.fy or args.fx, cy], [0, 0, 1]],
            np.float64,
        )
        Ks, Ds, Rs, Ts = [K], [np.zeros(5)], [np.eye(3)], [np.zeros(3)]
        print(
            f"  MONO MODE — markers are 2D only (metric 3D needs >=2 calibrated "
            f"cameras). Hands shown in 3D at a fixed anchor. fx={args.fx:.0f}"
        )
    else:
        raise SystemExit("--calib is required with 2+ cameras")

    out_dir = args.output_dir or os.path.join(
        "output_cosmik_demo", time.strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)
    # per-hand-iteration profiling (line-buffered → tail -f friendly)
    tlog = open(os.path.join(out_dir, "timing.log"), "w", buffering=1)
    tlog.write(
        "# wall_s body_hz body_ms yolo_ms nlf_ms sync_ms "
        "hand_ms prep_ms fwd_ms post_ms tri_ms stereo_R stereo_L\n"
    )
    print(f"  timing log: {os.path.join(out_dir, 'timing.log')}")

    viz = None
    if not args.replay:
        print("[1/4] Rerun viewer...")
        viz = Viz(
            args.rerun_mode,
            args.rerun_grpc_port,
            args.rerun_web_port,
            ncam,
            out_dir,
        )

    print(
        "[2/4] Loading SAM-3D-Body (hand decoder only — no YOLO, no MoGe2)..."
    )
    est = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(
            args.checkpoint_dir, "assets", "mhr_model.pt"
        ),
        detector_name="",
        fov_name="",
        device="cuda",
    )
    R_world_handcam = Rs[args.hand_cam].T  # cam→world (world = cam0 frame)

    if args.replay:
        print("[3/3] OFFLINE replay (no live cameras, no rerun)...")
        run_offline(args, Ks, Ds, Rs, Ts, ncam, est, out_dir, R_world_handcam)
        return

    print("[3/4] Cameras + workers (NLF warmup ~10 s)...")
    cams = [
        CamThread(
            i,
            args.cap_width,
            args.cap_height,
            rotate180=args.rotate180,
            lock_focus=args.lock_focus,
            focus=args.focus,
        )
        for i in cam_idx
    ]
    for c in cams:
        c.start()
    body = BodyWorker(cams, Ks, Ds, Rs, Ts, args.det_thr, args)
    body.start()
    hand_views = (
        [args.hand_cam] if (args.mono_hands or ncam < 2) else list(range(ncam))
    )
    if len(hand_views) >= 2:
        print(
            f"  STEREO hands: decoder on views {hand_views}, epipolar check "
            f"{args.hand_reproj_thr:.0f}px (--mono-hands to disable)"
        )
        # same clamp as HandWorker: 0 < k < nviews, else selection is off
        k = (
            args.hand_topk
            if args.hand_topk >= 0
            else (2 if len(hand_views) >= 3 else 0)
        )
        if 0 < k < len(hand_views):
            print(
                f"  per-hand view selection: top-{k} of "
                f"{len(hand_views)} views per hand (crop size x palm "
                f"visibility x NLF conf x ray diversity, hysteresis x"
                f"{args.hand_switch_bonus:.2f}) — --hand-topk 0 disables"
            )
    hands = HandWorker(
        cams,
        body,
        est.model,
        (Ks, Ds, Rs, Ts),
        args,
        hand_views,
        args.hand_cam,
    )
    hands.start()

    if args.emit_hand_port > 0:
        # reuse stream_demo's generic latest-payload TCP emitter (module already
        # imported at the top of the file for its env-flag side effects)
        threading.Thread(
            target=stream_demo._emit_server,
            args=(args.emit_hand_port,),
            daemon=True,
        ).start()

    rec = None
    if not args.no_record:
        rec = {
            "b3d": [],
            "b2d": [],
            "h2d": [],
            "h2dv": [],
            "hsrc": [],
            "g70": [],
            "ts": [],
        }
    vw = None

    print("[4/4] LIVE — Ctrl+C to stop.")
    last_body = -1
    last_hand_log = -1
    ema = None
    t_prev = None
    try:
        while True:
            res, nb = body.latest()
            if res is None or nb == last_body:
                time.sleep(0.002)
                continue
            last_body = nb
            t_wall = time.time()
            hres = hands.latest()

            # cam→world rotation of each hand's mono source view (per-hand
            # view selection: the two hands may come from different cameras)
            Rw_r = Rw_l = R_world_handcam
            if hres is not None:
                if hres.get("src_r") is not None:
                    Rw_r = Rs[hres["src_r"]].T
                if hres.get("src_l") is not None:
                    Rw_l = Rs[hres["src_l"]].T
            g70 = fuse_goliath70(res["kp3d"], hres, Rw_r, Rw_l)

            # stream the right hand (wrist-relative 3D) to the teleop MPC —
            # fused (stereo when trusted, world frame) with the raw mono
            # decoder as fallback; same 21x3 float32 payload either way
            if args.emit_hand_port > 0 and hres is not None:
                hr = g70[21:42]
                if np.isfinite(hr).all():
                    k = (hr - hr[_H_WRIST]).astype(np.float32)
                elif hres["k3r"] is not None:
                    k = (hres["k3r"] - hres["k3r"][_H_WRIST]).astype(
                        np.float32
                    )
                else:
                    k = None
                if k is not None:
                    stream_demo._EMIT["buf"] = k.tobytes()
                    stream_demo._EMIT["n"] = hands.n

            # overlays
            overlays = []
            for i, c in enumerate(cams):
                f, _, _ = c.latest()
                img = (
                    f if f is not None else np.zeros((720, 1280, 3), np.uint8)
                )
                img = draw_markers(
                    img.copy(), res["kp2d"][i], res["scores"][i], args.det_thr
                )
                pb = res["boxes"][i]  # locked YOLO person box
                if np.isfinite(pb).all():
                    x, y, w, h = pb.astype(int)
                    cv2.rectangle(
                        img, (x, y), (x + w, y + h), (0, 255, 120), 2
                    )
                if hres is not None and i in hres["views"]:
                    vkr, vkl, vrb, vlb = hres["views"][i]
                    if vkr is not None:  # raw 2D detection
                        img = draw_hand_skeleton(img, vkr)
                    if vkl is not None:
                        img = draw_hand_skeleton(img, vkl)
                    for box in (vrb, vlb):
                        if box is not None and np.isfinite(box).all():
                            cv2.rectangle(
                                img,
                                tuple(box[:2].astype(int)),
                                tuple(box[2:].astype(int)),
                                (255, 170, 80),
                                2,
                            )
                # view-selection badge: which hands THIS camera is decoding
                # (green = selected; grey = skipped this frame)
                if hres is not None and hres.get("sel"):
                    for k, (hand, lab) in enumerate((("r", "R"), ("l", "L"))):
                        on = i in hres["sel"][hand]
                        cv2.putText(
                            img,
                            lab,
                            (img.shape[1] - 76 + 38 * k, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.1,
                            (60, 220, 60) if on else (90, 90, 90),
                            3,
                            cv2.LINE_AA,
                        )
                # reproject the fused metric-3D hand (what actually drives the
                # robot) — hollow skeleton, to compare against the solid raw
                # detection: any divergence is triangulation/fusion error
                if args.reproj_hands and ncam >= 2:
                    for sl in (slice(21, 42), slice(42, 63)):
                        rp = reproject_points(
                            g70[sl], Ks[i], Ds[i], Rs[i], Ts[i]
                        )
                        img = draw_hand_skeleton(
                            img, rp, hollow=True, line_w=1
                        )
                    cv2.putText(
                        img,
                        "hand: solid=2D detect  hollow=3D reproj",
                        (14, img.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                overlays.append(img)

            dt = (t_wall - t_prev) if t_prev else None
            t_prev = t_wall
            if dt and dt > 0:
                ema = 1.0 / dt if ema is None else 0.85 * ema + 0.15 / dt
            cv2.putText(
                overlays[0],
                f"NLF markers + hand decoder  {ema or 0:4.1f} Hz",
                (14, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlays[0],
                f"NLF markers + hand decoder  {ema or 0:4.1f} Hz",
                (14, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # strip of the decoder's actual input tiles: [v0 R | v0 L | v1 R ...]
            # with the detected 21-kp hand skeleton drawn ON each tile (the 2D
            # keypoints mapped into tile coords), to judge the fit up close
            crop_strip = None
            if hres is not None and hres.get("crops"):
                tiles = []
                for v in sorted(hres["crops"]):
                    hv = hres["views"].get(v, (None, None, None, None))
                    kps, boxes = (hv[0], hv[1]), (hv[2], hv[3])
                    for ti, (tile, lab) in enumerate(
                        zip(hres["crops"][v], ("R", "L"))
                    ):
                        if tile is None:  # hand not selected in this view
                            continue
                        tile = cv2.cvtColor(
                            tile, cv2.COLOR_RGB2BGR
                        )  # draw in BGR
                        kp, box = kps[ti], boxes[ti]
                        if (
                            kp is not None
                            and box is not None
                            and np.isfinite(box).all()
                        ):
                            tp = _full_to_tile(kp, box, tile.shape[:2])
                            draw_hand_skeleton(
                                tile, tp, line_w=1, joint_r=2, tip_r=3
                            )
                        cv2.putText(
                            tile,
                            f"cam{v} {lab}",
                            (6, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        tiles.append(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
                if tiles:
                    crop_strip = cv2.hconcat(tiles)
                    if crop_strip.shape[0] > 256:  # bandwidth: cap at 256 tall
                        s = 256.0 / crop_strip.shape[0]
                        crop_strip = cv2.resize(crop_strip, None, fx=s, fy=s)

            viz.log(
                nb,
                t_wall,
                overlays,
                res["kp3d"],
                g70,
                res["ms"],
                hres["ms"] if hres else None,
                res["sync_ms"],
                ema or 0.0,
                cam_fps=[c.fps for c in cams],
                jpeg_quality=args.jpeg_quality,
                hand_crops=crop_strip,
                hand_batch=(
                    hres.get("tms", {}).get("batch") if hres else None
                ),
            )

            if rec is not None:
                rec["b3d"].append(res["kp3d"])
                rec["b2d"].append(res["kp2d"])
                h2d = np.full((42, 2), np.nan, np.float32)
                if hres is not None:
                    if hres["kp_r"] is not None:
                        h2d[:21] = hres["kp_r"]
                    if hres["kp_l"] is not None:
                        h2d[21:] = hres["kp_l"]
                rec["h2d"].append(h2d)
                rec["h2dv"].append(
                    hres["kp2d_views"]
                    if hres is not None
                    else np.full((ncam, 42, 2), np.nan, np.float32)
                )
                rec["hsrc"].append(
                    [
                        -1 if hres is None or hres.get(k) is None else hres[k]
                        for k in ("src_r", "src_l")
                    ]
                )
                rec["g70"].append(g70)
                rec["ts"].append(t_wall)
            if args.save_video:
                if vw is None:
                    h, w = overlays[0].shape[:2]
                    vw = cv2.VideoWriter(
                        os.path.join(out_dir, "overlay_cam0.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        25,
                        (w, h),
                    )
                vw.write(overlays[0])

            # profiling: one timing.log line per hand-worker iteration
            if hres is not None and hands.n != last_hand_log:
                last_hand_log = hands.n
                t = hres.get("tms") or {}
                nr = (
                    int(np.isfinite(hres["X_r"]).all(1).sum())
                    if hres.get("X_r") is not None
                    else -1
                )
                nl = (
                    int(np.isfinite(hres["X_l"]).all(1).sum())
                    if hres.get("X_l") is not None
                    else -1
                )
                tlog.write(
                    f"{t_wall:.3f} {ema or 0:.1f} {res['ms']:.1f} "
                    f"{res['yolo_ms']:.1f} {res['nlf_ms']:.1f} "
                    f"{res['sync_ms']:.1f} {hres['ms']:.1f} "
                    f"{t.get('prep_ms', 0):.1f} {t.get('fwd_ms', 0):.1f} "
                    f"{t.get('post_ms', 0):.1f} {t.get('tri_ms', 0):.1f} "
                    f"{nr} {nl}\n"
                )

            if nb % 60 == 0:
                cams_fps = " ".join(
                    f"cam{i} {c.fps:.0f}" for i, c in enumerate(cams)
                )
                st = ""
                if hres is not None and hres.get("X_r") is not None:
                    nr = int(np.isfinite(hres["X_r"]).all(1).sum())
                    nl = int(np.isfinite(hres["X_l"]).all(1).sum())
                    st = f", stereo hand jts R {nr}/21 L {nl}/21"
                fa = (hres or {}).get("forearm") or {}
                if fa.get("r") or fa.get("l"):
                    st += ", forearm " + " ".join(
                        f"{k.upper()} {v * 100:.0f}cm"
                        for k, v in fa.items()
                        if v
                    )
                if hres is not None and hres.get("sel"):
                    st += ", hand views " + " ".join(
                        f"{h.upper()}{s}" for h, s in hres["sel"].items()
                    )
                print(
                    f"  body {ema or 0:4.1f} Hz  (total {res['ms']:.0f} ms: "
                    f"yolo {res['yolo_ms']:.0f} + nlf {res['nlf_ms']:.0f}, "
                    f"hands {hres['ms'] if hres else 0:.0f} ms, "
                    f"capture: {cams_fps} fps, "
                    f"sync spread {res['sync_ms']:.0f} ms{st})",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        _STOP.set()
        tlog.close()
        if vw is not None:
            vw.release()
        if rec is not None and rec["ts"]:
            np.save(
                os.path.join(out_dir, "markers_3d.npy"), np.stack(rec["b3d"])
            )
            np.save(
                os.path.join(out_dir, "markers_2d.npy"), np.stack(rec["b2d"])
            )
            np.save(
                os.path.join(out_dir, "hands_2d.npy"), np.stack(rec["h2d"])
            )
            np.save(
                os.path.join(out_dir, "hands_2d_views.npy"),
                np.stack(rec["h2dv"]),
            )
            np.save(
                os.path.join(out_dir, "hands_src.npy"),
                np.asarray(rec["hsrc"], np.int16),
            )
            np.save(
                os.path.join(out_dir, "goliath70_3d.npy"), np.stack(rec["g70"])
            )
            np.save(
                os.path.join(out_dir, "timestamps.npy"), np.asarray(rec["ts"])
            )
            print(f"  saved {len(rec['ts'])} frames to {out_dir}/")


if __name__ == "__main__":
    main()
