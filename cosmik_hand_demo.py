#!/usr/bin/env python3
"""COSMIK body + SAM hand-decoder fingers — multi-camera metric 3D keypoints.

Replaces the SAM3D/YOLO body with the COSMIK approach: RTMPose "body26"
(Halpe-26) runs on every calibrated camera view, the 2D keypoints are
triangulated into METRIC 3D (removing the monocular scale ambiguity), and the
dedicated SAM hand decoder adds faithful fingers from one view. No IK here —
this produces and records the keypoints (Rerun UI + .npy), ready to feed the
ACADOS pipeline later.

Architecture (decoupled rates — the body never waits for the hands):
  capture thread per camera  → latest frame + timestamp (soft sync)
  body worker    (~25-30 Hz) → rtmlib body26 per view → triangulate 26×3 metric
  hand worker    (~15-20 Hz) → SAM hand decoder on the hand-cam wrist crops
  main loop      (body rate) → fuse (fingers re-anchored at the metric wrists),
                               Rerun logging, recording

Outputs (in --output_dir):
  body26_3d.npy      (T, 26, 3)      metric 3D, cam0/world frame
  body26_2d.npy      (T, ncam, 26, 2)
  hands_2d.npy       (T, 42, 2)      hand-cam pixels (right 0-20, left 21-41)
  goliath70_3d.npy   (T, 70, 3)      body26 mapped to Goliath + anchored fingers
  timestamps.npy     (T,)
  overlay_cam0.mp4   (with --save-video)

Calibration (--calib) accepts:
  - stereo npz  : keys K1,D1,K2,D2,R,T          (2 cameras, cam0 = reference)
  - multi npz   : keys K0..K{n},D0..,R0..,T0..  (R0=I, T0=0 for the reference)

Run (sam3d env; pip install rtmlib):
  # multi-camera (metric 3D):
  python cosmik_hand_demo.py --cams 0,1 --calib stereo_params.npz \
      --checkpoint_dir ./checkpoints/sam-3d-body-dinov3 --rerun-mode native
  # MONO mode (no calib yet): body26 2D + hand-decoder fingers, no metric body 3D
  python cosmik_hand_demo.py --cams 0 --fx 540 --rerun-mode native
"""

import os
import sys

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# stream_demo sets the TensorRT/speed env flags before torch is imported.
import stream_demo  # noqa: F401  (import side-effects only)

import argparse
import threading
import time

import cv2
import numpy as np
import torch
from notebook.utils import setup_sam_3d_body

# Reuse the proven hand-decoder step (elbow→wrist crops, batched, un-flipped)
# from the Rerun demo — nothing is modified there.
from rerun_demo import _H_WRIST, _hand_decoder_step

# ═══════════════════════════════════════════════════════════════════════════
# HALPE-26 (rtmlib body26) — names, skeleton, COSMIK-26 → Goliath-70 map
# ═══════════════════════════════════════════════════════════════════════════

BODY26_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "mid_hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
]

# COSMIK/RT-COSMIK body26 skeleton (same edge list as their visualisation cfg)
BODY26_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 2), (5, 18), (6, 18),
    (17, 18), (5, 7), (7, 9), (6, 8), (8, 10), (18, 19),
    (11, 13), (13, 15), (15, 20), (15, 22), (15, 24),
    (12, 14), (14, 16), (16, 21), (16, 23), (16, 25),
    (12, 19), (11, 19),
]

# COSMIK-26 index → Goliath-70 index (same mapping as the comfi IK scripts;
# head/mid_hip/small_toes have no Goliath slot and are skipped).
COSMIK2GOLIATH = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 62, 10: 41, 11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 16: 14,
                  18: 69, 20: 15, 21: 18, 24: 17, 25: 20,
                  22: 16, 23: 19}

L_WRI26, R_WRI26 = 9, 10


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION + TRIANGULATION (self-contained, NaN-safe)
# ═══════════════════════════════════════════════════════════════════════════

def load_calibration(path, ncam):
    """Return (Ks, Ds, Rs, Ts) lists, cam0 = reference frame (R0=I, T0=0)."""
    z = np.load(path)
    keys = set(z.keys())
    if {"K1", "K2", "R", "T"} <= keys:                      # stereo npz format
        if ncam != 2:
            raise ValueError(f"stereo calib is for 2 cameras, got --cams with {ncam}")
        Ks = [z["K1"].astype(np.float64), z["K2"].astype(np.float64)]
        Ds = [z.get("D1", np.zeros(5)).astype(np.float64),
              z.get("D2", np.zeros(5)).astype(np.float64)]
        Rs = [np.eye(3), z["R"].astype(np.float64)]
        Ts = [np.zeros(3), z["T"].astype(np.float64).reshape(3)]
        return Ks, Ds, Rs, Ts
    if f"K{ncam - 1}" in keys:                              # multi-cam npz format
        Ks = [z[f"K{i}"].astype(np.float64) for i in range(ncam)]
        Ds = [z.get(f"D{i}", np.zeros(5)).astype(np.float64) for i in range(ncam)]
        Rs = [z[f"R{i}"].astype(np.float64) if f"R{i}" in keys else np.eye(3)
              for i in range(ncam)]
        Ts = [z[f"T{i}"].astype(np.float64).reshape(3) if f"T{i}" in keys else np.zeros(3)
              for i in range(ncam)]
        return Ks, Ds, Rs, Ts
    raise ValueError(f"Unrecognized calibration file {path} — found keys {sorted(keys)}; "
                     "expected K1,D1,K2,D2,R,T (stereo) or K0..,D0..,R0..,T0.. (multi)")


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
                pts2d[i][v].reshape(-1, 1, 2).astype(np.float64), Ks[i], Ds[i])
            norm[i][v] = und.reshape(-1, 2)
    out = np.full((J, 3), np.nan, np.float32)
    for j in range(J):
        rows = []
        for i in range(ncam):
            if scores[i, j] > thr and np.isfinite(norm[i, j]).all():
                x, y = norm[i, j]
                rows.append(x * Ps[i][2] - Ps[i][0])
                rows.append(y * Ps[i][2] - Ps[i][1])
        if len(rows) >= 4:                                   # >= 2 views
            _, _, vt = np.linalg.svd(np.asarray(rows))
            X = vt[-1]
            if abs(X[3]) > 1e-12:
                out[j] = (X[:3] / X[3]).astype(np.float32)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# WORKERS (shared latest-value slots, drop-old everywhere)
# ═══════════════════════════════════════════════════════════════════════════

_STOP = threading.Event()


class CamThread(threading.Thread):
    """Grab continuously; keep only the latest frame (+ wall-clock timestamp)."""

    def __init__(self, index, width, height):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open camera {index}")
        # MJPG so several 720p webcams fit on the USB bus
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fcc_s = "".join(chr((fcc >> (8 * i)) & 0xFF) for i in range(4))
        print(f"  cam {index}: {w}x{h} fourcc={fcc_s} "
              f"(nominal {self.cap.get(cv2.CAP_PROP_FPS):.0f} fps)")
        self.frame, self.ts, self.n = None, 0.0, 0
        self.fps = 0.0                 # MEASURED capture rate (EMA)
        self._lock = threading.Lock()

    def run(self):
        t_prev = None
        while not _STOP.is_set():
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            now = time.time()
            if t_prev is not None and now > t_prev:
                inst = 1.0 / (now - t_prev)
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            t_prev = now
            with self._lock:
                self.frame, self.ts, self.n = f, now, self.n + 1
        self.cap.release()

    def latest(self):
        with self._lock:
            return (None, 0.0, -1) if self.frame is None else \
                (self.frame.copy(), self.ts, self.n)


class BodyWorker(threading.Thread):
    """rtmlib body26 on every view → triangulated 26×3 metric (latest slot)."""

    _DET_MODELS = {
        # tiny is plenty to frame one person; keypoint quality comes from RTMPose.
        # The x variant (comfi's offline config) costs a periodic ~100-500 ms
        # detection spike that wrecks the live cadence.
        "tiny": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
                 (416, 416)),
        "m": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
              (640, 640)),
        "x": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
              (640, 640)),
    }

    def __init__(self, cams, Ks, Ds, Rs, Ts, det_thr, device,
                 det_model="m", det_freq=10, tracking=True):
        super().__init__(daemon=True)
        from functools import partial
        import onnxruntime
        from rtmlib import Custom, PoseTracker
        print(f"  onnxruntime providers: {onnxruntime.get_available_providers()}")
        det_url, det_size = self._DET_MODELS[det_model]
        custom = partial(
            Custom, to_openpose=False,
            det_class="YOLOX", det=det_url, det_input_size=det_size,
            pose_class="RTMPose",
            pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
            pose_input_size=(192, 256),
            backend="onnxruntime", device=device,
        )
        # one tracker per camera (they hold per-stream state); with tracking the
        # person bbox propagates between frames and the detector only runs every
        # det_freq frames — cheap on GPU, and avoids per-frame detection cost
        self.trackers = [PoseTracker(custom, det_frequency=det_freq, tracking=tracking,
                                     backend="onnxruntime", device=device)
                         for _ in cams]
        self.cams = cams
        self.calib = (Ks, Ds, Rs, Ts)
        self.det_thr = det_thr
        self.result = None            # dict: kp2d (ncam,26,2), scores, kp3d, ts, ms
        self.n = 0
        self._lock = threading.Lock()

    @staticmethod
    def _main_person(kps, scores):
        """Largest bbox extent × mean score → (26,2), (26,) or (None, None)."""
        if kps is None or len(kps) == 0:
            return None, None
        best, best_v = None, -1.0
        for k, s in zip(kps, scores):
            v = np.isfinite(k).all(1) & (s > 0.1)
            if v.sum() < 6:
                continue
            # np.ptp(): the ndarray.ptp() method was removed in NumPy 2.0
            ext = float(np.ptp(k[v, 0]) * np.ptp(k[v, 1])) * float(s.mean())
            if ext > best_v:
                best_v, best = ext, (k, s)
        return best if best is not None else (None, None)

    def run(self):
        Ks, Ds, Rs, Ts = self.calib
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
            if any(f is None for f in frames) or all(n == l for n, l in zip(ns, last_ns)):
                time.sleep(0.002)
                continue
            last_ns = ns
            t0 = time.perf_counter()
            kp2d = np.full((ncam, 26, 2), np.nan, np.float32)
            sc = np.zeros((ncam, 26), np.float32)
            for i, f in enumerate(frames):
                kps, scores = self.trackers[i](f)
                k, s = self._main_person(kps, scores)
                if k is not None:
                    kp2d[i] = k[:26]
                    sc[i] = s[:26]
            if ncam >= 2:
                kp2d_masked = kp2d.copy()
                kp2d_masked[sc < self.det_thr] = np.nan
                kp3d = triangulate_multiview(kp2d_masked, sc, Ks, Ds, Rs, Ts,
                                             thr=self.det_thr)
            else:
                # MONO mode: no metric body 3D possible with a single view
                kp3d = np.full((26, 3), np.nan, np.float32)
            ms = (time.perf_counter() - t0) * 1e3
            with self._lock:
                self.result = {"kp2d": kp2d, "scores": sc, "kp3d": kp3d,
                               "frames_ts": tss, "ms": ms,
                               "sync_ms": (max(tss) - min(tss)) * 1e3}
                self.n += 1

    def latest(self):
        with self._lock:
            return self.result, self.n


class HandWorker(threading.Thread):
    """SAM hand decoder on the hand camera, guided by that view's body26 wrists."""

    def __init__(self, cam, body, model, cam_int, args, view_idx):
        super().__init__(daemon=True)
        self.cam, self.body, self.model = cam, body, model
        self.cam_int, self.args, self.view = cam_int, args, view_idx
        self.result = None            # dict: kp_r/kp_l 2D, k3r/k3l cam-frame 3D, ms
        self.n = 0
        self._lock = threading.Lock()

    def run(self):
        last_body = -1
        while not _STOP.is_set():
            res, nb = self.body.latest()
            frame, _, _ = self.cam.latest()
            if res is None or frame is None or nb == last_body:
                time.sleep(0.003)
                continue
            last_body = nb
            # COCO-17 layout expected by _hand_decoder_step = body26[:17]
            k17 = res["kp2d"][self.view][:17].copy()
            k17[res["scores"][self.view][:17] < self.args.det_thr] = np.nan
            t0 = time.perf_counter()
            kp_r, kp_l, k3r, k3l, rbox, lbox = _hand_decoder_step(
                self.model, frame, k17, self.cam_int, self.args)
            ms = (time.perf_counter() - t0) * 1e3
            with self._lock:
                self.result = {"kp_r": kp_r, "kp_l": kp_l,
                               "k3r": k3r, "k3l": k3l, "ms": ms,
                               "rbox": rbox, "lbox": lbox}
                self.n += 1

    def latest(self):
        with self._lock:
            return self.result


# ═══════════════════════════════════════════════════════════════════════════
# FUSION — fingers re-anchored at the metric wrists
# ═══════════════════════════════════════════════════════════════════════════

def fuse_goliath70(kp3d26, hands, R_world_handcam):
    """(26,3) metric body + decoder hands → (70,3) Goliath, world/cam0 frame.

    Decoder hand offsets are rotated from the hand-camera frame into the world
    frame, then anchored at the triangulated (metric) wrist. The decoder's own
    hand size is kept (MHR average-hand scale, roughly metric).
    """
    g = np.full((70, 3), np.nan, np.float32)
    for c, gi in COSMIK2GOLIATH.items():
        g[gi] = kp3d26[c]
    if hands is None:
        return g
    for dec, sl, wri26, disp in ((hands["k3r"], slice(21, 42), R_WRI26, (+0.15, 0, 0.5)),
                                 (hands["k3l"], slice(42, 63), L_WRI26, (-0.15, 0, 0.5))):
        if dec is None:
            continue
        off = (dec - dec[_H_WRIST]) @ R_world_handcam.T
        if np.isfinite(kp3d26[wri26]).all():
            g[sl] = kp3d26[wri26] + off              # metric wrist anchor
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
            rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=url)
            print(f"  Rerun UI: http://localhost:{web_port}  (tunnel {web_port} AND {grpc_port})")
        elif mode == "native":
            rr.spawn(port=grpc_port)
        else:
            rr.save(os.path.join(output_dir, "session.rrd"))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Vertical(
                *[rrb.Spatial2DView(origin=f"cams/cam{i}", name=f"Camera {i}")
                  for i in range(ncam)],
                rrb.TimeSeriesView(origin="timing", name="Latency (ms)"),
            ),
            rrb.Spatial3DView(origin="world", name="3D (metric)",
                              background=rrb.Background(color=[25, 25, 25])),
            column_shares=[2, 3],
        )))

    def log(self, seq, t_wall, overlays, kp3d26, g70, body_ms, hand_ms, sync_ms,
            fps, cam_fps=None, jpeg_quality=75):
        rr = self.rr
        rr.set_time("frame", sequence=seq)
        rr.set_time("time", timestamp=t_wall)
        rr.log("timing/body_ms", rr.Scalars(float(body_ms)))
        if hand_ms is not None:
            rr.log("timing/hand_ms", rr.Scalars(float(hand_ms)))
        rr.log("timing/sync_ms", rr.Scalars(float(sync_ms)))
        rr.log("timing/body_fps", rr.Scalars(float(fps)))
        for i, cf in enumerate(cam_fps or []):
            rr.log(f"timing/cam{i}_capture_fps", rr.Scalars(float(cf)))
        for i, img in enumerate(overlays):
            rr.log(f"cams/cam{i}", rr.Image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=jpeg_quality))
        # 3D: body26 skeleton (metric) + Goliath fingers
        v = np.isfinite(kp3d26).all(1)
        if v.any():
            rr.log("world/body/joints", rr.Points3D(kp3d26[v], radii=0.015,
                                                    colors=[0, 230, 0]))
            strips = [[kp3d26[a].tolist(), kp3d26[b].tolist()]
                      for a, b in BODY26_EDGES if v[a] and v[b]]
            if strips:
                rr.log("world/body/bones",
                       rr.LineStrips3D(strips, colors=[255, 230, 0], radii=0.006))
        hv = np.isfinite(g70[21:63]).all(1)
        if hv.any():
            rr.log("world/hands/joints", rr.Points3D(g70[21:63][hv], radii=0.006,
                                                     colors=[80, 170, 255]))
            strips = []
            for base in (21, 42):
                wrist = base + _H_WRIST
                for f0 in range(5):
                    chain = [wrist, base + 4 * f0 + 3, base + 4 * f0 + 2,
                             base + 4 * f0 + 1, base + 4 * f0]
                    strips += [[g70[a].tolist(), g70[b].tolist()]
                               for a, b in zip(chain[:-1], chain[1:])
                               if np.isfinite(g70[[a, b]]).all()]
            if strips:
                rr.log("world/hands/bones",
                       rr.LineStrips3D(strips, colors=[80, 170, 255], radii=0.003))


def draw_body26(img, kp, sc, thr):
    for a, b in BODY26_EDGES:
        if sc[a] > thr and sc[b] > thr:
            cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)),
                     (0, 200, 255), 2, cv2.LINE_AA)
    for j in range(26):
        if sc[j] > thr:
            cv2.circle(img, tuple(kp[j].astype(int)), 3, (0, 140, 255), -1, cv2.LINE_AA)
    return img


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="COSMIK body26 (multi-cam metric) + SAM hand decoder")
    p.add_argument("--cams", default="0", help="comma-separated camera indices")
    p.add_argument("--calib", default="", help="calibration npz (see docstring); "
                   "required for >=2 cameras, optional in mono mode")
    p.add_argument("--fx", type=float, default=540.0,
                   help="[mono] focal length in px (default 540 ≈ 720p webcam)")
    p.add_argument("--fy", type=float, default=0)
    p.add_argument("--cx", type=float, default=0)
    p.add_argument("--cy", type=float, default=0)
    p.add_argument("--hand-cam", type=int, default=0,
                   help="which view (position in --cams) runs the hand decoder")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--checkpoint_dir", default="./checkpoints/sam-3d-body-dinov3")
    p.add_argument("--cap-width", type=int, default=1280)
    p.add_argument("--cap-height", type=int, default=720)
    p.add_argument("--det-thr", type=float, default=0.3)
    p.add_argument("--det-model", choices=["tiny", "m", "x"], default="m",
                   help="YOLOX person detector size (m: good range/speed balance "
                        "on GPU; tiny: lightest, short range; x: heaviest)")
    p.add_argument("--det-freq", type=int, default=10,
                   help="run the detector every N frames (tracking covers the rest)")
    p.add_argument("--no-tracking", action="store_true",
                   help="disable bbox tracking: detect on every det-freq schedule "
                        "regardless (use if tracking drifts and loses you)")
    p.add_argument("--box-offset", type=float, default=0.35)
    p.add_argument("--box-size", type=float, default=1.0)
    p.add_argument("--box-scale-mode", choices=["stable", "forearm"], default="stable",
                   help="stable: shoulder-width floor on the hand box (no foreshortening "
                        "shrink); forearm: original behaviour")
    p.add_argument("--box-shoulder-frac", type=float, default=0.5)
    p.add_argument("--hand-res", type=int, default=0)
    p.add_argument("--rerun-mode", choices=["web", "native", "save"], default="web")
    p.add_argument("--rerun-grpc-port", type=int, default=9876)
    p.add_argument("--rerun-web-port", type=int, default=9090)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument("--output_dir", default="")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--emit-hand-port", type=int, default=0,
                   help="if >0: stream the RIGHT hand 21x3 wrist-relative 3D over "
                        "TCP (for orca_teleop/retarget_mpc.py --listen)")
    args = p.parse_args()

    cam_idx = [int(x) for x in args.cams.split(",")]
    ncam = len(cam_idx)
    if args.calib:
        Ks, Ds, Rs, Ts = load_calibration(args.calib, ncam)
        base = np.linalg.norm(Ts[1]) if ncam > 1 else 0.0
        print(f"  calib: {ncam} cameras, baseline cam0-cam1 ≈ {base * 100:.1f} cm")
    elif ncam == 1:
        # MONO mode: intrinsics from --fx (used by the hand decoder only)
        cx = args.cx if args.cx > 0 else args.cap_width / 2.0
        cy = args.cy if args.cy > 0 else args.cap_height / 2.0
        K = np.array([[args.fx, 0, cx], [0, args.fy or args.fx, cy], [0, 0, 1]],
                     np.float64)
        Ks, Ds, Rs, Ts = [K], [np.zeros(5)], [np.eye(3)], [np.zeros(3)]
        print(f"  MONO MODE — body26 is 2D only (metric 3D needs >=2 calibrated "
              f"cameras). Hands shown in 3D at a fixed anchor. fx={args.fx:.0f}")
    else:
        raise SystemExit("--calib is required with 2+ cameras")

    out_dir = args.output_dir or os.path.join(
        "output_cosmik_demo", time.strftime("%Y%m%d_%H%M%S"))
    if not args.no_record or args.rerun_mode == "save":
        os.makedirs(out_dir, exist_ok=True)

    print("[1/4] Rerun viewer...")
    viz = Viz(args.rerun_mode, args.rerun_grpc_port, args.rerun_web_port, ncam, out_dir)

    print("[2/4] Loading SAM-3D-Body (hand decoder only — no YOLO, no MoGe2)...")
    est = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(args.checkpoint_dir, "assets", "mhr_model.pt"),
        detector_name="", fov_name="", device="cuda",
    )
    hand_cam_int = torch.tensor([Ks[args.hand_cam]], dtype=torch.float32)
    R_world_handcam = Rs[args.hand_cam].T          # cam→world (world = cam0 frame)

    print("[3/4] Cameras + workers...")
    cams = [CamThread(i, args.cap_width, args.cap_height) for i in cam_idx]
    for c in cams:
        c.start()
    body = BodyWorker(cams, Ks, Ds, Rs, Ts, args.det_thr, device="cuda",
                      det_model=args.det_model, det_freq=args.det_freq,
                      tracking=not args.no_tracking)
    body.start()
    hands = HandWorker(cams[args.hand_cam], body, est.model, hand_cam_int,
                       args, args.hand_cam)
    hands.start()

    if args.emit_hand_port > 0:
        # reuse stream_demo's generic latest-payload TCP emitter
        import stream_demo as _sd
        threading.Thread(target=_sd._emit_server, args=(args.emit_hand_port,),
                         daemon=True).start()

    rec = None
    if not args.no_record:
        rec = {"b3d": [], "b2d": [], "h2d": [], "g70": [], "ts": []}
    vw = None

    print("[4/4] LIVE — Ctrl+C to stop.")
    last_body = -1
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

            g70 = fuse_goliath70(res["kp3d"], hres, R_world_handcam)

            # stream the right hand (wrist-relative camera-frame 3D) to the teleop MPC
            if args.emit_hand_port > 0 and hres is not None and hres["k3r"] is not None:
                import stream_demo as _sd
                k = (hres["k3r"] - hres["k3r"][20]).astype(np.float32)
                _sd._EMIT["buf"] = k.tobytes()
                _sd._EMIT["n"] = hands.n

            # overlays
            overlays = []
            for i, c in enumerate(cams):
                f, _, _ = c.latest()
                img = f if f is not None else np.zeros((720, 1280, 3), np.uint8)
                img = draw_body26(img.copy(), res["kp2d"][i], res["scores"][i],
                                  args.det_thr)
                if i == args.hand_cam and hres is not None:
                    from body_hand_decoder_extractor import _draw_hand
                    if hres["kp_r"] is not None:
                        img = _draw_hand(img, hres["kp_r"])
                    if hres["kp_l"] is not None:
                        img = _draw_hand(img, hres["kp_l"])
                    for box in (hres.get("rbox"), hres.get("lbox")):
                        if box is not None and np.isfinite(box).all():
                            cv2.rectangle(img, tuple(box[:2].astype(int)),
                                          tuple(box[2:].astype(int)),
                                          (255, 170, 80), 2)
                overlays.append(img)

            dt = (t_wall - t_prev) if t_prev else None
            t_prev = t_wall
            if dt and dt > 0:
                ema = 1.0 / dt if ema is None else 0.85 * ema + 0.15 / dt
            cv2.putText(overlays[0], f"COSMIK body26 + hand decoder  {ema or 0:4.1f} Hz",
                        (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlays[0], f"COSMIK body26 + hand decoder  {ema or 0:4.1f} Hz",
                        (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            viz.log(nb, t_wall, overlays, res["kp3d"], g70,
                    res["ms"], hres["ms"] if hres else None, res["sync_ms"],
                    ema or 0.0, cam_fps=[c.fps for c in cams],
                    jpeg_quality=args.jpeg_quality)

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
                rec["g70"].append(g70)
                rec["ts"].append(t_wall)
            if args.save_video:
                if vw is None:
                    h, w = overlays[0].shape[:2]
                    vw = cv2.VideoWriter(os.path.join(out_dir, "overlay_cam0.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                vw.write(overlays[0])

            if nb % 60 == 0:
                cams_fps = " ".join(f"cam{i} {c.fps:.0f}" for i, c in enumerate(cams))
                print(f"  body {ema or 0:4.1f} Hz  (rtm+tri {res['ms']:.0f} ms, "
                      f"hands {hres['ms'] if hres else 0:.0f} ms, "
                      f"capture: {cams_fps} fps, "
                      f"sync spread {res['sync_ms']:.0f} ms)", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        _STOP.set()
        if vw is not None:
            vw.release()
        if rec is not None and rec["ts"]:
            np.save(os.path.join(out_dir, "body26_3d.npy"), np.stack(rec["b3d"]))
            np.save(os.path.join(out_dir, "body26_2d.npy"), np.stack(rec["b2d"]))
            np.save(os.path.join(out_dir, "hands_2d.npy"), np.stack(rec["h2d"]))
            np.save(os.path.join(out_dir, "goliath70_3d.npy"), np.stack(rec["g70"]))
            np.save(os.path.join(out_dir, "timestamps.npy"), np.asarray(rec["ts"]))
            print(f"  saved {len(rec['ts'])} frames to {out_dir}/")


if __name__ == "__main__":
    main()
