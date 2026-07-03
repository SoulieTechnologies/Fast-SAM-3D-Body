#!/usr/bin/env python3
"""Live SAM3D demo with a Rerun UI — camera → SAM3D → Rerun viewer (video with
skeleton overlay + 3D skeleton alone + latency plots), records everything to disk,
and optionally feeds the live ACADOS IK retargeting process (comfi-examples).

Panels (InstantHMR-style layout):
  - Camera      : annotated frame (2D skeleton + FPS HUD)
  - 3D Skeleton : SAM3D 70-keypoint 3D pose, alone in a 3D view
  - Retarget    : (when --emit-port) markers logged by the IK process (Proc B)
  - Latency     : per-stage ms + FPS time series

Recording (in --output_dir, on Ctrl+C / end of video):
  raw.mp4               camera frames as captured
  overlay.mp4           annotated frames
  joints_2d.npy         (T,70,2)  SAM3D 2D keypoints
  joints_3d.npy         (T,70,3)  SAM3D 3D, camera frame
  joints_3d_world.npy   (T,70,3)  gravity-aligned (NaN until alignment locks)
  timestamps.npy        (T,)      capture wall-clock times

Run (server, sam3d env):
  python rerun_demo.py --source 0 --gpu 7 --fx 900 --emit-port 8090
  # Rerun UI (laptop):  ssh -L 9090:localhost:9090 ... → http://localhost:9090
  # or on a machine with a screen:  --rerun-mode native

Proc B (acados env, for the retargeting — see run_rerun_demo.sh to launch both):
  python scripts/run_ik_live_rerun.py --emit-port 8090 \
      --rerun-url rerun+http://127.0.0.1:9876/proxy
  # meshcat (URDF human):  ssh -L 7000:localhost:7000 → http://localhost:7000/static/

Requires:  pip install "rerun-sdk>=0.28"   (written against 0.33)
"""

import os
import sys

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# stream_demo sets the TensorRT/speed env flags at import time (before torch) and
# provides the camera receiver, keypoint emitter, person selection and intrinsics
# helpers — reuse them instead of duplicating.
import stream_demo as sd
from stream_demo import (
    _EMIT, _RECV, _STOP,
    _canonical_M, _emit_server, _estimate_intrinsics, _quiet, _receiver,
    _select_person,
)

import argparse
import threading
import time

import cv2
import numpy as np
import torch
from notebook.utils import setup_sam_3d_body
from visualize_skeleton_video import ALL_BONES, draw_option_b, draw_skeleton

# The ~15fps YOLO-body + dedicated-hand-decoder pipeline (--inference-type bodyhand)
# reuses the building blocks of body_hand_decoder_extractor.py.
from body_hand_decoder_extractor import (
    HAND_SRC, L_ELBOW, L_WRIST, R_ELBOW, R_WRIST,
    _draw_body, _draw_hand, _hand_box, _largest,
)
from sam_3d_body.models.meta_arch.sam3d_body import _prepare_hand_batches_gpu

# COCO-17 (YOLO-pose) index → Goliath-70 index (wrists live in the hand blocks).
_COCO2GOLIATH = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                 9: 62, 10: 41, 11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 16: 14}


# ═══════════════════════════════════════════════════════════════════════════
# RERUN VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════

# ALL_BONES colors are OpenCV BGR → RGB for Rerun.
_BONES_RGB = [(a, b, (c[2], c[1], c[0])) for a, b, c in ALL_BONES]


class RerunViz:
    """Rerun logging: camera panel, 3D skeleton panel, timing plots.

    modes:
      web    — serve gRPC + web viewer (browser over SSH tunnel, like meshcat)
      native — spawn the local Rerun viewer window
      save   — no viewer, write a .rrd file into the output dir
    """

    def __init__(self, mode, grpc_port, web_port, output_dir, with_retarget):
        import rerun as rr
        import rerun.blueprint as rrb
        self.rr = rr

        rr.init("fastsam3d_live")
        self.grpc_url = None
        if mode == "web":
            self.grpc_url = rr.serve_grpc(grpc_port=grpc_port)
            rr.serve_web_viewer(web_port=web_port, open_browser=False,
                                connect_to=self.grpc_url)
            print(f"  Rerun UI:  ssh -L {web_port}:localhost:{web_port} ...  "
                  f"→ http://localhost:{web_port}")
            print(f"  Rerun gRPC (for the IK process): {self.grpc_url}")
        elif mode == "native":
            rr.spawn()
            # also serve gRPC so the IK process can join the same viewer
            self.grpc_url = rr.serve_grpc(grpc_port=grpc_port)
        else:  # save
            path = os.path.join(output_dir, "session.rrd")
            rr.save(path)
            print(f"  Rerun recording → {path}")

        # SAM3D camera frame is right-handed, Y down.
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        views_3d = [rrb.Spatial3DView(origin="world", name="3D Skeleton",
                                      background=rrb.Background(color=[25, 25, 25]))]
        if with_retarget:
            views_3d.append(rrb.Spatial3DView(origin="retarget", name="Retargeted (ACADOS IK)",
                                              background=rrb.Background(color=[25, 25, 35])))
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/camera/image", name="Camera"),
                rrb.TimeSeriesView(origin="timing", name="Latency (ms) / FPS"),
                row_shares=[3, 1],
            ),
            *views_3d,
            column_shares=[2] + [2] * len(views_3d),
        )))

    def log_frame(self, frame_idx, t_wall, annotated_bgr, kp3d, K,
                  infer_ms, e2e_ms, fps, jpeg_quality=75):
        rr = self.rr
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", timestamp=t_wall)

        rr.log("timing/infer_ms", rr.Scalars(float(infer_ms)))
        rr.log("timing/e2e_ms", rr.Scalars(float(e2e_ms)))
        rr.log("timing/fps", rr.Scalars(float(fps)))

        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        rr.log("world/camera/image", rr.Image(rgb).compress(jpeg_quality=jpeg_quality))
        if K is not None:
            h, w = annotated_bgr.shape[:2]
            rr.log("world/camera", rr.Pinhole(
                width=w, height=h,
                focal_length=[float(K[0, 0]), float(K[1, 1])],
                principal_point=[float(K[0, 2]), float(K[1, 2])],
                image_plane_distance=0.6,
            ))

        if kp3d is None or not np.isfinite(kp3d).any():
            rr.log("world/skeleton", rr.Clear(recursive=True))
            return
        valid = np.isfinite(kp3d).all(axis=1)
        rr.log("world/skeleton/joints", rr.Points3D(
            positions=kp3d[valid], radii=0.012, colors=[0, 230, 0]))
        strips, colors = [], []
        for a, b, rgb_c in _BONES_RGB:
            if valid[a] and valid[b]:
                strips.append([kp3d[a].tolist(), kp3d[b].tolist()])
                colors.append(rgb_c)
        if strips:
            rr.log("world/skeleton/bones",
                   rr.LineStrips3D(strips, colors=colors, radii=0.005))


# ═══════════════════════════════════════════════════════════════════════════
# RECORDER
# ═══════════════════════════════════════════════════════════════════════════

class Recorder:
    """Write raw + overlay videos and accumulate keypoints; finalized on exit."""

    def __init__(self, output_dir, fps):
        os.makedirs(output_dir, exist_ok=True)
        self.dir = output_dir
        self.fps = fps
        self._raw = None
        self._ovl = None
        self.kp2d, self.kp3d, self.kp3d_world, self.ts = [], [], [], []

    def add(self, raw_bgr, overlay_bgr, kp2d, kp3d, kp3d_world, t_wall):
        if self._raw is None:
            h, w = raw_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._raw = cv2.VideoWriter(os.path.join(self.dir, "raw.mp4"),
                                        fourcc, self.fps, (w, h))
            self._ovl = cv2.VideoWriter(os.path.join(self.dir, "overlay.mp4"),
                                        fourcc, self.fps, (w, h))
        self._raw.write(raw_bgr)
        self._ovl.write(overlay_bgr)
        nan2 = np.full((70, 2), np.nan, np.float32)
        nan3 = np.full((70, 3), np.nan, np.float32)
        self.kp2d.append(nan2 if kp2d is None else kp2d.astype(np.float32))
        self.kp3d.append(nan3 if kp3d is None else kp3d.astype(np.float32))
        self.kp3d_world.append(nan3 if kp3d_world is None else kp3d_world.astype(np.float32))
        self.ts.append(t_wall)

    def close(self):
        if self._raw is not None:
            self._raw.release()
            self._ovl.release()
        if self.kp2d:
            np.save(os.path.join(self.dir, "joints_2d.npy"), np.stack(self.kp2d))
            np.save(os.path.join(self.dir, "joints_3d.npy"), np.stack(self.kp3d))
            np.save(os.path.join(self.dir, "joints_3d_world.npy"), np.stack(self.kp3d_world))
            np.save(os.path.join(self.dir, "timestamps.npy"), np.asarray(self.ts))
            print(f"  Recording saved to {self.dir}/  "
                  f"({len(self.kp2d)} frames @ nominal {self.fps:.1f} fps — "
                  f"see timestamps.npy for exact timing)")


# ═══════════════════════════════════════════════════════════════════════════
# BODYHAND PIPELINE (YOLO body + dedicated SAM hand decoder, ~15fps design)
# ═══════════════════════════════════════════════════════════════════════════

def _bodyhand_step(est, frame_bgr, cam_int, args):
    """One frame of the YOLO-body + hand-decoder pipeline.

    Returns (kp70_2d, body17_2d, kp_r21, kp_l21, hands3d70) or None if nobody
    detected. hands3d70 is (70,3) with only the hand blocks filled (each hand
    re-anchored near the origin for the 3D panel — there is NO 3D body here).
    """
    model = est.model
    with torch.no_grad(), _quiet():
        dr = est.detector.run_human_detection(
            frame_bgr, det_cat_id=0, bbox_thr=0.5, nms_thr=0.3,
            default_to_full_image=False)
    boxes = dr["boxes"] if isinstance(dr, dict) else dr
    kps = dr.get("keypoints") if isinstance(dr, dict) else None
    sel = _largest(boxes)
    if sel is None or kps is None or len(kps) <= sel:
        return None
    k = kps[sel]                                   # (17,3) x,y,conf
    body = k[:, :2].copy()
    body[k[:, 2] < 0.3] = np.nan

    kp_r = kp_l = None
    h3d = np.full((70, 3), np.nan, np.float32)
    rbox = _hand_box(k[R_WRIST, :2], k[R_ELBOW, :2], args.box_offset, args.box_size)
    lbox = _hand_box(k[L_WRIST, :2], k[L_ELBOW, :2], args.box_offset, args.box_size)
    if rbox is not None and lbox is not None and cam_int is not None:
        out_hw = ((args.hand_res, args.hand_res) if args.hand_res > 0
                  else (model.cfg.MODEL.IMAGE_SIZE[1], model.cfg.MODEL.IMAGE_SIZE[0]))
        with torch.no_grad(), _quiet():
            bl, br, _ = _prepare_hand_batches_gpu(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), lbox[None], rbox[None],
                cam_int, output_size=out_hw, padding=0.9, device="cuda")
            bh = model._merge_hand_batches(bl, br)
            model._initialize_batch(bh)
            merged = model.forward_step(bh, decoder_type="hand")
            lh, rh = model._split_hand_outputs(merged, batch_size=1)
        kp_r = rh["mhr_hand"]["pred_keypoints_2d"][0].detach().cpu().numpy()[HAND_SRC]
        kp_l = lh["mhr_hand"]["pred_keypoints_2d"][0].detach().cpu().numpy()[HAND_SRC].copy()
        kp_l[:, 0] = frame_bgr.shape[1] - kp_l[:, 0] - 1     # un-flip left hand
        # 3D fingers for the 3D panel (each hand anchored at its wrist near origin)
        for out, sl, mirror, anchor in ((rh, slice(21, 42), False, (+0.15, 0, 0.5)),
                                        (lh, slice(42, 63), True, (-0.15, 0, 0.5))):
            k3 = out["mhr_hand"].get("pred_keypoints_3d")
            if k3 is None:
                continue
            k3 = k3[0].detach().cpu().numpy()[HAND_SRC].copy()
            if mirror:
                k3[:, 0] *= -1                               # un-flip left hand
            k3 = k3 - k3[20] + np.asarray(anchor, np.float32)  # wrist → anchor
            h3d[sl] = k3

    kp70 = np.full((70, 2), np.nan, np.float32)
    for c, g in _COCO2GOLIATH.items():
        kp70[g] = body[c]
    if kp_r is not None:
        kp70[21:42] = kp_r
    if kp_l is not None:
        kp70[42:63] = kp_l
    return kp70, body, kp_r, kp_l, h3d


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run(args, estimator, cam_int, viz, rec):
    net = args.recv_port > 0
    cap = None
    is_cam = False
    if not net:
        is_cam = args.source.isdigit()
        cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open source: {args.source}")
        if is_cam:
            # Explicitly request the capture resolution — OpenCV often defaults a
            # webcam to 640x480, which shrinks the hands and degrades everything.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  camera capture: {w}x{h}"
                  + ("" if (w, h) == (args.cap_width, args.cap_height)
                     else f"  (requested {args.cap_width}x{args.cap_height} — camera negotiated down)"))

    kw = {"inference_type": args.inference_type}
    if cam_int is not None:
        kw["cam_int"] = cam_int
    auto_est = cam_int is None and getattr(estimator, "fov_estimator", None) is not None
    K_np = cam_int[0].numpy() if cam_int is not None else None

    img_diag = None
    cen = None
    ema = None
    M = None            # gravity+facing alignment (locked once after warmup)
    align_buf = []
    frame_idx = 0
    last_recv = -1

    while not _STOP.is_set():
        # ── acquire ──────────────────────────────────────────────────────
        if net:
            if _RECV["n"] == last_recv or _RECV["frame"] is None:
                time.sleep(0.003)
                continue
            last_recv = _RECV["n"]
            frame = _RECV["frame"]
        else:
            ok, frame = cap.read()
            if not ok:
                if is_cam:
                    break
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
        t_wall = time.time()
        raw = frame.copy()
        frame_idx += 1
        if img_diag is None:
            img_diag = float(np.hypot(frame.shape[1], frame.shape[0]))

        if auto_est:    # one-time MoGe2 calibration on the first frame
            with torch.no_grad():
                K_np = estimator.fov_estimator.get_cam_intrinsics(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).squeeze().cpu().numpy()
            kw["cam_int"] = torch.tensor([K_np], dtype=torch.float32)
            print(f"  AUTO intrinsics: fx={K_np[0,0]:.0f} fy={K_np[1,1]:.0f} "
                  f"cx={K_np[0,2]:.0f} cy={K_np[1,2]:.0f}", flush=True)
            auto_est = False

        # ── inference ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        kp = yolo_kp = kp3d = world = None
        body17 = kp_r = kp_l = None
        if args.inference_type == "bodyhand":
            # YOLO body + dedicated hand decoder (no SAM body pass, no 3D body)
            res = _bodyhand_step(estimator, frame, kw.get("cam_int"), args)
            infer_ms = (time.perf_counter() - t0) * 1e3
            if res is not None:
                kp, body17, kp_r, kp_l, kp3d = res    # kp3d: hands-only (70,3)
        else:
            with torch.no_grad(), _quiet():
                out = estimator.process_one_image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), **kw)
            infer_ms = (time.perf_counter() - t0) * 1e3
            sel = _select_person(out, cen, img_diag) if out else None
            if sel is not None:
                kp, yolo_kp, kp3d, cen = sel
                # gravity-aligned world 3D → IK process (Proc B) + recording
                if M is None and np.isfinite(kp3d[[0, 5, 6, 9, 10, 13, 14]]).all():
                    align_buf.append(kp3d)
                    if len(align_buf) >= args.warmup:
                        M = _canonical_M(align_buf)
                        print(f"  gravity alignment locked after {len(align_buf)} frames",
                              flush=True)
                if M is not None:
                    world = (M @ kp3d.T).T.astype(np.float32)
                    if args.emit_port > 0:
                        _EMIT["buf"] = world.tobytes()
                        _EMIT["n"] = frame_idx    # frame index → synced Rerun timelines

        # ── overlay ──────────────────────────────────────────────────────
        if args.inference_type == "bodyhand":
            if body17 is not None:
                frame = _draw_body(frame, body17)
                if kp_r is not None:
                    frame = _draw_hand(frame, kp_r)
                if kp_l is not None:
                    frame = _draw_hand(frame, kp_l)
            hud = "YOLO body + hand decoder"
        elif kp is not None:
            if yolo_kp is not None:
                frame = draw_option_b(frame, yolo_kp, kp,
                                      frame.shape[1], frame.shape[0], args.hand_scale)
            else:
                valid = ~np.isnan(kp).any(axis=1)
                frame = draw_skeleton(frame, kp, valid, frame.shape[1], frame.shape[0])
            hud = "SAM3D 3D pose"
        else:
            hud = "SAM3D 3D pose"
        e2e_ms = (time.perf_counter() - t0) * 1e3
        ema = 1e3 / e2e_ms if ema is None else 0.85 * ema + 0.15 * (1e3 / e2e_ms)
        cv2.putText(frame, f"{hud}  {ema:4.1f} FPS", (14, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"{hud}  {ema:4.1f} FPS", (14, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # ── log + record ─────────────────────────────────────────────────
        viz.log_frame(frame_idx, t_wall, frame, kp3d, K_np,
                      infer_ms, e2e_ms, ema, jpeg_quality=args.jpeg_quality)
        if rec is not None:
            rec.add(raw, frame, kp, kp3d, world, t_wall)

        if frame_idx % 30 == 0:
            print(f"  frame {frame_idx}  {ema:4.1f} FPS  "
                  f"(infer {infer_ms:.0f} ms, e2e {e2e_ms:.0f} ms)", flush=True)

    if cap is not None:
        cap.release()


def main():
    p = argparse.ArgumentParser(
        description="Live SAM3D → Rerun UI + recording + ACADOS IK feed")
    p.add_argument("--source", default="0", help="webcam index or video path")
    p.add_argument("--loop", action="store_true", help="loop a video source forever")
    p.add_argument("--cap-width", type=int, default=1280,
                   help="requested webcam capture width (default 1280)")
    p.add_argument("--cap-height", type=int, default=720,
                   help="requested webcam capture height (default 720)")
    p.add_argument("--inference-type", choices=["body", "full", "bodyhand"],
                   default="body",
                   help="body: fast, fingers regressed by the body head (coarse); "
                        "full: whole SAM pipeline incl. refinement (slow); "
                        "bodyhand: YOLO body + dedicated SAM hand decoder — faithful "
                        "fingers at ~15fps, but NO 3D body (IK/retarget disabled)")
    p.add_argument("--box-offset", type=float, default=0.35,
                   help="[bodyhand] push hand-box centre along elbow→wrist by this × forearm")
    p.add_argument("--box-size", type=float, default=1.0,
                   help="[bodyhand] hand-box side = this × forearm length")
    p.add_argument("--hand-res", type=int, default=0,
                   help="[bodyhand] backbone input for hand crops (0=model default 512; "
                        "256 needs the 256 engine + TRT_INPUT_SIZE=256)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--checkpoint_dir",
                   default="/home/users/theo/code/checkpoints/sam-3d-body-dinov3")
    p.add_argument("--detector_model", default="./checkpoints/yolo/yolo11m-pose.pt")
    p.add_argument("--intrinsics", default="", help="npz with 'K' — skips MoGe2")
    p.add_argument("--fx", type=float, default=0, help="fixed focal (0 = MoGe2 auto)")
    p.add_argument("--fy", type=float, default=0)
    p.add_argument("--cx", type=float, default=0)
    p.add_argument("--cy", type=float, default=0)
    p.add_argument("--recv-port", type=int, default=0,
                   help="if >0: receive camera frames over TCP (stream_client.py on the Mac)")
    p.add_argument("--hand-scale", type=float, default=1.0)
    # Rerun
    p.add_argument("--rerun-mode", choices=["web", "native", "save"], default="web",
                   help="web: browser viewer over a tunnel; native: local window; "
                        "save: .rrd file only")
    p.add_argument("--rerun-grpc-port", type=int, default=9876)
    p.add_argument("--rerun-web-port", type=int, default=9090)
    p.add_argument("--jpeg-quality", type=int, default=75,
                   help="JPEG quality of frames sent to the Rerun viewer")
    # IK feed
    p.add_argument("--emit-port", type=int, default=8090,
                   help="TCP port streaming gravity-aligned (70,3) 3D to the IK "
                        "process (0 disables)")
    p.add_argument("--warmup", type=int, default=30,
                   help="frames before locking the gravity alignment")
    # Recording
    p.add_argument("--output_dir", default="",
                   help="recording dir (default ./output_rerun_demo/<timestamp>)")
    p.add_argument("--no-record", action="store_true")
    args = p.parse_args()

    if args.inference_type == "bodyhand" and args.emit_port > 0:
        print("  NOTE: bodyhand mode has no 3D body → IK/retarget feed disabled "
              "(use --inference-type body for the ACADOS pipeline)")
        args.emit_port = 0

    out_dir = args.output_dir or os.path.join(
        "output_rerun_demo", time.strftime("%Y%m%d_%H%M%S"))
    if not args.no_record or args.rerun_mode == "save":
        os.makedirs(out_dir, exist_ok=True)

    # ── intrinsics: prefer a known calibration (fast + accurate, no MoGe2) ──
    K = None
    if args.intrinsics:
        K = np.load(args.intrinsics)["K"].astype(np.float32)
    elif args.fx > 0:
        w, h = 1280.0, 720.0
        if args.recv_port == 0:
            probe = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
            w = probe.get(cv2.CAP_PROP_FRAME_WIDTH) or w
            h = probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or h
            probe.release()
        cx = args.cx if args.cx > 0 else w / 2.0
        cy = args.cy if args.cy > 0 else h / 2.0
        K = np.array([[args.fx, 0, cx], [0, args.fy or args.fx, cy], [0, 0, 1]],
                     np.float32)
    cam_int = torch.tensor([K], dtype=torch.float32) if K is not None else None
    if K is not None:
        print(f"  fixed intrinsics: fx={K[0,0]:.0f} fy={K[1,1]:.0f} "
              f"cx={K[0,2]:.0f} cy={K[1,2]:.0f}")

    print("[1/4] Rerun viewer...")
    viz = RerunViz(args.rerun_mode, args.rerun_grpc_port, args.rerun_web_port,
                   out_dir, with_retarget=args.emit_port > 0)

    print("[2/4] Loading SAM-3D-Body...")
    det = args.detector_model
    if det.endswith(".pt") and os.path.exists(det.replace(".pt", ".engine")):
        det = det.replace(".pt", ".engine")
    estimator = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(args.checkpoint_dir, "assets", "mhr_model.pt"),
        detector_name="yolo_pose", detector_model=det,
        fov_name="" if K is not None else "moge2", device="cuda",
    )
    if K is None and not (args.recv_port > 0 or args.source.isdigit()):
        print("      no intrinsics given — estimating once with MoGe2...")
        probe = cv2.VideoCapture(args.source)
        cam_int = _estimate_intrinsics(estimator, probe)
        probe.release()

    print("[3/4] Ports...")
    if args.recv_port > 0:
        threading.Thread(target=_receiver, args=(args.recv_port,), daemon=True).start()
    if args.emit_port > 0:
        threading.Thread(target=_emit_server, args=(args.emit_port,), daemon=True).start()

    rec = None
    if not args.no_record:
        fps = 30.0
        if args.recv_port == 0:
            probe = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
            fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
            probe.release()
        rec = Recorder(out_dir, fps)
        print(f"      recording to {out_dir}/")

    print("[4/4] LIVE — Ctrl+C to stop.")
    try:
        run(args, estimator, cam_int, viz, rec)
    except KeyboardInterrupt:
        pass
    finally:
        _STOP.set()
        if rec is not None:
            rec.close()


if __name__ == "__main__":
    main()
