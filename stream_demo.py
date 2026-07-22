#!/usr/bin/env python3
"""Live SAM3D skeleton demo — run SAM3D in real time on a webcam or a (looping)
video and serve the annotated frames as an MJPEG stream, viewable in a browser
over an SSH tunnel. Ideal when the GPU is on a remote server and the screen is a
laptop reached via SSH (no X11 / XQuartz needed).

Server:
  python stream_demo.py --source /home/users/theo/code/test_input/take_01/cam0.mp4 --gpu 7
  python stream_demo.py --source 0 --gpu 7            # a webcam plugged into the server
Local (laptop):
  ssh -L 8080:localhost:8080 theo@clear-antares.tailb614a0.ts.net
  # then open  http://localhost:8080  in the browser
"""

import os
import sys

# ── TensorRT / speed flags — before importing torch (same as realtime_extractor) ──
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
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
from notebook.utils import setup_sam_3d_body
from utils.visualize_skeleton_video import draw_skeleton, draw_option_b

# Shared latest JPEG frame (written by inference thread, read by HTTP handler).
_LATEST = {"jpg": None}
# Latest frame RECEIVED from a remote camera client (net mode) — drop-old semantics.
# decode = last JPEG-decode cost; idle = last blocking-recv wait (proxy for network gap).
_RECV = {"frame": None, "n": 0, "decode": 0.0, "idle": 0.0}
# Per-frame compute accumulators for the inference thread (reset every 30 frames).
_PROF = {"infer": 0.0, "select+draw": 0.0, "encode": 0.0, "n": 0}
# Latest gravity-aligned (70,3) Goliath 3D as raw float32 bytes, for the IK process.
_EMIT = {"buf": None, "n": 0}
_STOP = threading.Event()


def _emit_server(port):
    """TCP server: stream the latest (70,3) float32 Goliath 3D to the IK process.

    Sends [4-byte big-endian n][70*3*4 bytes float32] each time a new pose is ready;
    the IK client keeps only the latest (drop-old).
    """
    import socket
    import struct

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(
        f"  keypoint emitter listening on tcp/{port} (waiting for IK client)..."
    )
    while not _STOP.is_set():
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"  IK client connected: {addr}")
        last = -1
        try:
            while not _STOP.is_set():
                if _EMIT["buf"] is not None and _EMIT["n"] != last:
                    last = _EMIT["n"]
                    conn.sendall(struct.pack(">I", last) + _EMIT["buf"])
                time.sleep(0.005)
        except Exception:
            print("  IK client disconnected")
            try:
                conn.close()
            except Exception:
                pass


def _canonical_M(buf):
    """Rotation putting the median pose upright (nose→ankle=+Z) & facing +Y (right-handed).

    buf: list of (70,3) mono 3D frames. Returns 3x3 float32 or None if not enough data.
    """
    a = np.stack(
        [f for f in buf if np.isfinite(f[[0, 5, 6, 9, 10, 13, 14]]).all()]
    )
    if len(a) < 5:
        return None
    med = np.median(a, axis=0)
    up = (med[5] + med[6]) / 2 - (med[13] + med[14]) / 2
    if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
        return None
    up /= np.linalg.norm(up)
    left = med[5] - med[6]
    left = left - up * (left @ up)
    left /= np.linalg.norm(left)
    return np.stack([np.cross(left, up), left, up]).astype(np.float32)


def _receiver(port):
    """TCP server: receive length-prefixed JPEG frames from the Mac camera client.

    Always keeps only the LATEST frame (drop-old) so the server never lags behind
    a slow inference — it processes the most recent frame and skips the backlog.
    Wire protocol: [4-byte big-endian length][JPEG bytes] repeated.
    """
    import socket
    import struct

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(
        f"  camera receiver listening on tcp/{port} (waiting for Mac client)..."
    )
    while not _STOP.is_set():
        conn, addr = srv.accept()
        print(f"  camera client connected: {addr}")
        buf = b""
        try:
            while not _STOP.is_set():
                while len(buf) < 4:
                    d = conn.recv(65536)
                    if not d:
                        raise ConnectionError
                    buf += d
                (ln,) = struct.unpack(">I", buf[:4])
                buf = buf[4:]
                _t_idle = time.perf_counter()
                while len(buf) < ln:
                    d = conn.recv(65536)
                    if not d:
                        raise ConnectionError
                    buf += d
                _RECV["idle"] = (
                    time.perf_counter() - _t_idle
                )  # blocking-recv wait ≈ network gap
                _t_dec = time.perf_counter()
                img = cv2.imdecode(
                    np.frombuffer(buf[:ln], np.uint8), cv2.IMREAD_COLOR
                )
                _RECV["decode"] = time.perf_counter() - _t_dec
                buf = buf[ln:]
                if img is not None:
                    _RECV["frame"] = img
                    _RECV["n"] += 1
        except Exception:
            print("  camera client disconnected")
            try:
                conn.close()
            except Exception:
                pass


def _quiet():
    if os.environ.get("SAM3D_PROFILE", "0") == "1":
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def _select_person(outputs, prev_centroid, img_diag, w_pen=5.0):
    """Pick the MAIN subject (largest body bbox + temporal continuity), not a passer-by.

    Returns (sam_kp2d (70,2), yolo_kp2d (17,2) or None, sam_kp3d (70,3), centroid (2,)) or None.
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
        yk = p.get("yolo_keypoints", None)
        if yk is not None:
            yk = np.asarray(yk, dtype=np.float32)  # (17,3) x,y,conf
            yxy = yk[:, :2].copy()
            yxy[yk[:, 2] < 0.3] = np.nan
        else:
            yxy = None
        cands.append((size, bb.mean(0), k2, yxy, k3))
    if not cands:
        return None
    key = (
        (lambda x: x[0])
        if prev_centroid is None
        else (
            lambda x: x[0]
            / (1.0 + w_pen * np.linalg.norm(x[1] - prev_centroid) / img_diag)
        )
    )
    _, c, k2, yxy, k3 = max(cands, key=key)
    return k2, yxy, k3, c


def _torso_ratio(yolo_kp, sam2d):
    """YOLO torso length / SAM3D torso length (to scale SAM hands to the YOLO body)."""
    y_ms = (yolo_kp[5] + yolo_kp[6]) / 2  # COCO shoulders
    y_mh = (yolo_kp[11] + yolo_kp[12]) / 2  # COCO hips
    s_ms = (sam2d[5] + sam2d[6]) / 2  # Goliath shoulders
    s_mh = (sam2d[9] + sam2d[10]) / 2  # Goliath hips
    if np.isnan([y_ms, y_mh, s_ms, s_mh]).any():
        return None
    yl = np.linalg.norm(y_ms - y_mh)
    sl = np.linalg.norm(s_ms - s_mh)
    return yl / sl if sl > 1e-3 else None


def _estimate_intrinsics(estimator, cap, n=10):
    """One-time MoGe2 intrinsics estimate over a few frames (like realtime_extractor)."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = (
        np.linspace(0, max(total - 1, 1), n, dtype=int) if total > 1 else [0]
    )
    Ks = []
    for i in idxs:
        if total > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        with torch.no_grad():
            K = (
                estimator.fov_estimator.get_cam_intrinsics(
                    cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        Ks.append(K)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not Ks:
        return None
    K = np.mean(Ks, axis=0)
    print(
        f"  intrinsics fx={K[0,0]:.0f} fy={K[1,1]:.0f} cx={K[0,2]:.0f} cy={K[1,2]:.0f}"
    )
    return torch.tensor([K], dtype=torch.float32)


def inference_loop(args, estimator, cam_int):
    net = args.recv_port > 0
    cap = None
    if not net:
        is_cam = args.source.isdigit()
        cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open source: {args.source}")
    kw = {"inference_type": "body"}
    if cam_int is not None:
        kw["cam_int"] = cam_int
    # No fixed intrinsics? Auto-estimate ONCE with MoGe2 from the first real frame
    # (e.g. an uncalibrated Mac webcam), then freeze → full FPS, no per-frame MoGe2.
    auto_est = (
        cam_int is None
        and getattr(estimator, "fov_estimator", None) is not None
    )
    print(
        f"  inference: cam_int={'FIXED' if cam_int is not None else ('AUTO (MoGe2 once on 1st frame)' if auto_est else 'NONE -> per-frame MoGe2 SLOW')}"
    )
    img_diag = None
    cen = None
    ema = None
    M = None  # gravity+facing alignment (computed once after warmup)
    align_buf = []
    _n = 0
    last_recv = -1
    while not _STOP.is_set():
        if net:
            if _RECV["n"] == last_recv or _RECV["frame"] is None:
                time.sleep(0.003)  # wait for a fresh received frame
                continue
            last_recv = _RECV["n"]
            frame = _RECV["frame"]
        else:
            ok, frame = cap.read()
            if not ok:
                if is_cam:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                continue
        _n += 1
        if img_diag is None:
            img_diag = float(np.hypot(frame.shape[1], frame.shape[0]))
        if auto_est:  # one-time MoGe2 calibration
            with torch.no_grad():
                K = (
                    estimator.fov_estimator.get_cam_intrinsics(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )
            kw["cam_int"] = torch.tensor([K], dtype=torch.float32)
            print(
                f"  AUTO intrinsics: fx={K[0,0]:.0f} fy={K[1,1]:.0f} "
                f"cx={K[0,2]:.0f} cy={K[1,2]:.0f}",
                flush=True,
            )
            auto_est = False
        t0 = time.perf_counter()
        with torch.no_grad(), _quiet():
            out = estimator.process_one_image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), **kw
            )
        t1 = time.perf_counter()
        dt = t1 - t0
        ema = 1.0 / dt if ema is None else 0.8 * ema + 0.2 * (1.0 / dt)
        sel = _select_person(out, cen, img_diag) if out else None
        _PROF["infer"] += dt
        if sel is not None:
            kp, yolo_kp, kp3d, cen = (
                sel  # main subject only (ignores passers-by)
            )
            # emit gravity-aligned 3D to the IK process (Proc B)
            if args.emit_port > 0:
                if (
                    M is None
                    and np.isfinite(kp3d[[0, 5, 6, 9, 10, 13, 14]]).all()
                ):
                    align_buf.append(kp3d)
                    if len(align_buf) >= args.warmup:
                        M = _canonical_M(align_buf)
                        print(
                            f"  gravity alignment locked after {len(align_buf)} frames",
                            flush=True,
                        )
                if M is not None:
                    world = (M @ kp3d.T).T.astype(np.float32)
                    _EMIT["buf"] = world.tobytes()
                    _EMIT["n"] += 1
            if yolo_kp is not None:
                # pixel-accurate YOLO body + SAM3D hands anchored at the YOLO wrists.
                # hand_scale=1.0 keeps the NATIVE SAM hand size (scaling to the ~15%-larger
                # YOLO body would inflate/distort the fingers).
                frame = draw_option_b(
                    frame,
                    yolo_kp,
                    kp,
                    frame.shape[1],
                    frame.shape[0],
                    args.hand_scale,
                )
            else:
                valid = ~np.isnan(kp).any(axis=1)
                frame = draw_skeleton(
                    frame, kp, valid, frame.shape[1], frame.shape[0]
                )
        t2 = time.perf_counter()
        _PROF["select+draw"] += t2 - t1
        cv2.putText(
            frame,
            f"SAM3D 3D pose  {ema:4.1f} FPS",
            (14, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"SAM3D 3D pose  {ema:4.1f} FPS",
            (14, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        ok2, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.mjpeg_quality]
        )
        if ok2:
            _LATEST["jpg"] = buf.tobytes()
        _PROF["encode"] += time.perf_counter() - t2
        _PROF["n"] += 1
        if _n % 30 == 0:
            k = max(_PROF["n"], 1)
            print(
                f"  inference {ema:4.1f} FPS | per-frame: "
                f"infer {1e3*_PROF['infer']/k:5.1f}ms  "
                f"select+draw {1e3*_PROF['select+draw']/k:4.1f}ms  "
                f"encode {1e3*_PROF['encode']/k:4.1f}ms  "
                f"| decode(rx-thread) {1e3*_RECV['decode']:.1f}ms  "
                f"recv-idle {1e3*_RECV['idle']:.1f}ms",
                flush=True,
            )
            for key in ("infer", "select+draw", "encode", "n"):
                _PROF[key] = 0
    if cap is not None:
        cap.release()


_PAGE = (
    b"<html><head><title>SAM3D live</title></head>"
    b"<body style='margin:0;background:#101014;text-align:center'>"
    b"<img src='/stream' style='max-width:100vw;max-height:100vh'></body></html>"
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_PAGE)
            return
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        try:
            while not _STOP.is_set():
                jpg = _LATEST["jpg"]
                if jpg is not None:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: "
                        + str(len(jpg)).encode()
                        + b"\r\n\r\n"
                        + jpg
                        + b"\r\n"
                    )
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    p = argparse.ArgumentParser(description="Live SAM3D skeleton MJPEG demo")
    p.add_argument(
        "--source", default="0", help="webcam index or video path (looped)"
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--checkpoint_dir",
        default="/home/users/theo/code/checkpoints/sam-3d-body-dinov3",
    )
    p.add_argument(
        "--detector_model", default="./checkpoints/yolo/yolo11m-pose.pt"
    )
    p.add_argument(
        "--intrinsics",
        default="",
        help="npz with 'K' (e.g. cam_params/cam0_intrinsics.npz) — no MoGe2",
    )
    p.add_argument(
        "--fx",
        type=float,
        default=0,
        help="fixed focal (0 = estimate with MoGe2)",
    )
    p.add_argument("--fy", type=float, default=0)
    p.add_argument(
        "--cx",
        type=float,
        default=0,
        help="principal point (0 = image centre)",
    )
    p.add_argument("--cy", type=float, default=0)
    p.add_argument(
        "--recv-port",
        type=int,
        default=0,
        help="if >0: receive camera frames over TCP from a remote Mac client "
        "(stream_client.py) instead of reading --source",
    )
    p.add_argument(
        "--hand-scale",
        type=float,
        default=1.0,
        help="scale of SAM3D hands about the YOLO wrist (1.0 = native size; "
        ">1 enlarges to match the bigger YOLO body but can distort fingers)",
    )
    p.add_argument(
        "--emit-port",
        type=int,
        default=0,
        help="if >0: stream gravity-aligned (70,3) Goliath 3D over TCP to the IK process",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="frames to accumulate before locking the gravity alignment (--emit-port)",
    )
    p.add_argument(
        "--mjpeg-quality",
        type=int,
        default=70,
        help="JPEG quality of the browser stream (lower = less bandwidth over a relay)",
    )
    args = p.parse_args()

    # ── Intrinsics — prefer KNOWN calibration (no MoGe2 = fast + accurate) ──────
    K = None
    if args.intrinsics:
        K = np.load(args.intrinsics)["K"].astype(np.float32)  # npz with "K"
    elif args.fx > 0:
        probe = cv2.VideoCapture(
            int(args.source) if args.source.isdigit() else args.source
        )
        w = probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
        h = probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
        probe.release()
        cx = args.cx if args.cx > 0 else w / 2.0
        cy = args.cy if args.cy > 0 else h / 2.0
        K = np.array(
            [[args.fx, 0, cx], [0, args.fy or args.fx, cy], [0, 0, 1]],
            np.float32,
        )
    have_fixed = K is not None
    cam_int = torch.tensor([K], dtype=torch.float32) if have_fixed else None
    if have_fixed:
        print(
            f"  fixed intrinsics: fx={K[0,0]:.0f} fy={K[1,1]:.0f} cx={K[0,2]:.0f} cy={K[1,2]:.0f}"
        )

    print("[1/3] Loading SAM-3D-Body...")
    det = args.detector_model
    if det.endswith(".pt") and os.path.exists(det.replace(".pt", ".engine")):
        det = det.replace(".pt", ".engine")
    estimator = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(
            args.checkpoint_dir, "assets", "mhr_model.pt"
        ),
        detector_name="yolo_pose",
        detector_model=det,  # exposes COCO-17 body keypoints (free)
        fov_name="" if have_fixed else "moge2",
        device="cuda",  # no MoGe2 when fixed
    )
    if not have_fixed:
        if args.recv_port > 0 or args.source.isdigit():
            print(
                "  No fixed intrinsics → will AUTO-estimate once with MoGe2 on the "
                "first frame (webcam/net), then run at full FPS."
            )
        elif not args.source.isdigit():
            print("[2/3] No intrinsics given — estimating once with MoGe2...")
            probe = cv2.VideoCapture(args.source)
            cam_int = _estimate_intrinsics(estimator, probe)
            probe.release()

    if args.recv_port > 0:
        threading.Thread(
            target=_receiver, args=(args.recv_port,), daemon=True
        ).start()
    if args.emit_port > 0:
        threading.Thread(
            target=_emit_server, args=(args.emit_port,), daemon=True
        ).start()

    th = threading.Thread(
        target=inference_loop, args=(args, estimator, cam_int), daemon=True
    )
    th.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[3/3] Streaming on port {args.port}.")
    print(
        f"      Local:  ssh -L {args.port}:localhost:{args.port} theo@clear-antares.tailb614a0.ts.net"
    )
    print(f"      Then open:  http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _STOP.set()


if __name__ == "__main__":
    main()
