#!/usr/bin/env python3
"""Real-time 3D body keypoint extraction with TensorRT-accelerated backbone.

Active optimizations:
  - TensorRT FP16 backbone (~30ms vs ~350ms PyTorch)
  - Body-only inference — hand decoder skipped
  - No MHR correctives, no intermediate-layer predictions
  - Fixed camera intrinsics — no per-frame MoGe2
  - Async frame prefetch thread

Target: ~15 FPS @ 720p on RTX A5000 (single person, body-only)

Setup (once per machine):
  python convert_backbone_tensorrt.py --all
  python convert_yolo_pose_trt.py --model yolo11m-pose.pt --imgsz 640 --half
"""

import os
import sys
import time

# Optimization flags — must be set before importing sam_3d_body.
# All use setdefault so they can be overridden from the shell.
os.environ.setdefault("USE_TRT_BACKBONE", "1")
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")
os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")
# torch.compile adds ~1-2 min startup with no steady-state gain in "default" mode.
os.environ.setdefault("USE_COMPILE", "0")
# The decoder/MHR head mixes fp16/fp32 submodules; fp16 causes dtype-mismatch errors.
# The TRT backbone still runs fp16 internally.
os.environ.setdefault("LAYER_DTYPE", "fp32")
os.environ.setdefault("GPU_HAND_PREP", "1")
# 999 = produce a prediction only at the final decoder layer, skipping intermediate ones.
os.environ.setdefault("INTERM_PRED_INTERVAL", "999")
os.environ.setdefault("KEYPOINT_PROMPT_INTERM_INTERVAL", "999")
# "reduce-overhead" compile mode uses CUDA graphs, which crash on TRT-allocated tensors.
os.environ.setdefault("COMPILE_MODE", "default")
os.environ.setdefault("MHR_USE_CUDA_GRAPH", "0")
os.environ.setdefault(
    "TRT_BACKBONE_PATH",
    "/home/users/theo/code/checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine",
)

# Pin GPU before importing torch — CUDA_VISIBLE_DEVICES is silently ignored
# after torch initialises CUDA, so --gpu would have no effect if set later.
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
import queue
import threading

import cv2
import numpy as np
import torch
from notebook.utils import setup_sam_3d_body

_PROFILE = os.environ.get("SAM3D_PROFILE", "0") == "1"


def _quiet_inference():
    """Suppress per-frame model stdout unless SAM3D_PROFILE=1."""
    return contextlib.nullcontext() if _PROFILE else contextlib.redirect_stdout(io.StringIO())


def main(args):
    """Run real-time body keypoint extraction and save joints_2d/3d.npy."""
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Real-time SAM-3D-Body Extractor (TensorRT)")
    print("=" * 60)

    use_fixed_intrinsics = args.fx > 0

    cam_int = None
    if use_fixed_intrinsics:
        cam_int = torch.tensor([[
            [args.fx,   0.0, args.cx],
            [  0.0, args.fy, args.cy],
            [  0.0,   0.0,   1.0],
        ]], dtype=torch.float32)
        print(f"Fixed intrinsics (provided): fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")
    else:
        print("No intrinsics provided — will estimate with MoGe2 on a few frames at startup")

    print("\n[1/4] Loading model...")
    t_load = time.time()

    detector_model = args.detector_model
    if detector_model.endswith(".pt") and os.path.exists(detector_model.replace(".pt", ".engine")):
        detector_model = detector_model.replace(".pt", ".engine")
        print(f"  TRT engine: {detector_model}")

    estimator = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(args.checkpoint_dir, "assets", "mhr_model.pt"),
        # yolo_pose keeps the COCO-17 keypoints the pose model already computes
        # (same forward pass, no extra cost) — used to anchor skeleton scale.
        detector_name="yolo_pose",
        detector_model=detector_model,
        fov_name="" if use_fixed_intrinsics else "moge2",
        device="cuda",
    )
    print(f"  Loaded in {time.time() - t_load:.1f}s")

    print(f"\n[2/4] Opening source: {args.source}")
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        if args.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        total_frames = -1
    else:
        cap = cv2.VideoCapture(args.source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        raise ValueError(f"Cannot open: {args.source}")

    vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"  {vid_w}x{vid_h} @ {fps_in:.1f} FPS" +
          (f", {total_frames} frames" if total_frames > 0 else ""))

    if use_fixed_intrinsics and args.cx == 0 and args.cy == 0:
        cam_int[0, 0, 2] = vid_w / 2.0
        cam_int[0, 1, 2] = vid_h / 2.0
        print(f"  Auto principal point: cx={vid_w/2:.0f}, cy={vid_h/2:.0f}")

    # Estimate intrinsics once from a sample of frames using MoGe2.
    # Much faster than per-frame estimation (one-time ~5s cost vs 80ms/frame).
    if not use_fixed_intrinsics:
        print("\n  Estimating camera intrinsics from video sample (MoGe2)...")
        n_samples = 10
        n_avail   = max(total_frames, 1) if total_frames > 0 else 1
        indices   = np.linspace(0, n_avail - 1, min(n_samples, n_avail), dtype=int)
        K_list    = []
        for idx in indices:
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                K = estimator.fov_estimator.get_cam_intrinsics(rgb).squeeze().cpu().numpy()
            K_list.append(K)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to beginning

        if K_list:
            K_mean = np.mean(K_list, axis=0)
            cam_int = torch.tensor([K_mean], dtype=torch.float32)
            print(f"  Estimated intrinsics (mean of {len(K_list)} frames):")
            print(f"    fx={K_mean[0,0]:.1f}  fy={K_mean[1,1]:.1f}"
                  f"  cx={K_mean[0,2]:.1f}  cy={K_mean[1,2]:.1f}")
        else:
            print("  WARNING: could not read frames for intrinsic estimation — falling back to per-frame MoGe2")

    writer = None
    if args.output_video:
        writer = cv2.VideoWriter(
            args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (vid_w, vid_h)
        )

    print("\n[3/4] Warmup (3 frames)...")
    inference_kwargs = {
        "inference_type": "body" if args.body_only else "full",
        **({"cam_int": cam_int} if cam_int is not None else {}),
    }
    if cam_int is not None:
        print(f"  Running with fixed intrinsics for all frames.")
    for _ in range(3):
        ret, frame = cap.read()
        if not ret:
            break
        with torch.no_grad(), _quiet_inference():
            estimator.process_one_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), **inference_kwargs)
    if not args.source.isdigit():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"\n[4/4] Running {'body-only' if args.body_only else 'full'} inference...")
    all_joints_2d = []
    all_joints_3d = []
    all_yolo_2d   = []   # (T, 17, 2) pixel-accurate COCO keypoints for scale anchoring
    frame_times   = []
    frame_idx     = 0

    prefetch_q  = queue.Queue(maxsize=2)
    stop_event  = threading.Event()

    def _reader():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                prefetch_q.put((None, None))
                break
            prefetch_q.put((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame))

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
        while True:
            rgb, frame = prefetch_q.get()
            if rgb is None:
                break

            t0 = time.perf_counter()
            with torch.no_grad(), _quiet_inference():
                outputs = estimator.process_one_image(rgb, **inference_kwargs)
            dt = time.perf_counter() - t0

            frame_times.append(dt)

            if outputs:
                all_joints_2d.append(outputs[0]["pred_keypoints_2d"])
                all_joints_3d.append(outputs[0]["pred_keypoints_3d"])
                yk = outputs[0].get("yolo_keypoints", None)
                if yk is not None:
                    all_yolo_2d.append(np.asarray(yk, dtype=np.float32)[:, :2])  # drop conf
                else:
                    all_yolo_2d.append(np.full((17, 2), np.nan, dtype=np.float32))
            else:
                all_joints_2d.append(np.full((70, 2), np.nan, dtype=np.float32))
                all_joints_3d.append(np.full((70, 3), np.nan, dtype=np.float32))
                all_yolo_2d.append(np.full((17, 2), np.nan, dtype=np.float32))

            if not args.headless:
                cv2.putText(frame, f"FPS: {1.0/dt:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                disp = cv2.resize(frame, (vid_w // 2, vid_h // 2)) if vid_w > 1920 else frame
                cv2.imshow("Real-time SAM3D", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer:
                writer.write(frame)

            if frame_idx % 20 == 0:
                avg = 1.0 / np.mean(frame_times[-20:]) if frame_times else 0
                suffix = f"/{total_frames}" if total_frames > 0 else ""
                print(f"  frame {frame_idx}{suffix}  {1.0/dt:.1f} FPS (avg {avg:.1f})")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        stop_event.set()
        reader_thread.join(timeout=2)
        cap.release()

    if writer:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    if all_joints_2d:
        joints_2d = np.array(all_joints_2d, dtype=np.float32)
        joints_3d = np.array(all_joints_3d, dtype=np.float32)
        path_2d   = os.path.join(args.output_dir, "joints_2d.npy")
        path_3d   = os.path.join(args.output_dir, "joints_3d.npy")
        np.save(path_2d, joints_2d)
        np.save(path_3d, joints_3d)

        path_yolo = os.path.join(args.output_dir, "joints_yolo_2d.npy")
        if all_yolo_2d:
            np.save(path_yolo, np.array(all_yolo_2d, dtype=np.float32))

        times = np.array(frame_times)
        print(f"\n{'=' * 60}")
        print(f"Results:")
        print(f"  Frames processed : {frame_idx}")
        print(f"  Average FPS      : {1.0 / times.mean():.1f}")
        print(f"  Median latency   : {np.median(times)*1000:.1f} ms")
        print(f"  P95 latency      : {np.percentile(times, 95)*1000:.1f} ms")
        print(f"  Min/Max FPS      : {1.0/times.max():.1f} / {1.0/times.min():.1f}")
        print(f"  Saved: {path_2d} {joints_2d.shape}")
        print(f"         {path_3d} {joints_3d.shape}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Real-time SAM-3D-Body extraction with TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python realtime_extractor.py --source video.mp4 --fx 760.7 --fy 759.2 --cx 648 --cy 351
  python realtime_extractor.py --source 0                          # webcam
  python realtime_extractor.py --source video.mp4 --no-body-only  # with hands
  python realtime_extractor.py --source video.mp4 --headless      # batch mode
        """,
    )
    p.add_argument("--source",         default="0",       help="Video path or webcam index")
    p.add_argument("--video",          dest="source",     help="Alias for --source")
    p.add_argument("--output_dir",     default="./output")
    p.add_argument("--output_video",   default="",        help="Save output video with FPS overlay")
    p.add_argument("--checkpoint_dir", default="/home/users/theo/code/checkpoints/sam-3d-body-dinov3")
    p.add_argument("--detector_model", default="./checkpoints/yolo/yolo11m-pose.pt")
    p.add_argument("--gpu",            type=int, default=0, help="GPU index")
    p.add_argument("--body-only",      action="store_true", default=True,
                   help="Skip hand decoder (default: on)")
    p.add_argument("--no-body-only",   action="store_false", dest="body_only",
                   help="Full inference including hands")
    p.add_argument("--headless",       action="store_true", help="No display window")
    p.add_argument("--fx",  type=float, default=0, help="Focal length X  (0 = use MoGe2)")
    p.add_argument("--fy",  type=float, default=0, help="Focal length Y")
    p.add_argument("--cx",  type=float, default=0, help="Principal point X  (0 = image centre)")
    p.add_argument("--cy",  type=float, default=0, help="Principal point Y")
    p.add_argument("--width",  type=int, default=0, help="Webcam capture width")
    p.add_argument("--height", type=int, default=0, help="Webcam capture height")
    main(p.parse_args())
