#!/usr/bin/env python3
"""Pipeline-parallel SAM-3D-Body extraction across two GPUs.

Architecture:
  Reader thread   — decode + cvtColor frames on CPU        → rgb_queue
  Backbone thread — YOLO + backbone on GPU_A               → embeddings_queue
  Decoder thread  — body decoder on GPU_B                  → results_queue
  Main thread     — collect results in order, save .npy

Throughput target : max(backbone, decoder) ≈ 27 ms → ~37 FPS
Latency target    : backbone + transfer + decoder  ≈ 47 ms

Note: work in progress — multi-GPU device mapping needs further fixes.

Setup (once per machine):
  python convert_backbone_tensorrt.py --all
  python convert_yolo_pose_trt.py --model yolo11m-pose.pt --imgsz 640 --half

Usage:
  python realtime_extractor_parallel.py \\
    --source video.mp4 --fx 760.72 --fy 759.24 --cx 648 --cy 351 \\
    --gpu-backbone 7 --gpu-decoder 1
"""

import os
import sys
import contextlib
import io
import queue
import threading
import time

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

if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import cv2
import numpy as np
import torch
from notebook.utils import setup_sam_3d_body
from sam_3d_body.utils import recursive_to

_PROFILE = os.environ.get("SAM3D_PROFILE", "0") == "1"


def _quiet():
    """Suppress per-frame model stdout unless SAM3D_PROFILE=1."""
    return contextlib.nullcontext() if _PROFILE else contextlib.redirect_stdout(io.StringIO())


_DECODER_BATCH_KEYS = [
    "bbox_center", "bbox_scale", "ori_img_size",
    "cam_int", "affine_trans", "img_size", "ray_cond",
    "img", "person_valid",
]


def _transfer_batch(batch: dict, device: str) -> dict:
    """Copy decoder-relevant batch tensors to device."""
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
        if k in _DECODER_BATCH_KEYS
    }


def _move_nonpersistent_buffers(module: torch.nn.Module, device: str):
    """Move persistent=False buffers that .to(device) skips."""
    for mod in module.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.device != torch.device(device):
                mod._buffers[name] = buf.to(device)


def main(args):
    """Run pipeline-parallel body keypoint extraction across two GPUs."""
    os.makedirs(args.output_dir, exist_ok=True)

    dev_a = f"cuda:{args.gpu_backbone}"
    dev_b = f"cuda:{args.gpu_decoder}"
    print(f"Backbone GPU : {dev_a}")
    print(f"Decoder  GPU : {dev_b}")

    cam_int = None
    if args.fx > 0:
        cam_int = torch.tensor([[
            [args.fx,   0.0, args.cx],
            [  0.0, args.fy, args.cy],
            [  0.0,   0.0,   1.0],
        ]], dtype=torch.float32)

    print("\n[1/4] Loading backbone model …")
    detector_model = args.detector_model
    if detector_model.endswith(".pt") and os.path.exists(detector_model.replace(".pt", ".engine")):
        detector_model = detector_model.replace(".pt", ".engine")

    estimator_a = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(args.checkpoint_dir, "assets", "mhr_model.pt"),
        detector_name="yolo",
        detector_model=detector_model,
        fov_name="" if cam_int is not None else "moge2",
        device=dev_a,
    )
    model_a = estimator_a.model
    _move_nonpersistent_buffers(model_a, dev_a)

    print("\n[2/4] Loading decoder model …")
    estimator_b = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(args.checkpoint_dir, "assets", "mhr_model.pt"),
        detector_name="",
        fov_name="",
        device=dev_b,
    )
    model_b = estimator_b.model
    _move_nonpersistent_buffers(model_b, dev_b)

    print(f"\n[3/4] Opening source: {args.source}")
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {args.source}")
    vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {vid_w}x{vid_h} @ {fps_in:.1f} FPS, {total} frames")

    if cam_int is not None and args.cx == 0 and args.cy == 0:
        cam_int[0, 0, 2] = vid_w / 2.0
        cam_int[0, 1, 2] = vid_h / 2.0

    rgb_q        = queue.Queue(maxsize=4)
    embeddings_q = queue.Queue(maxsize=2)
    results_q    = queue.Queue()
    stop         = threading.Event()

    def _reader():
        idx = 0
        while not stop.is_set():
            ret, bgr = cap.read()
            if not ret:
                rgb_q.put((None, None, None))
                return
            rgb_q.put((idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bgr))
            idx += 1

    def _backbone_worker():
        from sam_3d_body.data.utils.prepare_batch import prepare_batch

        while not stop.is_set():
            frame_idx, rgb, bgr = rgb_q.get()
            if rgb is None:
                embeddings_q.put((None,) * 6)
                return

            with torch.no_grad(), _quiet():
                det   = estimator_a.detector.run_human_detection(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    bbox_thr=0.5, nms_thr=0.3, default_to_full_image=False,
                )
                boxes = det["boxes"] if isinstance(det, dict) else det
                if len(boxes) == 0:
                    embeddings_q.put((frame_idx, None, None, None, None, bgr))
                    continue

                batch = prepare_batch(rgb, estimator_a.transform, boxes, None, None)
                batch = recursive_to(batch, dev_a)
                if cam_int is not None:
                    batch["cam_int"] = cam_int.to(dev_a).clone()

                model_a._initialize_batch(batch)
                model_a.hand_batch_idx = []
                model_a.body_batch_idx = list(range(batch["img"].shape[0] * batch["img"].shape[1]))

                emb, ci, kp = model_a.forward_backbone_only(batch)

            embeddings_q.put((
                frame_idx,
                emb.cpu(), ci.cpu(), kp.cpu(),
                _transfer_batch(batch, "cpu"),
                bgr,
            ))

    def _decoder_worker():
        while not stop.is_set():
            item      = embeddings_q.get()
            frame_idx = item[0]
            if frame_idx is None:
                results_q.put((None, None, None))
                return

            _, emb_cpu, ci_cpu, kp_cpu, batch_cpu, _ = item

            if emb_cpu is None:
                results_q.put((frame_idx,
                               np.full((70, 2), np.nan, np.float32),
                               np.full((70, 3), np.nan, np.float32)))
                continue

            with torch.no_grad(), _quiet():
                batch = _transfer_batch(batch_cpu, dev_b)
                model_b._initialize_batch(batch)
                model_b.hand_batch_idx = []
                model_b.body_batch_idx = list(range(batch["img"].shape[0] * batch["img"].shape[1]))

                output = model_b.forward_decoder_body(
                    emb_cpu.to(dev_b), ci_cpu.to(dev_b), kp_cpu.to(dev_b), batch
                )

                out  = recursive_to(recursive_to(output["mhr"], "cpu"), "numpy")
                kp2d = out["pred_keypoints_2d"][0]
                kp3d = out["pred_keypoints_3d"][0]

            results_q.put((frame_idx, kp2d, kp3d))

    print(f"\n[4/4] Running pipeline inference …")
    for t in [
        threading.Thread(target=_reader,          daemon=True),
        threading.Thread(target=_backbone_worker, daemon=True),
        threading.Thread(target=_decoder_worker,  daemon=True),
    ]:
        t.start()

    all_joints_2d = {}
    all_joints_3d = {}
    t_start       = time.perf_counter()

    try:
        while True:
            frame_idx, kp2d, kp3d = results_q.get()
            if frame_idx is None:
                break
            all_joints_2d[frame_idx] = kp2d
            all_joints_3d[frame_idx] = kp3d
            if frame_idx % 20 == 0:
                n      = len(all_joints_2d)
                elapsed = time.perf_counter() - t_start
                print(f"  frame {frame_idx}/{total}  avg {n/elapsed:.1f} FPS")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        stop.set()
        cap.release()

    n         = max(all_joints_2d.keys()) + 1 if all_joints_2d else 0
    joints_2d = np.stack([all_joints_2d.get(i, np.full((70, 2), np.nan, np.float32)) for i in range(n)])
    joints_3d = np.stack([all_joints_3d.get(i, np.full((70, 3), np.nan, np.float32)) for i in range(n)])
    path_2d   = os.path.join(args.output_dir, "joints_2d.npy")
    path_3d   = os.path.join(args.output_dir, "joints_3d.npy")
    np.save(path_2d, joints_2d)
    np.save(path_3d, joints_3d)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  Frames : {n}  |  Avg FPS : {n/elapsed:.1f}")
    print(f"  Saved  : {path_2d} {joints_2d.shape}")
    print(f"           {path_3d} {joints_3d.shape}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pipeline-parallel SAM-3D-Body (2 GPUs)")
    p.add_argument("--source",          default="0")
    p.add_argument("--video",           dest="source")
    p.add_argument("--output_dir",      default="./output")
    p.add_argument("--checkpoint_dir",  default="/home/users/theo/code/checkpoints/sam-3d-body-dinov3")
    p.add_argument("--detector_model",  default="./checkpoints/yolo/yolo11m-pose.pt")
    p.add_argument("--gpu-backbone",    type=int, default=7)
    p.add_argument("--gpu-decoder",     type=int, default=1)
    p.add_argument("--fx",  type=float, default=0)
    p.add_argument("--fy",  type=float, default=0)
    p.add_argument("--cx",  type=float, default=0)
    p.add_argument("--cy",  type=float, default=0)
    main(p.parse_args())
