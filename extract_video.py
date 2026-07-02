#!/usr/bin/env python3
"""Extract Goliath 70 keypoints (2D + 3D) from a video using SAM-3D-Body.

Saves:
  <output_dir>/joints_2d.npy  — shape (T, 70, 2), pixel coordinates
  <output_dir>/joints_3d.npy  — shape (T, 70, 3), body-relative 3D coords

Usage:
  python extract_video.py --video path/to/video.mp4 --output_dir ./output
"""

import argparse
import os
import sys
import time

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

import cv2
import numpy as np
import torch
from notebook.utils import setup_sam_3d_body


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/3] Loading model...")
    estimator = setup_sam_3d_body(
        detector_name=args.detector,
        detector_model=args.detector_model,
        local_checkpoint_path=args.local_checkpoint,
    )

    # Build camera intrinsic matrix if provided
    import torch
    cam_int = None
    if args.fx > 0:
        K = np.array([[args.fx, 0, args.cx],
                      [0, args.fy, args.cy],
                      [0,       0,       1]], dtype=np.float32)
        cam_int = torch.from_numpy(K).unsqueeze(0)
        print(f"  Fixed intrinsics: fx={args.fx} fy={args.fy} cx={args.cx} cy={args.cy}")

    print(f"[2/3] Processing video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {w}x{h} @ {fps:.1f} FPS, {total} frames")

    all_joints_2d = []
    all_joints_3d = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        outputs = estimator.process_one_image(
            frame,
            bbox_thr=args.bbox_thresh,
            cam_int=cam_int,
        )
        dt = time.time() - t0

        if outputs:
            person = outputs[0]
            kp2d = person["pred_keypoints_2d"]  # (70, 2)
            kp3d = person["pred_keypoints_3d"]  # (70, 3)
        else:
            kp2d = np.full((70, 2), np.nan, dtype=np.float32)
            kp3d = np.full((70, 3), np.nan, dtype=np.float32)

        all_joints_2d.append(kp2d)
        all_joints_3d.append(kp3d)

        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}/{total} ({dt:.3f}s)")
        frame_idx += 1

    cap.release()

    joints_2d = np.array(all_joints_2d, dtype=np.float32)
    joints_3d = np.array(all_joints_3d, dtype=np.float32)

    path_2d = os.path.join(args.output_dir, "joints_2d.npy")
    path_3d = os.path.join(args.output_dir, "joints_3d.npy")
    np.save(path_2d, joints_2d)
    np.save(path_3d, joints_3d)

    print(f"\n[3/3] Saved:")
    print(f"  2D keypoints: {path_2d}  shape={joints_2d.shape}")
    print(f"  3D keypoints: {path_3d}  shape={joints_3d.shape}")
    print(f"\nTo visualize:")
    print(f"  python visualize_skeleton_video.py --npy {path_2d} --video {args.video}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract SAM-3D-Body keypoints from video")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--output_dir", default="./output", help="Output directory")
    p.add_argument("--local_checkpoint", default="./checkpoints/sam-3d-body-dinov3",
                   help="Local checkpoint directory (containing model.ckpt)")
    p.add_argument("--detector", default="yolo", help="Detector name")
    p.add_argument("--detector_model", default="./checkpoints/yolo/yolo11n.pt",
                   help="YOLO model path")
    p.add_argument("--bbox_thresh", type=float, default=0.8, help="Detection threshold")
    p.add_argument("--fx", type=float, default=0, help="Focal length x (0 = use MoGe2)")
    p.add_argument("--fy", type=float, default=0, help="Focal length y (0 = use MoGe2)")
    p.add_argument("--cx", type=float, default=0, help="Principal point x")
    p.add_argument("--cy", type=float, default=0, help="Principal point y")
    args = p.parse_args()
    main(args)
