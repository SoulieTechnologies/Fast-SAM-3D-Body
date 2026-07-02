#!/usr/bin/env python3
"""Hand-ONLY extractor — skips the body decoder entirely.

Pipeline per frame:
  1. YOLO11-pose detects the person + wrist keypoints (COCO 9=left, 10=right wrist).
  2. Hand boxes are cropped directly around the wrists (no body decoder needed).
  3. The backbone runs on the 2 hand crops + the HAND DECODER only (forward_step "hand").
  4. The 21 hand keypoints per hand are mapped back to the full image (left hand un-flipped)
     and overlaid.

This isolates the hand decoder so you can judge its raw pose quality and its FPS when it is
the ONLY decoder running. The hand decoder is kept at FULL quality:
  - FP32 decoder (LAYER_DTYPE=fp32)
  - NO body-decoder speed hacks (correctives ON, keypoint-prompt ON, all decoder iterations ON)
  - TRT FP16 backbone (the standard backbone; unchanged)

Usage (GPU 7):
  python hand_only_extractor.py --source /home/users/theo/code/test_input/cam_0_426.mp4 \
      --gpu 7 --start 100 --output output/hands_only_426.mp4 --fx 900
"""
import os
import sys

# --- backbone: TRT FP16 (standard, fast). Decoder: FULL quality, no shortcuts. ---
os.environ.setdefault("USE_TRT_BACKBONE", "1")
os.environ.setdefault("LAYER_DTYPE", "fp32")          # decoder in fp32 (best)
os.environ.setdefault("GPU_HAND_PREP", "1")
os.environ.setdefault("USE_COMPILE", "0")
os.environ.setdefault("COMPILE_MODE", "default")
os.environ.setdefault("MHR_USE_CUDA_GRAPH", "0")
# NOTE: deliberately NOT setting MHR_NO_CORRECTIVES / SKIP_KEYPOINT_PROMPT /
# INTERM_PRED_INTERVAL / KEYPOINT_PROMPT_INTERM_INTERVAL — those degrade the decoder for speed.
os.environ.setdefault(
    "TRT_BACKBONE_PATH",
    "/home/users/theo/code/checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine",
)

import argparse
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--gpu", type=int, default=0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre.parse_known_args()[0].gpu)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contextlib
import io
import time

import cv2
import numpy as np
import torch
from notebook.utils import setup_sam_3d_body
from sam_3d_body.models.meta_arch.sam3d_body import _prepare_hand_batches_gpu
from visualize_skeleton_video import draw_skeleton

# Goliath-70 hand slices: right hand = 21..41 (wrist=41), left hand = 42..62 (wrist=62).
# The hand decoder ALWAYS outputs a right hand (the left crop is mirrored), so the valid
# joints live at indices 21..41 in BOTH outputs; the left hand is un-flipped afterwards.
R_SLICE = slice(21, 42)
L_SLICE = slice(42, 63)
HAND_SRC = slice(21, 42)   # where the decoder puts the (right-hand) joints in its 70-vector


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _largest_person(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="/home/users/theo/code/test_input/cam_0_426.mp4")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default="output/hands_only.mp4")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    p.add_argument("--checkpoint-dir", default="/home/users/theo/code/checkpoints/sam-3d-body-dinov3")
    p.add_argument("--detector-model", default="./checkpoints/yolo/yolo11m-pose.pt")
    p.add_argument("--fx", type=float, default=900.0)
    p.add_argument("--fy", type=float, default=0.0)
    p.add_argument("--cx", type=float, default=0.0)
    p.add_argument("--cy", type=float, default=0.0)
    p.add_argument("--hand-box-scale", type=float, default=3.0, help="wrist box = body_size/scale")
    p.add_argument("--draw-box", action="store_true", help="draw the cropped hand boxes")
    args = p.parse_args()

    det = args.detector_model
    if det.endswith(".pt") and os.path.exists(det.replace(".pt", ".engine")):
        det = det.replace(".pt", ".engine")

    print(f"Loading estimator (GPU {args.gpu}) — hand decoder FP32, full quality...")
    est = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(args.checkpoint_dir, "assets", "mhr_model.pt"),
        detector_name="yolo_pose", detector_model=det,
        fov_name="", device="cuda",
    )
    model = est.model

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    vw, vh = int(cap.get(3)), int(cap.get(4))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    fy = args.fy if args.fy > 0 else args.fx
    cx = args.cx if args.cx > 0 else vw / 2.0
    cy = args.cy if args.cy > 0 else vh / 2.0
    cam_int = torch.tensor([[[args.fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32)
    print(f"Intrinsics fx={args.fx} fy={fy} cx={cx:.0f} cy={cy:.0f}  ({vw}x{vh})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (vw, vh))

    n = 0
    infer_times = []      # hand-decoder inference only (detector + prep excluded)
    full_times = []       # whole per-frame time
    n_hands = 0
    printed_shape = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and n >= args.max_frames:
            break
        n += 1
        t_frame = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. detector → boxes + wrist keypoints (BGR in)
        with torch.no_grad(), _quiet():
            det_res = est.detector.run_human_detection(
                frame, det_cat_id=0, bbox_thr=0.5, nms_thr=0.3, default_to_full_image=False)
        boxes = det_res["boxes"] if isinstance(det_res, dict) else det_res
        kps = det_res.get("keypoints") if isinstance(det_res, dict) else None
        sel = _largest_person(boxes)
        if sel is None or kps is None or len(kps) <= sel:
            writer.write(frame)
            continue

        yolo_kp = kps[sel:sel + 1]         # (1,17,3)
        yolo_box = boxes[sel:sel + 1]      # (1,4)

        # 2. hand boxes from wrists (no body decoder)
        tmp_batch = {}
        with torch.no_grad(), _quiet():
            left_xyxy, right_xyxy = model._get_hand_box_from_yolo_pose(
                yolo_kp, yolo_box, tmp_batch, hand_box_scale=args.hand_box_scale)

            # 3. crop + backbone + HAND DECODER only
            out_hw = (model.cfg.MODEL.IMAGE_SIZE[1], model.cfg.MODEL.IMAGE_SIZE[0])
            batch_lhand, batch_rhand, left_xyxy_flipped = _prepare_hand_batches_gpu(
                rgb, left_xyxy, right_xyxy, cam_int, output_size=out_hw, padding=0.9, device="cuda")
            batch_hands = model._merge_hand_batches(batch_lhand, batch_rhand)
            model._initialize_batch(batch_hands)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            merged = model.forward_step(batch_hands, decoder_type="hand")
            torch.cuda.synchronize()
            infer_times.append(time.perf_counter() - t0)
            lhand, rhand = model._split_hand_outputs(merged, batch_size=1)

        kp_r = rhand["mhr_hand"]["pred_keypoints_2d"][0].detach().cpu().numpy()   # (70,2) full-image
        kp_l = lhand["mhr_hand"]["pred_keypoints_2d"][0].detach().cpu().numpy()   # (70,2) flipped-image
        if not printed_shape:
            print(f"  mhr_hand pred_keypoints_2d shape: {kp_r.shape}", flush=True)
            printed_shape = True

        # 4. assemble into a 70-vector; un-flip the left hand's x
        full = np.full((70, 2), np.nan, dtype=np.float32)
        full[R_SLICE] = kp_r[HAND_SRC]
        l = kp_l[HAND_SRC].copy()
        l[:, 0] = vw - l[:, 0] - 1
        full[L_SLICE] = l
        n_hands += 1

        valid = np.zeros(70, dtype=bool)
        valid[21:63] = np.isfinite(full[21:63]).all(axis=1)
        frame = draw_skeleton(frame, full, valid, vw, vh, hand_thickness=2, hand_radius=4)

        if args.draw_box:
            for b, col in ((right_xyxy[0], (0, 255, 0)), (left_xyxy[0], (255, 128, 0))):
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 1)

        # overlay FPS (rolling)
        if len(infer_times) >= 5:
            dec_ms = 1e3 * np.mean(infer_times[-30:])
            cv2.putText(frame, f"HAND DECODER only  {1000.0/dec_ms:.1f} FPS (decode {dec_ms:.0f}ms)",
                        (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f"HAND DECODER only  {1000.0/dec_ms:.1f} FPS (decode {dec_ms:.0f}ms)",
                        (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        full_times.append(time.perf_counter() - t_frame)
        if n % 30 == 0:
            print(f"  frame {n}: decoder {1e3*np.mean(infer_times[-30:]):.1f}ms  "
                  f"full {1e3*np.mean(full_times[-30:]):.1f}ms", flush=True)

    writer.release()
    cap.release()
    if infer_times:
        warm = min(5, len(infer_times) - 1)
        dec = np.mean(infer_times[warm:])
        fl = np.mean(full_times[warm:]) if len(full_times) > warm else float("nan")
        print("\n" + "=" * 60)
        print(f"HAND-ONLY  ({n_hands} frames with hands / {n} total)")
        print(f"  hand decoder (backbone+decoder on 2 crops): {1e3*dec:.1f} ms  -> {1.0/dec:.1f} FPS")
        print(f"  full frame (detector+crop+decode+draw):     {1e3*fl:.1f} ms  -> {1.0/fl:.1f} FPS")
        print(f"  output: {args.output}")
        print("=" * 60)


if __name__ == "__main__":
    main()
