#!/usr/bin/env python3
"""Per-stage profiler for the SAM-3D-Body model (backbone / body decoder / detector / …).

Runs the SAME optimized realtime path (TRT FP16 backbone, body-only) over N frames
of a video, captures the model's own internal timing prints (which are GPU-accurate
only when SAM3D_PROFILE=1 because _psync() then does torch.cuda.synchronize()), and
prints the AVERAGE milliseconds per stage.

Usage (GPU 7):
  python profile_sam3d_stages.py --source /home/users/theo/code/test_input/cam_0_426.mp4 \
      --gpu 7 --frames 40 --fx 900
"""
import os
import sys
import time

# Same optimization flags as realtime_extractor — set BEFORE importing torch/sam_3d_body.
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
# CRITICAL: enables the model's internal prints AND the per-stage cuda syncs (_psync),
# without which GPU stage timings are meaningless (async kernel launches).
os.environ["SAM3D_PROFILE"] = "1"

import argparse
import contextlib
import io
import re
from collections import defaultdict

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--gpu", type=int, default=0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre.parse_known_args()[0].gpu)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from notebook.utils import setup_sam_3d_body

# Matches lines like "  [forward_pose_branch] backbone: 0.0281s"
_LINE = re.compile(r"\[([\w_]+)\]\s+(.+?):\s+([\d.]+)s\s*$")

# Human-friendly grouping of the raw stage names into the pipeline parts.
_GROUP = {
    "human_detection": "1. detector (YOLO-pose)",
    "fov_estimation": "1b. intrinsics (MoGe2, if any)",
    "load_image": "0. load/convert image",
    "mask_processing": "2. mask (SAM2, off)",
    "prepare_batch": "3. prepare batch (crop/norm)",
    "initialize_batch": "3b. initialize batch",
    "data_preprocess": "4. data_preprocess",
    "ray_condition": "5. ray conditioning",
    "backbone": "6. BACKBONE (ViT-H, TRT)",
    "mask_condition": "7. mask conditioning",
    "decoder_condition": "8. decoder conditioning",
    "forward_decoder_body": "9. BODY DECODER",
    "forward_decoder_hand": "9b. HAND DECODER",
    "forward_decoders_combined (body+hand)": "9. body+hand decoders",
    "postprocess_output": "10. postprocess (cpu/numpy)",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="/home/users/theo/code/test_input/cam_0_426.mp4")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--start", type=int, default=0, help="seek to this frame first (skip empty intro)")
    p.add_argument("--checkpoint-dir", default="/home/users/theo/code/checkpoints/sam-3d-body-dinov3")
    p.add_argument("--detector-model", default="./checkpoints/yolo/yolo11m-pose.pt")
    p.add_argument("--fx", type=float, default=900.0, help="fixed fx (0 => MoGe2)")
    p.add_argument("--fy", type=float, default=0.0)
    p.add_argument("--cx", type=float, default=0.0)
    p.add_argument("--cy", type=float, default=0.0)
    p.add_argument("--full", action="store_true", help="full inference (body+hand) instead of body-only")
    args = p.parse_args()

    det = args.detector_model
    if det.endswith(".pt") and os.path.exists(det.replace(".pt", ".engine")):
        det = det.replace(".pt", ".engine")

    use_fixed = args.fx > 0
    print(f"Loading estimator (GPU {args.gpu}, {'body-only' if not args.full else 'full'})...")
    estimator = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(args.checkpoint_dir, "assets", "mhr_model.pt"),
        detector_name="yolo_pose",
        detector_model=det,
        fov_name="" if use_fixed else "moge2",
        device="cuda",
    )

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    vw, vh = int(cap.get(3)), int(cap.get(4))
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    cam_int = None
    if use_fixed:
        fy = args.fy if args.fy > 0 else args.fx
        cx = args.cx if args.cx > 0 else vw / 2.0
        cy = args.cy if args.cy > 0 else vh / 2.0
        cam_int = torch.tensor([[[args.fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32)
        print(f"Fixed intrinsics: fx={args.fx} fy={fy} cx={cx:.0f} cy={cy:.0f}  ({vw}x{vh})")

    kw = {"inference_type": "body" if not args.full else "full"}
    if cam_int is not None:
        kw["cam_int"] = cam_int

    acc = defaultdict(list)   # stage name -> [seconds per frame]
    wall = []
    n = 0
    processed = 0
    while processed < args.warmup + args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap_out = io.StringIO()
        t0 = time.perf_counter()
        with torch.no_grad(), contextlib.redirect_stdout(cap_out):
            res = estimator.process_one_image(rgb, **kw)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        processed += 1
        if processed <= args.warmup:      # discard warmup (TRT/cudnn autotune, allocs)
            print(f"  [warmup {processed}] persons={len(res)}  {1e3*dt:.1f}ms", file=sys.stderr)
            continue
        if len(res) == 0:
            print(f"  [frame {processed}] NO PERSON DETECTED — skipped", file=sys.stderr)
            continue
        wall.append(dt)
        n += 1
        for line in cap_out.getvalue().splitlines():
            m = _LINE.search(line.strip())
            if m:
                acc[m.group(2)].append(float(m.group(3)))
    cap.release()

    if n == 0:
        raise SystemExit("no frames profiled")

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print(f"SAM-3D-Body per-stage timing — mean over {n} frames "
          f"({'body-only' if not args.full else 'full'})")
    print("=" * 66)

    total_line = None
    rows = []
    for name, vals in acc.items():
        mean_ms = 1e3 * float(np.mean(vals))
        if name == "TOTAL" or "TOTAL" in name:
            total_line = mean_ms
            continue
        rows.append((mean_ms, name, len(vals)))
    rows.sort(reverse=True)

    wall_ms = 1e3 * float(np.mean(wall))
    print(f"{'stage':42s} {'ms':>8s}  {'%wall':>6s}")
    print("-" * 66)
    for mean_ms, name, cnt in rows:
        label = _GROUP.get(name, name)
        note = "" if cnt == n else f"  (only {cnt}/{n} frames)"
        print(f"{label:42s} {mean_ms:8.2f}  {100*mean_ms/wall_ms:5.1f}%{note}")
    print("-" * 66)
    print(f"{'WALL (perf_counter, GPU-synced)':42s} {wall_ms:8.2f}  {100.0:5.1f}%")
    print(f"{'≈ throughput':42s} {1000.0/wall_ms:8.2f}  fps")
    if total_line:
        print(f"{'(model-reported process_one_image TOTAL)':42s} {total_line:8.2f}")


if __name__ == "__main__":
    main()
