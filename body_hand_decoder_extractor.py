#!/usr/bin/env python3
"""Fast body + dedicated-hand-decoder fingers.

Architecture (the ~15 fps target design):
  1. YOLO11-pose  → full body (COCO-17) + wrist locations   [cheap, already running]
  2. crop 512² around each wrist
  3. backbone + HAND DECODER only on the 2 hand crops        [the ONLY ViT-H work]
  4. overlay: YOLO body skeleton + dedicated-hand-decoder fingers (21 kp/hand)

We deliberately SKIP the SAM body decoder (and its backbone pass) — the body comes
from YOLO (optionally COSMIK-augmented later). Only the hands go through the ViT-H
backbone, which is what makes ~15 fps reachable.

v1 = FP32 hand decoder, full quality (validate finger quality + measure FPS).
Next: TRT FP16 on the hand decoder + its backbone pass, and/or 256² hand crops → ~15 fps.

Usage (GPU 7):
  python body_hand_decoder_extractor.py --source .../cam_0_426.mp4 --gpu 7 \
      --start 100 --output output/body_hand_426.mp4 --fx 674.5
"""

import os
import sys

# Backbone TRT FP16; hand decoder FP32 full quality (no speed hacks that hurt fingers).
os.environ.setdefault("USE_TRT_BACKBONE", "1")
os.environ.setdefault("LAYER_DTYPE", "fp32")
os.environ.setdefault("GPU_HAND_PREP", "1")
os.environ.setdefault("USE_COMPILE", "0")
os.environ.setdefault("MHR_USE_CUDA_GRAPH", "0")
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

# ── COCO-17 body skeleton (YOLO-pose order) ──────────────────────────────────
COCO_EDGES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),  # torso
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # legs
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),  # head/neck
]
L_WRIST, R_WRIST = 9, 10  # COCO wrist indices
L_ELBOW, R_ELBOW = 7, 8  # COCO elbow indices
HAND_SRC = slice(
    21, 42
)  # the decoder's (right-hand) 21 joints in its 70-vector


def _hand_box(wrist, elbow, off, sz):
    """Hand box CENTERED ON THE HAND, not the wrist.

    YOLO only gives the wrist, so a wrist-centered box (old approach) is too big and
    off-centre → the hand decoder mis-places/inflates the hand. Instead push the box
    centre along the forearm (elbow→wrist) direction toward where the hand actually is,
    and size it to the forearm length. Verified: decoder wrist lands ~2px from the YOLO
    wrist (vs ~53px before), box ~77px (vs 175px).
    """
    d = wrist - elbow
    L = float(np.linalg.norm(d))
    if L < 1e-3 or not np.isfinite(d).all():
        return None
    c = wrist + (d / L) * (off * L)
    s = sz * L
    return np.array(
        [c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2],
        dtype=np.float32,
    )


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _largest(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(a))


# finger bone connectivity within a 21-vector: wrist(20) → each finger MCP→tip.
# decoder order (Goliath right hand 21..41): [th4,th3,th2,thMCP, ff4,ff3,ff2,ffMCP, mf..., rf..., pf..., wrist]
_FINGERS = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
    (16, 17, 18, 19),
]
_WRIST_L = 20
_FCOL = [
    (0, 128, 255),
    (255, 153, 255),
    (255, 178, 102),
    (51, 51, 255),
    (0, 255, 0),
]


def _draw_hand(frame, pts21):
    for (tip, j3, j2, mcp), col in zip(_FINGERS, _FCOL):
        chain = [_WRIST_L, mcp, j2, j3, tip]
        for a, b in zip(chain[:-1], chain[1:]):
            pa, pb = pts21[a], pts21[b]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                cv2.line(
                    frame,
                    tuple(pa.astype(int)),
                    tuple(pb.astype(int)),
                    col,
                    2,
                    cv2.LINE_AA,
                )
    for p in pts21:
        if np.isfinite(p).all():
            cv2.circle(
                frame,
                tuple(p.astype(int)),
                3,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
    return frame


def _draw_body(frame, kp17):
    for a, b in COCO_EDGES:
        pa, pb = kp17[a], kp17[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(
                frame,
                tuple(pa.astype(int)),
                tuple(pb.astype(int)),
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
    for p in kp17:
        if np.isfinite(p).all():
            cv2.circle(
                frame, tuple(p.astype(int)), 4, (0, 140, 255), -1, cv2.LINE_AA
            )
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", default="/home/users/theo/code/test_input/cam_0_426.mp4"
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default="output/body_hand.mp4")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--checkpoint-dir",
        default="/home/users/theo/code/checkpoints/sam-3d-body-dinov3",
    )
    p.add_argument(
        "--detector-model", default="./checkpoints/yolo/yolo11m-pose.pt"
    )
    p.add_argument("--fx", type=float, default=674.5)
    p.add_argument("--fy", type=float, default=0.0)
    p.add_argument("--cx", type=float, default=0.0)
    p.add_argument("--cy", type=float, default=0.0)
    p.add_argument(
        "--box-offset",
        type=float,
        default=0.35,
        help="push hand-box centre along elbow→wrist by this × forearm length",
    )
    p.add_argument(
        "--box-size",
        type=float,
        default=1.0,
        help="hand-box side = this × forearm length",
    )
    p.add_argument(
        "--hand-res",
        type=int,
        default=0,
        help="backbone input size for hand crops (0=model default 512; 256 needs the 256 TRT engine + TRT_INPUT_SIZE=256)",
    )
    args = p.parse_args()

    det = args.detector_model
    if det.endswith(".pt") and os.path.exists(det.replace(".pt", ".engine")):
        det = det.replace(".pt", ".engine")

    print(
        f"Loading estimator (GPU {args.gpu}) — YOLO body + hand decoder FP32..."
    )
    est = setup_sam_3d_body(
        local_checkpoint_path=args.checkpoint_dir,
        local_mhr_path=os.path.join(
            args.checkpoint_dir, "assets", "mhr_model.pt"
        ),
        detector_name="yolo_pose",
        detector_model=det,
        fov_name="",
        device="cuda",
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
    cam_int = torch.tensor(
        [[[args.fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32
    )
    print(f"Intrinsics fx={args.fx} fy={fy} cx={cx:.0f} cy={cy:.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (vw, vh)
    )
    if args.hand_res > 0:
        out_hw = (args.hand_res, args.hand_res)
    else:
        out_hw = (model.cfg.MODEL.IMAGE_SIZE[1], model.cfg.MODEL.IMAGE_SIZE[0])
    print(f"hand crop backbone input: {out_hw}")

    n = 0
    t_yolo, t_hand, t_full = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        n += 1
        t0 = time.perf_counter()

        # 1. YOLO body + wrists
        with torch.no_grad(), _quiet():
            dr = est.detector.run_human_detection(
                frame,
                det_cat_id=0,
                bbox_thr=0.5,
                nms_thr=0.3,
                default_to_full_image=False,
            )
        torch.cuda.synchronize()
        t_yolo.append(time.perf_counter() - t0)
        boxes = dr["boxes"] if isinstance(dr, dict) else dr
        kps = dr.get("keypoints") if isinstance(dr, dict) else None
        sel = _largest(boxes)
        if sel is None or kps is None or len(kps) <= sel:
            writer.write(frame)
            t_full.append(time.perf_counter() - t0)
            continue
        yolo_kp = kps[sel : sel + 1]  # (1,17,3)

        # 2-3. crop around wrists (hand-centred box) + backbone + HAND DECODER only
        k = yolo_kp[0]  # (17,3)
        rxyxy = _hand_box(
            k[R_WRIST, :2], k[R_ELBOW, :2], args.box_offset, args.box_size
        )
        lxyxy = _hand_box(
            k[L_WRIST, :2], k[L_ELBOW, :2], args.box_offset, args.box_size
        )
        if rxyxy is None or lxyxy is None:
            writer.write(frame)
            t_full.append(time.perf_counter() - t0)
            continue
        rxyxy = rxyxy[None]
        lxyxy = lxyxy[None]
        th = time.perf_counter()
        with torch.no_grad(), _quiet():
            bl, br, _ = _prepare_hand_batches_gpu(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                lxyxy,
                rxyxy,
                cam_int,
                output_size=out_hw,
                padding=0.9,
                device="cuda",
            )  # noqa
            bh = model._merge_hand_batches(bl, br)
            model._initialize_batch(bh)
            torch.cuda.synchronize()
            merged = model.forward_step(bh, decoder_type="hand")
            torch.cuda.synchronize()
            lh, rh = model._split_hand_outputs(merged, batch_size=1)
        t_hand.append(time.perf_counter() - th)

        kp_r = (
            rh["mhr_hand"]["pred_keypoints_2d"][0]
            .detach()
            .cpu()
            .numpy()[HAND_SRC]
        )
        kp_l = (
            lh["mhr_hand"]["pred_keypoints_2d"][0]
            .detach()
            .cpu()
            .numpy()[HAND_SRC]
            .copy()
        )
        kp_l[:, 0] = vw - kp_l[:, 0] - 1  # un-flip left hand

        # 4. overlay: YOLO body + hand-decoder fingers
        body = yolo_kp[0][:, :2].copy()
        body[yolo_kp[0][:, 2] < 0.3] = np.nan
        frame = _draw_body(frame, body)
        frame = _draw_hand(frame, kp_r)
        frame = _draw_hand(frame, kp_l)

        t_full.append(time.perf_counter() - t0)
        fps = 1.0 / np.mean(t_full[-30:])
        cv2.putText(
            frame,
            f"YOLO body + hand-decoder  {fps:.1f} FPS",
            (14, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"YOLO body + hand-decoder  {fps:.1f} FPS",
            (14, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        if n % 30 == 0:
            print(
                f"  frame {n}: yolo {1e3*np.mean(t_yolo[-30:]):.0f}ms  "
                f"hand {1e3*np.mean(t_hand[-30:]):.0f}ms  full {1e3*np.mean(t_full[-30:]):.0f}ms "
                f"({fps:.1f} fps)",
                flush=True,
            )

    writer.release()
    cap.release()
    if t_full:
        w = min(5, len(t_full) - 1)
        print("\n" + "=" * 60)
        print(f"BODY(YOLO) + HAND-DECODER  ({n} frames)")
        print(f"  yolo body : {1e3*np.mean(t_yolo[w:]):.1f} ms")
        print(
            f"  hand path : {1e3*np.mean(t_hand[w:]):.1f} ms  (backbone 2 crops + hand decoder)"
        )
        print(
            f"  full      : {1e3*np.mean(t_full[w:]):.1f} ms  -> {1.0/np.mean(t_full[w:]):.1f} FPS"
        )
        print(f"  output    : {args.output}")
        print("=" * 60)


if __name__ == "__main__":
    main()
