# Real-time Body + Fine Hand Tracking (~14.6 FPS)

A real-time extraction pipeline built on top of [Fast SAM 3D Body](README.md) that
combines a **cheap YOLO body skeleton** with the **dedicated SAM hand decoder** to get
faithful finger tracking at **~14.6 FPS on an RTX A5000** (720p, single person) — the
finger quality of the full pipeline, at nearly the frame-rate of body-only.

<p align="center"><i>Body (YOLO-pose) + dedicated hand decoder cropped around the wrists.</i></p>

## The idea

The full SAM-3D-Body pipeline gives excellent fingers but runs at ~2–6 FPS, because the
ViT-H backbone runs on the body crop **and** both hand crops, plus a heavy refinement pass.
Body-only mode is fast (~16 FPS) but the fingers (regressed by the body head) are coarse.

This pipeline gets the best of both:

1. **YOLO11-pose** → full body (COCO-17) + wrist locations (~6 ms, essentially free).
2. Crop a tight box **around each wrist** (centred on the hand via the elbow→wrist
   direction — a wrist-centred box is too big/off and inflates the hand).
3. Run the backbone + **hand decoder only** on the 2 hand crops (skip the SAM body decoder).
4. Overlay the YOLO body + the 21-keypoint hand-decoder fingers.

## Optimization ladder (verified: each step keeps finger quality identical)

| Config | FPS | Notes |
|---|---:|---|
| 512² backbone, FP32 decoder | 7.7 | baseline (full-quality hand decoder) |
| **256² TRT FP16 backbone** | 11.5 | hand crop carries only ~77 px of real info → 256² loses ~2 px |
| **+ `MHR_NO_CORRECTIVES`** | 12.8 | mesh pose-correctives are free to skip (don't touch keypoints) |
| **+ `torch.compile` decoder** | **14.6** | ~9 ms off the decoder floor |

What did **not** help: FP16 decoder (memory-bound, and the MHR sparse op has no fp16 CUDA
kernel); 128² crops (8×8 tokens too coarse — the decoder can't localize, output degenerates).
The backbone is already TRT-FP16 + batched over the 2 hands; the decoder is the floor.

## Setup

```bash
# 1. Environment (see setup_env.sh for the full SAM-3D-Body install)
bash setup_env.sh

# 2. Download the SAM-3D-Body checkpoint into ./checkpoints/sam-3d-body-dinov3/
#    (model.ckpt + assets/mhr_model.pt) — see the main README / HuggingFace.

# 3. Build the TensorRT engines (once, ~15 min each):
python convert_backbone_tensorrt.py            # 512² backbone (body-only / full modes)
python build_backbone_256.py                   # 256² backbone (this hand pipeline)
python convert_yolo_pose_trt.py --model yolo11m-pose.pt --imgsz 640 --half
```

Engines land in `checkpoints/sam-3d-body-dinov3/backbone_trt/`.

## Usage

**Real-time body + fine hands (the 14.6 FPS pipeline):**
```bash
TRT_INPUT_SIZE=256 MHR_NO_CORRECTIVES=1 USE_COMPILE=1 COMPILE_MODE=default \
TRT_BACKBONE_PATH=checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16_256.engine \
python body_hand_decoder_extractor.py \
    --source path/to/video.mp4 --gpu 0 \
    --hand-res 256 --fx 674.5 --output out.mp4
```
> First run pays a one-time ~1–2 min `torch.compile` cost. Pass the camera intrinsics via
> `--fx` (a wrong value can make the SAM decoder diverge to NaN); omit for MoGe2 auto-estimate.

## Key scripts

| Script | What it does |
|---|---|
| `body_hand_decoder_extractor.py` | **The 14.6 FPS pipeline** — YOLO body + dedicated hand decoder |
| `build_backbone_256.py` | Builds the 256² TRT FP16 backbone engine used above |
| `extract_two_cameras.py` | Two-camera triangulation → metric 3D (removes monocular scale ambiguity). Single-GPU, B2 hand refinement, full mono outputs — the reference version |
| `extract_dualgpu.py` | Same triangulation, one view per GPU in true parallel (~2x faster, **needs 2 GPUs**; body-only, no B2 hands) |
| `stream_demo.py` | Live webcam → server inference → MJPEG browser stream (also the TCP keypoint emitter used by the realtime demo) |
| `convert_backbone_tensorrt.py`, `convert_yolo_pose_trt.py` | TensorRT engine builders |

## Notes

- Checkpoints and TRT engines are **not** shipped in this repo (see `.gitignore`) — build them
  locally as above.
- The monocular skeleton has a global scale ambiguity (inherent to single-view). For metric 3D,
  use the two-camera triangulation scripts.

Built on [Fast SAM 3D Body](https://github.com/yangtiming/Fast-SAM-3D-Body) (USC PSI Lab).
