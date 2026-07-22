#!/usr/bin/env python3
"""Build a 256x256 TRT FP16 backbone engine (for the hand-decoder pipeline).

Hand crops carry only ~77px of real info → a 256² backbone loses ~2px on the
keypoints vs 512² (measured) while running the backbone AND decoder much faster
(16x16 tokens instead of 32x32). Reuses convert_backbone_tensorrt.py, overriding
the size constants.

Run:  CUDA_VISIBLE_DEVICES=7 python scripts/trt/build_backbone_256.py
Output: checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16_256.engine
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_backbone_tensorrt as C

d = C.TRT_OUTPUT_DIR
C.IMAGE_SIZE = (256, 256)
C.OUTPUT_SIZE = (16, 16)
C.ONNX_PATH = os.path.join(d, "backbone_dinov3_256.onnx")
C.TRT_PATH_FP16 = os.path.join(d, "backbone_dinov3_fp16_256.engine")
C.TRT_PATH = C.TRT_PATH_FP16

print(f"Building 256² engine → {C.TRT_PATH}")
bb = C.load_backbone()
ok1 = C.step1_export_onnx(bb, batch_sizes=[1, 2, 4])
ok2 = C.step2_convert_tensorrt(batch_sizes=[1, 2, 4]) if ok1 else False
print("DONE 256 build:", "OK" if ok2 else "FAILED")
