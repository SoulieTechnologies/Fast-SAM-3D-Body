#!/usr/bin/env python3
"""Project 3D joints to 2D and overlay the Goliath skeleton on a video.

Requires camera intrinsics (--cam-fx/fy/cx/cy) matching the recording camera.

Usage:
  python visualize.py \\
      --npy output/joints_3d.npy \\
      --video video.mp4 \\
      --cam-fx 760.7 --cam-fy 759.2 --cam-cx 648 --cam-cy 351
"""

import argparse
import cv2
import numpy as np
import os

KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint",
    "right_forefinger4", "right_forefinger3", "right_forefinger2", "right_forefinger_third_joint",
    "right_middle_finger4", "right_middle_finger3", "right_middle_finger2", "right_middle_finger_third_joint",
    "right_ring_finger4", "right_ring_finger3", "right_ring_finger2", "right_ring_finger_third_joint",
    "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2", "right_pinky_finger_third_joint",
    "right_wrist",
    "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint",
    "left_forefinger4", "left_forefinger3", "left_forefinger2", "left_forefinger_third_joint",
    "left_middle_finger4", "left_middle_finger3", "left_middle_finger2", "left_middle_finger_third_joint",
    "left_ring_finger4", "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint",
    "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2", "left_pinky_finger_third_joint",
    "left_wrist",
    "left_olecranon", "right_olecranon",
    "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion",
    "neck",
]

_IDX = {n: i for i, n in enumerate(KP_NAMES)}
_i   = _IDX.__getitem__

_GREEN  = (0, 255, 0)
_ORANGE = (0, 128, 255)
_BLUE   = (255, 153, 51)
_PINK   = (255, 153, 255)
_CYAN   = (255, 178, 102)
_RED    = (51, 51, 255)

BODY_BONES = [
    (_i("left_ankle"),      _i("left_knee"),       _GREEN),
    (_i("left_knee"),       _i("left_hip"),        _GREEN),
    (_i("right_ankle"),     _i("right_knee"),      _ORANGE),
    (_i("right_knee"),      _i("right_hip"),       _ORANGE),
    (_i("left_hip"),        _i("right_hip"),       _BLUE),
    (_i("left_shoulder"),   _i("left_hip"),        _BLUE),
    (_i("right_shoulder"),  _i("right_hip"),       _BLUE),
    (_i("left_shoulder"),   _i("right_shoulder"),  _BLUE),
    (_i("left_shoulder"),   _i("left_elbow"),      _GREEN),
    (_i("right_shoulder"),  _i("right_elbow"),     _ORANGE),
    (_i("left_elbow"),      _i("left_wrist"),      _GREEN),
    (_i("right_elbow"),     _i("right_wrist"),     _ORANGE),
    (_i("left_eye"),        _i("right_eye"),       _BLUE),
    (_i("nose"),            _i("left_eye"),        _BLUE),
    (_i("nose"),            _i("right_eye"),       _BLUE),
    (_i("left_eye"),        _i("left_ear"),        _BLUE),
    (_i("right_eye"),       _i("right_ear"),       _BLUE),
    (_i("left_ear"),        _i("left_shoulder"),   _BLUE),
    (_i("right_ear"),       _i("right_shoulder"),  _BLUE),
    (_i("left_ankle"),      _i("left_big_toe"),    _GREEN),
    (_i("left_ankle"),      _i("left_small_toe"),  _GREEN),
    (_i("left_ankle"),      _i("left_heel"),       _GREEN),
    (_i("right_ankle"),     _i("right_big_toe"),   _ORANGE),
    (_i("right_ankle"),     _i("right_small_toe"), _ORANGE),
    (_i("right_ankle"),     _i("right_heel"),      _ORANGE),
]


def _hand_bones(side, c_thumb, c_index, c_middle, c_ring, c_pinky):
    """Build finger bone connections for one hand."""
    wrist = _i(f"{side}_wrist")
    bones = []
    for finger, col in [
        ("thumb",         c_thumb),
        ("forefinger",    c_index),
        ("middle_finger", c_middle),
        ("ring_finger",   c_ring),
        ("pinky_finger",  c_pinky),
    ]:
        mcp = _i(f"{side}_{finger}_third_joint")
        j2  = _i(f"{side}_{finger}2")
        j3  = _i(f"{side}_{finger}3")
        j4  = _i(f"{side}_{finger}4")
        bones += [(wrist, mcp, col), (mcp, j2, col), (j2, j3, col), (j3, j4, col)]
    return bones


HAND_BONES = (
    _hand_bones("left",  _ORANGE, _PINK, _CYAN, _RED, _GREEN) +
    _hand_bones("right", _ORANGE, _PINK, _CYAN, _RED, _GREEN)
)

ALL_BONES = BODY_BONES + HAND_BONES


def project_3d_to_2d(joints_3d, fx, fy, cx, cy):
    """Pinhole-project (N, 3) 3D joints to (N, 2) pixel coordinates.

    Returns (pts2d, valid) where valid masks points with z > 0.
    """
    pts2d = np.zeros((joints_3d.shape[0], 2), dtype=np.float32)
    valid = joints_3d[:, 2] > 0.01
    pts2d[valid, 0] = fx * joints_3d[valid, 0] / joints_3d[valid, 2] + cx
    pts2d[valid, 1] = fy * joints_3d[valid, 1] / joints_3d[valid, 2] + cy
    return pts2d, valid


def draw_skeleton(frame, pts2d, valid, img_w, img_h,
                  body_thickness=2, hand_thickness=1,
                  body_radius=4, hand_radius=2):
    """Draw skeleton bones and joint dots onto frame (in-place copy)."""
    overlay = frame.copy()

    for a, b, color in ALL_BONES:
        if not (valid[a] and valid[b]):
            continue
        pa = (int(pts2d[a, 0]), int(pts2d[a, 1]))
        pb = (int(pts2d[b, 0]), int(pts2d[b, 1]))
        if not (0 <= pa[0] < img_w and 0 <= pa[1] < img_h):
            continue
        if not (0 <= pb[0] < img_w and 0 <= pb[1] < img_h):
            continue
        is_hand = (a >= 21 or b >= 21) and a < 63 and b < 63
        cv2.line(overlay, pa, pb, color,
                 hand_thickness if is_hand else body_thickness, cv2.LINE_AA)

    for i, pt2d in enumerate(pts2d):
        if not valid[i]:
            continue
        pt = (int(pt2d[0]), int(pt2d[1]))
        if not (0 <= pt[0] < img_w and 0 <= pt[1] < img_h):
            continue
        is_hand = 21 <= i < 63
        r = hand_radius if is_hand else body_radius
        cv2.circle(overlay, pt, r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, pt, r, (0, 0, 0),       1,  cv2.LINE_AA)

    return overlay


def main():
    p = argparse.ArgumentParser(description="Overlay Goliath skeleton on video (3D → 2D projection)")
    p.add_argument("--npy",    required=True, help="(T, 70, 3) joints .npy")
    p.add_argument("--video",  required=True, help="Original video path")
    p.add_argument("--output", default="skeleton_overlay.mp4")
    p.add_argument("--cam-fx", type=float, default=2726.9)
    p.add_argument("--cam-fy", type=float, default=2726.9)
    p.add_argument("--cam-cx", type=float, default=1080.0)
    p.add_argument("--cam-cy", type=float, default=1920.0)
    p.add_argument("--scale",  type=float, default=1.0,
                   help="Scale intrinsics if video was resized")
    p.add_argument("--show",   action="store_true", help="Display frames live")
    args = p.parse_args()

    joints = np.load(args.npy)
    assert joints.ndim == 3 and joints.shape[1] == 70 and joints.shape[2] == 3
    print(f"Joints: {joints.shape}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {args.video}")

    vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {vid_w}x{vid_h} @ {fps:.1f} FPS, {total} frames")

    fx = args.cam_fx * args.scale
    fy = args.cam_fy * args.scale
    cx = args.cam_cx * args.scale
    cy = args.cam_cy * args.scale

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer   = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h))
    n_joints = joints.shape[0]

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < n_joints:
            pts2d, valid = project_3d_to_2d(joints[frame_idx], fx, fy, cx, cy)
            frame = draw_skeleton(frame, pts2d, valid, vid_w, vid_h)

        writer.write(frame)

        if args.show:
            disp = cv2.resize(frame, (vid_w // 2, vid_h // 2)) if vid_w > 1920 else frame
            cv2.imshow("Skeleton", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}/{total}")

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
