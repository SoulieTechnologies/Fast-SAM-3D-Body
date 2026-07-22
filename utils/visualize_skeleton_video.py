#!/usr/bin/env python3
"""Overlay the 70-point Goliath skeleton on a video from pre-computed 2D joints.

Usage:
  python utils/visualize_skeleton_video.py \\
      --npy output/joints_2d.npy \\
      --video video.mp4 \\
      --output skeleton_overlay.mp4
"""

import argparse
import cv2
import numpy as np
import os

KP_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "right_thumb4",
    "right_thumb3",
    "right_thumb2",
    "right_thumb_third_joint",
    "right_forefinger4",
    "right_forefinger3",
    "right_forefinger2",
    "right_forefinger_third_joint",
    "right_middle_finger4",
    "right_middle_finger3",
    "right_middle_finger2",
    "right_middle_finger_third_joint",
    "right_ring_finger4",
    "right_ring_finger3",
    "right_ring_finger2",
    "right_ring_finger_third_joint",
    "right_pinky_finger4",
    "right_pinky_finger3",
    "right_pinky_finger2",
    "right_pinky_finger_third_joint",
    "right_wrist",
    "left_thumb4",
    "left_thumb3",
    "left_thumb2",
    "left_thumb_third_joint",
    "left_forefinger4",
    "left_forefinger3",
    "left_forefinger2",
    "left_forefinger_third_joint",
    "left_middle_finger4",
    "left_middle_finger3",
    "left_middle_finger2",
    "left_middle_finger_third_joint",
    "left_ring_finger4",
    "left_ring_finger3",
    "left_ring_finger2",
    "left_ring_finger_third_joint",
    "left_pinky_finger4",
    "left_pinky_finger3",
    "left_pinky_finger2",
    "left_pinky_finger_third_joint",
    "left_wrist",
    "left_olecranon",
    "right_olecranon",
    "left_cubital_fossa",
    "right_cubital_fossa",
    "left_acromion",
    "right_acromion",
    "neck",
]

_IDX = {n: i for i, n in enumerate(KP_NAMES)}
_i = _IDX.__getitem__

_GREEN = (0, 255, 0)
_ORANGE = (0, 128, 255)
_BLUE = (255, 153, 51)
_PINK = (255, 153, 255)
_CYAN = (255, 178, 102)
_RED = (51, 51, 255)

BODY_BONES = [
    (_i("left_ankle"), _i("left_knee"), _GREEN),
    (_i("left_knee"), _i("left_hip"), _GREEN),
    (_i("right_ankle"), _i("right_knee"), _ORANGE),
    (_i("right_knee"), _i("right_hip"), _ORANGE),
    (_i("left_hip"), _i("right_hip"), _BLUE),
    (_i("left_shoulder"), _i("left_hip"), _BLUE),
    (_i("right_shoulder"), _i("right_hip"), _BLUE),
    (_i("left_shoulder"), _i("right_shoulder"), _BLUE),
    (_i("left_shoulder"), _i("left_elbow"), _GREEN),
    (_i("right_shoulder"), _i("right_elbow"), _ORANGE),
    (_i("left_elbow"), _i("left_wrist"), _GREEN),
    (_i("right_elbow"), _i("right_wrist"), _ORANGE),
    (_i("left_eye"), _i("right_eye"), _BLUE),
    (_i("nose"), _i("left_eye"), _BLUE),
    (_i("nose"), _i("right_eye"), _BLUE),
    (_i("left_eye"), _i("left_ear"), _BLUE),
    (_i("right_eye"), _i("right_ear"), _BLUE),
    (_i("left_ear"), _i("left_shoulder"), _BLUE),
    (_i("right_ear"), _i("right_shoulder"), _BLUE),
    (_i("left_ankle"), _i("left_big_toe"), _GREEN),
    (_i("left_ankle"), _i("left_small_toe"), _GREEN),
    (_i("left_ankle"), _i("left_heel"), _GREEN),
    (_i("right_ankle"), _i("right_big_toe"), _ORANGE),
    (_i("right_ankle"), _i("right_small_toe"), _ORANGE),
    (_i("right_ankle"), _i("right_heel"), _ORANGE),
]


def _hand_bones(side, c_thumb, c_index, c_middle, c_ring, c_pinky):
    """Build finger bone connections for one hand."""
    wrist = _i(f"{side}_wrist")
    bones = []
    for finger, col in [
        ("thumb", c_thumb),
        ("forefinger", c_index),
        ("middle_finger", c_middle),
        ("ring_finger", c_ring),
        ("pinky_finger", c_pinky),
    ]:
        mcp = _i(f"{side}_{finger}_third_joint")
        j2 = _i(f"{side}_{finger}2")
        j3 = _i(f"{side}_{finger}3")
        j4 = _i(f"{side}_{finger}4")
        bones += [
            (wrist, mcp, col),
            (mcp, j2, col),
            (j2, j3, col),
            (j3, j4, col),
        ]
    return bones


HAND_BONES = _hand_bones(
    "left", _ORANGE, _PINK, _CYAN, _RED, _GREEN
) + _hand_bones("right", _ORANGE, _PINK, _CYAN, _RED, _GREEN)

ALL_BONES = BODY_BONES + HAND_BONES


def draw_skeleton(
    frame,
    pts2d,
    valid,
    img_w,
    img_h,
    body_thickness=3,
    hand_thickness=2,
    body_radius=5,
    hand_radius=3,
):
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
        cv2.line(
            overlay,
            pa,
            pb,
            color,
            hand_thickness if is_hand else body_thickness,
            cv2.LINE_AA,
        )

    for i, pt2d in enumerate(pts2d):
        if not valid[i]:
            continue
        pt = (int(pt2d[0]), int(pt2d[1]))
        if not (0 <= pt[0] < img_w and 0 <= pt[1] < img_h):
            continue
        is_hand = 21 <= i < 63
        r = hand_radius if is_hand else body_radius
        cv2.circle(overlay, pt, r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, pt, r, (0, 0, 0), 1, cv2.LINE_AA)

    return overlay


# ── Option B: pixel-accurate YOLO body + SAM3D hands ─────────────────────────
# COCO-17 keypoint order from yolo11m-pose.
_COCO = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_sho": 5,
    "r_sho": 6,
    "l_elb": 7,
    "r_elb": 8,
    "l_wri": 9,
    "r_wri": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_kne": 13,
    "r_kne": 14,
    "l_ank": 15,
    "r_ank": 16,
}
_c = _COCO.__getitem__

COCO_BONES = [
    (_c("nose"), _c("l_eye"), _BLUE),
    (_c("nose"), _c("r_eye"), _BLUE),
    (_c("l_eye"), _c("l_ear"), _BLUE),
    (_c("r_eye"), _c("r_ear"), _BLUE),
    (_c("l_sho"), _c("r_sho"), _BLUE),
    (_c("l_sho"), _c("l_hip"), _BLUE),
    (_c("r_sho"), _c("r_hip"), _BLUE),
    (_c("l_hip"), _c("r_hip"), _BLUE),
    (_c("l_sho"), _c("l_elb"), _GREEN),
    (_c("l_elb"), _c("l_wri"), _GREEN),
    (_c("r_sho"), _c("r_elb"), _ORANGE),
    (_c("r_elb"), _c("r_wri"), _ORANGE),
    (_c("l_hip"), _c("l_kne"), _GREEN),
    (_c("l_kne"), _c("l_ank"), _GREEN),
    (_c("r_hip"), _c("r_kne"), _ORANGE),
    (_c("r_kne"), _c("r_ank"), _ORANGE),
]

# SAM3D Goliath wrist indices, paired to their YOLO COCO wrist + elbow.
# Each hand is anchored at the wrist and scaled about it to match the body.
_HANDS = [
    {
        "sam_wrist": 41,
        "sam_range": range(21, 42),
        "yolo_wrist": _c("r_wri"),
        "yolo_elbow": _c("r_elb"),
    },
    {
        "sam_wrist": 62,
        "sam_range": range(42, 63),
        "yolo_wrist": _c("l_wri"),
        "yolo_elbow": _c("l_elb"),
    },
]


def compute_hand_scale(joints, yolo_kp):
    """Global scale to bring SAM3D hands up to the YOLO body scale.

    Uses the torso-height ratio (mid-shoulder→mid-hip), the most stable segment,
    median over all frames. One scalar for the whole clip → no jitter.
    """
    ratios = []
    for f in range(min(len(joints), len(yolo_kp))):
        s = joints[f, :, :2]
        y = yolo_kp[f]
        y_ms = (y[5] + y[6]) / 2
        y_mh = (y[11] + y[12]) / 2
        s_ms = (s[5] + s[6]) / 2
        s_mh = (s[9] + s[10]) / 2
        if np.isnan([y_ms, y_mh, s_ms, s_mh]).any():
            continue
        sl = np.linalg.norm(s_ms - s_mh)
        yl = np.linalg.norm(y_ms - y_mh)
        if sl > 1e-3 and yl > 1e-3:
            ratios.append(yl / sl)
    return float(np.median(ratios)) if ratios else 1.0


def draw_option_b(frame, yolo_kp, sam_pts2d, img_w, img_h, hand_scale=1.0):
    """Draw pixel-accurate YOLO COCO-17 body + SAM3D 42 hand joints.

    Hands keep SAM3D's internal structure but are anchored at the YOLO wrist
    and scaled about it by hand_scale, so they connect cleanly to the body
    and match its size.
    """
    overlay = frame.copy()

    def ok(p):
        return (
            not np.isnan(p).any() and 0 <= p[0] < img_w and 0 <= p[1] < img_h
        )

    # Transform each SAM3D hand: anchor wrist to YOLO wrist, scale about it.
    hand_pts = sam_pts2d.copy()
    for h in _HANDS:
        yw = yolo_kp[h["yolo_wrist"]]
        sw = sam_pts2d[h["sam_wrist"]]
        if np.isnan(yw).any() or np.isnan(sw).any():
            continue
        for i in h["sam_range"]:
            hand_pts[i] = yw + (sam_pts2d[i] - sw) * hand_scale

    # YOLO body bones
    for a, b, color in COCO_BONES:
        pa, pb = yolo_kp[a], yolo_kp[b]
        if ok(pa) and ok(pb):
            cv2.line(
                overlay,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                color,
                3,
                cv2.LINE_AA,
            )

    # SAM3D hand bones (transformed)
    for a, b, color in HAND_BONES:
        pa, pb = hand_pts[a], hand_pts[b]
        if ok(pa) and ok(pb):
            cv2.line(
                overlay,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                color,
                2,
                cv2.LINE_AA,
            )

    # Joint dots
    for i in range(17):
        p = yolo_kp[i]
        if ok(p):
            cv2.circle(
                overlay,
                (int(p[0]), int(p[1])),
                5,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                overlay, (int(p[0]), int(p[1])), 5, (0, 0, 0), 1, cv2.LINE_AA
            )
    for i in range(21, 63):
        p = hand_pts[i]
        if ok(p):
            cv2.circle(
                overlay,
                (int(p[0]), int(p[1])),
                3,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                overlay, (int(p[0]), int(p[1])), 3, (0, 0, 0), 1, cv2.LINE_AA
            )

    return overlay


def reproject(joints_3d, fx, fy, cx, cy):
    """Reproject (T, 70, 3) 3D camera-space joints → (T, 70, 2) pixel coords."""
    T, J, _ = joints_3d.shape
    out = np.full((T, J, 2), np.nan, dtype=np.float32)
    z = joints_3d[:, :, 2]
    valid = z > 1e-4
    out[:, :, 0] = np.where(
        valid, joints_3d[:, :, 0] / np.where(valid, z, 1) * fx + cx, np.nan
    )
    out[:, :, 1] = np.where(
        valid, joints_3d[:, :, 1] / np.where(valid, z, 1) * fy + cy, np.nan
    )
    return out


def main():
    p = argparse.ArgumentParser(
        description="Overlay Goliath skeleton on video"
    )
    p.add_argument(
        "--npy",
        required=True,
        help="(T, 70, 2) joints_2d.npy  OR  (T, 70, 3) joints_3d.npy",
    )
    p.add_argument("--video", required=True, help="Original video path")
    p.add_argument("--output", default="skeleton_overlay.mp4")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale skeleton around its centroid (e.g. 1.25)",
    )
    p.add_argument(
        "--yolo",
        default="",
        help="joints_yolo_2d.npy — Option B: draw pixel-accurate YOLO body + SAM3D hands",
    )
    p.add_argument(
        "--fx",
        type=float,
        default=0,
        help="Focal length x — required when --npy is joints_3d.npy",
    )
    p.add_argument("--fy", type=float, default=0)
    p.add_argument(
        "--cx",
        type=float,
        default=0,
        help="Principal point x (0 = image centre)",
    )
    p.add_argument("--cy", type=float, default=0)
    args = p.parse_args()

    joints = np.load(args.npy)
    assert (
        joints.ndim == 3 and joints.shape[1] == 70
    ), "Expected shape (T, 70, 2 or 3)"

    yolo_kp = None
    hand_scale = 1.0
    if args.yolo:
        yolo_kp = np.load(args.yolo)
        assert (
            yolo_kp.ndim == 3 and yolo_kp.shape[1] == 17
        ), "Expected YOLO shape (T, 17, 2)"
        hand_scale = compute_hand_scale(joints, yolo_kp)
        print(
            f"Option B: YOLO body + SAM3D hands  {yolo_kp.shape}  hand_scale={hand_scale:.3f}"
        )

    if joints.shape[2] == 3:
        # 3D input — reproject with provided intrinsics
        if args.fx <= 0:
            raise ValueError(
                "--npy contains 3D joints: please provide --fx/--fy/--cx/--cy"
            )
        cap_probe = cv2.VideoCapture(args.video)
        vid_w_p = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h_p = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_probe.release()
        cx = args.cx if args.cx > 0 else vid_w_p / 2.0
        cy = args.cy if args.cy > 0 else vid_h_p / 2.0
        fy = args.fy if args.fy > 0 else args.fx
        joints = reproject(joints, args.fx, fy, cx, cy)
        print(
            f"Reprojected 3D→2D: fx={args.fx} fy={fy} cx={cx:.0f} cy={cy:.0f}"
        )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {args.video}")

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {vid_w}x{vid_h} @ {fps:.1f} FPS, {total} frames")
    print(f"Joints: {joints.shape}")

    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h)
    )
    n_joints = joints.shape[0]

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < n_joints:
            pts2d = joints[frame_idx, :, :2].copy()
            if yolo_kp is not None and frame_idx < len(yolo_kp):
                # Option B — pixel-accurate YOLO body + SAM3D hands
                frame = draw_option_b(
                    frame, yolo_kp[frame_idx], pts2d, vid_w, vid_h, hand_scale
                )
            else:
                valid = ~np.isnan(pts2d).any(axis=1)
                if args.scale != 1.0 and valid.any():
                    center = pts2d[valid].mean(axis=0)
                    pts2d[valid] = (
                        center + (pts2d[valid] - center) * args.scale
                    )
                frame = draw_skeleton(frame, pts2d, valid, vid_w, vid_h)

        writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}/{total}")

    cap.release()
    writer.release()
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
