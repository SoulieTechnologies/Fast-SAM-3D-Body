"""
Stereo calibration using synchronized image pairs.

Requires calibrate_single.py to have been run for both cameras first.

Usage:
    python calibrate_stereo.py

Outputs:
    calibration_data/stereo_params.npz
        K1, D1, K2, D2   — intrinsics
        R, T              — rotation & translation from cam0 to cam1
        E, F              — essential & fundamental matrices
        R1, R2, P1, P2, Q — rectification output
        rms               — stereo reprojection error
"""

import glob
import cv2
import numpy as np
import board_config

board, dictionary = board_config.make_board()
detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)


# Load per-camera intrinsics
def load_intrinsics(cam):
    data = np.load(f"calibration_data/{cam}_intrinsics.npz")
    return data["K"], data["D"], tuple(data["img_size"])


K1, D1, img_size = load_intrinsics("cam0")
K2, D2, _ = load_intrinsics("cam1")

paths0 = sorted(glob.glob("images/cam0/*.png"))
paths1 = sorted(glob.glob("images/cam1/*.png"))
assert len(paths0) == len(paths1), "Unequal number of images in cam0 / cam1"

obj_points_all, img_points0_all, img_points1_all = [], [], []

for p0, p1 in zip(paths0, paths1):
    img0 = cv2.imread(p0)
    img1 = cv2.imread(p1)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    def get_charuco(gray):
        ch_corners, ch_ids, _, _ = board_config.detect_charuco(
            gray, board, dictionary, min_corners=6
        )
        return ch_corners, ch_ids

    ch0, ids0 = get_charuco(gray0)
    ch1, ids1 = get_charuco(gray1)

    if ch0 is None or ch1 is None:
        print(f"  skip: {p0}")
        continue

    # Keep only corners visible in BOTH images
    ids0_set = set(ids0.ravel())
    ids1_set = set(ids1.ravel())
    common_ids = sorted(ids0_set & ids1_set)
    if len(common_ids) < 6:
        print(f"  skip (only {len(common_ids)} common corners): {p0}")
        continue

    id_map0 = {int(ids0[i].item()): ch0[i] for i in range(len(ids0))}
    id_map1 = {int(ids1[i].item()): ch1[i] for i in range(len(ids1))}

    pts0 = np.array([id_map0[i] for i in common_ids], dtype=np.float32)
    pts1 = np.array([id_map1[i] for i in common_ids], dtype=np.float32)

    # 3-D object points from board layout
    obj_pts = board.getChessboardCorners()[common_ids].reshape(-1, 1, 3)

    obj_points_all.append(obj_pts)
    img_points0_all.append(pts0)
    img_points1_all.append(pts1)
    print(f"  ok ({len(common_ids)} common corners): {p0}")

print(f"\nUsing {len(obj_points_all)} frame pairs for stereo calibration")
assert len(obj_points_all) >= 10, "Need at least 10 valid pairs."

flags = cv2.CALIB_FIX_INTRINSIC  # intrinsics already refined per-camera

rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    obj_points_all,
    img_points0_all,
    img_points1_all,
    K1,
    D1,
    K2,
    D2,
    img_size,
    flags=flags,
    criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
)

print(f"\nStereo RMS reprojection error: {rms:.4f} px")
print(f"R =\n{R}")
print(f"T = {T.ravel()}")

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2, img_size, R, T, alpha=0
)

import os

os.makedirs("calibration_data", exist_ok=True)
out = "calibration_data/stereo_params.npz"
np.savez(
    out,
    K1=K1,
    D1=D1,
    K2=K2,
    D2=D2,
    R=R,
    T=T,
    E=E,
    F=F,
    R1=R1,
    R2=R2,
    P1=P1,
    P2=P2,
    Q=Q,
    rms=rms,
    img_size=np.array(img_size),
)
print(f"\nSaved → {out}")
