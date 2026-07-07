"""
Calibrate each camera individually → K (3x3) + D (distortion vector).

Usage:
    python calibrate_single.py --cam cam0
    python calibrate_single.py --cam cam1

Outputs:
    calibration_data/cam0_intrinsics.npz
    calibration_data/cam1_intrinsics.npz
"""
import argparse
import glob
import cv2
import numpy as np
import board_config

parser = argparse.ArgumentParser()
parser.add_argument("--cam", default="cam0", choices=["cam0", "cam1"])
parser.add_argument("--flags", type=int, default=0,
                    help="cv2.calibrateCamera flags (e.g. cv2.CALIB_FIX_K3)")
args = parser.parse_args()

board, dictionary = board_config.make_board()
detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

paths = sorted(glob.glob(f"images/{args.cam}/*.png"))
assert paths, f"No images found in images/{args.cam}/"

all_corners, all_ids, img_size = [], [], None

for path in paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]

    ch_corners, ch_ids, _, _ = board_config.detect_charuco(
        gray, board, dictionary, min_corners=6)
    if ch_corners is None:
        print(f"  skip (board not found / too few corners): {path}")
        continue

    all_corners.append(ch_corners)
    all_ids.append(ch_ids)
    print(f"  ok ({len(ch_corners)} corners): {path}")

print(f"\nUsing {len(all_corners)} / {len(paths)} images for {args.cam}")
assert len(all_corners) >= 10, "Need at least 10 valid frames for reliable calibration."

# calibrateCameraCharuco was removed in OpenCV >= 4.8 — build per-frame object
# points from the board's chessboard corners and use plain calibrateCamera.
chess = board.getChessboardCorners()            # (Ncorners, 3) float32, metres
obj_all = [chess[ids.ravel()] for ids in all_ids]
rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
    obj_all, all_corners, img_size, None, None, flags=args.flags
)

print(f"\nRMS reprojection error: {rms:.4f} px")
print(f"K =\n{K}")
print(f"D = {D.ravel()}")

out = f"calibration_data/{args.cam}_intrinsics.npz"
import os
os.makedirs("calibration_data", exist_ok=True)
np.savez(out, K=K, D=D, rms=rms, img_size=np.array(img_size))
print(f"\nSaved → {out}")
