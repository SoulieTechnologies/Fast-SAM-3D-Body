"""
Sanity-check the calibration: reprojection error per image + epipolar lines.

Usage:
    python verify_calibration.py
"""
import glob
import cv2
import numpy as np
from triangulate import StereoTriangulator
import board_config

board, dictionary = board_config.make_board()
detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

tri = StereoTriangulator("calibration_data/stereo_params.npz")

paths0 = sorted(glob.glob("images/cam0/*.png"))
paths1 = sorted(glob.glob("images/cam1/*.png"))

errors = []
for p0, p1 in zip(paths0, paths1):
    img0 = cv2.imread(p0)
    img1 = cv2.imread(p1)
    r0, r1 = tri.rectify_images(img0, img1)

    # Draw epipolar lines on the first pair and show it
    if not errors:
        h = max(r0.shape[0], r1.shape[0])
        combined = np.zeros((h, r0.shape[1] + r1.shape[1], 3), dtype=np.uint8)
        combined[:r0.shape[0], :r0.shape[1]] = r0
        combined[:r1.shape[0], r0.shape[1]:] = r1
        for y in range(0, h, 30):
            cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)
        cv2.imshow("Rectified pair (epipolar lines should align)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Per-image reprojection error
    def corners(gray):
        c, ids, _ = aruco_detector.detectMarkers(gray)
        if ids is None: return None, None
        retval, cc, ci = cv2.aruco.interpolateCornersCharuco(c, ids, gray, board)
        return (cc, ci) if retval >= 6 else (None, None)

    cc0, ci0 = corners(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY))
    cc1, ci1 = corners(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    if cc0 is None or cc1 is None:
        continue

    ids0_set = set(ci0.ravel())
    ids1_set = set(ci1.ravel())
    common = sorted(ids0_set & ids1_set)
    if len(common) < 4:
        continue

    id_map0 = {int(ci0[i]): cc0[i].ravel() for i in range(len(ci0))}
    id_map1 = {int(ci1[i]): cc1[i].ravel() for i in range(len(ci1))}
    pts0 = np.array([id_map0[i] for i in common], dtype=np.float32)
    pts1 = np.array([id_map1[i] for i in common], dtype=np.float32)
    obj  = board.getChessboardCorners()[common].reshape(-1, 3)

    pts3d = tri.triangulate(pts0, pts1)
    proj0 = tri.project(pts3d, cam=0)
    proj1 = tri.project(pts3d, cam=1)

    err0 = np.linalg.norm(pts0 - proj0, axis=1).mean()
    err1 = np.linalg.norm(pts1 - proj1, axis=1).mean()
    errors.append((err0 + err1) / 2)
    print(f"{p0}  err0={err0:.3f}  err1={err1:.3f}")

if errors:
    print(f"\nMean reprojection error: {np.mean(errors):.4f} px  "
          f"(max {np.max(errors):.4f} px)")
