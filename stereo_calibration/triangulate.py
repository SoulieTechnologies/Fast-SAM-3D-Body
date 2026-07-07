"""
Triangulation helper for FastSAM3D.

Usage example:
    from triangulate import StereoTriangulator
    tri = StereoTriangulator("calibration_data/stereo_params.npz")
    pts3d = tri.triangulate(pts_left, pts_right)  # Nx2 float32 arrays
"""
import cv2
import numpy as np


class StereoTriangulator:
    def __init__(self, stereo_npz: str):
        data = np.load(stereo_npz)
        self.K1  = data["K1"]
        self.D1  = data["D1"]
        self.K2  = data["K2"]
        self.D2  = data["D2"]
        self.R   = data["R"]
        self.T   = data["T"]
        self.P1  = data["P1"]
        self.P2  = data["P2"]
        self.R1  = data["R1"]
        self.R2  = data["R2"]
        self.Q   = data["Q"]
        self.img_size = tuple(data["img_size"])

        # Undistort + rectify maps (computed once)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, self.img_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, self.img_size, cv2.CV_32FC1
        )

    def rectify_images(self, img0, img1):
        r0 = cv2.remap(img0, self.map1x, self.map1y, cv2.INTER_LINEAR)
        r1 = cv2.remap(img1, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return r0, r1

    def triangulate(self, pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
        """
        Triangulate matched 2-D points from both cameras.

        pts0, pts1: (N, 2) float32 pixel coordinates in the ORIGINAL (unrectified) images.
        Returns: (N, 3) float64 3-D points in cam0 coordinate frame.
        """
        # Undistort to normalized image coords
        pts0_ud = cv2.undistortPoints(
            pts0.reshape(-1, 1, 2), self.K1, self.D1, R=self.R1, P=self.P1
        ).reshape(-1, 2)
        pts1_ud = cv2.undistortPoints(
            pts1.reshape(-1, 1, 2), self.K2, self.D2, R=self.R2, P=self.P2
        ).reshape(-1, 2)

        pts4d = cv2.triangulatePoints(
            self.P1, self.P2,
            pts0_ud.T.astype(np.float32),
            pts1_ud.T.astype(np.float32),
        )
        pts3d = (pts4d[:3] / pts4d[3]).T  # (N, 3)
        return pts3d

    def project(self, pts3d: np.ndarray, cam: int = 0):
        """Re-project 3-D points onto cam0 or cam1 (for reprojection error check)."""
        rvec, _ = cv2.Rodrigues(np.eye(3) if cam == 0 else self.R)
        tvec = np.zeros((3, 1)) if cam == 0 else self.T
        K = self.K1 if cam == 0 else self.K2
        D = self.D1 if cam == 0 else self.D2
        pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, D)
        return pts2d.reshape(-1, 2)
