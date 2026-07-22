#!/usr/bin/env python3
"""End-to-end synthetic test of calibrate_multi's pairwise + chaining solver.

Simulates the real capture workflow on the 4-camera ceiling rig (2 front,
2 side): board shown to the front pair {0,1}, then to the LEFT pair {0,2},
then to the RIGHT pair {1,3} — cam3 NEVER shares a frame with cam0, so its
extrinsics must come out of the 0 -> 1 -> 3 chain. Exact synthetic corners
-> the recovered R,T must match ground truth to numerical precision.

Runs anywhere with opencv (python3 tools/test_calibrate_chain.py).
"""

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "stereo_calibration"))

import calibrate_multi as cm
from tools.test_view_selection import look_at

# ground truth rig (world frame), same geometry as the real room
GT = {
    0: look_at([-0.6, 1.6, 2.5]),  # front-left  = reference
    1: look_at([+0.6, 1.6, 2.5]),  # front-right
    2: look_at([-2.5, 1.6, 0.0]),  # left side
    3: look_at([+2.5, 1.6, 0.0]),
}  # right side
K = np.array([[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]])
D = np.zeros(5)
IMG_SIZE = (1280, 720)

# 8x5 synthetic "chessboard" corners, 4 cm pitch, board frame (z=0)
gx, gy = np.meshgrid(np.arange(8), np.arange(5))
CHESS = np.stack(
    [gx.ravel() * 0.04, gy.ravel() * 0.04, np.zeros(40)], 1
).astype(np.float64)


def rot(rx, ry, rz):
    r, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    return r


def make_dets(zones, n_per_zone=12, seed=0):
    """dets[cam][frame_name] = {corner_id: (1,2) pixel}, per visibility zone."""
    rng = np.random.default_rng(seed)
    dets = {c: {} for c in GT}
    f = 0
    for center, facing, cams in zones:
        for _ in range(n_per_zone):
            Rb = rot(*facing) @ rot(*rng.uniform(-0.35, 0.35, 3))
            tb = np.asarray(center) + rng.uniform(-0.15, 0.15, 3)
            pts_w = CHESS @ Rb.T + tb
            name = f"frame_{f:04d}.png"
            f += 1
            for c in cams:
                Rc, Tc = GT[c]
                rvec, _ = cv2.Rodrigues(Rc)
                px, _ = cv2.projectPoints(pts_w, rvec, Tc, K, D)
                dets[c][name] = {
                    i: px[i].astype(np.float32) for i in range(len(CHESS))
                }
    return dets


def gt_rel(c, ref=0):
    """Ground-truth world(=cam ref)->cam c: R_rel, T_rel."""
    Rc, Tc = GT[c]
    R0, T0 = GT[ref]
    R = Rc @ R0.T
    return R, Tc - R @ T0


def main():
    zones = [
        ((-0.1, 0.2, 0.9), (0.0, 0.4, 0.0), (0, 1)),  # front pair
        ((-1.2, 0.2, 1.0), (0.0, 0.9, 0.0), (0, 2)),  # left pair
        ((+1.2, 0.2, 1.0), (0.0, -0.9, 0.0), (1, 3)),
    ]  # right pair
    dets = make_dets(zones)
    assert not set(dets[0]) & set(dets[3]), "test premise: cam0/cam3 disjoint"

    cams = [0, 1, 2, 3]
    pairs = {}
    for i in range(4):
        for j in range(i + 1, 4):
            r = cm.pair_extrinsics(
                dets[i], dets[j], K, D, K, D, IMG_SIZE, CHESS, min_common=6
            )
            if r is not None:
                pairs[(i, j)] = r
    assert set(pairs) == {(0, 1), (0, 2), (1, 3)}, f"pairs: {sorted(pairs)}"
    for (a, b), (R, T, rms, n) in pairs.items():
        assert rms < 0.1, f"cam{a}<->cam{b} stereo rms {rms}"

    Rw, Tw, route = cm.chain_extrinsics(cams, pairs, 0)
    assert route[3] == [0, 1, 3], f"cam3 must chain through cam1: {route[3]}"
    for c in (1, 2, 3):
        Rg, Tg = gt_rel(c)
        ang = np.degrees(
            np.arccos(np.clip((np.trace(Rw[c] @ Rg.T) - 1) / 2, -1, 1))
        )
        dt = np.linalg.norm(Tw[c] - Tg)
        tag = "chained" if len(route[c]) > 2 else "direct"
        print(
            f"  cam{c} ({tag}): rot err {ang:.2e} deg, "
            f"trans err {dt * 1000:.4f} mm"
        )
        assert ang < 0.01 and dt < 1e-3, f"cam{c}: ang {ang} dt {dt}"

    # corrupt 5 of cam1's front-zone frames (corner ids shuffled = the
    # grazing-view misdetection failure mode): the PnP/consensus gate must
    # drop exactly those frames and keep the solve exact
    bad = [n for n in sorted(set(dets[0]) & set(dets[1]))][:5]
    rng = np.random.default_rng(1)
    for n in bad:
        ids = list(dets[1][n])
        perm = rng.permutation(ids)
        dets[1][n] = {i: dets[1][n][j] for i, j in zip(ids, perm)}
    r = cm.pair_extrinsics(
        dets[0], dets[1], K, D, K, D, IMG_SIZE, CHESS, min_common=6, la=0, lb=1
    )
    assert r is not None and r[2] < 0.1, f"rms after outliers: {r[2]}"
    assert r[3] <= 12 - 5 + 1, f"corrupted frames not dropped: {r[3]} inliers"
    Rg, Tg = gt_rel(1)
    ang = np.degrees(
        np.arccos(np.clip((np.trace(r[0] @ Rg.T) - 1) / 2, -1, 1))
    )
    assert ang < 0.01 and np.linalg.norm(r[1] - Tg) < 1e-3
    print(
        f"  outlier rejection ok ({12 - r[3]} frames dropped, "
        f"rms {r[2]:.4f} px)"
    )

    # disconnect cam3 entirely -> must fail with a clear message
    try:
        cm.chain_extrinsics(
            cams, {k: v for k, v in pairs.items() if 3 not in k}, 0
        )
        raise AssertionError("disconnected cam3 must raise")
    except SystemExit as e:
        assert "3" in str(e)
    print("all calibrate-chain tests passed")


if __name__ == "__main__":
    main()
