#!/usr/bin/env python3
"""Goliath-70 → COSMIK-26 mapping (for the one-time LSTM model scaling).

COSMIK 26-point layout is derived from the Goliath 70 keypoints: most points map
1:1 by index, two are midpoints (mid-ears, mid-hip).

Usage (convert a (T,70,3) array to (T,26,3)):
  python goliath_to_cosmik.py --npy output_twocam/joints_3d_tri.npy --out cosmik_26.npy
or import:
  from goliath_to_cosmik import goliath_to_cosmik
  cosmik = goliath_to_cosmik(joints70)   # (...,70,3) -> (...,26,3)
"""

import argparse
import numpy as np

# Direct index map: COSMIK index -> Goliath index.
_DIRECT = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
    9: 62,   # L_wrist
    10: 41,  # R_wrist
    11: 9, 12: 10, 13: 11, 14: 12,
    15: 13, 16: 14,
    18: 69,  # neck
    20: 15, 21: 18, 22: 16, 23: 19, 24: 17, 25: 20,
}
# Computed points (averages of two Goliath joints).
_AVG = {
    17: (3, 4),    # mid_ears
    19: (9, 10),   # mid_hip
}
N_COSMIK = 26


def goliath_to_cosmik(goliath):
    """(..., 70, 3) Goliath keypoints -> (..., 26, 3) COSMIK keypoints (NaN-safe)."""
    goliath = np.asarray(goliath, dtype=np.float32)
    assert goliath.shape[-2] == 70, f"expected (...,70,3), got {goliath.shape}"
    out = np.full(goliath.shape[:-2] + (N_COSMIK, 3), np.nan, dtype=np.float32)
    for c, g in _DIRECT.items():
        out[..., c, :] = goliath[..., g, :]
    for c, (a, b) in _AVG.items():
        out[..., c, :] = (goliath[..., a, :] + goliath[..., b, :]) / 2.0
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert Goliath-70 npy to COSMIK-26")
    p.add_argument("--npy", required=True, help="(T,70,3) Goliath joints")
    p.add_argument("--out", default="cosmik_26.npy")
    a = p.parse_args()
    g = np.load(a.npy)
    c = goliath_to_cosmik(g)
    np.save(a.out, c)
    valid = np.isfinite(c).all(-1)
    print(f"{a.npy} {g.shape} -> {a.out} {c.shape}")
    print(f"  COSMIK points present (median per frame): {np.median(valid.sum(-1)):.0f}/26")
