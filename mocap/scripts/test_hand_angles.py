"""Synthetic 21-marker hand → validate the full-hand angle pipeline end to end.
Run: python scripts/test_hand_angles.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils import hand_angles as ha  # noqa: E402
from utils import hand_kinematics as hk  # noqa: E402

MCPS = {"index": [0.07, 0.03, 0], "middle": [0.075, 0.01, 0],
        "ring": [0.07, -0.01, 0], "pinky": [0.065, -0.03, 0],
        "thumb": [0.03, 0.05, 0]}
LENS = {"index": (0.040, 0.025, 0.020), "middle": (0.045, 0.028, 0.022),
        "ring": (0.040, 0.026, 0.020), "pinky": (0.030, 0.020, 0.017),
        "thumb": (0.035, 0.030, 0.025)}


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])   # +x → (c,0,−s)


def make_hand(flex):
    P = np.zeros((hk.N_LM, 3))
    x = np.array([1.0, 0, 0])
    for f in hk.FINGERS:
        m = np.array(MCPS[f], float)
        L1, L2, L3 = LENS[f]
        fm, fp, fd = 0.7 * flex, 1.0 * flex, 0.6 * flex
        pip = m + L1 * (_ry(fm) @ x)
        dip = pip + L2 * (_ry(fm + fp) @ x)
        tip = dip + L3 * (_ry(fm + fp + fd) @ x)
        L = hk.LM[f]
        P[L["mcp"]], P[L["pip"]], P[L["dip"]], P[L["tip"]] = m, pip, dip, tip
    return P


def make_sequence(T=240):
    flex = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, T)))   # 0→1→0 rad
    return np.stack([make_hand(fx) for fx in flex]), flex


def test_all_angles_present_and_sane():
    seq, flex = make_sequence()
    time, ang, qual = ha.hand_angle_series(seq, fps=120.0)
    names = ha.angle_names()
    assert len(names) == 20, len(names)
    for n in names:
        assert n in ang and np.isfinite(ang[n]).any(), n
        assert qual[n][0] > 0.9, (n, qual[n])          # >90% valid, no gaps
    # PIP flexion should track the applied fp = 1.0*flex (max ≈ 57°)
    for f in hk.FINGERS:
        pip = ang[f"{f}_pip"]
        assert 45 < np.nanmax(pip) < 70, (f, np.nanmax(pip))
        assert np.nanmin(pip) < 5                       # returns to straight
        # DIP applied fd = 0.6*flex (max ≈ 34°)
        dip = ang[f"{f}_dip"]
        assert 25 < np.nanmax(dip) < 45, (f, np.nanmax(dip))
    print("  20 angles present; PIP≈57°, DIP≈34° peaks; all return to ~0  ok")


def test_mcp_flex_monotone_with_flexion():
    seq, flex = make_sequence()
    _, ang, _ = ha.hand_angle_series(seq, fps=120.0)
    # MCP flexion at the peak frame > at the start frame, per finger
    peak = int(np.argmax(flex))
    for f in hk.FINGERS:
        mf = ang[f"{f}_mcp_flex"]
        assert mf[peak] > mf[0] + 10, (f, mf[0], mf[peak])
    print("  MCP flexion rises with applied flexion  ok")


def test_cleaning_rejects_swap_and_dropout():
    seq, _ = make_sequence()
    L = hk.LM["index"]
    seq[50:55, L["dip"]] = np.nan                       # a 5-frame dropout
    seq[100, L["dip"]] = seq[100, L["pip"]] + 0.5       # a gross swap/jump
    _, ang, qual = ha.hand_angle_series(seq, fps=120.0)
    a = ang["index_dip"]
    assert np.isfinite(a[52])                            # short gap interpolated
    assert np.all(np.abs(a[np.isfinite(a)]) <= ha.ANAT_MAX["dip"])  # swap gone
    print("  cleaning: short gap filled, swap rejected  ok")


def main():
    print("hand_angles — synthetic 21-marker tests")
    test_all_angles_present_and_sane()
    test_mcp_flex_monotone_with_flexion()
    test_cleaning_rejects_swap_and_dropout()
    print("all hand-angle tests passed")


if __name__ == "__main__":
    main()
