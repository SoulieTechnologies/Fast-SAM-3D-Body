"""Full-hand joint angles from a labelled 21-marker sequence.

Input: seq (T, 21, 3) in the fastsam3d landmark order (see hand_kinematics.LM):
per finger [tip, DIP, PIP, MCP] + wrist. Output: a time series (degrees) for
EVERY articulation — this is the reference the SAM3D angles are compared to.

Per finger we report 4 angles:
    <finger>_mcp_flex   MCP flexion   (palm-frame; thumb → CMC flexion)
    <finger>_mcp_abd    MCP abduction (palm-frame; thumb → CMC abduction)
    <finger>_pip        PIP flexion   (inter-segment; thumb → MCP)
    <finger>_dip        DIP flexion   (inter-segment; thumb → IP)
→ 20 articulation angles for the hand.

Cleaning per joint (same rationale as the derivations): reject frames whose
adjacent bone length deviates >bone_tol from its median (marker swap/jump),
drop anatomically impossible values, median-smooth single-frame spikes, and
interpolate only SHORT gaps (real dropouts stay NaN).
"""
import numpy as np

from utils import hand_kinematics as hk
from utils import io_motive as io

FINGERS = hk.FINGERS
JOINTS = ("mcp_flex", "mcp_abd", "pip", "dip")

# conservative anatomical ceilings (deg); |value| above → tracking error
ANAT_MAX = {"mcp_flex": 95.0, "mcp_abd": 30.0, "pip": 115.0, "dip": 90.0}
# which two chain bones (0 wrist-MCP,1 MCP-PIP,2 PIP-DIP,3 DIP-tip) gate a joint
JOINT_BONES = {"mcp_flex": (0, 1), "mcp_abd": (0, 1), "pip": (1, 2), "dip": (2, 3)}


def bone_length_series(seq, finger):
    """(T,4) lengths of wrist-MCP, MCP-PIP, PIP-DIP, DIP-tip for one finger."""
    L = hk.LM[finger]
    chain = [hk.WRIST, L["mcp"], L["pip"], L["dip"], L["tip"]]
    return np.stack([np.linalg.norm(seq[:, b] - seq[:, a], axis=1)
                     for a, b in zip(chain[:-1], chain[1:])], axis=1)


def hand_angle_series(seq, fps, hand="right", bone_tol=0.30, gap_ms=120.0,
                      smooth_win=5, clean=True, bone_ref=None):
    """seq (T,21,3) → (time (T,), angles dict {name:(T,) deg}, quality dict).
    `angles` keys are '<finger>_<joint>'. `quality` gives per-angle
    (valid_fraction, longest_gap_ms)."""
    T = len(seq)
    time = np.arange(T) / fps
    raw = {}
    # per-frame angles
    per = {f: {j: np.full(T, np.nan) for j in JOINTS} for f in FINGERS}
    for t in range(T):
        P = seq[t]
        try:
            basis = hk.palm_basis(P[hk.WRIST], P[hk.LM["index"]["mcp"]],
                                  P[hk.LM["pinky"]["mcp"]], hand=hand)
        except (ValueError, FloatingPointError):
            basis = None
        for f in FINGERS:
            fa = hk.finger_flexion(P, f)
            per[f]["pip"][t] = np.degrees(fa["pip"])
            per[f]["dip"][t] = np.degrees(fa["dip"])
            if basis is not None:
                fl, ab = hk.mcp_flex_abd(P, f, basis)
                per[f]["mcp_flex"][t] = np.degrees(fl)
                per[f]["mcp_abd"][t] = np.degrees(ab)

    quality = {}
    gap = int(gap_ms / 1000.0 * fps)
    for f in FINGERS:
        blen = bone_length_series(seq, f)
        # reference bone lengths: the KNOWN seed lengths when given (rejects a
        # mislabelled marker whose bone deviates from the true length), else the
        # take median
        bref = np.asarray(bone_ref[f]) if bone_ref else np.nanmedian(blen, axis=0)
        good_bone = np.abs(blen - bref) <= bone_tol * bref
        for j in JOINTS:
            a = per[f][j].copy()
            if clean:
                b0, b1 = JOINT_BONES[j]
                bad = ~(good_bone[:, b0] & good_bone[:, b1])
                lim = ANAT_MAX[j]
                bad |= ~(np.abs(a) <= lim)
                a[bad] = np.nan
                a = io.reject_spikes(a, win=11, thr=25.0)
                a = io.smooth_series(a, win=smooth_win)
                a = io.fill_short_gaps(a, max_gap=gap)
            name = f"{f}_{j}"
            raw[name] = a
            nmiss, longest = io.gap_report(a)
            quality[name] = (1.0 - nmiss / T, longest / fps * 1000.0)
    return time, raw, quality


def angle_names():
    """Canonical ordered list of the 20 articulation-angle column names."""
    return [f"{f}_{j}" for f in FINGERS for j in JOINTS]
