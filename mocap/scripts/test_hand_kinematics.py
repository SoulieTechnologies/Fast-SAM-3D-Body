"""Tests that PROVE, numerically, the paper derivations behind the STA math.

Run standalone (numpy only):  python3 mocap_validation/test_hand_kinematics.py

Each test corresponds to a step of the pen-and-paper derivation for one finger
(the PIP joint of a synthetic index finger, lengths L1=40, L2=25, L3=20).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import hand_kinematics as hk  # noqa: E402


# a synthetic index finger, straight along +x then optionally bent at the PIP
L1, L2, L3 = 40.0, 25.0, 20.0


def _finger(bend_deg=0.0):
    """Build a 21×3 hand array; only the index finger + wrist/knuckles matter.
    The index is straight (MCP→PIP along +x), then the middle+distal phalanges
    are flexed by bend_deg about the PIP in the −y direction."""
    P = np.zeros((hk.N_LM, 3))
    P[hk.WRIST] = [-20.0, 0.0, 0.0]
    mcp = np.array([0.0, 0.0, 0.0])
    pip = mcp + [L1, 0.0, 0.0]
    t = np.radians(bend_deg)
    d = np.array([np.cos(t), -np.sin(t), 0.0])
    dip = pip + L2 * d
    tip = dip + L3 * d
    L = hk.LM["index"]
    P[L["mcp"]], P[L["pip"]], P[L["dip"]], P[L["tip"]] = mcp, pip, dip, tip
    # knuckles for the palm frame
    P[hk.LM["index"]["mcp"]] = [0.0, 0.5, 0.0]
    P[hk.LM["pinky"]["mcp"]] = [0.0, -0.5, 0.0]
    return P


def _pip_angle(P):
    return hk.finger_flexion(P, "index")["pip"]


def test_scale_invariance():
    u = np.array([2.0, 1.0, 0.5])
    assert abs(hk.included_angle(u, 3.0 * u)) < 1e-12
    v = np.array([0.0, 1.0, 1.0])
    assert abs(hk.included_angle(u, v) - hk.included_angle(5.0 * u, v)) < 1e-12
    print("  scale invariance: angle depends only on direction  ok")


def test_axial_slip_harmless():
    """Slide the DIP marker ALONG the middle phalanx → PIP angle unchanged."""
    P = _finger(bend_deg=35.0)
    a0 = _pip_angle(P)
    L = hk.LM["index"]
    bhat = (P[L["dip"]] - P[L["pip"]])
    bhat = bhat / np.linalg.norm(bhat)
    P2 = P.copy()
    P2[L["dip"]] = P[L["dip"]] + 4.0 * bhat        # 4 mm axial slip
    assert abs(_pip_angle(P2) - a0) < 1e-10
    print("  axial slip (4 mm): PIP angle exactly unchanged  ok")


def test_perpendicular_slip_biases_by_s_over_L():
    """Perp slip s of the DIP marker biases the PIP angle by dφ ≈ s / L2."""
    P = _finger(bend_deg=35.0)
    L = hk.LM["index"]
    b = P[L["dip"]] - P[L["pip"]]
    bhat = b / np.linalg.norm(b)
    a = P[L["pip"]] - P[L["mcp"]]
    perp = a - np.dot(a, bhat) * bhat              # in-plane ⊥ to b
    perp = perp / np.linalg.norm(perp)

    s = 1e-3
    ap = _pip_angle(_shift(P, L["dip"], +s * perp))
    am = _pip_angle(_shift(P, L["dip"], -s * perp))
    dphi_ds = abs(ap - am) / (2 * s)               # central difference
    assert abs(dphi_ds - 1.0 / L2) < 0.02 / L2, (dphi_ds, 1.0 / L2)
    # a 3 mm slip on a 25 mm bone ≈ 6.9° of angle error
    err_deg = np.degrees(3.0 / L2)
    assert 6.0 < err_deg < 7.5
    print(f"  perp slip: dφ/ds = {dphi_ds:.5f} ≈ 1/L = {1/L2:.5f}; "
          f"3 mm → {err_deg:.1f}°  ok")


def test_constraint_blind_to_perpendicular():
    """The bone length changes at O(s²) under a perpendicular slip (constraint
    can't see it) but at O(s) under an axial slip (constraint removes it)."""
    P = _finger(bend_deg=35.0)
    L = hk.LM["index"]
    b = P[L["dip"]] - P[L["pip"]]
    L2meas = np.linalg.norm(b)
    bhat = b / L2meas
    a = P[L["pip"]] - P[L["mcp"]]
    perp = a - np.dot(a, bhat) * bhat
    perp = perp / np.linalg.norm(perp)

    s = 1e-2
    dperp = np.linalg.norm(_finger_bone(_shift(P, L["dip"], s * perp))) - L2meas
    dax = np.linalg.norm(_finger_bone(_shift(P, L["dip"], s * bhat))) - L2meas
    assert abs(dperp) < 5.0 * s ** 2               # O(s²): ~1e-4
    assert abs(dax - s) < 1e-9                      # O(s), exactly s
    print(f"  length change: perp {dperp:.2e} (O(s²)), "
          f"axial {dax:.2e} (=s)  ok")


def test_constrain_chain_enforces_lengths():
    """Rigid projection restores exact bone lengths and pulls a noisy chain
    back toward the clean one."""
    P = _finger(bend_deg=50.0)
    L = hk.LM["index"]
    chain = [L["mcp"], L["pip"], L["dip"], L["tip"]]
    clean = P[chain]
    lengths = np.array([L1, L2, L3])
    rng = np.random.default_rng(0)
    noisy = clean + rng.normal(0, 1.2, clean.shape)
    fixed = hk.constrain_chain(noisy, lengths)
    got = np.linalg.norm(np.diff(fixed, axis=0), axis=1)
    assert np.allclose(got, lengths, atol=1e-6), got
    assert (np.linalg.norm(fixed - clean, axis=1).mean()
            < np.linalg.norm(noisy - clean, axis=1).mean())
    print("  constrain_chain: lengths exact, closer to ground truth  ok")


def test_mcp_flex_abd_separation():
    """Pure flexion → zero abduction, and pure abduction → zero flexion."""
    P = np.zeros((hk.N_LM, 3))
    P[hk.WRIST] = [0.0, 0.0, 0.0]
    P[hk.LM["index"]["mcp"]] = [1.0, 0.5, 0.0]
    P[hk.LM["pinky"]["mcp"]] = [1.0, -0.5, 0.0]
    basis = hk.palm_basis(P[hk.WRIST], P[hk.LM["index"]["mcp"]],
                          P[hk.LM["pinky"]["mcp"]])
    # e_z (palm normal) for this frame is world −z; flexion lives in x–z plane
    Pf = P.copy()
    Pf[hk.LM["middle"]["mcp"]] = [0.0, 0.0, 0.0]
    Pf[hk.LM["middle"]["pip"]] = [1.0, 0.0, -0.5]      # bent toward palm, no y
    flex, abd = hk.mcp_flex_abd(Pf, "middle", basis)
    assert abs(abd) < 1e-9 and abs(flex) > 0.1

    Pa = P.copy()
    Pa[hk.LM["middle"]["mcp"]] = [0.0, 0.0, 0.0]
    Pa[hk.LM["middle"]["pip"]] = [1.0, 0.5, 0.0]       # spread sideways, no z
    flex2, abd2 = hk.mcp_flex_abd(Pa, "middle", basis)
    assert abs(flex2) < 1e-9 and abs(abd2) > 0.1
    print("  MCP decomposition: flexion ⟂ abduction cleanly separated  ok")


def test_sta_metrics_and_neutral():
    T = 200
    t = np.linspace(0, np.radians(80), T)
    seq = np.stack([_finger(bend_deg=np.degrees(ti)) for ti in t])
    L = hk.LM["index"]
    # inject axial STA on the middle phalanx growing with flexion
    slip = 0.05 * t / t.max()                      # up to 0.05 units
    seq[:, L["dip"]] += slip[:, None] * np.array([1.0, 0.0, 0.0])
    sig = hk.sta_bone_std(seq, "index")
    assert sig[1] > sig[0] and sig[1] > sig[2]     # middle bone is the noisy one
    flex = np.array([hk.finger_flexion(seq[i], "index")["pip"] for i in range(T)])
    blen = np.linalg.norm(seq[:, L["dip"]] - seq[:, L["pip"]], axis=1)
    slope = hk.sta_flexion_slope(blen, flex)
    assert slope > 0                               # length grows with flexion
    # neutral referencing removes a constant offset
    ang = {"pip": 1.2}
    neut = {"pip": 0.3}
    assert abs(hk.to_neutral(ang, neut)["pip"] - 0.9) < 1e-12
    print(f"  STA: σ_mid={sig[1]:.4f}>σ_others, slope={slope:.4f}>0; "
          "neutral ref ok")


# helpers
def _shift(P, idx, delta):
    Q = P.copy()
    Q[idx] = P[idx] + delta
    return Q


def _finger_bone(P):
    L = hk.LM["index"]
    return P[L["dip"]] - P[L["pip"]]


def main():
    print("hand_kinematics — derivation tests")
    test_scale_invariance()
    test_axial_slip_harmless()
    test_perpendicular_slip_biases_by_s_over_L()
    test_constraint_blind_to_perpendicular()
    test_constrain_chain_enforces_lengths()
    test_mcp_flex_abd_separation()
    test_sta_metrics_and_neutral()
    print("all hand-kinematics tests passed")


if __name__ == "__main__":
    main()
