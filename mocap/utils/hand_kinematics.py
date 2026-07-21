"""Hand joint-angle kinematics for the mocap-vs-fastsam3d validation.

Pure numpy. Operates on 21 hand landmarks in the fastsam3d ordering, so the
SAME functions run on both sides of the comparison (fastsam3d keypoints and
the labelled mocap markers mapped onto these 21 slots) — identical angle
conventions are what make the RMSE meaningful.

Landmark order per hand (fastsam3d hand-decoder order):
    per finger [tip, dist(DIP/IP), pip(PIP / thumb MCP), mcp(MCP / thumb CMC)]
    thumb 0..3, index 4..7, middle 8..11, ring 12..15, pinky 16..19, wrist 20.

The three STA-handling ideas derived on paper live here:
  Point 1  constrain_chain   — rigid bone-length projection (MKO-lite)
  Point 2  sta_bone_std / sta_flexion_slope — quantify (do not correct) STA
  Point 3  to_neutral        — reference angles to the neutral pose

Key facts the tests pin down (see test_hand_kinematics.py):
  * the inter-segment angle is scale-invariant  → axial marker slip is harmless
  * a perpendicular slip s of an end marker biases the angle by dφ ≈ s / L
  * that perpendicular slip changes the bone length only at O(s²), so the
    length constraint is blind to exactly the component that biases the angle.
"""
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# landmark index of each named point, per finger
LM = {
    "thumb":  {"tip": 0,  "dip": 1,  "pip": 2,  "mcp": 3},
    "index":  {"tip": 4,  "dip": 5,  "pip": 6,  "mcp": 7},
    "middle": {"tip": 8,  "dip": 9,  "pip": 10, "mcp": 11},
    "ring":   {"tip": 12, "dip": 13, "pip": 14, "mcp": 15},
    "pinky":  {"tip": 16, "dip": 17, "pip": 18, "mcp": 19},
}
WRIST = 20
N_LM = 21

_EPS = 1e-9


def included_angle(u, v):
    """Angle between two vectors (rad); 0 = parallel. Scale-invariant: only
    the directions matter, so scaling |u| or |v| leaves it unchanged."""
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < _EPS or nv < _EPS:
        return np.nan
    c = float(np.dot(u, v)) / (nu * nv)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def palm_basis(wrist, mcp_index, mcp_pinky, hand="right"):
    """Right-handed palm coordinate frame (rows = e_x, e_y, e_z):
        e_x  proximal→distal  (wrist → midpoint of the index/pinky knuckles)
        e_z  palm normal, oriented DORSAL for a RIGHT hand (cross order chosen
             so that flexion toward the palm comes out POSITIVE)
        e_y  = e_z × e_x       (radial-ulnar)
    Returns a (3,3) matrix R with R @ d giving d's components in this frame.
    """
    mid = 0.5 * (mcp_index + mcp_pinky)
    ex = mid - wrist
    nx = np.linalg.norm(ex)
    if nx < _EPS:
        raise ValueError("degenerate palm: wrist coincides with knuckle midpoint")
    ex = ex / nx
    # dorsal-oriented normal so flexion comes out positive; mirror for a left hand
    n = np.cross(mcp_pinky - wrist, mcp_index - wrist)
    if hand == "left":
        n = -n
    nn = np.linalg.norm(n)
    if nn < _EPS:
        raise ValueError("degenerate palm: knuckles collinear with wrist")
    ez = n / nn
    ez = ez - np.dot(ez, ex) * ex          # re-orthogonalise against e_x
    ez = ez / np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    return np.stack([ex, ey, ez])


def finger_flexion(points, finger, wrist_idx=WRIST):
    """Inter-segment flexion angles (rad) at MCP, PIP, DIP of one finger.
    Convention: angle between consecutive segment vectors, 0 = straight.
    The MCP value uses (MCP−wrist) as a crude metacarpal direction — fine as
    long as BOTH sides use the same definition (see palm-frame variant below).
    """
    L = LM[finger]
    wrist = points[wrist_idx]
    mcp, pip = points[L["mcp"]], points[L["pip"]]
    dip, tip = points[L["dip"]], points[L["tip"]]
    meta = mcp - wrist
    prox = pip - mcp
    mid = dip - pip
    dist = tip - dip
    return {
        "mcp": included_angle(meta, prox),
        "pip": included_angle(prox, mid),
        "dip": included_angle(mid, dist),
    }


def mcp_flex_abd(points, finger, basis):
    """MCP flexion & abduction (rad) of the proximal phalanx, resolved in the
    palm frame `basis` (from palm_basis). Flexion = bending toward the palm
    (around e_y); abduction = sideways spread (around e_z). This is the clean
    2-DoF MCP decomposition; use it instead of finger_flexion()['mcp'] when you
    need flexion and abduction separated."""
    L = LM[finger]
    d = points[L["pip"]] - points[L["mcp"]]
    nd = np.linalg.norm(d)
    if nd < _EPS:
        return np.nan, np.nan
    dx, dy, dz = basis @ (d / nd)
    flexion = float(np.arctan2(-dz, dx))
    abduction = float(np.arctan2(dy, dx))
    return flexion, abduction


def hand_angles(points, wrist_idx=WRIST):
    """All per-finger angles for one 21×3 hand. Returns
    {finger: {'mcp','pip','dip','mcp_flex','mcp_abd'}} — 'mcp' is the plain
    inter-segment angle, 'mcp_flex'/'mcp_abd' the palm-frame decomposition."""
    basis = palm_basis(points[wrist_idx], points[LM["index"]["mcp"]],
                        points[LM["pinky"]["mcp"]])
    out = {}
    for f in FINGERS:
        ang = finger_flexion(points, f, wrist_idx)
        flex, abd = mcp_flex_abd(points, f, basis)
        ang["mcp_flex"], ang["mcp_abd"] = flex, abd
        out[f] = ang
    return out


# ── Point 1: rigid bone-length constraint (MKO-lite) ────────────────────────

def chain_indices(finger, with_metacarpal=False, wrist_idx=WRIST):
    """Landmark indices along a finger, MCP→tip (optionally wrist→MCP→tip)."""
    L = LM[finger]
    chain = [L["mcp"], L["pip"], L["dip"], L["tip"]]
    if with_metacarpal:
        chain = [wrist_idx] + chain
    return chain


def bone_lengths(seq, finger, with_metacarpal=False, wrist_idx=WRIST):
    """Median length of each bone over a sequence seq (T,21,3) — the rigid-bone
    estimate used as the constraint target."""
    chain = chain_indices(finger, with_metacarpal, wrist_idx)
    lens = []
    for a, b in zip(chain[:-1], chain[1:]):
        d = np.linalg.norm(seq[:, b] - seq[:, a], axis=1)
        lens.append(float(np.median(d)))
    return np.array(lens)


def constrain_chain(pts, lengths, n_iter=200):
    """Project a marker chain onto fixed bone lengths, staying close to the
    measured pts (a position-based / SHAKE relaxation of the constrained LS
    min Σ|p̂−p|² s.t. |p̂_k−p̂_{k−1}|=L_k). pts:(M,3) ordered along the chain,
    lengths:(M-1,). Returns the corrected (M,3).

    Removes the axial (length-changing) component of STA; by construction it
    does NOT touch the perpendicular component (that one leaves bone length
    unchanged) — which is exactly why STA must be *measured*, not corrected."""
    p = np.array(pts, dtype=float)
    for _ in range(n_iter):
        for k in range(len(lengths)):
            a, b = p[k], p[k + 1]
            d = b - a
            L = np.linalg.norm(d)
            if L < _EPS:
                continue
            corr = 0.5 * (L - lengths[k]) * d / L
            p[k] = a + corr
            p[k + 1] = b - corr
    return p


# ── Point 2: quantify STA (measure, do not correct) ─────────────────────────

def sta_bone_std(seq, finger, with_metacarpal=False, wrist_idx=WRIST):
    """σ_k = std of each bone's measured length over the sequence. A rigid bone
    gives 0; the observed value is the STA amplitude (axial component). Order-of-
    magnitude angle uncertainty of the reference: δθ_k ≈ σ_k / L_k (rad)."""
    chain = chain_indices(finger, with_metacarpal, wrist_idx)
    out = []
    for a, b in zip(chain[:-1], chain[1:]):
        d = np.linalg.norm(seq[:, b] - seq[:, a], axis=1)
        out.append(float(np.std(d)))
    return np.array(out)


def sta_flexion_slope(bone_len_series, flexion_series):
    """Slope b_k of |u_k| regressed on the joint flexion — the *systematic*
    STA (skin sliding with flexion). |b_k| ≈ 0 for a rigid bone."""
    x = np.asarray(flexion_series, dtype=float)
    y = np.asarray(bone_len_series, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


# ── Point 3: neutral-pose referencing (removes a constant placement bias) ────

def to_neutral(angles, neutral):
    """Δθ = θ − θ_neutral, per joint. A constant marker-placement bias cancels;
    you then compare angle CHANGES / ROM (Bland–Altman on Δθ). `angles` and
    `neutral` are matching {joint: value} dicts."""
    return {k: angles[k] - neutral[k] for k in angles if k in neutral}
