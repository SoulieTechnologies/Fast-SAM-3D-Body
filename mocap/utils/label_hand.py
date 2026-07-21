"""Geometric labelling of a flat, spread-hand seed frame: assign ~21 raw
markers to the 21 fastsam3d slots (per finger MCP/PIP/DIP/tip + wrist).

Works on the calibration pose (hand flat on a board, fingers spread). Strategy:
wrist = the isolated marker; then grow 5 chains inward from the 5 fingertips
(fingers are near-straight lines in the hand plane, so each chain follows
markers at a roughly constant lateral offset with decreasing distance-to-wrist).
Fingers are ordered thumb→pinky by the lateral position of their MCP, using
handedness. Validate with bone_sanity() before trusting the result.
"""
import numpy as np

from utils import hand_kinematics as hk


def _hand_frame(P, wrist):
    c = P.mean(0)
    Vt = np.linalg.svd(P - c)[2]
    long, lat, nrm = Vt[0], Vt[1], Vt[2]
    if (wrist - c) @ long > 0:            # orient so the wrist sits proximal (−)
        long = -long
    lat_sign = 1.0
    return c, long, lat, nrm, lat_sign


def label_flat_hand(P, handedness="right"):
    """P:(N,3) hand-cluster points (N≈21). Returns (slots, info):
    slots is a (21,3) array in fastsam3d order (NaN if unfilled); info holds the
    per-finger marker index chains for inspection."""
    N = len(P)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(D, 1e9)
    nn1 = D.min(axis=1)
    wrist_i = int(np.argmax(nn1))                 # most isolated = wrist

    idx = [i for i in range(N) if i != wrist_i]
    Q = P[idx]
    c, long, lat, nrm, _ = _hand_frame(P, P[wrist_i])
    along = (Q - c) @ long                         # proximal(−) → distal(+)
    lateral = (Q - c) @ lat
    dwrist = np.linalg.norm(Q - P[wrist_i], axis=1)

    # 5 fingertips = most-distal markers, laterally separated
    order = np.argsort(along)[::-1]
    tips = []
    for k in order:
        if all(abs(lateral[k] - lateral[t]) > 0.012 for t in tips):
            tips.append(k)
        if len(tips) == 5:
            break

    used = set(tips)
    chains = {}
    for tip in tips:
        chain = [tip]
        cur = tip
        for _ in range(3):                         # tip→DIP→PIP→MCP
            best, bcost = None, 1e9
            for j in range(len(Q)):
                if j in used:
                    continue
                step = along[cur] - along[j]        # must go proximal
                d = np.linalg.norm(Q[j] - Q[cur])
                if step <= 0 or not (0.012 <= d <= 0.055):
                    continue
                cost = d + 3.0 * abs(lateral[j] - lateral[cur])
                if cost < bcost:
                    best, bcost = j, cost
            if best is None:                        # fall back to nearest proximal
                cand = [j for j in range(len(Q)) if j not in used
                        and along[j] < along[cur]]
                if not cand:
                    break
                best = min(cand, key=lambda j: np.linalg.norm(Q[j] - Q[cur]))
            chain.append(best)
            used.add(best)
            cur = best
        chains[tip] = chain                         # [tip, DIP, PIP, MCP]

    # order fingers thumb→pinky by MCP lateral position (right hand: thumb = +lat
    # side... determined empirically, flip via handedness)
    mcp_lat = {tip: lateral[ch[-1]] for tip, ch in chains.items()}
    sign = 1 if handedness == "right" else -1
    fingers_by_lat = sorted(chains, key=lambda t: sign * mcp_lat[t])
    names = ["thumb", "index", "middle", "ring", "pinky"]

    slots = np.full((hk.N_LM, 3), np.nan)
    slots[hk.WRIST] = P[wrist_i]
    chain_out = {}
    for name, tip in zip(names, fingers_by_lat):
        ch = chains[tip]                            # [tip, DIP, PIP, MCP]
        L = hk.LM[name]
        for slot_key, m in zip(("tip", "dip", "pip", "mcp"), ch):
            slots[L[slot_key]] = Q[m]
        chain_out[name] = [idx[m] for m in ch]
    return slots, {"wrist": wrist_i, "chains": chain_out}


def bone_sanity(slots):
    """Per-finger bone lengths (mm) for a labelled (21,3) frame; used to check a
    labelling before trusting it. Returns {finger: [wrist-MCP, MCP-PIP, PIP-DIP,
    DIP-tip]}."""
    out = {}
    for f in hk.FINGERS:
        L = hk.LM[f]
        chain = [hk.WRIST, L["mcp"], L["pip"], L["dip"], L["tip"]]
        out[f] = [float(np.linalg.norm(slots[b] - slots[a]) * 1000)
                  for a, b in zip(chain[:-1], chain[1:])]
    return out
