"""Parse a Motive marker-export CSV into a clean per-frame point cloud, then
re-track markers by SPATIAL CONTINUITY (ignoring Motive's marker IDs).

Why ignore the IDs: Motive dumps every unlabeled trajectory in its own column
and hands out a NEW id each time a marker reappears after occlusion — a 3-marker
take can span 60+ id columns. So the ids are useless as anatomical labels. We
flatten each frame to the set of populated 3D points and rebuild persistent
tracks from geometry, exactly the "classify points by their position in the
cloud" approach. Occlusions become NaN gaps; ghost detections become short,
low-occupancy tracks that get dropped.

Pipeline:
    times, frames = read_motive_csv(path)      # frames: list of (n_i, 3)
    tracks = build_tracklets(frames, dt=1/fps) # (T, K, 3), NaN where missing
    tracks = keep_top_tracks(tracks, k)        # drop ghosts
    tracks = order_along_axis(tracks)          # anatomical order along a finger
"""
import csv

import numpy as np


def read_motive_csv(path):
    """Return (meta, times, frames):
        meta   dict with 'fps', 'units', 'take'
        times  (T,) seconds
        frames list of (n_i, 3) arrays — the populated markers of each frame,
               order-agnostic (ids discarded).
    """
    rows = list(csv.reader(open(path)))
    meta = _parse_meta(rows[0]) if rows else {}
    # find the "Frame,Time,X,Y,Z,..." header row
    h = next(i for i, r in enumerate(rows)
             if r[:2] == ["Frame", "Time"] or (r and r[0] == "Frame"))
    ncol = len(rows[h])
    nmark = (ncol - 2) // 3
    times, frames = [], []
    for r in rows[h + 1:]:
        if not r or r[0] == "" or r[0] == "Frame":
            continue
        try:
            times.append(float(r[1]))
        except (ValueError, IndexError):
            continue
        pts = []
        for m in range(nmark):
            c = 2 + 3 * m
            x = r[c] if c < len(r) else ""
            if x == "" or r[c + 1] == "" or r[c + 2] == "":
                continue
            pts.append((float(x), float(r[c + 1]), float(r[c + 2])))
        frames.append(np.array(pts, dtype=float).reshape(-1, 3))
    return meta, np.array(times), frames


_JOINT_ALIASES = {
    "tip": "tip", "nail": "tip", "fingertip": "tip", "end": "tip",
    "dip": "dip", "distal": "dip", "ip": "dip",
    "pip": "pip", "middle": "pip", "proximalinter": "pip",
    "mcp": "mcp", "knuckle": "mcp", "cmc": "mcp", "base": "mcp",
}
_FINGER_ALIASES = {
    "thumb": "thumb", "thb": "thumb", "pollex": "thumb",
    "index": "index", "idx": "index", "ind": "index", "fore": "index",
    "middle": "middle", "mid": "middle", "long": "middle",
    "ring": "ring", "rng": "ring",
    "pinky": "pinky", "little": "pinky", "pnk": "pinky", "small": "pinky",
}


def default_resolver(label):
    """Map a marker label string → 'wrist' or '<finger>_<joint>' (fastsam3d
    slot), or None if it doesn't look like a hand marker. Case/spacing-robust;
    handles many naming conventions (RIndexPIP, index_pip, Idx3, wrist, ...)."""
    s = "".join(c for c in label.lower() if c.isalnum())
    if not s:
        return None
    if "wrist" in s or "carp" in s:
        return "wrist"
    finger = next((v for k, v in _FINGER_ALIASES.items() if k in s), None)
    if finger is None:
        return None
    # thumb: cmc/base→mcp, its "pip"=MCP, "dip"=IP
    joint = None
    for k, v in _JOINT_ALIASES.items():
        if k in s:
            joint = v
            break
    if joint is None:                       # numeric convention 1..4 = MCP..tip
        for d, v in (("4", "tip"), ("3", "dip"), ("2", "pip"), ("1", "mcp")):
            if d in s:
                joint = v
                break
    return f"{finger}_{joint}" if joint else None


def read_labeled(path, resolver=default_resolver):
    """Read a LABELLED Motive/SOMA export → (meta, times, seq) with seq a
    (T,21,3) array in fastsam3d slot order (NaN where a marker is missing).
    Uses the marker-name header row + `resolver` to place each named column.
    Unresolved labels are ignored (reported via labels_seen())."""
    from utils import hand_kinematics as hk
    rows = list(csv.reader(open(path)))
    meta = _parse_meta(rows[0]) if rows else {}
    h = next(i for i, r in enumerate(rows) if r and r[0] == "Frame")
    names_row = _find_names_row(rows, h)
    nmark = (len(rows[h]) - 2) // 3
    slot = {"wrist": hk.WRIST}
    for f in hk.FINGERS:
        for j, key in (("tip", "tip"), ("dip", "dip"), ("pip", "pip"),
                       ("mcp", "mcp")):
            slot[f"{f}_{j}"] = hk.LM[f][key]
    col_slot = {}
    for m in range(nmark):
        lab = names_row[2 + 3 * m] if names_row else ""
        s = resolver(lab) if lab else None
        if s in slot:
            col_slot[m] = slot[s]
    times, seq = [], []
    for r in rows[h + 1:]:
        if not r or r[0] in ("", "Frame"):
            continue
        try:
            times.append(float(r[1]))
        except (ValueError, IndexError):
            continue
        P = np.full((hk.N_LM, 3), np.nan)
        for m, sl in col_slot.items():
            c = 2 + 3 * m
            if c + 2 < len(r) and r[c] != "" and r[c + 1] != "" and r[c + 2] != "":
                P[sl] = (float(r[c]), float(r[c + 1]), float(r[c + 2]))
        seq.append(P)
    return meta, np.array(times), np.array(seq)


def labels_seen(path, resolver=default_resolver):
    """List (raw_label, resolved_slot) for every marker column — use to check a
    file's naming convention before trusting read_labeled()."""
    from utils import hand_kinematics as hk  # noqa: F401
    rows = list(csv.reader(open(path)))
    h = next(i for i, r in enumerate(rows) if r and r[0] == "Frame")
    names_row = _find_names_row(rows, h)
    nmark = (len(rows[h]) - 2) // 3
    out = []
    for m in range(nmark):
        lab = names_row[2 + 3 * m] if names_row else ""
        out.append((lab, resolver(lab) if lab else None))
    return out


def _find_names_row(rows, h):
    """The marker-name header row sits a few rows above the Frame row; pick the
    nearest one above h that has the same column count and non-empty labels."""
    ncol = len(rows[h])
    for i in range(h - 1, max(-1, h - 6), -1):
        r = rows[i]
        if len(r) == ncol and any(r[2:]):
            return r
    return None


def _parse_meta(row):
    kv = {row[i]: row[i + 1] for i in range(0, len(row) - 1, 2)}
    fps = kv.get("Capture Frame Rate") or kv.get("Export Frame Rate")
    return {"fps": float(fps) if fps else None,
            "units": kv.get("Length Units", "Meters"),
            "take": kv.get("Take Name", "")}


def build_tracklets(frames, dt, max_speed=3.0, max_gap=30, max_points=None):
    """Greedy nearest-neighbour tracking with a velocity gate and gap tolerance.

    frames    list of (n_i,3) point clouds (from read_motive_csv)
    dt        seconds per frame
    max_speed m/s — a marker can't move more than max_speed*dt between frames
    max_gap   frames a track may stay unmatched (occluded) before it closes
    max_points if set, ignore frames with more than this many detections
              (garbage/reflection bursts); None keeps all.

    Returns tracks (T, K, 3) with NaN where a track has no point that frame.
    Tracks are in discovery order; use keep_top_tracks/order_along_axis next.
    """
    gate = max_speed * dt
    T = len(frames)
    active = []   # each: dict(last_xyz, vel, last_t, gap, col: list length T)
    for t, pts in enumerate(frames):
        if max_points is not None and len(pts) > max_points:
            pts = np.empty((0, 3))
        used = np.zeros(len(pts), dtype=bool)
        # predict and match existing tracks to the nearest unused detection
        for tr in active:
            if tr["gap"] > max_gap:
                continue
            pred = tr["last"] + tr["vel"] * (tr["gap"] + 1)
            if len(pts) == 0:
                tr["gap"] += 1
                continue
            d = np.linalg.norm(pts - pred, axis=1)
            d[used] = np.inf
            j = int(np.argmin(d))
            # allow a larger gate after an occlusion (prediction drifts)
            if d[j] <= gate * (1 + tr["gap"]):
                tr["vel"] = (pts[j] - tr["last"]) / (tr["gap"] + 1)
                tr["last"] = pts[j]
                tr["gap"] = 0
                tr["xyz"][t] = pts[j]
                used[j] = True
            else:
                tr["gap"] += 1
        # unmatched detections seed new tracks
        for j in range(len(pts)):
            if used[j]:
                continue
            xyz = np.full((T, 3), np.nan)
            xyz[t] = pts[j]
            active.append({"last": pts[j].copy(), "vel": np.zeros(3),
                           "gap": 0, "xyz": xyz})
    if not active:
        return np.full((T, 0, 3), np.nan)
    return np.stack([tr["xyz"] for tr in active], axis=1)


def track_fixed_k(frames, k, dt, max_speed=3.0, reacquire_mult=10,
                  seed_max_diam=0.20):
    """Track a KNOWN number k of persistent markers as anchored slots — far
    more robust than free tracklets when k is small and dropout is heavy.

    Seeds the k slots from the first frame that has exactly k points, then each
    frame greedily matches the present points to slots by predicted position
    (last + velocity), with a gate that widens after an occlusion so a marker
    is re-acquired instead of spawning a ghost track. Unmatched slots → NaN this
    frame; unmatched points (reflections) are ignored.

    Returns (T, k, 3) with NaN gaps. Apply order_along_axis() for anatomy.
    """
    gate = max_speed * dt
    T = len(frames)
    out = np.full((T, k, 3), np.nan)
    slots = None
    vel = np.zeros((k, 3))
    miss = np.zeros(k, dtype=int)
    for t, pts in enumerate(frames):
        if slots is None:
            # seed only on exactly k points that form a hand-sized cluster —
            # rejects "1 real marker + reflections scattered across the room"
            if len(pts) == k:
                diam = np.max([np.linalg.norm(a - b) for a in pts for b in pts])
                if diam <= seed_max_diam:
                    slots = pts.astype(float).copy()
                    out[t] = slots
                    miss[:] = 0
            continue
        pred = slots + vel
        assigned = -np.ones(k, dtype=int)
        if len(pts):
            D = np.linalg.norm(pred[:, None, :] - pts[None, :, :], axis=2)
            usedp = set()
            for si in np.argsort(D.min(axis=1)):   # closest slots first
                gj = [D[si, j] if j not in usedp else np.inf
                      for j in range(len(pts))]
                j = int(np.argmin(gj))
                g = min(gate * (1 + miss[si]), gate * reacquire_mult)
                if gj[j] <= g:
                    assigned[si] = j
                    usedp.add(j)
        for si in range(k):
            if assigned[si] >= 0:
                p = pts[assigned[si]]
                vel[si] = (p - slots[si]) / (miss[si] + 1)
                slots[si] = p
                out[t, si] = p
                miss[si] = 0
            else:
                miss[si] += 1
    return out


def track_chain(frames, k, dt, max_speed=3.0, seed_max_diam=0.20,
                reacquire_mult=12, bone_weight=1.0, bone_tol=0.4):
    """Chain-aware fixed-k tracker: seeds the full labelled set from one frame
    with all k markers, fixes the anatomical CHAIN ORDER there, and re-acquires
    a marker that reappears after occlusion by BOTH its predicted position AND
    its bone length to the tracked neighbours (the finger is a chain of ~rigid
    bones). This is the "start from the full cloud, relabel the lost ones when
    they come back" idea — the bone-length prior disambiguates which lost marker
    returned, far better than position alone after a long gap.

    Returns (T, k, 3) already ordered proximal→distal (chain order), NaN gaps.
    """
    gate = max_speed * dt
    T = len(frames)
    out = np.full((T, k, 3), np.nan)
    slots = None
    vel = np.zeros((k, 3))
    miss = np.zeros(k, dtype=int)
    bone = None                                   # (k-1,) reference bone lengths
    for t, pts in enumerate(frames):
        if slots is None:
            if len(pts) == k:
                diam = np.max([np.linalg.norm(a - b) for a in pts for b in pts])
                if diam <= seed_max_diam:
                    c = pts.mean(0)
                    axis = np.linalg.svd(pts - c)[2][0]
                    order = np.argsort((pts - c) @ axis)   # chain order
                    slots = pts[order].astype(float).copy()
                    bone = np.linalg.norm(np.diff(slots, axis=0), axis=1)
                    out[t] = slots
            continue
        pred = slots + vel
        n = len(pts)
        assigned = -np.ones(k, dtype=int)
        if n:
            cost = np.full((k, n), np.inf)
            for si in range(k):
                g = min(gate * (1 + miss[si]), gate * reacquire_mult)
                dpos = np.linalg.norm(pts - pred[si], axis=1)
                for j in range(n):
                    if dpos[j] > g:
                        continue                  # position is the HARD gate
                    # soft bone-length prior, only from neighbours that are
                    # currently tracked (miss==0) — during a whole-hand dropout
                    # no neighbour is fresh, so this degrades to position-only
                    pen = 0.0
                    for nb, bi in ((si - 1, si - 1), (si + 1, si)):
                        if 0 <= nb < k and miss[nb] == 0:
                            pen += abs(np.linalg.norm(pts[j] - pred[nb]) - bone[bi])
                    cost[si, j] = dpos[j] + bone_weight * pen
            # greedy: assign lowest-cost slot/detection pairs first
            order = np.dstack(np.unravel_index(np.argsort(cost, axis=None),
                                               cost.shape))[0]
            takens, takenj = set(), set()
            for si, j in order:
                if cost[si, j] == np.inf:
                    break
                if si in takens or j in takenj:
                    continue
                assigned[si] = j
                takens.add(si)
                takenj.add(j)
        for si in range(k):
            if assigned[si] >= 0:
                p = pts[assigned[si]]
                vel[si] = (p - slots[si]) / (miss[si] + 1)
                slots[si] = p
                out[t, si] = p
                miss[si] = 0
            else:
                miss[si] += 1
    return out


def fill_short_gaps(series, max_gap):
    """Linearly interpolate NaN runs no longer than max_gap frames; leave
    longer gaps as NaN (never fabricate data across a real dropout). Works on
    a 1-D array (angle) or (T,3) points along axis 0. Returns a copy."""
    s = np.array(series, dtype=float)
    flat = s.reshape(len(s), -1)
    good = np.isfinite(flat).all(axis=1)
    if good.sum() < 2:
        return s
    idx = np.where(good)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                flat[t] = (1 - w) * flat[a] + w * flat[b]
    return flat.reshape(s.shape)


def smooth_series(series, win=5):
    """Rolling median (odd window) that ignores NaN — knocks out isolated
    single-frame spikes without bridging real gaps."""
    s = np.asarray(series, dtype=float)
    h = win // 2
    out = s.copy()
    for t in range(len(s)):
        lo, hi = max(0, t - h), min(len(s), t + h + 1)
        w = s[lo:hi]
        w = w[np.isfinite(w)]
        if w.size:
            out[t] = np.median(w)
    return out


def reject_position_spikes(seq, win=15, thr=0.03):
    """NaN a marker's position when it jumps more than `thr` metres from its
    local median position (window `win`). A wrong chain assignment teleports a
    marker (right bone length, wrong place) — undetectable by bone length but
    obvious as a position spike. Returns a cleaned copy of seq (T,k,3)."""
    out = np.array(seq, dtype=float)
    T, k = out.shape[:2]
    h = win // 2
    for s in range(k):
        col = out[:, s, :]
        for t in range(T):
            if not np.isfinite(col[t, 0]):
                continue
            w = col[max(0, t - h):min(T, t + h + 1)]
            w = w[np.isfinite(w[:, 0])]
            if len(w) >= 4:
                med = np.median(w, axis=0)
                if np.linalg.norm(col[t] - med) > thr:
                    out[t, s] = np.nan
    return out


def reject_spikes(series, win=11, thr=25.0):
    """NaN points that deviate more than `thr` from the local rolling median
    (window `win`). For slow motions this removes mislabel spikes while keeping
    genuine (smooth) movement. Operates on a 1-D angle series."""
    s = np.asarray(series, dtype=float)
    out = s.copy()
    h = win // 2
    for t in range(len(s)):
        if not np.isfinite(s[t]):
            continue
        w = s[max(0, t - h):min(len(s), t + h + 1)]
        w = w[np.isfinite(w)]
        if w.size >= 3 and abs(s[t] - np.median(w)) > thr:
            out[t] = np.nan
    return out


def gap_report(series):
    """(#missing frames, longest gap length) for a 1-D series."""
    miss = ~np.isfinite(np.asarray(series, dtype=float))
    longest = cur = 0
    for m in miss:
        cur = cur + 1 if m else 0
        longest = max(longest, cur)
    return int(miss.sum()), longest


def _track_dir(frames, out, start, step, slots, dt, max_speed, reacquire_mult):
    """Propagate `slots` (k,3) from frame `start` in direction `step` (+1/-1),
    writing matched points into `out`. Position-only greedy match with a gate
    that widens after occlusion (same core as track_fixed_k)."""
    k = len(slots)
    vel = np.zeros((k, 3))
    miss = np.zeros(k, dtype=int)
    gate = max_speed * dt
    t = start + step
    while 0 <= t < len(frames):
        pts = frames[t]
        pred = slots + vel
        assigned = -np.ones(k, dtype=int)
        if len(pts):
            D = np.linalg.norm(pred[:, None, :] - pts[None, :, :], axis=2)
            usedp = set()
            for si in np.argsort(D.min(axis=1)):
                gj = [D[si, j] if j not in usedp else np.inf
                      for j in range(len(pts))]
                j = int(np.argmin(gj))
                g = min(gate * (1 + miss[si]), gate * reacquire_mult)
                if gj[j] <= g:
                    assigned[si] = j
                    usedp.add(j)
        for si in range(k):
            if assigned[si] >= 0:
                p = pts[assigned[si]]
                vel[si] = (p - slots[si]) / (miss[si] + 1)
                slots[si] = p
                out[t, si] = p
                miss[si] = 0
            else:
                miss[si] += 1
        t += step


def track_from_seed(frames, seed, seed_idx, dt, max_speed=3.0,
                    reacquire_mult=12):
    """Track k markers through a take from a MANUALLY LABELLED seed frame:
    seed (k,3) are the slot positions at frame seed_idx (e.g. the 21 fastsam3d
    slots labelled by hand on the flat calibration pose). Propagates forward AND
    backward from the seed so the whole take is covered. Returns (T,k,3)."""
    T, k = len(frames), len(seed)
    out = np.full((T, k, 3), np.nan)
    out[seed_idx] = seed
    _track_dir(frames, out, seed_idx, +1, np.array(seed, float),
               dt, max_speed, reacquire_mult)
    _track_dir(frames, out, seed_idx, -1, np.array(seed, float),
               dt, max_speed, reacquire_mult)
    return out


def _hand_neighbors(seed):
    """From a labelled 21-slot seed, the bone graph (fastsam3d topology): each
    slot -> [(neighbour_slot, bone_length)]. Bones (wrist-MCP, MCP-PIP, PIP-DIP,
    DIP-tip per finger) are ~rigid, so their seed lengths anchor the tracking."""
    from utils import hand_kinematics as hk
    nb = {i: [] for i in range(hk.N_LM)}
    for f in hk.FINGERS:
        L = hk.LM[f]
        chain = [hk.WRIST, L["mcp"], L["pip"], L["dip"], L["tip"]]
        for a, b in zip(chain[:-1], chain[1:]):
            d = float(np.linalg.norm(seed[a] - seed[b]))
            nb[a].append((b, d))
            nb[b].append((a, d))
    return nb


def _track_dir_struct(frames, out, start, step, slots, nb, dt, max_speed,
                      reacquire_mult, bone_w):
    """Structure-aware propagation: global (Hungarian) assignment of detections
    to the 21 slots, cost = position error + bone-length deviation to tracked
    neighbours. The rigid-bone prior stops the position-only swaps."""
    from scipy.optimize import linear_sum_assignment
    k = len(slots)
    vel = np.zeros((k, 3))
    miss = np.zeros(k, dtype=int)
    gate = max_speed * dt
    t = start + step
    BIG = 1e6
    while 0 <= t < len(frames):
        pts = frames[t]
        n = len(pts)
        assigned = -np.ones(k, dtype=int)
        if n:
            pred = slots + vel
            C = np.full((k, n), BIG)
            for si in range(k):
                g = min(gate * (1 + miss[si]), gate * reacquire_mult)
                dpos = np.linalg.norm(pts - pred[si], axis=1)
                near = np.where(dpos <= g)[0]
                for j in near:
                    pen = 0.0
                    for nbi, L in nb[si]:
                        if miss[nbi] == 0:
                            pen += abs(np.linalg.norm(pts[j] - pred[nbi]) - L)
                    C[si, j] = dpos[j] + bone_w * pen
            ri, ci = linear_sum_assignment(C)
            for s, j in zip(ri, ci):
                if C[s, j] < BIG:
                    assigned[s] = j
        for si in range(k):
            if assigned[si] >= 0:
                p = pts[assigned[si]]
                vel[si] = (p - slots[si]) / (miss[si] + 1)
                slots[si] = p
                out[t, si] = p
                miss[si] = 0
            else:
                miss[si] += 1
        t += step


def _kabsch(A, B):
    """Rigid R,t minimizing ||R@A.T ... || i.e. maps A→B (both (n,3))."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def _track_dir_rigid(frames, out, start, step, prev, nb, dt, max_speed,
                     reacquire_mult, bone_w, stop=None):
    """Rigid-frame propagation: each frame (1) a loose position match to find
    the hand's rigid motion (Kabsch on inliers), (2) predict all slots with that
    rigid transform, (3) a TIGHT match around the rigid prediction + bone prior.
    Separating hand transport from finger articulation kills the swaps.
    Stops before frame `stop` (exclusive) in the stepping direction if given."""
    from scipy.optimize import linear_sum_assignment
    k = len(prev)
    miss = np.zeros(k, dtype=int)
    gate = max_speed * dt
    BIG = 1e6
    t = start + step
    while 0 <= t < len(frames) and (stop is None or
                                    (t < stop if step > 0 else t > stop)):
        pts = frames[t]
        n = len(pts)
        pred = prev.copy()
        if n >= 6:
            # pass 1: loose match prev→pts to estimate the rigid hand motion
            D = np.linalg.norm(prev[:, None, :] - pts[None, :, :], axis=2)
            C = np.where(D <= gate * reacquire_mult, D, BIG)
            ri, ci = linear_sum_assignment(C)
            pairs = [(s, j) for s, j in zip(ri, ci) if C[s, j] < BIG]
            if len(pairs) >= 6:
                A = np.array([prev[s] for s, _ in pairs])
                B = np.array([pts[j] for _, j in pairs])
                res = np.linalg.norm(A - B, axis=1)
                inl = res <= max(np.median(res), gate)
                if inl.sum() >= 6:
                    R, tt = _kabsch(A[inl], B[inl])
                    pred = prev @ R.T + tt
        # pass 2: tight match around the rigid prediction + bone-length prior
        assigned = -np.ones(k, dtype=int)
        if n:
            C = np.full((k, n), BIG)
            for si in range(k):
                g = min(gate * (1 + miss[si]), gate * reacquire_mult)
                dpos = np.linalg.norm(pts - pred[si], axis=1)
                for j in np.where(dpos <= g)[0]:
                    pen = sum(abs(np.linalg.norm(pts[j] - pred[nbi]) - L)
                              for nbi, L in nb[si] if miss[nbi] == 0)
                    C[si, j] = dpos[j] + bone_w * pen
            ri, ci = linear_sum_assignment(C)
            for s, j in zip(ri, ci):
                if C[s, j] < BIG:
                    assigned[s] = j
        for si in range(k):
            if assigned[si] >= 0:
                prev[si] = pts[assigned[si]]
                out[t, si] = prev[si]
                miss[si] = 0
            else:
                prev[si] = pred[si]           # carry missing slot rigidly
                miss[si] += 1
        t += step


def track_hand_rigid(frames, seed, seed_idx, dt, max_speed=3.0,
                     reacquire_mult=12, bone_w=5.0):
    """Track 21 hand markers from a labelled seed with a per-frame rigid-hand
    model (Kabsch) + bone prior — robust to fast hand transport. Returns
    (T,21,3)."""
    T, k = len(frames), len(seed)
    out = np.full((T, k, 3), np.nan)
    out[seed_idx] = seed
    nb = _hand_neighbors(seed)
    _track_dir_rigid(frames, out, seed_idx, +1, np.array(seed, float), nb,
                     dt, max_speed, reacquire_mult, bone_w)
    _track_dir_rigid(frames, out, seed_idx, -1, np.array(seed, float), nb,
                     dt, max_speed, reacquire_mult, bone_w)
    return out


def _hand_constraints(seed):
    """Rigid distance constraints from the seed skeleton: the PALM (wrist + the
    four finger MCPs) is a rigid body (all pairwise distances), plus each
    finger's phalanx bones and the wrist->thumb-CMC bone. These fixed lengths
    are the model that recovers occluded points and rejects swaps/ghosts."""
    from utils import hand_kinematics as hk
    palm = [hk.WRIST] + [hk.LM[f]["mcp"] for f in
                         ("index", "middle", "ring", "pinky")]
    cons = []
    for i in range(len(palm)):
        for j in range(i + 1, len(palm)):
            cons.append((palm[i], palm[j],
                         float(np.linalg.norm(seed[palm[i]] - seed[palm[j]]))))
    for f in hk.FINGERS:
        L = hk.LM[f]
        chain = [L["mcp"], L["pip"], L["dip"], L["tip"]]
        if f == "thumb":
            chain = [hk.WRIST] + chain
        for a, b in zip(chain[:-1], chain[1:]):
            cons.append((a, b, float(np.linalg.norm(seed[a] - seed[b]))))
    return cons


def _track_dir_skel(frames, out, obs, start, step, prev, cons, dt, max_speed,
                    reacquire_mult, n_iter, stop):
    """Skeleton-model propagation (position-based). Each frame: associate
    detections to nodes (Hungarian, gated) -> anchor matched nodes -> project
    the rigid distance constraints (fills occluded nodes at model-consistent
    positions, immovable anchors). Unassociated detections (ghosts) are dropped;
    nodes with no detection are recovered from the skeleton."""
    from scipy.optimize import linear_sum_assignment
    k = len(prev)
    vel = np.zeros((k, 3))
    gate = max_speed * dt
    BIG = 1e6
    t = start + step
    while 0 <= t < len(frames) and (stop is None or
                                    (t < stop if step > 0 else t > stop)):
        pts = frames[t]
        pred = prev + vel
        anchored = np.zeros(k, dtype=bool)
        p = pred.copy()
        if len(pts):
            D = np.linalg.norm(pred[:, None, :] - pts[None, :, :], axis=2)
            C = np.where(D <= gate * reacquire_mult, D, BIG)
            ri, ci = linear_sum_assignment(C)
            for s, j in zip(ri, ci):
                if C[s, j] < BIG:
                    p[s] = pts[j]
                    anchored[s] = True
        # project rigid constraints: anchors immovable (inv-mass 0), others free
        for _ in range(n_iter):
            for a, b, L in cons:
                d = p[b] - p[a]
                dist = np.linalg.norm(d)
                if dist < 1e-9:
                    continue
                wa = 0.0 if anchored[a] else 1.0
                wb = 0.0 if anchored[b] else 1.0
                if wa + wb == 0:
                    continue
                corr = (dist - L) * d / dist
                p[a] = p[a] + wa / (wa + wb) * corr
                p[b] = p[b] - wb / (wa + wb) * corr
        vel = 0.6 * vel + 0.4 * (p - prev)
        prev = p
        for si in range(k):
            out[t, si] = p[si]
            obs[t, si] = anchored[si]
        t += step


def track_hand_skeleton(frames, seed, seed_idx, dt, max_speed=3.0,
                        reacquire_mult=12, n_iter=8):
    """Fit the seed hand SKELETON (rigid palm + fixed bones) to each frame's
    markers, recovering occluded points from the model and rejecting ghosts.
    Returns (seq (T,21,3), observed (T,21) bool = node backed by a real
    detection this frame vs recovered from the skeleton)."""
    T, k = len(frames), len(seed)
    out = np.full((T, k, 3), np.nan)
    obs = np.zeros((T, k), dtype=bool)
    out[seed_idx] = seed
    obs[seed_idx] = True
    cons = _hand_constraints(seed)
    _track_dir_skel(frames, out, obs, seed_idx, +1, np.array(seed, float),
                    cons, dt, max_speed, reacquire_mult, n_iter, None)
    _track_dir_skel(frames, out, obs, seed_idx, -1, np.array(seed, float),
                    cons, dt, max_speed, reacquire_mult, n_iter, None)
    return out, obs


def _fit_palm(pts, seed_palm, pred_palm, gate, tol_match=0.015, n_ransac=200,
              rng=None):
    """Register the rigid palm pentagon (seed_palm, 5x3) to the detection cloud
    `pts`, returning (R, t, match) where match[i] is the detection index for
    palm point i (or -1). Fast path: match the previous prediction; RANSAC
    fallback recovers the pose from scratch (independent of the past) when the
    fast match is poor — this is what lets the tracker recover after drift."""
    from scipy.optimize import linear_sum_assignment
    m = len(seed_palm)
    n = len(pts)
    if n < 3:
        return None
    BIG = 1e6

    def match_to(target, tol):
        """Injective nearest match (Hungarian) target(m,3) -> pts, gated by tol."""
        D = np.linalg.norm(target[:, None, :] - pts[None, :, :], axis=2)
        C = np.where(D <= tol, D, BIG)
        ri, ci = linear_sum_assignment(C)
        mt = [-1] * m
        for i, j in zip(ri, ci):
            if C[i, j] < BIG:
                mt[i] = int(j)
        return mt

    def refine(match):
        idx = [i for i in range(m) if match[i] >= 0]
        if len(idx) < 3:
            return None
        R, t = _kabsch(seed_palm[idx], pts[[match[i] for i in idx]])
        proj = seed_palm @ R.T + t
        res = np.mean([np.linalg.norm(proj[i] - pts[match[i]]) for i in idx])
        return R, t, res

    # fast path: injective match to the predicted palm
    match = match_to(pred_palm, gate)
    if sum(x >= 0 for x in match) >= 4:
        r = refine(match)
        if r and r[2] < tol_match:
            return r[0], r[1], match

    # RANSAC recovery: sample 3 palm<->det correspondences, score injective inliers
    rng = rng or np.random.default_rng(0)
    best, best_inl = None, 3
    for _ in range(n_ransac):
        si = rng.choice(m, 3, replace=False)
        dj = rng.choice(n, 3, replace=False)
        R, t = _kabsch(seed_palm[si], pts[dj])
        mt = match_to(seed_palm @ R.T + t, tol_match)
        inl = sum(x >= 0 for x in mt)
        if inl > best_inl:
            best_inl, best = inl, mt
    if best is not None:
        r = refine(best)
        if r:
            return r[0], r[1], best
    return None


def _track_dir_palm(frames, out, obs, start, step, prev, seed, cons, palm_slots,
                    dt, max_speed, reacquire_mult, n_iter, stop):
    from scipy.optimize import linear_sum_assignment
    k = len(prev)
    vel = np.zeros((k, 3))
    gate = max_speed * dt
    seed_palm = seed[palm_slots]
    rng = np.random.default_rng(0)
    BIG = 1e6
    t = start + step
    while 0 <= t < len(frames) and (stop is None or
                                    (t < stop if step > 0 else t > stop)):
        pts = frames[t]
        p = prev + vel
        anchored = np.zeros(k, dtype=bool)
        used = set()
        fit = _fit_palm(pts, seed_palm, p[palm_slots],
                        gate * reacquire_mult, rng=rng) if len(pts) else None
        rigid = p.copy()
        if fit is not None:
            R, tt, match = fit
            rigid = seed @ R.T + tt              # gross pose for all slots
            jump = np.linalg.norm(rigid[palm_slots[0]] - prev[palm_slots[0]])
            for i, slot in enumerate(palm_slots):
                if match[i] >= 0:
                    p[slot] = pts[match[i]]
                    anchored[slot] = True
                    used.add(match[i])
            if jump > 0.05:                       # recovery jump → reset fingers
                for si in range(k):
                    if not anchored[si]:
                        p[si] = rigid[si]
        # assign the remaining (finger) slots to the remaining detections
        free = [si for si in range(k) if not anchored[si]]
        dets = [j for j in range(len(pts)) if j not in used]
        if free and dets:
            C = np.full((len(free), len(dets)), BIG)
            for a, si in enumerate(free):
                for b, j in enumerate(dets):
                    d = np.linalg.norm(pts[j] - p[si])
                    if d <= gate * reacquire_mult:
                        C[a, b] = d
            ri, ci = linear_sum_assignment(C)
            for a, b in zip(ri, ci):
                if C[a, b] < BIG:
                    p[free[a]] = pts[dets[b]]
                    anchored[free[a]] = True
        # project the rigid constraints (fills unanchored nodes)
        for _ in range(n_iter):
            for a, b, L in cons:
                d = p[b] - p[a]
                dist = np.linalg.norm(d)
                if dist < 1e-9:
                    continue
                wa = 0.0 if anchored[a] else 1.0
                wb = 0.0 if anchored[b] else 1.0
                if wa + wb == 0:
                    continue
                corr = (dist - L) * d / dist
                p[a] = p[a] + wa / (wa + wb) * corr
                p[b] = p[b] - wb / (wa + wb) * corr
        vel = 0.6 * vel + 0.4 * (p - prev)
        prev = p.copy()
        out[t] = p
        obs[t] = anchored
        t += step


def track_hand_palm(frames, seed, seed_idx, dt, max_speed=3.0,
                    reacquire_mult=12, n_iter=8):
    """Global palm-registration tracker: each frame re-registers the rigid palm
    pentagon (wrist + 4 finger MCPs) to the cloud — from scratch via RANSAC when
    needed — so the hand pose RECOVERS after drift instead of persisting a wrong
    labelling. Fingers are then assigned from the corrected MCPs and the
    skeleton constraints fill occluded points. Returns (seq, observed)."""
    from utils import hand_kinematics as hk
    T, k = len(frames), len(seed)
    out = np.full((T, k, 3), np.nan)
    obs = np.zeros((T, k), dtype=bool)
    out[seed_idx] = seed
    obs[seed_idx] = True
    cons = _hand_constraints(seed)
    # DISTINCTIVE anchor: wrist + thumb base (radial) + all MCPs. Including the
    # thumb breaks the near-symmetry of the collinear knuckle line, so the
    # global registration can't lock a mirror-flipped (index<->pinky) pose.
    palm_slots = [hk.WRIST, hk.LM["thumb"]["mcp"]] + [
        hk.LM[f]["mcp"] for f in ("index", "middle", "ring", "pinky")]
    for sdir, stop in ((+1, None), (-1, None)):
        _track_dir_palm(frames, out, obs, seed_idx, sdir, np.array(seed, float),
                        np.array(seed, float), cons, palm_slots, dt, max_speed,
                        reacquire_mult, n_iter, stop)
    return out, obs


def _best_chain(mcp, cand, bones, tol=0.30, min_tol=0.008):
    """Given the MCP position and candidate detection positions `cand` (C,3),
    find the ordered (PIP,DIP,tip) indices whose bones best match `bones`
    (L_mcp-pip, L_pip-dip, L_dip-tip). Returns (idx_triple or None, cost). This
    is the per-finger 'try combinations, keep the best skeleton fit'."""
    C = len(cand)
    tols = [max(min_tol, tol * b) for b in bones]
    best, bcost = None, 1e9
    for a in range(C):
        e1 = abs(np.linalg.norm(cand[a] - mcp) - bones[0])
        if e1 > tols[0]:
            continue
        for b in range(C):
            if b == a:
                continue
            e2 = abs(np.linalg.norm(cand[b] - cand[a]) - bones[1])
            if e2 > tols[1]:
                continue
            for c in range(C):
                if c in (a, b):
                    continue
                e3 = abs(np.linalg.norm(cand[c] - cand[b]) - bones[2])
                if e3 > tols[2]:
                    continue
                cost = e1 + e2 + e3
                if cost < bcost:
                    bcost, best = cost, (a, b, c)
    return best, bcost


def track_hand_independent(frames, seed, seed_idx, dt, max_speed=3.0,
                           reacquire_mult=12):
    """Per-frame INDEPENDENT labelling (no temporal drift): each frame register
    the distinctive palm anchor (wrist+thumb+MCPs, RANSAC, warm-started but
    verified) to lock the 5 MCPs, then a per-finger chain search assigns
    PIP/DIP/tip. A frame whose palm doesn't register cleanly is left NaN rather
    than propagating a wrong label. Returns (seq, observed)."""
    from utils import hand_kinematics as hk
    T, k = len(frames), len(seed)
    out = np.full((T, k, 3), np.nan)
    obs = np.zeros((T, k), dtype=bool)
    palm_slots = [hk.WRIST, hk.LM["thumb"]["mcp"]] + [
        hk.LM[f]["mcp"] for f in ("index", "middle", "ring", "pinky")]
    seed_palm = seed[palm_slots]
    # per-finger seed bones (mcp->pip->dip->tip)
    fbones = {}
    for f in hk.FINGERS:
        L = hk.LM[f]
        ch = [L["mcp"], L["pip"], L["dip"], L["tip"]]
        fbones[f] = [float(np.linalg.norm(seed[a] - seed[b]))
                     for a, b in zip(ch[:-1], ch[1:])]
    gate = max_speed * dt
    rng = np.random.default_rng(0)
    prev_palm = seed_palm.copy()
    for t in range(T):
        pts = frames[t]
        if len(pts) < 5:
            continue
        fit = _fit_palm(pts, seed_palm, prev_palm, gate * reacquire_mult, rng=rng)
        if fit is None:
            prev_palm = seed_palm.copy()          # reset warm start on failure
            continue
        R, tt, match = fit
        if sum(x >= 0 for x in match) < 5:
            continue
        used = set()
        for i, slot in enumerate(palm_slots):
            if match[i] >= 0:
                out[t, slot] = pts[match[i]]
                obs[t, slot] = True
                used.add(match[i])
        prev_palm = np.array([out[t, s] if obs[t, s] else (seed_palm @ R.T + tt)[i]
                              for i, s in enumerate(palm_slots)])
        # per-finger chain search from the locked MCP
        for f in hk.FINGERS:
            L = hk.LM[f]
            mcp = out[t, L["mcp"]]
            if not np.all(np.isfinite(mcp)):
                continue
            reach = 1.4 * sum(fbones[f])
            ci = [j for j in range(len(pts)) if j not in used
                  and np.linalg.norm(pts[j] - mcp) < reach]
            if len(ci) < 3:
                continue
            cand = pts[ci]
            tri, cost = _best_chain(mcp, cand, fbones[f])
            if tri is not None:
                for slot_key, a in zip(("pip", "dip", "tip"), tri):
                    out[t, L[slot_key]] = cand[a]
                    obs[t, L[slot_key]] = True
                    used.add(ci[a])
    return out, obs


def track_hand_multiseed(frames, seeds, dt, max_speed=3.0, reacquire_mult=12,
                         bone_w=5.0):
    """Track with SEVERAL hand-labelled seeds — each seed owns the interval up to
    the midpoint toward its neighbours, so segments stay short and errors don't
    accumulate across the whole take. `seeds` = {frame_idx: (21,3)}.

    Workflow: run track_review to see where tracking breaks, re-label those
    frames with label_seed --append, then process with the multi-seed file."""
    T = len(frames)
    k = len(next(iter(seeds.values())))
    out = np.full((T, k, 3), np.nan)
    fidxs = sorted(seeds)
    nb0 = _hand_neighbors(seeds[fidxs[0]])
    for n, fi in enumerate(fidxs):
        out[fi] = seeds[fi]
        nb = _hand_neighbors(seeds[fi])
        lo = 0 if n == 0 else (fidxs[n - 1] + fi) // 2
        hi = T if n == len(fidxs) - 1 else (fi + fidxs[n + 1]) // 2 + 1
        _track_dir_rigid(frames, out, fi, +1, np.array(seeds[fi], float), nb,
                         dt, max_speed, reacquire_mult, bone_w, stop=hi)
        _track_dir_rigid(frames, out, fi, -1, np.array(seeds[fi], float), nb,
                         dt, max_speed, reacquire_mult, bone_w, stop=lo - 1)
    return out


def bone_health(seq, seed, tol=0.30):
    """Per-frame tracking health: fraction of the 20 hand bones whose length is
    within `tol` of the seed (a labelled seed defines the true bone lengths).
    Low health = the tracker has mislabelled markers there."""
    from utils import hand_kinematics as hk
    edges = []
    for f in hk.FINGERS:
        L = hk.LM[f]
        chain = [hk.WRIST, L["mcp"], L["pip"], L["dip"], L["tip"]]
        for a, b in zip(chain[:-1], chain[1:]):
            edges.append((a, b, np.linalg.norm(seed[a] - seed[b])))
    T = len(seq)
    health = np.zeros(T)
    for t in range(T):
        ok = tot = 0
        for a, b, Lref in edges:
            d = np.linalg.norm(seq[t, a] - seq[t, b])
            if np.isfinite(d):
                tot += 1
                if abs(d - Lref) <= tol * Lref:
                    ok += 1
        health[t] = ok / tot if tot else 0.0
    return health


def find_breaks(health, min_good=0.8, min_len=30):
    """Return (good_segments, break_frames): contiguous runs where health ≥
    min_good (length ≥ min_len), and the frames between them that need
    re-labelling (the middle of each bad run)."""
    good = health >= min_good
    segs, i, T = [], 0, len(health)
    while i < T:
        if good[i]:
            j = i
            while j < T and good[j]:
                j += 1
            if j - i >= min_len:
                segs.append((i, j))
            i = j
        else:
            i += 1
    breaks = []
    for (a0, b0), (a1, b1) in zip(segs[:-1], segs[1:]):
        breaks.append((b0 + a1) // 2)          # middle of the bad gap
    if not segs:
        breaks = [len(health) // 2]
    return segs, breaks


def track_hand_from_seed(frames, seed, seed_idx, dt, max_speed=3.0,
                         reacquire_mult=12, bone_w=2.0):
    """Track the 21 hand markers from a hand-labelled seed with a rigid-bone
    structure prior (Hungarian assignment) — resists the marker swaps that a
    position-only tracker suffers on a dense moving hand. Returns (T,21,3)."""
    T, k = len(frames), len(seed)
    out = np.full((T, k, 3), np.nan)
    out[seed_idx] = seed
    nb = _hand_neighbors(seed)
    _track_dir_struct(frames, out, seed_idx, +1, np.array(seed, float), nb,
                      dt, max_speed, reacquire_mult, bone_w)
    _track_dir_struct(frames, out, seed_idx, -1, np.array(seed, float), nb,
                      dt, max_speed, reacquire_mult, bone_w)
    return out


def occupancy(tracks):
    """Fraction of frames each track is present (finite)."""
    return np.isfinite(tracks[:, :, 0]).mean(axis=0)


def keep_top_tracks(tracks, k):
    """Keep the k tracks with the highest occupancy (drops ghost tracks)."""
    occ = occupancy(tracks)
    keep = np.argsort(occ)[::-1][:k]
    keep = keep[np.argsort(keep)]        # stable original order
    return tracks[:, keep, :]


def order_along_axis(tracks):
    """Order tracks along their common principal axis (a finger is ~collinear),
    proximal→distal by projection of each track's mean position. Returns the
    reordered tracks and the permutation."""
    mean = np.nanmean(tracks, axis=0)                 # (K,3)
    c = np.nanmean(mean, axis=0)
    u, s, vt = np.linalg.svd(mean - c)
    axis = vt[0]
    proj = (mean - c) @ axis
    order = np.argsort(proj)
    return tracks[:, order, :], order
