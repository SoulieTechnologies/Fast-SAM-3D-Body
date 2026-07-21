"""Pilot: index finger + wrist, 5 markers (wrist, MCP, PIP, DIP, nail).
Re-track by position, order proximal→distal, compute MCP/PIP/DIP flexion.

    python3 mocap_validation/pilot_index_wrist.py [path.csv] [--drop-last N]
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils import hand_kinematics as hk  # noqa: E402
from utils import io_motive as io  # noqa: E402


# conservative anatomical flexion ceilings (deg) — a residual value above these
# is a tracking error (marker swap that survived the bone-length gate), not a
# real joint angle, so it is dropped.
_ANAT_MAX = {"mcp": 100.0, "pip": 120.0, "dip": 95.0}


def orient_wrist_first(tracks):
    """order_along_axis gives an arbitrary end first; put the WRIST at index 0
    by orienting so the longest bone (wrist→MCP metacarpal) is the first one."""
    mean = np.nanmean(tracks, axis=0)
    bones = np.linalg.norm(np.diff(mean, axis=0), axis=1)
    if bones[-1] > bones[0]:                       # long bone at the far end
        tracks = tracks[:, ::-1, :]
    return tracks


def main(path, drop_last=100):
    meta, times, frames = io.read_motive_csv(path)
    fps = meta["fps"] or 120.0
    if drop_last:
        frames, times = frames[:-drop_last], times[:-drop_last]
    print(f"take={meta['take']!r} fps={fps} frames={len(frames)} "
          f"(dropped last {drop_last})")

    # chain-aware tracker: seeds the full 5-marker set, re-IDs a reappearing
    # marker by position + bone length to neighbours (already in chain order)
    tracks = io.track_chain(frames, k=5, dt=1.0 / fps, max_speed=3.0)
    tracks = orient_wrist_first(tracks)
    occ = io.occupancy(tracks)
    names = ["wrist", "MCP", "PIP", "DIP", "nail"]
    print("occupancy:", {n: round(o, 2) for n, o in zip(names, occ)})

    # per-frame bone lengths; a swap/jump makes a bone deviate wildly from its
    # median → mark that bone "bad" this frame (reject, don't trust the angle)
    labels = ["wrist–MCP", "MCP–PIP", "PIP–DIP", "DIP–nail"]
    blen = np.linalg.norm(np.diff(tracks, axis=1), axis=2)     # (T,4)
    bmed = np.nanmedian(blen, axis=0)
    good_bone = np.abs(blen - bmed) <= 0.30 * bmed             # (T,4)
    for i, lab in enumerate(labels):
        d = blen[good_bone[:, i], i]
        print(f"  {lab:9s}: {bmed[i]*1000:5.1f} mm  "
              f"(clean STA std {d.std()*1000:.1f} mm, "
              f"rejected {int((~good_bone[:, i]).sum())} frames)")

    # map the 5 ordered points into a fastsam3d 21-array and reuse the real
    # angle code (finger_flexion): MCP uses (MCP−wrist), PIP, DIP inter-segment
    T = len(frames)
    ang = {j: np.full(T, np.nan) for j in ("mcp", "pip", "dip")}
    # each joint needs its two adjacent bones clean (bone indices along chain:
    # 0 wrist–MCP, 1 MCP–PIP, 2 PIP–DIP, 3 DIP–nail)
    joint_bones = {"mcp": (0, 1), "pip": (1, 2), "dip": (2, 3)}
    L = hk.LM["index"]
    for t in range(T):
        if not np.all(np.isfinite(tracks[t])):
            continue
        w, mcp, pip, dip, tip = tracks[t]
        P = np.full((hk.N_LM, 3), np.nan)
        P[hk.WRIST], P[L["mcp"]], P[L["pip"]], P[L["dip"]], P[L["tip"]] = \
            w, mcp, pip, dip, tip
        fa = hk.finger_flexion(P, "index")
        for j in ang:
            b0, b1 = joint_bones[j]
            deg = np.degrees(fa[j])
            # reject: bone-length outlier (swap/jump) OR anatomically impossible
            if good_bone[t, b0] and good_bone[t, b1] and deg <= _ANAT_MAX[j]:
                ang[j][t] = deg

    # handle discontinuities: kill isolated spikes, bridge SHORT gaps only
    # (≤ gap_ms), leave real dropouts as NaN. gap_ms/1000*fps frames.
    gap_frames = int(0.12 * fps)                  # bridge gaps ≤ 120 ms
    clean = {}
    for j in ("mcp", "pip", "dip"):
        s = io.smooth_series(ang[j], win=5)
        s = io.fill_short_gaps(s, max_gap=gap_frames)
        clean[j] = s
        nmiss, longest = io.gap_report(s)
        a = s[np.isfinite(s)]
        print(f"  {j.upper()} flexion: {a.min():5.1f}–{a.max():5.1f}°  "
              f"(missing {nmiss}/{T}, longest gap {longest} fr "
              f"= {longest/fps*1000:.0f} ms)")

    out = path.rsplit(".", 1)[0] + "_angles.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    for j in ("mcp", "pip", "dip"):
        ax.plot(times, clean[j], lw=1.0, label=f"{j.upper()} flexion")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("flexion (°)")
    ax.set_title(f"{meta['take']} — index joint angles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    import pathlib
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else str(pathlib.Path.home()
                                    / "TheophileCodes/MOCAP/index_wrist.csv")
    n = 100
    if "--drop-last" in sys.argv:
        n = int(sys.argv[sys.argv.index("--drop-last") + 1])
    main(path, drop_last=n)
