"""Pilot: parse the 3-marker index take, re-track by position, order the 3
markers along the finger, compute the single inter-segment angle over time.

    python3 mocap_validation/pilot_index_3.py path/to/index_3_mocap.csv
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils import io_motive as io  # noqa: E402
from utils.hand_kinematics import included_angle  # noqa: E402


def main(path):
    meta, times, frames = io.read_motive_csv(path)
    fps = meta["fps"] or 120.0
    dt = 1.0 / fps
    print(f"take={meta['take']!r}  fps={fps}  units={meta['units']}  "
          f"frames={len(frames)}")
    npts = np.array([len(f) for f in frames])
    print(f"points/frame: median {int(np.median(npts[npts > 0]))}, "
          f"empty frames {int((npts == 0).sum())}/{len(frames)}")

    # anchor 3 persistent slots (robust to the heavy dropout), order on finger
    tracks = io.track_fixed_k(frames, k=3, dt=dt, max_speed=3.0)
    tracks, order = io.order_along_axis(tracks)
    print(f"3 anchored slots, ordered along finger axis; "
          f"occupancy {np.round(io.occupancy(tracks), 2)}")

    # bone lengths between consecutive ordered markers (rigid-bone check)
    for a, b in ((0, 1), (1, 2)):
        d = np.linalg.norm(tracks[:, b] - tracks[:, a], axis=1)
        d = d[np.isfinite(d)]
        print(f"  segment {a}-{b}: length median {d.mean()*1000:.1f} mm "
              f"(std {d.std()*1000:.1f} mm = STA proxy)")

    # inter-segment angle at the middle marker, all 3 present
    ang = np.full(len(frames), np.nan)
    for t in range(len(frames)):
        A, B, C = tracks[t, 0], tracks[t, 1], tracks[t, 2]
        if np.all(np.isfinite([A, B, C])):
            ang[t] = np.degrees(included_angle(B - A, C - B))
    valid = np.isfinite(ang)
    print(f"angle valid in {valid.sum()}/{len(frames)} frames; "
          f"range {np.nanmin(ang):.1f}–{np.nanmax(ang):.1f}°, "
          f"mean {np.nanmean(ang):.1f}°")

    out = path.rsplit(".", 1)[0] + "_angle.png"
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(times, npts, lw=0.5)
    ax[0].set_ylabel("points / frame")
    ax[0].set_title(f"{meta['take']} — detections and reconstructed angle")
    ax[1].plot(times, ang, lw=1.0)
    ax[1].set_ylabel("inter-segment angle (°)")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else str(__import__("pathlib").Path.home()
                  / "TheophileCodes/MOCAP/index_3_mocap.csv"))
