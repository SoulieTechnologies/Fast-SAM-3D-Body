"""Full-hand mocap pipeline: a labelled 21-marker take → all articulation
angles, logged to CSV and plotted. This CSV is the reference the SAM3D angles
are compared against.

    # inspect a file's marker naming first (tells read_labeled how to map)
    python scripts/process_hand.py data/take.csv --list-labels

    # process a labelled Motive/SOMA CSV (or a pre-labelled .npy of shape T×21×3)
    python scripts/process_hand.py data/take.csv
    python scripts/process_hand.py data/take.npy --fps 120

Outputs next to the input: <take>_angles.csv  and  <take>_angles.png
"""
import argparse
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils import hand_angles as ha  # noqa: E402
from utils import io_motive as io  # noqa: E402


def load(path, fps_override, seed_path=""):
    if seed_path:                                  # raw take + hand-labelled seed
        meta, _, frames = io.read_motive_csv(path)
        fps = fps_override or meta.get("fps") or 120.0
        z = np.load(seed_path)
        if "seeds" in z and len(z["seeds"]) > 1:
            seeds = dict(zip(z["frames"].astype(int), z["seeds"]))
            seq = io.track_hand_multiseed(frames, seeds, dt=1.0 / fps)
            tag = f"{len(seeds)} seeds @{sorted(seeds)}"
        else:
            seed = z["seeds"][0] if "seeds" in z else z["seed"]
            fidx = int(z["frames"][0]) if "seeds" in z else int(z["frame"])
            seq = io.track_hand_rigid(frames, seed, fidx, dt=1.0 / fps)
            tag = f"seed @{fidx}"
        occ = io.occupancy(seq)
        print(f"tracked {len(seq)} frames ({tag}); "
              f"per-slot occupancy min {occ.min():.2f} mean {occ.mean():.2f}")
        return fps, seq
    if path.endswith(".npy"):
        seq = np.load(path)
        assert seq.ndim == 3 and seq.shape[1:] == (21, 3), seq.shape
        fps = fps_override or 120.0
    else:
        meta, _, seq = io.read_labeled(path)
        fps = fps_override or meta.get("fps") or 120.0
        n = np.isfinite(seq[:, :, 0]).mean(axis=0)
        missing = [ha.hk.WRIST] + []  # informational
        got = int((n > 0).sum())
        print(f"read {len(seq)} frames, {got}/21 slots populated "
              f"(mean occupancy {n[n > 0].mean():.2f})")
        if got < 21:
            miss = [k for k in range(21) if n[k] == 0]
            print(f"  WARNING: {21 - got} slots never seen (indices {miss}) — "
                  "check --list-labels / naming convention")
    return fps, seq


def save_csv(out, time, angles):
    cols = ha.angle_names()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s"] + cols)
        for t in range(len(time)):
            w.writerow([t, f"{time[t]:.4f}"]
                       + [("" if not np.isfinite(angles[c][t])
                           else f"{angles[c][t]:.3f}") for c in cols])


def plot(out, time, angles, title):
    fig, axes = plt.subplots(len(ha.FINGERS), 1, figsize=(12, 12), sharex=True)
    for ax, f in zip(axes, ha.FINGERS):
        for j in ha.JOINTS:
            ax.plot(time, angles[f"{f}_{j}"], lw=0.9, label=j)
        ax.set_ylabel(f"{f}\n(°)")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--seed", default="",
                   help="raw take + <take>_seed.npz from label_seed.py "
                        "-> track from the labelled seed")
    p.add_argument("--hand", choices=["right", "left"], default="right")
    p.add_argument("--fps", type=float, default=0.0)
    p.add_argument("--list-labels", action="store_true")
    p.add_argument("--no-clean", action="store_true")
    args = p.parse_args()

    if args.list_labels:
        for lab, slot in io.labels_seen(args.path):
            print(f"  {lab!r:32s} -> {slot}")
        return

    fps, seq = load(args.path, args.fps, args.seed)
    time, angles, quality = ha.hand_angle_series(seq, fps, hand=args.hand,
                                                 clean=not args.no_clean)

    print("per-articulation quality (valid %, longest gap ms):")
    for name in ha.angle_names():
        v, g = quality[name]
        a = angles[name][np.isfinite(angles[name])]
        rng = f"{a.min():5.1f}–{a.max():5.1f}°" if a.size else "  no data"
        print(f"  {name:16s} valid {v*100:4.0f}%  gap {g:5.0f}ms  range {rng}")

    stem = args.path.rsplit(".", 1)[0]
    save_csv(stem + "_angles.csv", time, angles)
    plot(stem + "_angles.png", time, angles,
         f"{pathlib.Path(args.path).name} — hand joint angles")
    print(f"wrote {stem}_angles.csv and {stem}_angles.png")


if __name__ == "__main__":
    main()
