"""Show where the seed-based tracking loses the markers, so you know which
frames to re-label. Plots per-frame tracking health (fraction of hand bones at
their seed length) and lists the break frames to re-label with
`label_seed.py --append`.

    python scripts/track_review.py data/take_gabin_1.csv data/take_gabin_1_seed.npz
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils import io_motive as io  # noqa: E402


def load_seeds(z):
    if "seeds" in z:
        return list(z["frames"].astype(int)), z["seeds"]
    return [int(z["frame"])], z["seed"][None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("take")
    p.add_argument("seed")
    p.add_argument("--min-good", type=float, default=0.8)
    args = p.parse_args()

    meta, _, frames = io.read_motive_csv(args.take)
    fps = meta.get("fps") or 120.0
    z = np.load(args.seed)
    fidxs, seeds = load_seeds(z)

    if len(fidxs) == 1:
        seq = io.track_hand_rigid(frames, seeds[0], fidxs[0], dt=1.0 / fps)
    else:
        seq = io.track_hand_multiseed(frames, dict(zip(fidxs, seeds)),
                                      dt=1.0 / fps)
    health = io.bone_health(seq, seeds[0])
    segs, breaks = io.find_breaks(health, min_good=args.min_good)

    good_frac = (health >= args.min_good).mean()
    print(f"tracking health >= {args.min_good}: {good_frac*100:.0f}% of frames")
    print(f"{len(segs)} clean segments; {len(breaks)} break(s) to re-label:")
    for b in breaks:
        # suggest the nearest frame with ~21 points around the break
        best = b
        for d in range(0, 120):
            for bb in (b + d, b - d):
                if 0 <= bb < len(frames) and len(frames[bb]) == 21:
                    best = bb
                    break
            else:
                continue
            break
        print(f"   break ~frame {b} ({b/fps:.1f}s) -> re-label frame {best} "
              f"(21 pts):  python scripts/label_seed.py {args.take} "
              f"--frame {best} --append {args.seed}")

    out = args.take.rsplit(".", 1)[0] + "_health.png"
    t = np.arange(len(health)) / fps
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(t, health, lw=0.7)
    ax.axhline(args.min_good, color="r", ls="--", lw=0.8, label="min good")
    for a, b in segs:
        ax.axvspan(a / fps, b / fps, color="green", alpha=0.12)
    for b in breaks:
        ax.axvline(b / fps, color="orange", lw=1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("bone health\n(frac bones OK)")
    ax.set_title(f"{pathlib.Path(args.take).name} — tracking health "
                 "(green=clean, orange=re-label here)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
