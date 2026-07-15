#!/usr/bin/env python3
"""Which cameras decoded each hand over a cosmik_hand_demo recording.

Usage:
    python plot_view_selection.py output_cosmik_demo/<run>/

Reads hands_2d_views.npy (T, ncam, 42, 2): a (view, hand) block is finite
only when the decoder actually ran there — with --hand-topk that is exactly
the per-frame view selection. Prints per-view usage stats + the per-hand
switch rate (flapping check: should be well under a few %/frame with the
default hysteresis) and saves <run>/view_selection.png — one row per
(hand, view), filled where that camera was decoding that hand.
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="cosmik_hand_demo output dir (or the .npy)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    path = (args.run_dir if args.run_dir.endswith(".npy")
            else os.path.join(args.run_dir, "hands_2d_views.npy"))
    h = np.load(path)                                  # (T, ncam, 42, 2)
    ts_p = os.path.join(os.path.dirname(path), "timestamps.npy")
    t = (np.load(ts_p) if os.path.isfile(ts_p) else np.arange(len(h)))
    t = t - t[0]
    T, ncam = h.shape[:2]
    # a hand counts as decoded in a view when most of its 21 joints are finite
    act = {"R": np.isfinite(h[:, :, :21]).all(3).sum(2) > 10,   # (T, ncam)
           "L": np.isfinite(h[:, :, 21:]).all(3).sum(2) > 10}

    print(f"{T} frames, {ncam} cameras — decoder usage per (hand, view):")
    for hand, a in act.items():
        use = a.mean(0) * 100
        n_active = a.sum(1)
        sw = (a[1:] != a[:-1]).any(1) & (n_active[1:] > 0) & (n_active[:-1] > 0)
        print(f"  {hand}: " + "  ".join(f"cam{v} {use[v]:5.1f}%"
                                        for v in range(ncam))
              + f"   |  views/frame {n_active.mean():.2f}, "
                f"switches {sw.mean() * 100:.1f}%/frame")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for ax, (hand, a) in zip(axes, act.items()):
        for v in range(ncam):
            on = a[:, v]
            ax.fill_between(t, v - 0.4, v + 0.4, where=on, step="mid",
                            alpha=0.8, label=f"cam{v}")
        ax.set_yticks(range(ncam), [f"cam{v}" for v in range(ncam)])
        ax.set_ylabel(f"{hand} hand")
        ax.grid(True, axis="x", alpha=0.3)
    axes[1].set_xlabel("time (s)" if os.path.isfile(ts_p) else "frame")
    fig.suptitle("hand-decoder view selection (filled = view decoded)")
    fig.tight_layout()
    if args.show:
        plt.show()
    else:
        out = os.path.join(os.path.dirname(path), "view_selection.png")
        fig.savefig(out, dpi=120)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
