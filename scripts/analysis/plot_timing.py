#!/usr/bin/env python3
"""Analyze cosmik_hand_demo timing.log files: stats table + graphs.

Usage:
    python scripts/analysis/plot_timing.py output_cosmik_demo/<run>/timing.log
    python scripts/analysis/plot_timing.py run512/timing.log run256/timing.log   # compare runs

Prints a per-column stats table (mean/median/p95) for each log and saves
<log_dir>/timing.png with the time series (body/hand latency breakdown,
rates, stereo joint counts). --show opens the matplotlib window instead.
"""
import argparse
import os

import numpy as np

COLS = ["wall_s", "body_hz", "body_ms", "yolo_ms", "nlf_ms", "sync_ms",
        "hand_ms", "prep_ms", "fwd_ms", "post_ms", "tri_ms",
        "stereo_R", "stereo_L"]


def load(path):
    d = np.loadtxt(path, comments="#")
    if d.ndim == 1:                                  # single line
        d = d[None]
    if d.shape[1] != len(COLS):
        raise SystemExit(f"{path}: {d.shape[1]} columns, expected {len(COLS)} "
                         f"(old format?)")
    return {c: d[:, i] for i, c in enumerate(COLS)}


def stats_table(runs):
    """One row per (run, column): mean / median / p95."""
    show = ["body_hz", "body_ms", "yolo_ms", "nlf_ms", "sync_ms",
            "hand_ms", "prep_ms", "fwd_ms", "post_ms", "tri_ms"]
    w = max(len(os.path.dirname(p) or p) for p in runs) + 2
    print(f"{'':{w}} {'':>9} " + " ".join(f"{c:>8}" for c in show))
    for path, d in runs.items():
        name = os.path.dirname(path) or path
        n = len(d["wall_s"])
        dur = d["wall_s"][-1] - d["wall_s"][0] if n > 1 else 0
        hand_hz = (n - 1) / dur if dur > 0 else 0
        print(f"{name:{w}} ({n} hand iters, {dur:.0f}s, hand {hand_hz:.1f} Hz)")
        for stat, fn in (("mean", np.mean), ("median", np.median),
                         ("p95", lambda x: np.percentile(x, 95))):
            print(f"{'':{w}} {stat:>9} "
                  + " ".join(f"{fn(d[c]):8.1f}" for c in show))
        sr, sl = d["stereo_R"], d["stereo_L"]
        if (sr >= 0).any():
            print(f"{'':{w}} stereo jts R {np.mean(sr[sr >= 0]):.1f}/21 "
                  f"(21/21 on {np.mean(sr == 21) * 100:.0f}% of iters)   "
                  f"L {np.mean(sl[sl >= 0]):.1f}/21 "
                  f"(21/21 on {np.mean(sl == 21) * 100:.0f}%)")
        else:
            print(f"{'':{w}} stereo: OFF (mono hands)")
        print()


def plot(runs, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for path, d in runs.items():
        name = os.path.dirname(path) or path
        t = d["wall_s"] - d["wall_s"][0]

        ax = axes[0]                                  # hand latency breakdown
        ax.plot(t, d["hand_ms"], lw=1.2, label=f"{name} hand total")
        ax.plot(t, d["fwd_ms"], lw=0.8, alpha=0.8, label=f"{name} fwd")
        ax.plot(t, d["prep_ms"], lw=0.8, alpha=0.6, label=f"{name} prep")
        ax.set_ylabel("hand ms")
        ax.legend(fontsize=7, ncol=3)

        ax = axes[1]                                  # body latency breakdown
        ax.plot(t, d["body_ms"], lw=1.2, label=f"{name} body total")
        ax.plot(t, d["nlf_ms"], lw=0.8, alpha=0.8, label=f"{name} nlf")
        ax.plot(t, d["yolo_ms"], lw=0.8, alpha=0.6, label=f"{name} yolo")
        ax.set_ylabel("body ms")
        ax.legend(fontsize=7, ncol=3)

        ax = axes[2]                                  # rates
        ax.plot(t, d["body_hz"], lw=1.0, label=f"{name} body Hz")
        if len(t) > 1:
            dt = np.diff(d["wall_s"])
            ax.plot(t[1:], np.clip(1.0 / np.maximum(dt, 1e-3), 0, 60),
                    lw=0.6, alpha=0.6, label=f"{name} hand Hz (inst)")
        ax.set_ylabel("Hz")
        ax.legend(fontsize=7, ncol=2)

        ax = axes[3]                                  # stereo joint counts
        ax.plot(t, d["stereo_R"], lw=0.8, label=f"{name} R")
        ax.plot(t, d["stereo_L"], lw=0.8, alpha=0.7, label=f"{name} L")
        ax.set_ylabel("stereo jts /21")
        ax.set_ylim(-1.5, 22)
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=7, ncol=2)

    fig.align_ylabels(axes)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        out = os.path.join(os.path.dirname(list(runs)[0]) or ".", "timing.png")
        fig.savefig(out, dpi=130)
        print(f"saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("logs", nargs="+", help="timing.log path(s) — several to compare")
    p.add_argument("--show", action="store_true",
                   help="open the matplotlib window instead of saving a png")
    p.add_argument("--no-plot", action="store_true", help="stats table only")
    args = p.parse_args()

    runs = {path: load(path) for path in args.logs}
    stats_table(runs)
    if not args.no_plot:
        plot(runs, args.show)


if __name__ == "__main__":
    main()
