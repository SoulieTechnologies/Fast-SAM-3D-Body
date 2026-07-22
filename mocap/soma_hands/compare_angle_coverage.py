"""SOMA vs hybrid — the boss-facing result. Compares two angle CSVs (same 20
columns, produced by the SAME hand_angle_series so they are commensurable):

  * hybrid: data/<take>_angles.csv               (track_hand_hybrid)
  * soma:   <labelled>_soma_angles.csv           (soma_labeled_to_angles.py)

Reports, per angle and overall:
  * valid-fraction (coverage) for each method  -> does SOMA label MORE frames?
  * agreement where BOTH are valid: RMSE + bias -> do they AGREE when both label?
and writes a grouped bar chart of coverage.

This is what shows the boss whether SOMA beats the geometric tracker. Testable
now: hybrid CSV exists; feed any SOMA CSV (even a partial pilot) once trained.

    python compare_angle_coverage.py \
        --hybrid ../data/take_gabin_1_angles.csv \
        --soma   $WORK/.../take_gabin_1_labeled_soma_angles.csv \
        --out ../data/soma_vs_hybrid.png
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import utils.hand_angles as ha  # noqa: E402


def load_angles(path):
    a = np.genfromtxt(path, delimiter=",", names=True)
    cols = [c for c in ha.angle_names() if c in a.dtype.names]
    return {c: np.asarray(a[c], float) for c in cols}


def align(h, s):
    """Truncate to the shorter series (both start at take/label frame 0)."""
    n = min(len(next(iter(h.values()))), len(next(iter(s.values()))))
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hybrid", required=True)
    p.add_argument("--soma", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    H, S = load_angles(args.hybrid), load_angles(args.soma)
    cols = [c for c in ha.angle_names() if c in H and c in S]
    n = align(H, S)

    print(f"{'angle':16s} {'hyb cov':>8s} {'soma cov':>9s} {'both':>6s} "
          f"{'RMSE':>7s} {'bias':>7s}")
    rows = []
    for c in cols:
        hv, sv = H[c][:n], S[c][:n]
        hcov, scov = np.isfinite(hv).mean(), np.isfinite(sv).mean()
        both = np.isfinite(hv) & np.isfinite(sv)
        if both.sum() >= 5:
            d = sv[both] - hv[both]
            rmse, bias = float(np.sqrt(np.mean(d**2))), float(np.mean(d))
        else:
            rmse = bias = np.nan
        rows.append((c, hcov, scov, both.mean(), rmse, bias))
        print(f"{c:16s} {hcov:8.2f} {scov:9.2f} {both.mean():6.2f} "
              f"{rmse:7.1f} {bias:7.1f}")

    hcov_m = np.mean([r[1] for r in rows])
    scov_m = np.mean([r[2] for r in rows])
    rmse_m = np.nanmean([r[4] for r in rows])
    print(f"\nMEAN coverage: hybrid {hcov_m:.2f}  soma {scov_m:.2f}  "
          f"(soma {'WINS' if scov_m > hcov_m else 'loses'} on coverage)")
    print(f"MEAN agreement RMSE (both-valid): {rmse_m:.1f} deg")

    out = args.out or "soma_vs_hybrid.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(cols))
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.bar(x - 0.2, [r[1] for r in rows], 0.4, label=f"hybrid ({hcov_m:.0%})")
        ax.bar(x + 0.2, [r[2] for r in rows], 0.4, label=f"soma ({scov_m:.0%})")
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("coverage (valid frame fraction)")
        ax.set_title("SOMA vs hybrid — per-angle coverage")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        print(f"saved {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
