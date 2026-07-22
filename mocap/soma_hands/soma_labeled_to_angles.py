"""Bridge: a SOMA-labelled c3d -> our 21-slot sequence -> the 20 hand angles,
reusing utils.hand_angles so SOMA output is directly comparable to
track_hand_hybrid on the SAME metric.

SOMA writes a labelled c3d whose point labels are our marker names (LINDEX_TIP,
LWRIST, ...). We map them back to the 21 fastsam3d slots via the same
name_to_slot() used to build the layout, assemble a T x 21 x 3 array (NaN where
a label is absent that frame), and run hand_angle_series.

    python soma_labeled_to_angles.py $WORK/.../take_gabin_1_labeled.c3d --hand left

RUN ON crslab (needs ezc3d). Writes <take>_soma_angles.csv (+ .png).
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils import hand_angles as ha  # noqa: E402
from utils import hand_kinematics as hk  # noqa: E402
from build_hand_superset import name_to_slot  # noqa: E402


def read_labeled_c3d(path, hand):
    from ezc3d import c3d
    c = c3d(str(path))
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    pts = c["data"]["points"]              # 4 x nMarkers x nFrames
    rate = c["parameters"]["POINT"]["RATE"]["value"][0]
    unit = (c["parameters"]["POINT"]["UNITS"]["value"] or ["mm"])[0]
    scale = 0.001 if unit.lower().startswith("mm") else 1.0
    T = pts.shape[2]
    seq = np.full((T, 21, 3), np.nan)
    for m, lab in enumerate(labels):
        try:
            h, slot = name_to_slot(lab.strip())
        except Exception:
            continue                       # ghost / body / unknown label
        if h != hand:
            continue
        xyz = pts[:3, m, :].T * scale      # T x 3, metres
        good = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)
        seq[good, slot] = xyz[good]
    return seq, float(rate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("labeled_c3d")
    p.add_argument("--hand", choices=["left", "right"], default="left")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seq, fps = read_labeled_c3d(args.labeled_c3d, args.hand)
    occ = np.isfinite(seq).all(2).mean(0)
    print(f"{seq.shape[0]} frames @ {fps:g}fps; per-slot occupancy "
          f"min {occ.min():.2f} mean {occ.mean():.2f}")

    # angles is a dict {name: (T,) deg}; assemble the ordered column matrix
    time, angles, quality = ha.hand_angle_series(seq, fps, hand=args.hand)
    out = args.out or str(pathlib.Path(args.labeled_c3d).with_suffix("")) + "_soma_angles.csv"
    cols = ha.angle_names()
    mat = np.column_stack([angles[c] for c in cols])
    arr = np.column_stack([np.arange(len(time)), time, mat])
    hdr = "frame,time," + ",".join(cols)
    np.savetxt(out, arr, delimiter=",", header=hdr, comments="")
    valid = np.mean([quality[c][0] for c in cols])
    print(f"wrote {out}  ({len(cols)} angle columns; mean valid-fraction {valid:.2f})")
    print("Score against track_hand_hybrid with the same bone-health/clean-% "
          "metric for the SOMA-vs-hybrid comparison.")


if __name__ == "__main__":
    main()
