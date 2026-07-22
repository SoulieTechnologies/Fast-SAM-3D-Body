"""Resolve the GRAB npz format so the Strategy-C hand-pose sampler reads the
right keys. Run this FIRST on crslab once GRAB is downloaded — it prints the
keys/shapes of a GRAB sequence and confirms patches/sample_hand_sequences_grab.py
can extract left/right hand poses.

    python inspect_grab.py --grab-dir $SUPPORT/smplx/amass_neutral/GRAB
    python inspect_grab.py --npz path/to/one_grab_seq.npz

The sampler already handles three layouts: {lhand_pose,rhand_pose},
concatenated pose_hand(90), and AMASS-X fullpose(165). This tells you which one
your GRAB uses (and flags if it's none of them, so we adapt the sampler).
"""
import argparse
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "patches"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grab-dir", default=None)
    p.add_argument("--npz", default=None)
    args = p.parse_args()

    if args.npz:
        npz = args.npz
    else:
        hits = sorted(glob.glob(str(pathlib.Path(args.grab_dir) / "**" / "*.npz"),
                                recursive=True))
        if not hits:
            print(f"no npz under {args.grab_dir}")
            return
        print(f"{len(hits)} GRAB npz found; inspecting {hits[0]}")
        npz = hits[0]

    d = np.load(npz, allow_pickle=True)
    print("\nkeys / shapes:")
    for k in d.keys():
        try:
            v = np.asarray(d[k])
            print(f"  {k:20s} {v.shape} {v.dtype}")
        except Exception:
            print(f"  {k:20s} (unreadable)")

    # try the sampler's reader
    from sample_hand_sequences_grab import _load_hand_pose
    try:
        lp, rp = _load_hand_pose(npz)
        print(f"\n_load_hand_pose OK: left {lp.shape} right {rp.shape} "
              f"(expect (T,45) each)")
        print(f"  left-hand pose range [{lp.min():.2f}, {lp.max():.2f}] rad "
              f"(nonzero frac {np.mean(np.abs(lp) > 1e-3):.2f})")
        if lp.shape[1] != 45 or rp.shape[1] != 45:
            print("  WARNING: expected 45 dims/hand (15 joints × 3); check the "
                  "PCA-vs-full convention.")
        else:
            print("  -> sampler ready for Strategy C. Set --strategy C in train_hands.py.")
    except Exception as e:
        print(f"\n_load_hand_pose FAILED: {type(e).__name__}: {e}")
        print("  -> tell me these keys and I adapt patches/sample_hand_sequences_grab.py")


if __name__ == "__main__":
    main()
