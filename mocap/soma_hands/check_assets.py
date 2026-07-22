"""Verify the licensed assets SOMA needs are present on crslab before we start.

Nothing here downloads (all require personal MPI accounts). It only reports what
is found / missing so we know the exact gap.

    python check_assets.py --support-base $SUPPORT --amass-dir $SUPPORT/smplx/amass_neutral

Expected under --support-base:
    smplx/neutral/model.npz            SMPL-X neutral (locked-head for SOMA)
    smplx/{male,female}/model.npz      gender models (for the final MoSh solve)
    smplx/extra_smplx_data.tar.bz2 ->  extracted extra data
    smplx/amass_neutral/<DS>/...       AMASS SMPL-X neutral params (train/vald)
    smplx/amass_neutral/GRAB/...       GRAB (hand articulation) -- key for hands
"""
import argparse
import glob
import pathlib

WANT_AMASS = ["CMU", "Transitions", "PosePrior", "HumanEva", "ACCAD", "TotalCapture"]


def check(label, path, is_glob=False):
    if is_glob:
        hits = glob.glob(path)
        ok = len(hits) > 0
        extra = f"{len(hits)} files" if ok else "MISSING"
    else:
        ok = pathlib.Path(path).exists()
        extra = "ok" if ok else "MISSING"
    print(f"  [{'x' if ok else ' '}] {label:34s} {path}  ({extra})")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--support-base", required=True)
    p.add_argument("--amass-dir", default=None)
    args = p.parse_args()
    sb = pathlib.Path(args.support_base)
    amass = pathlib.Path(args.amass_dir) if args.amass_dir else sb / "smplx" / "amass_neutral"

    print("SMPL-X models:")
    check("neutral (SOMA)", sb / "smplx" / "neutral" / "model.npz")
    check("male", sb / "smplx" / "male" / "model.npz")
    check("female", sb / "smplx" / "female" / "model.npz")
    check("extra_smplx_data", str(sb / "smplx" / "extra_smplx_data*"), is_glob=True)

    print("\nAMASS SMPL-X neutral params:")
    for ds in WANT_AMASS:
        check(ds, str(amass / ds / "**" / "*.npz"), is_glob=True)

    print("\nGRAB (hand articulation — needed for good finger labels):")
    grab_ok = check("GRAB", str(amass / "GRAB" / "**" / "*.npz"), is_glob=True)

    print("\npython deps:")
    for m in ["torch", "smplx", "ezc3d", "moshpp", "soma", "psbody.mesh"]:
        try:
            __import__(m)
            print(f"  [x] {m}")
        except Exception as e:
            print(f"  [ ] {m}  ({type(e).__name__})")

    if not grab_ok:
        print("\nGRAB not found: download SMPL-X params from grab.is.tue.mpg.de "
              "(or the AMASS 'GRAB' subset) and extract under the amass dir.")


if __name__ == "__main__":
    main()
