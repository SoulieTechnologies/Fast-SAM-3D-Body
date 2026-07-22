"""Lay our raw Motive takes into the folder structure SOMA/MoSh++ expect:

    <work>/mocap/<ds_name>/<subject>/<sequence>.c3d
    <work>/mocap/<ds_name>/<subject>/settings.json   # {"gender": "..."}

SOMA reads unlabelled point-cloud c3d directly, so we just place (symlink) the
Motive-exported c3d and write the per-subject settings.json. We also print the
POINT unit + count so the run config (`mocap.unit`) is set correctly — our
Motive export is in METRES, SOMA's default run conf assumes mm.

    python prepare_soma_data.py \
        --c3d ../data/take_gabin_1.c3d --c3d ../data/take_gabin_0.c3d \
        --ds-name clear_hands --subject gabin --gender male \
        --work-dir $WORK/mocap

RUN ON crslab (needs ezc3d). NOT on the Mac.
"""
import argparse
import json
import os
import pathlib


def inspect_c3d(path):
    from ezc3d import c3d
    c = c3d(str(path))
    pts = c["data"]["points"]  # 4 x nMarkers x nFrames
    unit = c["parameters"]["POINT"]["UNITS"]["value"]
    unit = unit[0] if unit else "?"
    n_markers = pts.shape[1]
    n_frames = pts.shape[2]
    rate = c["parameters"]["POINT"]["RATE"]["value"][0]
    return unit, n_markers, n_frames, rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c3d", action="append", required=True,
                   help="raw Motive c3d (repeatable)")
    p.add_argument("--ds-name", default="clear_hands")
    p.add_argument("--subject", default="gabin")
    p.add_argument("--gender", choices=["male", "female", "neutral"],
                   default="male")
    p.add_argument("--work-dir", required=True,
                   help="SOMA mocap root; files go <work>/<ds>/<subject>/")
    p.add_argument("--copy", action="store_true",
                   help="copy instead of symlink (default symlink; c3d are big)")
    args = p.parse_args()

    dst = pathlib.Path(args.work_dir) / args.ds_name / args.subject
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "settings.json", "w") as f:
        json.dump({"gender": args.gender}, f, indent=2)

    for c in args.c3d:
        src = pathlib.Path(c).resolve()
        link = dst / src.name
        if link.exists() or link.is_symlink():
            link.unlink()
        if args.copy:
            import shutil
            shutil.copy2(src, link)
        else:
            os.symlink(src, link)
        try:
            unit, nm, nf, rate = inspect_c3d(src)
            print(f"{src.name}: unit={unit!r}  markers={nm}  frames={nf}  "
                  f"rate={rate:g}fps  ->  {link}")
            if unit and unit.lower() in ("m", "meter", "metre", "meters"):
                print("  NOTE: units are METRES -> set mocap.unit=m in the run "
                      "config (SOMA default is mm).")
        except Exception as e:  # ezc3d missing or unreadable header
            print(f"{src.name}: linked -> {link}  (header read failed: {e})")

    print(f"\nsettings.json (gender={args.gender}) written to {dst}")
    print("Next: build the superset, then generate synthetic data + train.")


if __name__ == "__main__":
    main()
