"""
N-camera calibration → the multi-cam npz that cosmik_hand_demo --calib expects
(K0..K{n}, D0..D{n}, R0..R{n}, T0..T{n}; cam0 is the reference: R0=I, T0=0, so
every R{i},T{i} maps cam0/world coordinates into camera i — exactly the
convention load_calibration() and triangulate_multiview() use).

This is the 2-camera calibrate_stereo.py generalised to any number of cameras.
The 2-cam scripts are left untouched (validated path).

Pipeline (self-contained — no need to run calibrate_single first):
  1. per-camera INTRINSICS from images/cam{c}/*.png (cv2.calibrateCamera on the
     ChArUco corners), unless calibration_data/cam{c}_intrinsics.npz exists
     (then it is reused, matching calibrate_single.py's output).
  2. per-camera EXTRINSICS relative to cam0 via stereoCalibrate(cam0, cam{c})
     with CALIB_FIX_INTRINSIC, over the frame pairs (matched by filename) where
     BOTH cameras saw >= --min-common shared ChArUco corners.

Capture requirement: cam0 must share board views with EVERY other camera —
during capture, place the board where cam0 and cam{c} both see it. For a wide
ring where some camera never overlaps cam0, calibrate that camera against a
neighbour it does overlap and chain the transforms by hand (not automated here).

Usage:
    python calibrate_multi.py --cams 0,1,2,3
    python calibrate_multi.py --cams 0,1,2 --out calibration_data/multi_params.npz

Output (default calibration_data/multi_params.npz), directly usable as:
    python cosmik_hand_demo.py --cams 0,1,2,3 --calib calibration_data/multi_params.npz
"""
import argparse
import glob
import os

import cv2
import numpy as np

import board_config


def _detect(gray, board, dictionary, min_corners):
    ch, ids, _, _ = board_config.detect_charuco(
        gray, board, dictionary, min_corners=min_corners)
    return ch, ids


def camera_intrinsics(cam, board, dictionary, min_corners, flags=0):
    """Per-camera K, D from images/cam{cam}/*.png (or a cached npz)."""
    cached = f"calibration_data/cam{cam}_intrinsics.npz"
    if os.path.isfile(cached):
        d = np.load(cached)
        print(f"  cam{cam}: reusing {cached} (rms {float(d['rms']):.3f} px)")
        return d["K"].astype(np.float64), d["D"].astype(np.float64), \
            tuple(int(x) for x in d["img_size"]), float(d["rms"])

    paths = sorted(glob.glob(f"images/cam{cam}/*.png"))
    if not paths:
        raise SystemExit(f"no images in images/cam{cam}/ — run "
                         f"capture_calibration_multi.py first")
    corners, ids, img_size = [], [], None
    for p in paths:
        gray = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        ch, ii = _detect(gray, board, dictionary, min_corners)
        if ch is not None:
            corners.append(ch)
            ids.append(ii)
    if len(corners) < 10:
        raise SystemExit(f"cam{cam}: only {len(corners)} usable frames "
                         f"(need >= 10) — capture more board views")
    chess = board.getChessboardCorners()
    obj = [chess[ii.ravel()] for ii in ids]
    rms, K, D, _, _ = cv2.calibrateCamera(obj, corners, img_size, None, None,
                                          flags=flags)
    print(f"  cam{cam}: intrinsics from {len(corners)}/{len(paths)} frames, "
          f"rms {rms:.3f} px")
    os.makedirs("calibration_data", exist_ok=True)
    np.savez(f"calibration_data/cam{cam}_intrinsics.npz",
             K=K, D=D, rms=rms, img_size=np.array(img_size))
    return K.astype(np.float64), D.astype(np.float64), img_size, rms


def extrinsics_to_ref(cam, K0, D0, Kc, Dc, img_size, board, dictionary,
                      min_corners, min_common):
    """R, T (cam0 -> cam{cam}) from stereoCalibrate over shared board frames."""
    obj_all, p0_all, pc_all = [], [], []
    base0 = {os.path.basename(p): p for p in glob.glob("images/cam0/*.png")}
    basec = {os.path.basename(p): p for p in glob.glob(f"images/cam{cam}/*.png")}
    chess = board.getChessboardCorners()
    for name in sorted(set(base0) & set(basec)):
        g0 = cv2.cvtColor(cv2.imread(base0[name]), cv2.COLOR_BGR2GRAY)
        gc = cv2.cvtColor(cv2.imread(basec[name]), cv2.COLOR_BGR2GRAY)
        ch0, id0 = _detect(g0, board, dictionary, min_corners)
        chc, idc = _detect(gc, board, dictionary, min_corners)
        if ch0 is None or chc is None:
            continue
        m0 = {int(id0[i]): ch0[i] for i in range(len(id0))}
        mc = {int(idc[i]): chc[i] for i in range(len(idc))}
        common = sorted(set(m0) & set(mc))
        if len(common) < min_common:
            continue
        obj_all.append(chess[common].reshape(-1, 1, 3))
        p0_all.append(np.array([m0[i] for i in common], np.float32))
        pc_all.append(np.array([mc[i] for i in common], np.float32))
    if len(obj_all) < 6:
        raise SystemExit(f"cam{cam}: only {len(obj_all)} frames share >= "
                         f"{min_common} corners with cam0 — cam0 and cam{cam} "
                         f"must both see the board more often")
    rms, *_, R, T, _, _ = cv2.stereoCalibrate(
        obj_all, p0_all, pc_all, K0, D0, Kc, Dc, img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5))
    print(f"  cam0->cam{cam}: {len(obj_all)} shared frames, stereo rms "
          f"{rms:.3f} px, baseline {np.linalg.norm(T) * 100:.1f} cm")
    return R.astype(np.float64), T.reshape(3).astype(np.float64), rms


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", default="0,1,2,3",
                    help="comma-separated camera ids (or the same /dev paths "
                         "given to capture_calibration_multi — a path maps to "
                         "its POSITION, matching the images/cam{pos} folders); "
                         "the first is the reference")
    ap.add_argument("--out", default="calibration_data/multi_params.npz")
    ap.add_argument("--min-corners", type=int, default=6,
                    help="min ChArUco corners to accept a detection")
    ap.add_argument("--min-common", type=int, default=6,
                    help="min corners shared cam0<->cam{c} to use a frame pair")
    ap.add_argument("--intr-flags", type=int, default=0,
                    help="cv2.calibrateCamera flags for the intrinsics step")
    args = ap.parse_args()

    _toks = [x.strip() for x in args.cams.split(",")]
    cams = [int(x) if x.isdigit() else i for i, x in enumerate(_toks)]
    if cams[0] != 0:
        print(f"NOTE: reference camera is cam{cams[0]}, but load_calibration "
              f"treats index 0 as the world frame — keep cam0 first.")
    board, dictionary = board_config.make_board()

    print(f"[1/2] Intrinsics for {len(cams)} cameras...")
    K, D, img_size = {}, {}, None
    for c in cams:
        res = camera_intrinsics(c, board, dictionary, args.min_corners,
                                args.intr_flags)
        K[c], D[c], img_size = res[0], res[1], res[2]

    print(f"[2/2] Extrinsics relative to cam{cams[0]}...")
    R = {cams[0]: np.eye(3)}
    T = {cams[0]: np.zeros(3)}
    for c in cams[1:]:
        R[c], T[c], _ = extrinsics_to_ref(
            c, K[cams[0]], D[cams[0]], K[c], D[c], img_size, board, dictionary,
            args.min_corners, args.min_common)

    out = {}
    for i, c in enumerate(cams):                         # save in 0..n order
        out[f"K{i}"] = K[c]
        out[f"D{i}"] = D[c]
        out[f"R{i}"] = R[c]
        out[f"T{i}"] = T[c]
    out["cam_ids"] = np.array(cams)
    out["img_size"] = np.array(img_size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **out)
    print(f"\nSaved {len(cams)}-camera calibration → {args.out}")
    print(f"Run: python cosmik_hand_demo.py --cams {args.cams} --calib {args.out} "
          f"--cap-width {img_size[0]} --cap-height {img_size[1]}")


if __name__ == "__main__":
    main()
