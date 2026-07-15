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
  2. PAIRWISE extrinsics via stereoCalibrate(cam{a}, cam{b}) with
     CALIB_FIX_INTRINSIC over every camera pair that shares >= 6 frames
     (matched by filename, >= --min-common shared ChArUco corners each),
     then CHAINED to cam0 over the strongest links (max-shared-frames
     spanning tree): a camera that never sees the board together with cam0
     is still calibrated through any camera it does overlap with.

Capture requirement: the cameras must form a CONNECTED graph of shared board
views (e.g. front pair together, front-left + side-left together, front-right
+ side-right together). NOT all cameras at once — any 2+ per saved set is
enough. Direct overlap with cam0 is still best (each chain hop compounds its
stereo error); the tool prints which cameras were chained and through what.

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


def detect_all(cam, board, dictionary, min_corners):
    """{frame basename: {corner id: (1,2) pixel}} for images/cam{cam}/*.png."""
    out = {}
    for p in sorted(glob.glob(f"images/cam{cam}/*.png")):
        gray = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
        ch, ii = _detect(gray, board, dictionary, min_corners)
        if ch is not None:
            out[os.path.basename(p)] = {int(ii[k]): ch[k]
                                        for k in range(len(ii))}
    return out


def pair_extrinsics(deta, detb, Ka, Da, Kb, Db, img_size, chess, min_common,
                    max_frames=40):
    """(R, T, rms, n_frames) with x_b = R @ x_a + T, or None if < 6 shared
    frames. deta/detb: detect_all outputs for the two cameras.

    Shared frames are evenly subsampled to max_frames: stereoCalibrate
    optimises 6 board-pose params PER VIEW, so its LM step scales badly
    with the view count (150 views ~ 900 params = tens of minutes) while
    the extrinsics themselves stop improving after a few dozen varied
    views."""
    obj_all, pa_all, pb_all = [], [], []
    for name in sorted(set(deta) & set(detb)):
        common = sorted(set(deta[name]) & set(detb[name]))
        if len(common) < min_common:
            continue
        obj_all.append(chess[common].reshape(-1, 1, 3).astype(np.float32))
        pa_all.append(np.array([deta[name][i] for i in common],
                               np.float32).reshape(-1, 1, 2))
        pb_all.append(np.array([detb[name][i] for i in common],
                               np.float32).reshape(-1, 1, 2))
    if len(obj_all) < 6:
        return None
    n_shared = len(obj_all)
    if n_shared > max_frames:
        idx = np.linspace(0, n_shared - 1, max_frames).astype(int)
        obj_all = [obj_all[i] for i in idx]
        pa_all = [pa_all[i] for i in idx]
        pb_all = [pb_all[i] for i in idx]
    rms, *_, R, T, _, _ = cv2.stereoCalibrate(
        obj_all, pa_all, pb_all, Ka, Da, Kb, Db, img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5))
    return (R.astype(np.float64), T.reshape(3).astype(np.float64), rms,
            n_shared)


def chain_extrinsics(cams, pairs, ref):
    """World(=cam{ref})->cam R,T for every camera, chaining pairwise links.

    pairs: {(a, b): (R, T, rms, n)} with x_b = R @ x_a + T. Greedy
    max-shared-frames spanning tree from ref (strong links first, so a solid
    two-hop chain beats a weak direct link). Returns (Rw, Tw, route) dicts;
    raises SystemExit naming any camera not connected to ref.
    """
    adj = {}
    for (a, b), (R, T, _, n) in pairs.items():
        adj.setdefault(a, []).append((b, R, T, n))
        adj.setdefault(b, []).append((a, R.T, -R.T @ T, n))    # inverted link
    Rw, Tw = {ref: np.eye(3)}, {ref: np.zeros(3)}
    route = {ref: [ref]}
    while len(Rw) < len(cams):
        best = None                    # (n_shared, from, to, R_ab, T_ab)
        for a in list(Rw):
            for b, R, T, n in adj.get(a, []):
                if b not in Rw and (best is None or n > best[0]):
                    best = (n, a, b, R, T)
        if best is None:
            missing = [c for c in cams if c not in Rw]
            raise SystemExit(
                f"cams {missing} share no board frames with any calibrated "
                f"camera — capture sets linking them (any pair works, "
                f"e.g. side cam together with its nearest front cam)")
        _, a, b, R, T = best
        Rw[b] = R @ Rw[a]
        Tw[b] = R @ Tw[a] + T
        route[b] = route[a] + [b]
    return Rw, Tw, route


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
    ap.add_argument("--max-pair-frames", type=int, default=40,
                    help="evenly subsample the shared frames of each pair to "
                         "this many before stereoCalibrate (its LM step has 6 "
                         "board-pose params PER VIEW; beyond a few dozen "
                         "varied views the extrinsics stop improving and the "
                         "solve time explodes)")
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

    print(f"[1/3] Intrinsics for {len(cams)} cameras...")
    K, D, img_size = {}, {}, None
    for c in cams:
        res = camera_intrinsics(c, board, dictionary, args.min_corners,
                                args.intr_flags)
        K[c], D[c], img_size = res[0], res[1], res[2]

    print("[2/3] Corner detections (cached per frame)...")
    dets = {c: detect_all(c, board, dictionary, args.min_corners)
            for c in cams}
    chess = board.getChessboardCorners()

    print(f"[3/3] Pairwise extrinsics, chained to cam{cams[0]}...")
    pairs = {}
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            a, b = cams[i], cams[j]
            r = pair_extrinsics(dets[a], dets[b], K[a], D[a], K[b], D[b],
                                img_size, chess, args.min_common,
                                args.max_pair_frames)
            if r is None:
                print(f"  cam{a}<->cam{b}: <6 shared frames (skipped)")
                continue
            pairs[(a, b)] = r
            used = min(r[3], args.max_pair_frames)
            print(f"  cam{a}<->cam{b}: {r[3]} shared frames ({used} used), "
                  f"stereo rms {r[2]:.3f} px, "
                  f"baseline {np.linalg.norm(r[1]) * 100:.1f} cm")
    R, T, route = chain_extrinsics(cams, pairs, cams[0])
    for c in cams[1:]:
        if len(route[c]) > 2:
            hops = " -> ".join(f"cam{x}" for x in route[c])
            print(f"  cam{c}: CHAINED {hops} (each hop compounds its stereo "
                  f"error — direct shared views with cam{cams[0]} would "
                  f"tighten it)")

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
