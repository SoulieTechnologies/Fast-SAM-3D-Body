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
        gray, board, dictionary, min_corners=min_corners
    )
    return ch, ids


def camera_intrinsics(cam, board, dictionary, min_corners, flags=0):
    """Per-camera K, D from images/cam{cam}/*.png (or a cached npz)."""
    cached = f"calibration_data/cam{cam}_intrinsics.npz"
    if os.path.isfile(cached):
        d = np.load(cached)
        print(f"  cam{cam}: reusing {cached} (rms {float(d['rms']):.3f} px)")
        return (
            d["K"].astype(np.float64),
            d["D"].astype(np.float64),
            tuple(int(x) for x in d["img_size"]),
            float(d["rms"]),
        )

    paths = sorted(glob.glob(f"images/cam{cam}/*.png"))
    if not paths:
        raise SystemExit(
            f"no images in images/cam{cam}/ — run "
            f"capture_calibration_multi.py first"
        )
    corners, ids, img_size = [], [], None
    for p in paths:
        gray = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        ch, ii = _detect(gray, board, dictionary, min_corners)
        if ch is not None:
            corners.append(ch)
            ids.append(ii)
    if len(corners) < 10:
        raise SystemExit(
            f"cam{cam}: only {len(corners)} usable frames "
            f"(need >= 10) — capture more board views"
        )
    chess = board.getChessboardCorners()
    obj = [chess[ii.ravel()] for ii in ids]
    rms, K, D, _, _ = cv2.calibrateCamera(
        obj, corners, img_size, None, None, flags=flags
    )
    print(
        f"  cam{cam}: intrinsics from {len(corners)}/{len(paths)} frames, "
        f"rms {rms:.3f} px"
    )
    os.makedirs("calibration_data", exist_ok=True)
    np.savez(
        f"calibration_data/cam{cam}_intrinsics.npz",
        K=K,
        D=D,
        rms=rms,
        img_size=np.array(img_size),
    )
    return K.astype(np.float64), D.astype(np.float64), img_size, rms


def detect_all(cam, board, dictionary, min_corners):
    """{frame basename: {corner id: (1,2) pixel}} for images/cam{cam}/*.png."""
    out = {}
    for p in sorted(glob.glob(f"images/cam{cam}/*.png")):
        gray = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
        ch, ii = _detect(gray, board, dictionary, min_corners)
        if ch is not None:
            out[os.path.basename(p)] = {
                int(ii[k]): ch[k] for k in range(len(ii))
            }
    return out


def count_shared(deta, detb, min_common):
    """Frames where BOTH cameras saw >= min_common common ChArUco corners."""
    return sum(
        1
        for name in set(deta) & set(detb)
        if len(set(deta[name]) & set(detb[name])) >= min_common
    )


def _pnp_pose(det, K, D, chess):
    """Board pose (R, t) in one camera + reprojection rms (px), K fixed.
    High rms here = bad corner detections OR a wrong K (autofocus drift)."""
    ids = sorted(det)
    obj = chess[ids].reshape(-1, 3).astype(np.float64)
    img = np.array([det[i] for i in ids], np.float64).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img.reshape(-1, 1, 2), K, D)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, 1))))
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), err


PNP_THR_PX = 3.0  # per-frame per-camera PnP reprojection gate
POSE_TOL_M = 0.05  # per-frame relative pose vs consensus: translation
POSE_TOL_DEG = 3.0  # ... and rotation


def pair_extrinsics(
    deta,
    detb,
    Ka,
    Da,
    Kb,
    Db,
    img_size,
    chess,
    min_common,
    max_frames=40,
    la="a",
    lb="b",
):
    """(R, T, rms, n_inliers) with x_b = R @ x_a + T, or None.

    stereoCalibrate has NO outlier rejection — a handful of frames with
    grazing-angle ChArUco misdetections or a focal drifted by autofocus
    poisons the whole solve (seen live: rms 100+ px). So each shared frame
    is vetted first: solvePnP per camera (K fixed) gives a per-frame board
    pose + reprojection error, the per-frame RELATIVE pose (R_b R_a^T) must
    agree with the consensus (medoid) within POSE_TOL, and only the inliers
    reach stereoCalibrate (evenly subsampled to max_frames — its LM has 6
    board-pose params per view, 150 views ran for tens of minutes). The
    printed per-camera PnP median identifies WHICH camera is bad: a high
    value on one camera = its detections or its K (autofocus?) are off.
    """
    frames = []  # (obj, pa, pb, pose_a, pose_b)
    for name in sorted(set(deta) & set(detb)):
        common = sorted(set(deta[name]) & set(detb[name]))
        if len(common) < min_common:
            continue
        da = {i: deta[name][i] for i in common}
        db = {i: detb[name][i] for i in common}
        frames.append(
            (
                chess[common].reshape(-1, 1, 3).astype(np.float32),
                np.array([da[i] for i in common], np.float32).reshape(
                    -1, 1, 2
                ),
                np.array([db[i] for i in common], np.float32).reshape(
                    -1, 1, 2
                ),
                _pnp_pose(da, Ka, Da, chess),
                _pnp_pose(db, Kb, Db, chess),
            )
        )
    if len(frames) < 6:
        return None
    err_a = np.median([f[3][2] for f in frames if f[3]])
    err_b = np.median([f[4][2] for f in frames if f[4]])
    print(
        f"    cam{la}<->cam{lb}: PnP median reproj cam{la} {err_a:.2f}px "
        f"cam{lb} {err_b:.2f}px"
        + (
            "   <- HIGH: that camera's detections or K are off "
            "(autofocus? grazing views?)"
            if max(err_a, err_b) > PNP_THR_PX
            else ""
        )
    )
    # per-frame relative pose, gated by PnP quality
    rel = []
    for k, (_, _, _, pa, pb) in enumerate(frames):
        if (
            pa is None
            or pb is None
            or pa[2] > PNP_THR_PX
            or pb[2] > PNP_THR_PX
        ):
            continue
        R = pb[0] @ pa[0].T
        rel.append((k, R, pb[1] - R @ pa[1]))
    if len(rel) < 6:
        print(
            f"    cam{la}<->cam{lb}: only {len(rel)} frames pass the "
            f"{PNP_THR_PX:.0f}px PnP gate (of {len(frames)}) — SKIPPED"
        )
        return None
    # consensus = translation medoid; keep frames agreeing in R and T
    ts = np.stack([t for _, _, t in rel])
    med = rel[int(np.argmin(((ts[None] - ts[:, None]) ** 2).sum(-1).sum(1)))]
    inl = []
    for k, R, t in rel:
        ang = np.degrees(
            np.arccos(np.clip((np.trace(R @ med[1].T) - 1) / 2, -1, 1))
        )
        if np.linalg.norm(t - med[2]) <= POSE_TOL_M and ang <= POSE_TOL_DEG:
            inl.append(k)
    if len(inl) < 6:
        print(
            f"    cam{la}<->cam{lb}: only {len(inl)} pose-consistent "
            f"frames — SKIPPED"
        )
        return None
    n_inl = len(inl)
    if n_inl > max_frames:
        inl = [
            inl[i] for i in np.linspace(0, n_inl - 1, max_frames).astype(int)
        ]
    dropped = len(frames) - n_inl
    if dropped:
        print(
            f"    cam{la}<->cam{lb}: dropped {dropped}/{len(frames)} "
            f"outlier frames"
        )
    rms, *_, R, T, _, _ = cv2.stereoCalibrate(
        [frames[k][0] for k in inl],
        [frames[k][1] for k in inl],
        [frames[k][2] for k in inl],
        Ka,
        Da,
        Kb,
        Db,
        img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(
            cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
            100,
            1e-5,
        ),
    )
    return (R.astype(np.float64), T.reshape(3).astype(np.float64), rms, n_inl)


def chain_extrinsics(cams, pairs, ref):
    """World(=cam{ref})->cam R,T for every camera, chaining pairwise links.

    pairs: {(a, b): (R, T, rms, n)} with x_b = R @ x_a + T. Greedy
    MIN-STEREO-RMS spanning tree from ref: a clean two-hop chain (0.4 px)
    beats a rotten direct link, and a poisoned pair never contaminates the
    cameras it touches. Returns (Rw, Tw, route) dicts; raises SystemExit
    naming any camera not connected to ref.
    """
    adj = {}
    for (a, b), (R, T, rms, _) in pairs.items():
        adj.setdefault(a, []).append((b, R, T, rms))
        adj.setdefault(b, []).append((a, R.T, -R.T @ T, rms))  # inverted link
    Rw, Tw = {ref: np.eye(3)}, {ref: np.zeros(3)}
    route = {ref: [ref]}
    while len(Rw) < len(cams):
        best = None  # (rms, from, to, R_ab, T_ab)
        for a in list(Rw):
            for b, R, T, rms in adj.get(a, []):
                if b not in Rw and (best is None or rms < best[0]):
                    best = (rms, a, b, R, T)
        if best is None:
            missing = [c for c in cams if c not in Rw]
            raise SystemExit(
                f"cams {missing} have no USABLE link to any calibrated "
                f"camera (no shared frames, or every shared pair was "
                f"rejected) — recapture sets linking them (any pair works, "
                f"e.g. side cam together with its nearest front cam)"
            )
        _, a, b, R, T = best
        Rw[b] = R @ Rw[a]
        Tw[b] = R @ Tw[a] + T
        route[b] = route[a] + [b]
    return Rw, Tw, route


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--cams",
        default="0,1,2,3",
        help="comma-separated camera ids (or the same /dev paths "
        "given to capture_calibration_multi — a path maps to "
        "its POSITION, matching the images/cam{pos} folders); "
        "the first is the reference",
    )
    ap.add_argument("--out", default="calibration_data/multi_params.npz")
    ap.add_argument(
        "--min-corners",
        type=int,
        default=6,
        help="min ChArUco corners to accept a detection",
    )
    ap.add_argument(
        "--max-link-rms",
        type=float,
        default=2.0,
        help="reject a pairwise link whose stereo rms (px) is "
        "above this — better to fail loudly than to save "
        "extrinsics that are metres off",
    )
    ap.add_argument(
        "--max-pair-frames",
        type=int,
        default=40,
        help="evenly subsample the shared frames of each pair to "
        "this many before stereoCalibrate (its LM step has 6 "
        "board-pose params PER VIEW; beyond a few dozen "
        "varied views the extrinsics stop improving and the "
        "solve time explodes)",
    )
    ap.add_argument(
        "--min-common",
        type=int,
        default=6,
        help="min corners shared cam0<->cam{c} to use a frame pair",
    )
    ap.add_argument(
        "--intr-flags",
        type=int,
        default=0,
        help="cv2.calibrateCamera flags for the intrinsics step",
    )
    args = ap.parse_args()

    _toks = [x.strip() for x in args.cams.split(",")]
    cams = [int(x) if x.isdigit() else i for i, x in enumerate(_toks)]
    if cams[0] != 0:
        print(
            f"NOTE: reference camera is cam{cams[0]}, but load_calibration "
            f"treats index 0 as the world frame — keep cam0 first."
        )
    board, dictionary = board_config.make_board()

    print(f"[1/3] Intrinsics for {len(cams)} cameras...")
    K, D, img_size = {}, {}, None
    for c in cams:
        res = camera_intrinsics(
            c, board, dictionary, args.min_corners, args.intr_flags
        )
        K[c], D[c], img_size = res[0], res[1], res[2]

    print("[2/3] Corner detections (cached per frame)...")
    dets = {
        c: detect_all(c, board, dictionary, args.min_corners) for c in cams
    }
    chess = board.getChessboardCorners()

    # connectivity warmup BEFORE any solve: which pairs share board frames
    # (the capture only ever needs 2 cameras at a time — this shows whether
    # the resulting graph reaches cam0 everywhere, and what will be chained)
    shared = {}
    print("  shared-frame overlap:")
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            a, b = cams[i], cams[j]
            shared[(a, b)] = count_shared(dets[a], dets[b], args.min_common)
            mark = "" if shared[(a, b)] >= 6 else "   <- too few, not solvable"
            print(f"    cam{a}<->cam{b}: {shared[(a, b)]:4d} frames{mark}")

    print(f"[3/3] Pairwise extrinsics, chained to cam{cams[0]}...")
    pairs = {}
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            a, b = cams[i], cams[j]
            if shared[(a, b)] < 6:
                continue
            r = pair_extrinsics(
                dets[a],
                dets[b],
                K[a],
                D[a],
                K[b],
                D[b],
                img_size,
                chess,
                args.min_common,
                args.max_pair_frames,
                la=a,
                lb=b,
            )
            if r is None:
                continue
            print(
                f"  cam{a}<->cam{b}: {r[3]} inlier frames, stereo rms "
                f"{r[2]:.3f} px, baseline {np.linalg.norm(r[1]) * 100:.1f} cm"
            )
            if r[2] > args.max_link_rms:
                print(
                    f"  cam{a}<->cam{b}: rms > {args.max_link_rms:.1f} px "
                    f"-> link REJECTED (won't poison the chain)"
                )
                continue
            pairs[(a, b)] = r
    R, T, route = chain_extrinsics(cams, pairs, cams[0])
    for c in cams[1:]:
        if len(route[c]) > 2:
            hops = " -> ".join(f"cam{x}" for x in route[c])
            print(
                f"  cam{c}: CHAINED {hops} (each hop compounds its stereo "
                f"error — direct shared views with cam{cams[0]} would "
                f"tighten it)"
            )

    out = {}
    for i, c in enumerate(cams):  # save in 0..n order
        out[f"K{i}"] = K[c]
        out[f"D{i}"] = D[c]
        out[f"R{i}"] = R[c]
        out[f"T{i}"] = T[c]
    out["cam_ids"] = np.array(cams)
    out["img_size"] = np.array(img_size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **out)
    print(f"\nSaved {len(cams)}-camera calibration → {args.out}")
    print(
        f"Run: python cosmik_hand_demo.py --cams {args.cams} --calib {args.out} "
        f"--cap-width {img_size[0]} --cap-height {img_size[1]}"
    )


if __name__ == "__main__":
    main()
