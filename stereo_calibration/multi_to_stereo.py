"""
Export a rectified 2-camera stereo_params.npz from an N-camera multi_params.npz.

Why: calibrate_multi.py emits K/D/R/T per camera (chained to cam0) — enough for
the live multiview demo (cosmik_hand_demo load_calibration) and DLT
triangulation, but NOT for the OFFLINE stereo path (extract_two_cameras.py ->
StereoTriangulator), which needs the RECTIFICATION (P1,P2,R1,R2,Q). This derives
the pair's relative pose from the shared multi calibration and runs
cv2.stereoRectify, writing the exact stereo_params.npz that calibrate_stereo.py
produces — no re-solve, so both systems share ONE world frame (cam0 of the
multi calib). That is what lets fastsam3d hands (2 cams) drop into the same
coordinates as the 4-cam cosmik body.

Convention (matches load_calibration + calibrate_multi): R_i, T_i map
cam0/world coords into camera i:  x_i = R_i x_world + T_i. For the stereo pair
(a = stereo reference = cam0 of the output, b = second cam) we need R,T with
x_b = R x_a + T:
    x_w = R_a^T (x_a - T_a)
    x_b = R_b x_w + T_b = (R_b R_a^T) x_a + (T_b - R_b R_a^T T_a)
=>  R = R_b R_a^T ,  T = T_b - R T_a

Usage:
    # cams 0 and 1 of a 0,1,2,3 multi calib become the stereo pair:
    python multi_to_stereo.py --multi calibration_data/multi_params.npz \
        --pair 0,1 --out calibration_data/stereo_params.npz

    # any pair, e.g. the two front cams are positions 0 and 2:
    python multi_to_stereo.py --multi calibration_data/multi_params.npz --pair 0,2

--pair takes the POSITIONS in the multi file (0-based, the same order given to
calibrate_multi --cams), NOT device ids. The first becomes stereo cam0.
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multi", default="calibration_data/multi_params.npz",
                    help="N-camera calib from calibrate_multi.py")
    ap.add_argument("--pair", default="0,1",
                    help="two camera POSITIONS in the multi file (first = "
                         "stereo cam0/reference)")
    ap.add_argument("--out", default="calibration_data/stereo_params.npz")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="cv2.stereoRectify free-scaling (0 = crop to valid "
                         "pixels, matches calibrate_stereo.py; 1 = keep all)")
    args = ap.parse_args()

    a, b = (int(x) for x in args.pair.split(","))
    z = np.load(args.multi)
    for i in (a, b):
        if f"K{i}" not in z:
            raise SystemExit(
                f"position {i} not in {args.multi} (keys: {sorted(z.keys())}) "
                f"— --pair uses 0-based POSITIONS in the multi --cams order")

    Ka, Da = z[f"K{a}"].astype(np.float64), z.get(f"D{a}", np.zeros(5)).astype(np.float64)
    Kb, Db = z[f"K{b}"].astype(np.float64), z.get(f"D{b}", np.zeros(5)).astype(np.float64)
    Ra = z[f"R{a}"].astype(np.float64) if f"R{a}" in z else np.eye(3)
    Rb = z[f"R{b}"].astype(np.float64) if f"R{b}" in z else np.eye(3)
    Ta = z[f"T{a}"].astype(np.float64).reshape(3) if f"T{a}" in z else np.zeros(3)
    Tb = z[f"T{b}"].astype(np.float64).reshape(3) if f"T{b}" in z else np.zeros(3)
    img_size = tuple(int(x) for x in z["img_size"])

    # relative pose: x_b = R x_a + T
    R = Rb @ Ra.T
    T = (Tb - R @ Ta).reshape(3, 1)
    baseline_cm = float(np.linalg.norm(T)) * 100
    print(f"pair cam{a}->cam{b}: baseline {baseline_cm:.1f} cm")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        Ka, Da, Kb, Db, img_size, R, T, alpha=args.alpha)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out,
             K1=Ka, D1=Da, K2=Kb, D2=Db,
             R=R, T=T,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             img_size=np.array(img_size),
             source_multi=os.path.abspath(args.multi),
             source_pair=np.array([a, b]))
    print(f"Saved rectified stereo (cam{a}=cam0, cam{b}=cam1) -> {args.out}")
    print(f"Run: python ../extract_two_cameras.py --stereo {args.out} ...")


if __name__ == "__main__":
    main()
