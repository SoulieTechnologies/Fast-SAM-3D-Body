"""
Live capture from N cameras with ChArUco overlay, for multi-camera calibration.

The 2-camera capture_calibration.py is left untouched; this is its N-camera
generalisation. Capture resolution MUST match the resolution the demo runs at
(intrinsics are resolution-dependent; cosmik_hand_demo hard-errors on mismatch).

On 's', every camera that currently detects the board saves a frame under
images/cam{c}/frame_{count}.png (same filename index across cameras, which is
how calibrate_multi.py pairs them). A shot is only committed when cam0 (the
reference) AND at least one other camera see the board — so cam0 always shares
the board with each partner, which is what the pairwise extrinsics need. Aim to
show the board to cam0 together with EACH other camera in turn.

Exposure/focus behave exactly like capture_calibration.py (V4L2 keeps settings
across sessions; --auto-exposure resets, --exposure/--gain set the same manual
values on all cameras; the per-view sharpness score is a focus aid).

Usage:
    python capture_calibration_multi.py --cams 0,2,4,6
    python capture_calibration_multi.py --cams 0,2,4,6 --width 1920 --height 1080

Controls:
    s    — save a frame set (cams that see the board; needs cam0 + >=1 other)
    -/+  — exposure down/up on ALL cameras (x1.4 steps)
    [/]  — gain down/up on ALL cameras
    a    — auto exposure on ALL cameras
    q    — quit
Then:  python calibrate_multi.py --cams 0,2,4,6
"""
import argparse
import os

import cv2
import numpy as np

import board_config

ap = argparse.ArgumentParser()
ap.add_argument("--cams", default="0,1,2,3",
                help="comma-separated camera device indices; first = reference")
ap.add_argument("--min_corners", type=int, default=6)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--auto-exposure", action="store_true")
ap.add_argument("--exposure", type=float, default=None)
ap.add_argument("--gain", type=float, default=None)
ap.add_argument("--rotate180", action="store_true",
                help="rotate every frame 180 (upside-down mount); the demo MUST "
                     "then run with the same flag")
args = ap.parse_args()

cam_ids = [int(x) for x in args.cams.split(",")]
board, dictionary = board_config.make_board()
N_CORNERS = (board_config.BOARD_COLS - 1) * (board_config.BOARD_ROWS - 1)

for c in cam_ids:
    os.makedirs(f"images/cam{c}", exist_ok=True)

caps = []
for c in cam_ids:
    cap = cv2.VideoCapture(c)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {c}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
        if args.gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, args.gain)
    elif args.auto_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"cam{c}: {w}x{h} auto_exp={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):g} "
          f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):g} "
          f"gain={cap.get(cv2.CAP_PROP_GAIN):g}")
    if w != args.width:
        print(f"  WARNING: cam{c} refused {args.width}x{args.height}")
    caps.append(cap)

count = 0
sharp_peak = [0.0] * len(cam_ids)
cur_exp = args.exposure if args.exposure is not None \
    else max(caps[0].get(cv2.CAP_PROP_EXPOSURE), 1.0)
cur_gain = args.gain if args.gain is not None else caps[0].get(cv2.CAP_PROP_GAIN)
manual_exp = args.exposure is not None


def set_exposure(exp=None, gain=None, auto=False):
    global cur_exp, cur_gain, manual_exp
    for cap in caps:
        if auto:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            if exp is not None:
                cap.set(cv2.CAP_PROP_EXPOSURE, exp)
            if gain is not None:
                cap.set(cv2.CAP_PROP_GAIN, gain)
    manual_exp = not auto
    if auto:
        print("exposure -> AUTO (all cams)")
    else:
        cur_exp = exp if exp is not None else cur_exp
        cur_gain = gain if gain is not None else cur_gain
        print(f"exposure -> {cur_exp:.0f}  gain -> {cur_gain:.0f} (all cams)")


def sharpness(gray, pts):
    h, w = gray.shape
    if pts is not None and len(pts):
        p = pts.reshape(-1, 2)
        x0, y0 = np.maximum(p.min(0).astype(int) - 20, 0)
        x1, y1 = np.minimum(p.max(0).astype(int) + 20, (w, h))
    else:
        x0, x1, y0, y1 = w // 3, 2 * w // 3, h // 3, 2 * h // 3
    roi = gray[y0:y1, x0:x1]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0


def overlay(idx, frame, gray):
    disp = frame.copy()
    ch, ids, mk, mkids = board_config.detect_charuco(
        gray, board, dictionary, min_corners=args.min_corners)
    if mkids is not None and len(mkids):
        cv2.aruco.drawDetectedMarkers(disp, mk, mkids)
    pts = None
    if mk is not None and len(mk):
        pts = np.concatenate([c.reshape(-1, 2) for c in mk])
    elif ch is not None:
        pts = ch
    s = sharpness(gray, pts)
    sharp_peak[idx] = max(sharp_peak[idx], s)
    near = s >= 0.9 * sharp_peak[idx] and sharp_peak[idx] > 0
    cv2.putText(disp, f"sharp {s:.0f} (peak {sharp_peak[idx]:.0f})", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if near else (0, 165, 255), 2)
    ok = ch is not None
    if ok:
        for pt in ch.reshape(-1, 2):
            cv2.circle(disp, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
        cv2.putText(disp, f"cam{cam_ids[idx]}  corners {len(ch)}/{N_CORNERS}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(disp, f"cam{cam_ids[idx]}  no board", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return disp, ok


def grid(images):
    """Tile N views into a roughly-square, screen-fitting mosaic."""
    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w = images[0].shape[:2]
    blank = np.zeros_like(images[0])
    cells = images + [blank] * (rows * cols - n)
    mosaic = cv2.vconcat([cv2.hconcat(cells[r * cols:(r + 1) * cols])
                          for r in range(rows)])
    if mosaic.shape[1] > 2400:
        s = 2400.0 / mosaic.shape[1]
        mosaic = cv2.resize(mosaic, None, fx=s, fy=s)
    return mosaic


print("Press 's' to save a frame set, 'q' to quit.")
while True:
    reads = [cap.read() for cap in caps]
    if not all(r[0] for r in reads):
        print("camera read failed")
        break
    frames = [cv2.rotate(f, cv2.ROTATE_180) if args.rotate180 else f
              for _, f in reads]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    disps, oks = zip(*[overlay(i, frames[i], grays[i]) for i in range(len(caps))])
    disps = list(disps)

    exp_txt = (f"exp {cur_exp:.0f} gain {cur_gain:.0f}" if manual_exp
               else "exp auto")
    n_ok = sum(oks)
    for d in disps:
        cv2.putText(d, f"saved {count}  seen {n_ok}/{len(caps)}  {exp_txt}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Multi-cam calibration  [s=save q=quit]", grid(disps))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key in (ord('+'), ord('=')):
        set_exposure(exp=min(cur_exp * 1.4, 5000))
    if key == ord('-'):
        set_exposure(exp=max(cur_exp / 1.4, 1.0))
    if key == ord(']'):
        set_exposure(gain=min((cur_gain or 0) + 10, 255))
    if key == ord('['):
        set_exposure(gain=max((cur_gain or 0) - 10, 0))
    if key == ord('a'):
        set_exposure(auto=True)
    if key == ord('s'):
        # need the reference AND at least one partner so cam0 shares the board
        if oks[0] and n_ok >= 2:
            saved = []
            for i, c in enumerate(cam_ids):
                if oks[i]:
                    cv2.imwrite(f"images/cam{c}/frame_{count:04d}.png", frames[i])
                    saved.append(c)
            print(f"saved set {count}: cams {saved}")
            count += 1
        else:
            print(f"not saved — cam{cam_ids[0]} + >=1 other must see the board "
                  f"(cam0 ok={oks[0]}, total seen={n_ok})")

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
print(f"Done. {count} frame sets saved. Next: python calibrate_multi.py "
      f"--cams {args.cams}")
