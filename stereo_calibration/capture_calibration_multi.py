"""
Live capture from N cameras with ChArUco overlay, for multi-camera calibration.

The 2-camera capture_calibration.py is left untouched; this is its N-camera
generalisation. Capture resolution MUST match the resolution the demo runs at
(intrinsics are resolution-dependent; cosmik_hand_demo hard-errors on mismatch).

On 's', every camera that currently detects the board saves a frame under
images/cam{c}/frame_{count}.png (same filename index across cameras, which is
how calibrate_multi.py pairs them). A shot is committed when ANY 2+ cameras
see the board: calibrate_multi solves every pair that shares frames and
CHAINS the transforms to cam0, so the cameras only need to form a connected
graph (e.g. front pair together, then front-left + side-left, then
front-right + side-right). Direct cam0 overlap still gives the tightest
extrinsics — prefer it when the geometry allows.

Exposure/focus behave exactly like capture_calibration.py (V4L2 keeps settings
across sessions; --auto-exposure resets, --exposure/--gain set the same manual
values on all cameras; the per-view sharpness score is a focus aid).

Usage:
    python capture_calibration_multi.py --cams 0,2,4,6
    python capture_calibration_multi.py --cams 0,2,4,6 --width 1920 --height 1080

Controls:
    s    — save a frame set (cams that see the board; needs >=2 cameras)
    -/+  — exposure down/up on ALL cameras (x1.4 steps)
    [/]  — gain down/up on ALL cameras
    a    — auto exposure on ALL cameras
    q    — quit
Then:  python calibrate_multi.py --cams 0,2,4,6
"""
import argparse
import glob
import os

import cv2
import numpy as np

import board_config

ap = argparse.ArgumentParser()
ap.add_argument("--cams", default="0,1,2,3",
                help="comma-separated camera indices OR /dev/v4l/by-id paths "
                     "(stable across reboots — recommended with 4 identical "
                     "cams); first = reference. A path is labelled/saved by "
                     "its POSITION (images/cam{pos})")
ap.add_argument("--min_corners", type=int, default=6)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--auto-exposure", action="store_true")
ap.add_argument("--lock-focus", action="store_true",
                help="disable autofocus on every camera (no-op on fixed-focus "
                     "models). REQUIRED with an autofocus camera: refocusing "
                     "shifts the effective focal length. The demo must then "
                     "run with the same --lock-focus/--focus")
ap.add_argument("--focus", type=float, default=None,
                help="fixed manual focus (V4L2 units, typically 0-255; "
                     "implies --lock-focus). Pick it with the sharpness "
                     "readout: corners N/88 + sharp value on the board")
ap.add_argument("--exposure", type=float, default=None)
ap.add_argument("--gain", type=float, default=None)
ap.add_argument("--display-width", type=int, default=1700,
                help="max preview mosaic width (px); the window is also "
                     "freely resizable with the mouse")
ap.add_argument("--display-height", type=int, default=900,
                help="max preview mosaic height (px) — lower it if the "
                     "bottom row is cut off by the taskbar")
ap.add_argument("--rotate180", action="store_true",
                help="rotate every frame 180 (upside-down mount); the demo MUST "
                     "then run with the same flag")
args = ap.parse_args()

# int token = device index (label = itself, unchanged); path token = opened
# as-is by V4L2, labelled by its position so folders stay images/cam{N}
_toks = [x.strip() for x in args.cams.split(",")]
cam_ids = [int(x) if x.isdigit() else x for x in _toks]
cam_labs = [c if isinstance(c, int) else i for i, c in enumerate(cam_ids)]
board, dictionary = board_config.make_board()
N_CORNERS = (board_config.BOARD_COLS - 1) * (board_config.BOARD_ROWS - 1)

for c in cam_labs:
    os.makedirs(f"images/cam{c}", exist_ok=True)

caps = []
for c in cam_ids:
    cap = cv2.VideoCapture(c)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {c}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # keep at most 1 buffered frame: with a slow 4x1080p preview loop the
    # default queue serves frames that are 100s of ms old, with a DIFFERENT
    # age per camera -> a hand-held board lands at different poses in the
    # same "simultaneous" set and the pairwise extrinsics are garbage
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
        if args.gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, args.gain)
    elif args.auto_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    if args.lock_focus or args.focus is not None:
        # autofocus = focal length drift (focus breathing) → K would be wrong
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if args.focus is not None:
            cap.set(cv2.CAP_PROP_FOCUS, args.focus)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"cam{c}: {w}x{h} auto_exp={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):g} "
          f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):g} "
          f"gain={cap.get(cv2.CAP_PROP_GAIN):g}"
          + (f" focus={cap.get(cv2.CAP_PROP_FOCUS):g}"
             f" af={cap.get(cv2.CAP_PROP_AUTOFOCUS):g}"
             if (args.lock_focus or args.focus is not None) else ""))
    if w != args.width:
        print(f"  WARNING: cam{c} refused {args.width}x{args.height}")
    caps.append(cap)

# resume numbering after any existing capture: sets ADD to the pool (drop
# a bad zone by deleting its frame range; calibrate_multi pairs by filename)
_existing = glob.glob("images/cam*/frame_*.png")
count = 1 + max((int(os.path.basename(f)[6:10]) for f in _existing),
                default=-1)
if count:
    print(f"resuming at set {count} ({len(_existing)} existing images kept)")
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
    cmap = {}
    if ok:
        cmap = {int(ids[k]): ch[k].reshape(2) for k in range(len(ids))}
        for pt in ch.reshape(-1, 2):
            cv2.circle(disp, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
        cv2.putText(disp, f"cam{cam_labs[idx]}  corners {len(ch)}/{N_CORNERS}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(disp, f"cam{cam_labs[idx]}  no board", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return disp, ok, cmap


def grid(images):
    """Tile N views into a roughly-square, screen-fitting mosaic."""
    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    blank = np.zeros_like(images[0])
    cells = images + [blank] * (rows * cols - n)
    mosaic = cv2.vconcat([cv2.hconcat(cells[r * cols:(r + 1) * cols])
                          for r in range(rows)])
    # fit BOTH dimensions on screen (4x1080p in 2x2 is 3840x2160 raw)
    s = min(1.0, args.display_width / mosaic.shape[1],
            args.display_height / mosaic.shape[0])
    if s < 1.0:
        mosaic = cv2.resize(mosaic, None, fx=s, fy=s)
    return mosaic


WIN = "Multi-cam calibration  [s=save q=quit]"
# WINDOW_NORMAL: the preview scales to the window -> drag it to any size
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
_win_sized = False

MOVE_TOL_PX = 4.0     # max corner motion since the previous preview frame
prev_corners = [{} for _ in caps]

print("Press 's' to save a frame set, 'q' to quit (hold the board STILL "
      "when saving — the cameras are only soft-synced).")
while True:
    # grab all cameras first (fast buffer dequeue, ~ms apart), THEN decode:
    # sequential read() would stagger the actual capture instants
    if not all(cap.grab() for cap in caps):
        print("camera grab failed")
        break
    reads = [cap.retrieve() for cap in caps]
    if not all(r[0] for r in reads):
        print("camera retrieve failed")
        break
    frames = [cv2.rotate(f, cv2.ROTATE_180) if args.rotate180 else f
              for _, f in reads]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    disps, oks, cmaps = zip(*[overlay(i, frames[i], grays[i])
                              for i in range(len(caps))])
    disps = list(disps)
    # board motion per cam = mean corner displacement vs previous iteration
    motion = []
    for i, cm in enumerate(cmaps):
        common = set(cm) & set(prev_corners[i])
        motion.append(float(np.mean([np.linalg.norm(cm[k] - prev_corners[i][k])
                                     for k in common])) if common else None)
        prev_corners[i] = cm

    exp_txt = (f"exp {cur_exp:.0f} gain {cur_gain:.0f}" if manual_exp
               else "exp auto")
    n_ok = sum(oks)
    for d in disps:
        cv2.putText(d, f"saved {count}  seen {n_ok}/{len(caps)}  {exp_txt}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    mosaic = grid(disps)
    if not _win_sized:                    # start at the fitted size once
        cv2.resizeWindow(WIN, mosaic.shape[1], mosaic.shape[0])
        _win_sized = True
    cv2.imshow(WIN, mosaic)

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
        moving = [f"cam{cam_labs[i]} {m:.0f}px" for i, m in enumerate(motion)
                  if oks[i] and m is not None and m > MOVE_TOL_PX]
        # any pair is useful: calibrate_multi chains pairwise links to cam0
        if moving:
            print(f"not saved — board MOVING ({', '.join(moving)}): the "
                  f"cameras are not hardware-synced, hold it still")
        elif n_ok >= 2:
            saved = []
            for i, c in enumerate(cam_labs):
                if oks[i]:
                    cv2.imwrite(f"images/cam{c}/frame_{count:04d}.png", frames[i])
                    saved.append(c)
            print(f"saved set {count}: cams {saved}")
            count += 1
        else:
            print(f"not saved — >=2 cameras must see the board "
                  f"(total seen={n_ok})")

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
print(f"Done. {count} frame sets saved. Next: python calibrate_multi.py "
      f"--cams {args.cams}")
