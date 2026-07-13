"""
Live capture from two cameras with ChArUco detection overlay.

Usage:
    python capture_calibration.py --cam0 0 --cam1 2

The capture resolution MUST match the resolution the demo runs at
(intrinsics are resolution-dependent; cosmik_hand_demo hard-errors on
mismatch). Default is 1920x1080 -> run the demo with
--cap-width 1920 --cap-height 1080.

Exposure: V4L2 cameras KEEP their settings across sessions, so one camera
can be stuck in manual exposure from a previous tool while the other is in
auto (one feed dark, one bright). --auto-exposure resets both to auto;
--exposure N (+ optional --gain N) sets both to the SAME manual values.
Actual per-camera values are printed at startup either way.

Focus: each view shows the ChArUco corner count and a sharpness score
(Laplacian variance on the board region when detected). Turn the focus
ring to maximize "sharp" — the peak seen so far is shown, so overshoot,
come back, and stop at the max (corner count should peak with it).

Controls:
    s  — save current frame pair (only saved when corners detected in BOTH)
    q  — quit
"""
import argparse
import os
import cv2
import numpy as np
import board_config

parser = argparse.ArgumentParser()
parser.add_argument("--cam0", type=int, default=0)
parser.add_argument("--cam1", type=int, default=2)
parser.add_argument("--min_corners", type=int, default=6,
                    help="Minimum ChArUco corners required to accept a frame")
parser.add_argument("--width", type=int, default=1920,
                    help="capture width — MUST match the resolution the demo runs at "
                         "(intrinsics are resolution-dependent)")
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--auto-exposure", action="store_true",
                    help="force auto exposure on BOTH cameras (fixes one-dark-"
                         "one-bright when a camera kept manual settings)")
parser.add_argument("--exposure", type=float, default=None,
                    help="manual exposure for BOTH cameras (V4L2 units, "
                         "typically 3..2047; overrides --auto-exposure)")
parser.add_argument("--gain", type=float, default=None,
                    help="manual gain for BOTH cameras (with --exposure)")
args = parser.parse_args()

board, dictionary = board_config.make_board()
N_CORNERS = (board_config.BOARD_COLS - 1) * (board_config.BOARD_ROWS - 1)

os.makedirs("images/cam0", exist_ok=True)
os.makedirs("images/cam1", exist_ok=True)

cap0 = cv2.VideoCapture(args.cam0)
cap1 = cv2.VideoCapture(args.cam1)
assert cap0.isOpened(), f"Cannot open camera {args.cam0}"
assert cap1.isOpened(), f"Cannot open camera {args.cam1}"

for i, cap in enumerate((cap0, cap1)):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # USB bandwidth
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: 1 = manual
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
        if args.gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, args.gain)
    elif args.auto_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # V4L2: 3 = auto

    print(f"cam{i}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  "
          f"auto_exp={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):g} "
          f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):g} "
          f"gain={cap.get(cv2.CAP_PROP_GAIN):g} "
          f"brightness={cap.get(cv2.CAP_PROP_BRIGHTNESS):g}")

if (cap0.get(cv2.CAP_PROP_FRAME_WIDTH) != args.width or
        cap1.get(cv2.CAP_PROP_FRAME_WIDTH) != args.width):
    print(f"WARNING: a camera refused {args.width}x{args.height} — the demo "
          f"must then run at the resolution shown above.")

count = 0
sharp_peak = [0.0, 0.0]


def sharpness(gray, pts):
    """Laplacian variance on the board bounding box (or center crop)."""
    h, w = gray.shape
    if pts is not None and len(pts):
        p = pts.reshape(-1, 2)
        x0, y0 = np.maximum(p.min(axis=0).astype(int) - 20, 0)
        x1, y1 = np.minimum(p.max(axis=0).astype(int) + 20, (w, h))
    else:
        x0, x1, y0, y1 = w // 3, 2 * w // 3, h // 3, 2 * h // 3
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def draw_overlay(idx, frame, gray):
    display = frame.copy()
    ch_corners, ch_ids, mk_corners, mk_ids = board_config.detect_charuco(
        gray, board, dictionary, min_corners=args.min_corners)
    if mk_ids is not None and len(mk_ids):
        cv2.aruco.drawDetectedMarkers(display, mk_corners, mk_ids)

    # focus assist: sharpness on the board region (markers > corners > center)
    pts = None
    if mk_corners is not None and len(mk_corners):
        pts = np.concatenate([c.reshape(-1, 2) for c in mk_corners])
    elif ch_corners is not None:
        pts = ch_corners
    sharp = sharpness(gray, pts)
    sharp_peak[idx] = max(sharp_peak[idx], sharp)
    near_peak = sharp >= 0.9 * sharp_peak[idx] and sharp_peak[idx] > 0
    cv2.putText(display,
                f"sharp: {sharp:5.0f}  (peak {sharp_peak[idx]:.0f})",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if near_peak else (0, 165, 255), 2)

    ok = ch_corners is not None
    if ok:
        for pt in ch_corners.reshape(-1, 2):
            cv2.circle(display, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
        cv2.putText(display, f"corners: {len(ch_corners)}/{N_CORNERS}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(display, "no board", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return display, ok


print("Press 's' to save a frame pair, 'q' to quit.")
while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if not ret0 or not ret1:
        print("Camera read failed.")
        break

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    disp0, ok0 = draw_overlay(0, frame0, gray0)
    disp1, ok1 = draw_overlay(1, frame1, gray1)

    cv2.putText(disp0, f"saved: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(disp1, f"saved: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    combined = cv2.hconcat([disp0, disp1])
    if combined.shape[1] > 2600:  # 2x1080p doesn't fit on screen
        combined = cv2.resize(combined, None, fx=0.66, fy=0.66)
    cv2.imshow("Stereo Calibration Capture  [s=save  q=quit]", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        if ok0 and ok1:
            cv2.imwrite(f"images/cam0/frame_{count:04d}.png", frame0)
            cv2.imwrite(f"images/cam1/frame_{count:04d}.png", frame1)
            print(f"Saved pair {count}")
            count += 1
        else:
            print("Board not detected in both cameras — frame not saved.")

cap0.release()
cap1.release()
cv2.destroyAllWindows()
print(f"Done. {count} frame pairs saved.")
