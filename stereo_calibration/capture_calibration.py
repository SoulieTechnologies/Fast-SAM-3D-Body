"""
Live capture from two cameras with ChArUco detection overlay.

Usage:
    python capture_calibration.py --cam0 0 --cam1 2

Controls:
    s  — save current frame pair (only saved when corners detected in BOTH)
    q  — quit
"""
import argparse
import os
import cv2
import board_config

parser = argparse.ArgumentParser()
parser.add_argument("--cam0", type=int, default=0)
parser.add_argument("--cam1", type=int, default=2)
parser.add_argument("--min_corners", type=int, default=6,
                    help="Minimum ChArUco corners required to accept a frame")
args = parser.parse_args()

board, dictionary = board_config.make_board()
detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

os.makedirs("images/cam0", exist_ok=True)
os.makedirs("images/cam1", exist_ok=True)

cap0 = cv2.VideoCapture(args.cam0)
cap1 = cv2.VideoCapture(args.cam1)
assert cap0.isOpened(), f"Cannot open camera {args.cam0}"
assert cap1.isOpened(), f"Cannot open camera {args.cam1}"

for cap in (cap0, cap1):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = 0


def detect_charuco(gray):
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None, None
    retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if retval < args.min_corners:
        return None, None
    return ch_corners, ch_ids


def draw_overlay(frame, gray):
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    display = frame.copy()
    if ids is not None and len(ids) >= 4:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
        retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if retval >= args.min_corners:
            cv2.aruco.drawDetectedCornersCharuco(display, ch_corners, ch_ids)
            cv2.putText(display, f"corners: {retval}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return display, True
    cv2.putText(display, "no board", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return display, False


print("Press 's' to save a frame pair, 'q' to quit.")
while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if not ret0 or not ret1:
        print("Camera read failed.")
        break

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    disp0, ok0 = draw_overlay(frame0, gray0)
    disp1, ok1 = draw_overlay(frame1, gray1)

    cv2.putText(disp0, f"saved: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(disp1, f"saved: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    combined = cv2.hconcat([disp0, disp1])
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
