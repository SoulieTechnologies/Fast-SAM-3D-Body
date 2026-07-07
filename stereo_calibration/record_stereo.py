"""
Record synchronized stereo video from both cameras.

Usage:
    python record_stereo.py --cam0 0 --cam1 1 --name subject_01

Output:
    recordings/<name>/cam0.mp4
    recordings/<name>/cam1.mp4

Controls:
    r  — start / stop recording
    q  — quit (saves if recording)
"""
import argparse
import os
import time
import threading
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--cam0", type=int, default=0)
parser.add_argument("--cam1", type=int, default=1)
parser.add_argument("--name", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
parser.add_argument("--fps", type=float, default=30.0)
args = parser.parse_args()

out_dir = os.path.join("recordings", args.name)
os.makedirs(out_dir, exist_ok=True)

cap0 = cv2.VideoCapture(args.cam0)
cap1 = cv2.VideoCapture(args.cam1)
for cap in (cap0, cap1):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer0 = writer1 = None
recording = False
frame_count = 0

# Lock-stepped reads in a background thread for tightest sync
frame0 = frame1 = None
lock = threading.Lock()
stop_event = threading.Event()

def grab_loop():
    global frame0, frame1
    while not stop_event.is_set():
        r0, f0 = cap0.read()
        r1, f1 = cap1.read()
        if r0 and r1:
            with lock:
                frame0, frame1 = f0, f1

grabber = threading.Thread(target=grab_loop, daemon=True)
grabber.start()

# Wait for first frames
time.sleep(0.2)

print(f"Output → {out_dir}/cam0.mp4  &  cam1.mp4")
print("Press 'r' to start/stop recording, 'q' to quit.")

WINNAME = "Stereo Record  [r=rec  q=quit]"
cv2.namedWindow(WINNAME, cv2.WINDOW_NORMAL)
placeholder = __import__("numpy").zeros((720, 2560, 3), dtype="uint8")
cv2.putText(placeholder, "Waiting for cameras...", (80, 360),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 255), 3)
cv2.imshow(WINNAME, placeholder)

while True:
    with lock:
        f0 = frame0.copy() if frame0 is not None else None
        f1 = frame1.copy() if frame1 is not None else None

    if f0 is None or f1 is None:
        cv2.imshow(WINNAME, placeholder)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        continue

    if recording and writer0:
        writer0.write(f0)
        writer1.write(f1)
        frame_count += 1

    disp0, disp1 = f0.copy(), f1.copy()
    status = f"REC  {frame_count} frames" if recording else "READY — press r"
    color  = (0, 0, 255) if recording else (0, 255, 0)
    for d in (disp0, disp1):
        cv2.putText(d, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imshow(WINNAME, cv2.hconcat([disp0, disp1]))
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        if not recording:
            writer0 = cv2.VideoWriter(os.path.join(out_dir, "cam0.mp4"), fourcc, args.fps, (1280, 720))
            writer1 = cv2.VideoWriter(os.path.join(out_dir, "cam1.mp4"), fourcc, args.fps, (1280, 720))
            frame_count = 0
            recording = True
            print("Recording started...")
        else:
            recording = False
            writer0.release(); writer1.release()
            print(f"Saved {frame_count} frames → {out_dir}/")

    elif key == ord('q'):
        if recording:
            writer0.release(); writer1.release()
            print(f"Saved {frame_count} frames → {out_dir}/")
        break

stop_event.set()
cap0.release(); cap1.release()
cv2.destroyAllWindows()
