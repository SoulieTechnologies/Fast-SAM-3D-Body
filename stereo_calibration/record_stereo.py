"""Record synchronized 1080p stereo video from two cameras, feed flipped 180
(to match the upside-down calibrated mounting). Saves locally to
recordings/<name>/cam0.mp4 & cam1.mp4 for offline inference.

    python record_stereo.py --cam0 0 --cam1 1 --name take_01

The 180 flip is BAKED into the files, so downstream you read them as-is:
    python extract_two_cameras.py --cam0 recordings/take_01/cam0.mp4 \
        --cam1 recordings/take_01/cam1.mp4 --stereo <stereo_params.npz> ...

Controls:  r = start/stop recording   q = quit (saves if recording)
"""
import argparse
import os
import threading
import time

import cv2

p = argparse.ArgumentParser()
p.add_argument("--cam0", type=int, default=0)
p.add_argument("--cam1", type=int, default=1)
p.add_argument("--name", default=time.strftime("%Y%m%d_%H%M%S"))
p.add_argument("--fps", type=float, default=30.0)
args = p.parse_args()

W, H = 1920, 1080
out_dir = os.path.join("recordings", args.name)
os.makedirs(out_dir, exist_ok=True)


def open_cam(idx):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {idx}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    print(f"  cam {idx}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


cap0, cap1 = open_cam(args.cam0), open_cam(args.cam1)

frame0 = frame1 = None
lock = threading.Lock()
stop = threading.Event()


def grab_loop():
    global frame0, frame1
    while not stop.is_set():
        cap0.grab()
        cap1.grab()                       # grab both, then retrieve both (sync)
        r0, f0 = cap0.retrieve()
        r1, f1 = cap1.retrieve()
        if r0 and r1:
            with lock:                    # flip 180 baked in
                frame0 = cv2.rotate(f0, cv2.ROTATE_180)
                frame1 = cv2.rotate(f1, cv2.ROTATE_180)


threading.Thread(target=grab_loop, daemon=True).start()
time.sleep(0.3)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer0 = writer1 = None
recording = False
n = 0
print(f"Output -> {out_dir}/cam0.mp4 & cam1.mp4  (1920x1080, flipped 180)")
print("r = start/stop,  q = quit")
WIN = "Stereo Record  [r=rec q=quit]"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

while True:
    with lock:
        f0 = None if frame0 is None else frame0.copy()
        f1 = None if frame1 is None else frame1.copy()
    if f0 is None or f1 is None:
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break
        continue

    if recording and writer0 is not None:
        writer0.write(f0)
        writer1.write(f1)
        n += 1

    d0, d1 = f0.copy(), f1.copy()
    txt = f"REC {n}" if recording else "READY - press r"
    col = (0, 0, 255) if recording else (0, 255, 0)
    for d in (d0, d1):
        cv2.putText(d, txt, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
    both = cv2.hconcat([d0, d1])
    cv2.imshow(WIN, cv2.resize(both, (1600, int(1600 * both.shape[0] / both.shape[1]))))

    k = cv2.waitKey(1) & 0xFF
    if k == ord("r"):
        if not recording:
            writer0 = cv2.VideoWriter(f"{out_dir}/cam0.mp4", fourcc, args.fps, (W, H))
            writer1 = cv2.VideoWriter(f"{out_dir}/cam1.mp4", fourcc, args.fps, (W, H))
            n, recording = 0, True
            print("recording...")
        else:
            recording = False
            writer0.release(); writer1.release()
            print(f"saved {n} frames -> {out_dir}/")
    elif k == ord("q"):
        break

stop.set()
if recording and writer0 is not None:
    writer0.release(); writer1.release()
    print(f"saved {n} frames -> {out_dir}/")
cap0.release(); cap1.release()
cv2.destroyAllWindows()
