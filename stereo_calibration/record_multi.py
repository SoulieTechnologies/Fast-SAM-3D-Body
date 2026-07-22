"""Record synchronized N-camera video for OFFLINE inference (record now, infer
later). Generalises record_stereo.py to any number of cameras and writes the
layout cosmik_hand_demo.py --replay expects.

Lockstep capture: every tick we grab ALL cameras then retrieve ALL (soft sync,
~ms apart), and a frame set is committed to disk ONLY when every camera
returned a fresh frame — so cam{i}.mp4 frame k is the same instant across all
files (what offline triangulation needs; a per-camera drop would shear the
geometry). Encoding runs in one writer thread per camera (bounded queues) so a
slow encoder never stalls capture; if a queue backs up the WHOLE set is dropped
(never a single camera), keeping the files index-aligned. Actual per-frame
timestamps are saved for the exact playback rate + sync spread.

The 180° flip (--rotate180) is BAKED into the files, matching the upside-down
calibrated mounting — so downstream reads them as-is (do NOT pass --rotate180
to cosmik_hand_demo --replay when it was baked here). --lock-focus/--focus must
match what capture_calibration_multi.py used (autofocus drifts K).

    python record_multi.py --cams 0,1,2,3 --name take_01 --rotate180 --lock-focus
    # then, later:
    python ../cosmik_hand_demo.py --cams 0,1,2,3 --calib calibration_data/multi_params.npz \
        --cap-width 1920 --cap-height 1080 --replay recordings/take_01
    # fastsam3d hands on just 2 of them (positions 0,1 of this recording):
    python ../cosmik_hand_demo.py --cams 0,1 --calib calibration_data/stereo_params.npz \
        --cap-width 1920 --cap-height 1080 --replay recordings/take_01

Controls:  r = start/stop recording   q = quit (saves if recording)
"""

import argparse
import json
import os
import queue
import threading
import time

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument(
    "--cams",
    default="0,1,2,3",
    help="comma-separated camera indices OR /dev/v4l/by-id paths "
    "(stable across reboots — recommended with identical cams); "
    "order MUST match the calibration --cams / K0..K{n}",
)
p.add_argument("--name", default=time.strftime("%Y%m%d_%H%M%S"))
p.add_argument(
    "--fps",
    type=float,
    default=30.0,
    help="requested camera fps AND the container fps written; the "
    "true rate is in timestamps.npy regardless",
)
p.add_argument("--width", type=int, default=1920)
p.add_argument("--height", type=int, default=1080)
p.add_argument(
    "--rotate180",
    action="store_true",
    help="bake a 180° flip into every file (upside-down mount) — "
    "match the calibration; then don't re-flip at replay",
)
p.add_argument(
    "--lock-focus",
    action="store_true",
    help="disable autofocus on every camera (match calibration; "
    "no-op on fixed-focus models)",
)
p.add_argument(
    "--focus",
    type=float,
    default=None,
    help="fixed manual focus (implies --lock-focus); use the SAME "
    "value as during calibration",
)
p.add_argument(
    "--exposure",
    type=float,
    default=None,
    help="fixed manual exposure on all cams (else camera default)",
)
p.add_argument("--gain", type=float, default=None)
p.add_argument(
    "--codec",
    default="mp4v",
    help="VideoWriter fourcc (mp4v = compatible default; try MJPG "
    "for lighter compression at a larger file)",
)
p.add_argument(
    "--queue",
    type=int,
    default=64,
    help="max buffered frame sets per camera before a set is "
    "dropped to protect capture (raise if you have RAM and see "
    "drops; a drop is reported live)",
)
p.add_argument("--display-width", type=int, default=1600)
args = p.parse_args()

W, H = args.width, args.height
_toks = [x.strip() for x in args.cams.split(",")]
cam_ids = [int(x) if x.isdigit() else x for x in _toks]
cam_labs = list(range(len(cam_ids)))  # files/positions are 0..n-1
ncam = len(cam_ids)
out_dir = os.path.join("recordings", args.name)
os.makedirs(out_dir, exist_ok=True)


def open_cam(cid):
    cap = cv2.VideoCapture(cid)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {cid}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # freshest frame, low latency
    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
        if args.gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, args.gain)
    if args.lock_focus or args.focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if args.focus is not None:
            cap.set(cv2.CAP_PROP_FOCUS, args.focus)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"  cam {cid}: {w}x{h}"
        + (
            f" focus={cap.get(cv2.CAP_PROP_FOCUS):g}"
            if (args.lock_focus or args.focus is not None)
            else ""
        )
    )
    if w != W or h != H:
        print(f"  WARNING: cam {cid} refused {W}x{H} — got {w}x{h}")
    return cap


caps = [open_cam(c) for c in cam_ids]

latest = [None] * ncam  # freshest (frame, retrieve_ts) per cam
lock = threading.Lock()
stop = threading.Event()
cap_fps = [0.0] * ncam


def grab_loop():
    tprev = [None] * ncam
    while not stop.is_set():
        # grab all first (near-simultaneous shutter), then decode
        for cap in caps:
            cap.grab()
        now = time.time()
        for i, cap in enumerate(caps):
            r, f = cap.retrieve()
            if not r or f is None:
                continue
            if args.rotate180:
                f = cv2.rotate(f, cv2.ROTATE_180)
            if tprev[i] is not None and now > tprev[i]:
                inst = 1.0 / (now - tprev[i])
                cap_fps[i] = (
                    inst if cap_fps[i] == 0 else 0.9 * cap_fps[i] + 0.1 * inst
                )
            tprev[i] = now
            with lock:
                latest[i] = (f, now)


threading.Thread(target=grab_loop, daemon=True).start()
time.sleep(0.3)

# one encoder thread per camera; a sentinel None flushes+closes the writer
fourcc = cv2.VideoWriter_fourcc(*args.codec)
queues = [queue.Queue(maxsize=args.queue) for _ in range(ncam)]


def writer_loop(i):
    path = os.path.join(out_dir, f"cam{i}.mp4")
    vw = cv2.VideoWriter(path, fourcc, args.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit(f"cannot open writer {path} (codec {args.codec}?)")
    while True:
        item = queues[i].get()
        if item is None:
            break
        vw.write(item)
    vw.release()


writers = []  # started on first record
recording = False
n = 0  # committed frame sets
dropped = 0
ts_rows = []  # (ncam,) retrieve wall-times per committed set

WIN = "Multi Record  [r=rec q=quit]"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
print(
    f"Output -> {out_dir}/cam0..cam{ncam-1}.mp4  ({W}x{H}"
    + (", flipped 180" if args.rotate180 else "")
    + ")"
)
print("r = start/stop,  q = quit")


def start_writers():
    global writers
    for q in queues:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
    writers = [
        threading.Thread(target=writer_loop, args=(i,), daemon=True)
        for i in range(ncam)
    ]
    for t in writers:
        t.start()


def stop_writers():
    for q in queues:
        q.put(None)
    for t in writers:
        t.join()


def grid(imgs):
    cols = int(np.ceil(np.sqrt(len(imgs))))
    rows = int(np.ceil(len(imgs) / cols))
    blank = np.zeros_like(imgs[0])
    cells = imgs + [blank] * (rows * cols - len(imgs))
    m = cv2.vconcat(
        [cv2.hconcat(cells[r * cols : (r + 1) * cols]) for r in range(rows)]
    )
    s = min(1.0, args.display_width / m.shape[1])
    return cv2.resize(m, None, fx=s, fy=s) if s < 1.0 else m


while True:
    with lock:
        snap = [
            None if latest[i] is None else (latest[i][0], latest[i][1])
            for i in range(ncam)
        ]
    if any(s is None for s in snap):
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break
        continue

    if recording:
        # commit the whole set atomically, or drop it all (keeps files aligned)
        if all(not queues[i].full() for i in range(ncam)):
            for i in range(ncam):
                queues[i].put_nowait(snap[i][0])
            ts_rows.append([snap[i][1] for i in range(ncam)])
            n += 1
        else:
            dropped += 1

    disps = []
    for i in range(ncam):
        d = snap[i][0].copy()
        txt = f"REC {n}" if recording else "READY - r"
        col = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(
            d, f"cam{i} {txt}", (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3
        )
        cv2.putText(
            d,
            f"{cap_fps[i]:.0f} fps",
            (12, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        disps.append(d)
    if dropped:
        cv2.putText(
            disps[0],
            f"DROPPED {dropped} sets (encoder slow)",
            (12, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    cv2.imshow(WIN, grid(disps))

    k = cv2.waitKey(1) & 0xFF
    if k == ord("r"):
        if not recording:
            n, dropped, ts_rows = 0, 0, []
            start_writers()
            recording = True
            print("recording...")
        else:
            recording = False
            stop_writers()
            np.save(
                os.path.join(out_dir, "timestamps.npy"),
                np.asarray(ts_rows, np.float64),
            )
            meta = {
                "cams": [str(c) for c in cam_ids],
                "ncam": ncam,
                "width": W,
                "height": H,
                "fps": args.fps,
                "rotate180": args.rotate180,
                "codec": args.codec,
                "frames": n,
                "dropped": dropped,
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as fp:
                json.dump(meta, fp, indent=2)
            print(f"saved {n} sets ({dropped} dropped) -> {out_dir}/")
    elif k == ord("q"):
        break

stop.set()
if recording:
    stop_writers()
    np.save(
        os.path.join(out_dir, "timestamps.npy"),
        np.asarray(ts_rows, np.float64),
    )
    print(f"saved {n} sets ({dropped} dropped) -> {out_dir}/")
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
print(
    f"Done. Infer later: python ../cosmik_hand_demo.py --cams {args.cams} "
    f"--calib calibration_data/multi_params.npz --cap-width {W} "
    f"--cap-height {H} --replay {out_dir}"
)
