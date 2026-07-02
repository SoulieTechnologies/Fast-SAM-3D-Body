#!/usr/bin/env python3
"""Camera client (run on the MacBook) — capture a webcam / video and stream JPEG
frames to the SAM3D server (stream_demo.py --recv-port ...).

Mac:
  python stream_client.py --host clear-antares.tailb614a0.ts.net --port 8091 --source 0
Server:
  python stream_demo.py --recv-port 8091 --fx 900 --cx 640 --cy 360 --gpu 7 --port 8080
View (browser):  http://clear-antares.tailb614a0.ts.net:8080   (or via ssh -L 8080:...)

Only needs: opencv-python + numpy.  Sends [4-byte length][JPEG] frames over TCP.
"""

import argparse
import socket
import struct
import time

import cv2


def main():
    p = argparse.ArgumentParser(description="Stream a camera to the SAM3D server")
    p.add_argument("--host", required=True, help="server host (Tailscale name or IP)")
    p.add_argument("--port", type=int, default=8091, help="server --recv-port")
    p.add_argument("--source", default="0", help="webcam index or video path (looped)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30, help="max send rate")
    p.add_argument("--quality", type=int, default=92, help="JPEG quality 1-100 (higher = crisper hands)")
    a = p.parse_args()

    is_cam = a.source.isdigit()
    cap = cv2.VideoCapture(int(a.source) if is_cam else a.source)
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {a.source}")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # send frames immediately (low latency)
    s.connect((a.host, a.port))
    print(f"connected to {a.host}:{a.port} — streaming '{a.source}' "
          f"@≤{a.fps:.0f} fps, q{a.quality}")

    period = 1.0 / a.fps
    n = 0
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                if is_cam:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop a video source
                continue
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            if not ok2:
                continue
            data = buf.tobytes()
            s.sendall(struct.pack(">I", len(data)) + data)
            n += 1
            if n % 60 == 0:
                print(f"  sent {n} frames")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except (BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
        print("disconnected / stopped")
    finally:
        cap.release()
        s.close()


if __name__ == "__main__":
    main()
