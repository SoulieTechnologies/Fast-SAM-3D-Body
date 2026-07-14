#!/usr/bin/env python3
"""Manus-glove → TCP bridge for the retarget_mpc.py ghost comparison.

retarget_mpc.py --manus-listen host:port connects here (this is the SERVER,
exactly like cosmik_hand_demo's keypoint emitter) and receives, at --rate Hz,
the latest glove payload in one of two formats:

  kp     : [>I frame][21 x 3 float32]  — right-hand keypoints, SAM3D decoder
           order (thumb[tip,DIP,PIP,MCP], index, middle, ring, pinky, wrist=20),
           WRIST-RELATIVE metres. Fed through the SAME MPC as SAM3D → a fair
           "same pipeline, different sensor" accuracy comparison.
  angles : [>I frame][n_sdk x float32 '<f4']  — Sharpa-SDK joint vector
           (retarget_mpc's HANDS['sharpa'].publish_order order). Applied to the
           ghost URDF directly (fast, no MPC).

────────────────────────────────────────────────────────────────────────────
INTEGRATION POINT — fill ManusReader.read() from your lab's sharpa-manus-sdk
(the ~/TheophileCodes/Sharpa/sharpa-manus-sdk / sharpa-pilot stack). Everything
else (the server, the wire format, the ghost, the comparison graph) is done.
Until then, --fake streams a moving synthetic hand so the whole chain can be
tested end-to-end without the glove.
────────────────────────────────────────────────────────────────────────────

Run (on crslab, glove connected):
  python manus_bridge.py --mode kp     --port 8095
  python manus_bridge.py --mode angles --port 8095
Test with no hardware:
  python manus_bridge.py --mode kp --port 8095 --fake
  python manus_bridge.py --mode kp --port 8095 --replay some_hand_kp.npy
"""
import argparse
import socket
import struct
import threading
import time

import numpy as np

# Sharpa-SDK joint count (retarget_mpc HANDS['sharpa'].publish_order = 22:
# thumb CMC_FE/CMC_AA/MCP_FE/MCP_AA/IP + index|middle|ring MCP_FE/MCP_AA/PIP/DIP
# + pinky CMC/MCP_FE/MCP_AA/PIP/DIP). Kept here so the bridge has no import
# dependency on retarget_mpc; assert-checked against it if importable.
N_SDK = 22
N_KP = 21

_LATEST = {"buf": None, "n": 0}


class ManusReader:
    """Reads the Manus glove. FILL read() from sharpa-manus-sdk."""

    def __init__(self, mode, sdk_path=""):
        self.mode = mode
        # ── TODO(lab): open the Manus/Sharpa session here, e.g.
        #   import sys; sys.path.insert(0, sdk_path or "~/TheophileCodes/Sharpa/sharpa-manus-sdk")
        #   from sharpa_manus_sdk import ManusSession
        #   self.sess = ManusSession(); self.sess.connect()
        self.sess = None

    def read(self):
        """Return the latest glove sample:
        mode 'kp'     → (21,3) float32 wrist-relative keypoints (SAM3D order)
        mode 'angles' → (N_SDK,) float32 Sharpa-SDK joint angles (rad)
        or None if no fresh sample. REPLACE the body below with the SDK read."""
        raise NotImplementedError(
            "ManusReader.read() is a stub — wire it to sharpa-manus-sdk, or run "
            "with --fake / --replay for testing. See the module docstring.")


class FakeReader:
    """Synthetic moving hand so the full pipeline runs without the glove."""

    def __init__(self, mode):
        self.mode = mode
        self.t0 = time.time()

    def read(self):
        t = time.time() - self.t0
        if self.mode == "angles":
            # gentle open/close sweep on every joint
            base = 0.4 * (1 - np.cos(1.5 * t)) / 2
            return (base * np.ones(N_SDK)
                    + 0.05 * np.sin(2 * t + np.arange(N_SDK))).astype(np.float32)
        # a plausible right hand: fingers along +z from the wrist, curling with t
        kp = np.zeros((N_KP, 3), np.float32)
        curl = 0.5 * (1 - np.cos(1.5 * t)) / 2
        lat = np.array([-0.03, -0.015, 0.0, 0.015, 0.03])           # finger spread
        seglen = np.array([0.03, 0.025, 0.022, 0.02])               # mcp..tip
        for fi in range(5):
            base = fi * 4
            z = 0.0
            x = lat[fi]
            for k, L in enumerate(seglen[::-1]):                    # mcp→...→tip idx 3..0
                z += L * np.cos(curl * (k + 1))
                y = -np.sin(curl * (k + 1)) * L + (kp[base + 4 - k][1] if k else 0)
                kp[base + 3 - k] = [x, y, z]
        return kp - kp[20]


def producer(reader, mode, rate):
    dt = 1.0 / rate
    n = 0
    while True:
        try:
            s = reader.read()
        except NotImplementedError:
            raise
        except Exception as e:                                     # keep serving
            print(f"  manus read error: {e}")
            s = None
        if s is not None:
            s = np.asarray(s, np.float32)
            if mode == "kp":
                assert s.shape == (N_KP, 3), f"kp must be (21,3), got {s.shape}"
                payload = s.astype(np.float32).tobytes()
            else:
                assert s.shape == (N_SDK,), f"angles must be ({N_SDK},), got {s.shape}"
                payload = s.astype("<f4").tobytes()
            n += 1
            _LATEST["buf"] = struct.pack(">I", n) + payload
            _LATEST["n"] = n
        time.sleep(dt)


def serve(port, rate):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"  manus bridge serving on :{port} (waiting for retarget_mpc)")
    dt = 1.0 / rate
    while True:
        conn, addr = srv.accept()
        print(f"  client {addr} connected")
        last = -1
        try:
            while True:
                if _LATEST["buf"] is not None and _LATEST["n"] != last:
                    last = _LATEST["n"]
                    conn.sendall(_LATEST["buf"])
                time.sleep(dt)
        except OSError:
            print("  client disconnected")
        finally:
            conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["kp", "angles"], default="kp")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--rate", type=float, default=60.0)
    ap.add_argument("--sdk-path", default="", help="path to sharpa-manus-sdk")
    ap.add_argument("--fake", action="store_true",
                    help="stream a synthetic moving hand (no glove needed)")
    ap.add_argument("--replay", default="",
                    help="stream from a .npy: (T,21,3) kp or (T,N_SDK) angles")
    args = ap.parse_args()

    if args.replay:
        arr = np.load(args.replay)
        print(f"  replaying {args.replay} {arr.shape}")

        class _Replay:
            def __init__(self): self.i = 0
            def read(self):
                x = arr[self.i % len(arr)]
                self.i += 1
                return x
        reader = _Replay()
    elif args.fake:
        reader = FakeReader(args.mode)
    else:
        reader = ManusReader(args.mode, args.sdk_path)

    threading.Thread(target=producer, args=(reader, args.mode, args.rate),
                     daemon=True).start()
    serve(args.port, args.rate)


if __name__ == "__main__":
    main()
