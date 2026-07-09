#!/usr/bin/env python3
"""SharpaWave driver over plain TCP — no ROS anywhere.

Consumes the command stream of `hand_teleop_node.py --no-ros --emit-q 8093`
([>I frame][22×float32], radians, SDK joint order) and forwards it to the
Wave SDK. This exists because rclpy (built for the system python 3.10) cannot
be imported inside the acados conda env (python 3.11) — the TCP hop replaces
the ROS topic, everything else (MPC, safety layers) stays in the node.

Safety here (the node already ramps/clamps/holds upstream):
  - SDK init like the official sample: POSITION mode, speed_coeff/current_coeff
    limits, control source SDK
  - first command sent with the SDK's interpolation mode (smooth approach)
  - no data → no writes (the hand holds); Ctrl-C → disable + stop

Run (SYSTEM python, hand on USB, Sharpa Pilot closed):
  python3 sharpa_tcp_driver.py                       # connects localhost:8093
  python3 sharpa_tcp_driver.py --dry                 # print instead of driving
"""
import argparse
import socket
import struct
import sys
import threading
import time

NJ = 22
_MSG = 4 + NJ * 4
_RX = {"q": None, "n": 0}


def _rx_thread(host, port):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            print(f"connected to {host}:{port}")
            buf = b""
            while True:
                d = s.recv(65536)
                if not d:
                    raise ConnectionError
                buf += d
                msg = None
                while len(buf) >= _MSG:
                    msg = buf[:_MSG]
                    buf = buf[_MSG:]
                if msg is not None:
                    # payload floats are LITTLE-endian (numpy native on x86 —
                    # matches the node's astype(float32).tobytes()); only the
                    # frame counter header is big-endian
                    q = list(struct.unpack(f"<{NJ}f", msg[4:]))
                    # garbage guard: a joint command can never exceed ±π
                    if all(abs(v) < 3.2 for v in q):
                        # q before n (readers key off n — no torn frame)
                        _RX["q"] = q
                        _RX["n"] = struct.unpack(">I", msg[:4])[0]
        except OSError:
            time.sleep(1.0)


class FakeWave:
    """--dry stand-in: same surface as the SDK hand, prints instead."""
    def set_joint_position(self, q, *a):
        self._last = q


def connect_sdk():
    sys.path.insert(0, "/opt/sharpa-wave-sdk/python")
    from sharpa import (SharpaWaveManager, DeviceType, ControlSource,
                        ControlMode)
    manager = SharpaWaveManager.get_instance()
    time.sleep(1)
    while True:
        infos = [i for i in (manager.get_all_devices() or [])
                 if i.device_type == DeviceType.HAND]
        if infos:
            break
        print("waiting for a SharpaWave hand...")
        time.sleep(1)
    wave = manager.connect(infos[0].sn)
    # same init as the official sample — POSITION mode + speed/current limits
    for call, val in (("set_control_mode", ControlMode.POSITION),
                      ("set_speed_coeff", 0.5), ("set_current_coeff", 0.6),
                      ("set_control_source", ControlSource.SDK)):
        err = getattr(wave, call)(val)
        if err.code != 0:
            raise RuntimeError(f"{call} failed: {err.message}")
    if hasattr(wave, "set_enable_state"):
        wave.set_enable_state(True)
    if not wave.start():
        raise RuntimeError("wave.start() failed")
    print(f"hand {infos[0].sn} connected + started")
    return manager, wave


def main():
    p = argparse.ArgumentParser(description="SharpaWave TCP driver (no ROS)")
    p.add_argument("--listen", default="localhost:8093",
                   help="host:port of hand_teleop_node --emit-q")
    p.add_argument("--dt", type=float, default=0.02, help="SDK write period (s)")
    p.add_argument("--dry", action="store_true", help="no hardware, print q")
    args = p.parse_args()

    manager = None
    if args.dry:
        wave = FakeWave()
    else:
        manager, wave = connect_sdk()

    host, port = args.listen.rsplit(":", 1)
    threading.Thread(target=_rx_thread, args=(host, int(port)), daemon=True).start()

    last_n, first, k = -1, True, 0
    try:
        while True:
            t0 = time.time()
            if _RX["q"] is not None and _RX["n"] != last_n:
                last_n = _RX["n"]
                # first command with SDK interpolation → smooth approach; the
                # node ramps from neutral anyway, this is belt-and-braces
                wave.set_joint_position(_RX["q"], first)
                first = False
                k += 1
                if k <= 3 or k % 250 == 0:
                    print(f"cmd #{k}  q[:5]={[round(v, 2) for v in _RX['q'][:5]]}",
                          flush=True)
            time.sleep(max(0.0, args.dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if manager is not None:
            try:
                if hasattr(wave, "set_enable_state"):
                    wave.set_enable_state(False)
                wave.stop()
                manager.disconnect_all()
                print("hand disabled + disconnected")
            except Exception as e:
                print(f"shutdown: {e}")


if __name__ == "__main__":
    main()
