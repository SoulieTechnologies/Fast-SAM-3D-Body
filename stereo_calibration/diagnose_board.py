"""Diagnose a physical ChArUco board: infer its geometry from what the camera
sees and test which (cols, rows, legacy, ratio) configuration actually yields
ChArUco corners. Run, point the camera at the board, read the verdict.

    python diagnose_board.py --cam 0
    python diagnose_board.py --image path/to/frame.png
"""
import argparse
import itertools

import cv2
import numpy as np

import board_config

parser = argparse.ArgumentParser()
parser.add_argument("--cam", type=int, default=0)
parser.add_argument("--image", default="", help="diagnose a saved image instead")
parser.add_argument("--frames", type=int, default=60, help="frames to scan for the best view")
args = parser.parse_args()

dictionary = cv2.aruco.getPredefinedDictionary(board_config.ARUCO_DICT)
ad = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

# ── 1. grab the frame with the most markers ─────────────────────────────────
best = (0, None, None, None)          # (n, gray, corners, ids)
if args.image:
    img = cv2.imread(args.image)
    assert img is not None, f"cannot read {args.image}"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c, i, _ = ad.detectMarkers(gray)
    best = (0 if i is None else len(i), gray, c, i)
else:
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"scanning {args.frames} frames — show the board to camera {args.cam}...")
    for _ in range(args.frames):
        ok, img = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c, i, _ = ad.detectMarkers(gray)
        n = 0 if i is None else len(i)
        if n > best[0]:
            best = (n, gray, c, i)
    cap.release()

n_mk, gray, mk_corners, mk_ids = best
assert n_mk >= 4, f"only {n_mk} markers seen — check dictionary/lighting/distance"
ids = mk_ids.ravel()
print(f"\nmarkers detected : {n_mk}")
print(f"marker IDs       : min={ids.min()}  max={ids.max()}")

# ── 2. infer geometry from the markers ──────────────────────────────────────
centers = np.array([c.reshape(4, 2).mean(0) for c in mk_corners])
sides = np.array([np.linalg.norm(c.reshape(4, 2)[0] - c.reshape(4, 2)[1])
                  for c in mk_corners])
marker_px = float(np.median(sides))
# nearest-neighbour distance between marker centers ≈ diagonal of one square
# for same-color-adjacent markers it's 2 squares horizontally; use the MIN
# plausible: diagonal neighbours are sqrt(2)*square apart
d = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
d[d == 0] = np.inf
nn = float(np.median(d.min(1)))
square_px_diag = nn / np.sqrt(2)          # if nearest are diagonal neighbours
square_px_row = nn / 2.0                  # if nearest are same-row neighbours
for name, sq in (("diagonal-adjacent", square_px_diag), ("row-adjacent", square_px_row)):
    print(f"marker/square ratio if {name}: {marker_px / sq:.2f}")
print(f"(configured ratio: {board_config.MARKER_SIZE / board_config.SQUARE_SIZE:.2f})")

# max marker id → number of squares on the board (markers fill half the squares)
n_squares_min = 2 * (ids.max() + 1)
print(f"max id {ids.max()} → the board has ~{n_squares_min} squares "
      f"(configured: {board_config.BOARD_COLS * board_config.BOARD_ROWS})")

# ── 3. brute-force candidate configurations ─────────────────────────────────
print("\ntesting configurations (cols x rows, legacy, ratio):")
ratios = sorted({round(board_config.MARKER_SIZE / board_config.SQUARE_SIZE, 2),
                 0.7, 0.75, 0.6, 0.8, 0.5})
cand_dims = sorted({(board_config.BOARD_COLS, board_config.BOARD_ROWS),
                    (11, 8), (11, 3), (8, 11), (3, 11), (7, 5), (5, 7),
                    (10, 7), (14, 9)})
results = []
for (cols, rows), legacy, ratio in itertools.product(cand_dims, (False, True), ratios):
    if cols * rows < n_squares_min:       # board too small for the observed ids
        continue
    b = cv2.aruco.CharucoBoard((cols, rows), 1.0, ratio, dictionary)
    if legacy:
        b.setLegacyPattern(True)
    det = cv2.aruco.CharucoDetector(b)
    ch_c, ch_i, _, _ = det.detectBoard(gray)
    n = 0 if ch_i is None else len(ch_i)
    if n > 0:
        results.append((n, cols, rows, legacy, ratio))
        print(f"  {cols}x{rows}  legacy={legacy}  ratio={ratio:.2f}  ->  {n} corners")

if not results:
    print("  NOTHING interpolates — likely a different dictionary or a non-ChArUco board.")
else:
    n, cols, rows, legacy, ratio = max(results)
    print(f"\nVERDICT: use BOARD_COLS={cols} BOARD_ROWS={rows} "
          f"legacy={legacy} marker/square ratio≈{ratio:.2f} ({n} corners)")
    print("→ update board_config.py accordingly (SQUARE_SIZE = measured square in "
          "meters, MARKER_SIZE = SQUARE_SIZE * ratio).")
