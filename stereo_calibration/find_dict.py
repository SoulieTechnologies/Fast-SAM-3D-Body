"""
Tries every common ArUco dictionary on camera 0 and prints which ones detect markers.
Run this while holding your ChArUco board in front of the camera.
"""

import cv2

DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cam", type=int, default=0)
args = parser.parse_args()

cap = cv2.VideoCapture(args.cam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
assert cap.isOpened()

print(
    "Hold the board in front of the camera. Press any key to capture and test."
)
while True:
    ret, frame = cap.read()
    cv2.imshow("Hold board here, press any key to test", frame)
    if cv2.waitKey(1) != -1:
        break

cv2.destroyAllWindows()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print()
for name, dict_id in DICTS.items():
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)
    n = len(ids) if ids is not None else 0
    if n > 0:
        print(f"  FOUND {n:2d} markers  →  {name}  ← use this one")
    else:
        print(f"  0 markers        {name}")
