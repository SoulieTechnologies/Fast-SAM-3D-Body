"""
Shows detected marker IDs so you can confirm the board's actual col/row dimensions.
Run while holding the board in front of the camera.
Press 'q' to quit.
"""
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cam", type=int, default=0)
args = parser.parse_args()

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

cap = cv2.VideoCapture(args.cam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Hold full board in view. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        id_list = sorted(ids.ravel().tolist())
        cv2.putText(frame, f"markers: {len(ids)}  max_id: {max(id_list)}  ids: {id_list[:8]}...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"\rDetected {len(ids)} markers, IDs: {id_list}          ", end="", flush=True)
    cv2.imshow("Board check", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
