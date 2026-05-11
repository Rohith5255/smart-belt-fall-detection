"""
find_cameras.py — Find which index is your external webcam.

Run:  .venv\Scripts\python.exe find_cameras.py

Saves cam_0.jpg, cam_1.jpg … for every working camera.
Open the JPGs — the one showing your external webcam view is the right index.
Then set  CAMERA_INDEX = <that number>  in config.py.
"""

import cv2
import time
import os

print("Scanning cameras (indices 0-4) ...\n")

found = []
for i in range(5):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"  [{i}]  not available")
        cap.release()
        continue
    ret, frame = cap.read()
    cap.release()
    if ret:
        path = f"cam_{i}.jpg"
        cv2.imwrite(path, frame)
        h, w = frame.shape[:2]
        print(f"  [{i}]  WORKS  {w}x{h}  → saved {path}")
        found.append(i)
    else:
        print(f"  [{i}]  opens but no frame")
    time.sleep(0.3)

print()
if len(found) == 0:
    print("No cameras found. Make sure cameras are connected.")
elif len(found) == 1:
    print(f"Only one camera found (index {found[0]}).")
    print(f"Set  CAMERA_INDEX = {found[0]}  in config.py")
else:
    print(f"Found {len(found)} cameras: {found}")
    print(f"Open the cam_X.jpg files to identify your external webcam.")
    print(f"Then set  CAMERA_INDEX = <that index>  in config.py")
