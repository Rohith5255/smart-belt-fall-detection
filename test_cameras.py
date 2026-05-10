"""
test_cameras.py — Identify which camera index is your external webcam.

Run:  python test_cameras.py

Saves cam_0.jpg, cam_1.jpg … for every working camera, then opens an
OpenCV window cycling through each one for 3 seconds so you can see
which index shows your external webcam view.

Once you know the right index, set in config.py:
    PREFERRED_CAMERA_INDEX = <that index>
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from utils.camera_utils import _camera_names, _is_builtin

MAX_PROBE = 5
SHOW_SECONDS = 3      # seconds to show each camera live

def main():
    print("\n[TestCameras] Probing indices 0 …", MAX_PROBE)

    names = _camera_names()
    working = []

    devnull = open(os.devnull, "w")
    old_fd  = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    try:
        for idx in range(MAX_PROBE + 1):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    name = names.get(idx, f"Camera {idx}")
                    tag  = "BUILTIN" if _is_builtin(name) else "EXTERNAL"
                    working.append((idx, name, tag, cap, frame))
                else:
                    cap.release()
            else:
                cap.release()
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        devnull.close()

    if not working:
        print("[TestCameras] No cameras found. Make sure cameras are connected.")
        return

    print(f"\n[TestCameras] Found {len(working)} working camera(s):\n")
    for idx, name, tag, cap, frame in working:
        h, w = frame.shape[:2]
        path = f"cam_{idx}.jpg"
        cv2.imwrite(path, frame)
        print(f"  [{idx}]  {name}  [{tag}]   {w}x{h}   → saved {path}")
    print()

    # ── Live preview cycle ────────────────────────────────────────────────
    print(f"[TestCameras] Showing each camera live for {SHOW_SECONDS}s …")
    print("              Press  Q  to skip to next camera.\n")

    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Test", 800, 500)

    for idx, name, tag, cap, _frame in working:
        label = f"[{idx}]  {name}  [{tag}] — is this your EXTERNAL webcam?"
        print(f"  Showing {label}")
        t_end = time.time() + SHOW_SECONDS
        while time.time() < t_end:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            cv2.putText(display, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0) if tag == "EXTERNAL" else (0, 165, 255),
                        2)
            cv2.putText(display,
                        f"Set PREFERRED_CAMERA_INDEX = {idx} in config.py if YES",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)
            cv2.imshow("Camera Test", display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    # Release all
    for _, _, _, cap, _ in working:
        cap.release()
    cv2.destroyAllWindows()

    print("\n[TestCameras] Done.")
    print("  → Look at the cam_N.jpg files or the live preview to identify")
    print("    your external webcam index.")
    print("  → Then set  PREFERRED_CAMERA_INDEX = <index>  in config.py")
    print("    (leave it at -1 to keep auto-detect)\n")


if __name__ == "__main__":
    main()
