"""
test_droidcam.py — Diagnose DroidCam green-frame issue with OpenCV.

Tests multiple backends and strategies in sequence, prints per-frame
channel stats, saves the first non-green frame as droidcam_test.jpg.

Run:  .venv\Scripts\python.exe test_droidcam.py
"""

import cv2
import numpy as np
import time
import os

CAM_IDX   = 1
N_FRAMES  = 30
OUT_FILE  = "droidcam_test.jpg"

def is_green(frame, threshold=50):
    """
    True if frame is DroidCam's no-signal placeholder (R≈0, G≈135, B≈0).
    NOTE: DroidCam green mean ≈ 45 — do NOT use mean>180 as the gate.
    Don't use std — MSMF H.264 decompression adds noise (~5-20 counts)
    even on solid-color frames, so std is never reliably < 5.
    Use channel ratios only: DroidCam no-signal has R<10, B<10, G>80.
    """
    r = float(frame[:, :, 0].mean())
    g = float(frame[:, :, 1].mean())
    b = float(frame[:, :, 2].mean())
    return r < 10 and b < 10 and g > 80

def probe(label, cap, n=N_FRAMES, warmup=0, discard=0):
    """
    Read n frames from cap, print per-frame channel stats.
    Returns (first_good_frame, green_count, total_read).
    """
    if not cap.isOpened():
        print(f"  [{label}] NOT OPENED")
        return None, 0, 0

    # Optional: discard frames before sampling
    for _ in range(discard):
        cap.read()

    if warmup:
        print(f"  [{label}] Sleeping {warmup}s for warm-up ...")
        time.sleep(warmup)

    green_count = 0
    good_frame  = None
    total       = 0

    for i in range(n):
        ret, frame = cap.read()
        total += 1
        if not ret or frame is None:
            print(f"  [{label}] frame {i:02d}: READ FAILED (ret={ret})")
            continue

        r = frame[:, :, 0].mean()
        g = frame[:, :, 1].mean()
        b = frame[:, :, 2].mean()
        mn = frame.mean()
        green = is_green(frame)
        tag = "GREEN" if green else "OK   "
        print(f"  [{label}] frame {i:02d}: mean={mn:5.1f}  R={r:5.1f} G={g:5.1f} B={b:5.1f}  [{tag}]")

        if green:
            green_count += 1
        elif good_frame is None:
            good_frame = frame.copy()

    return good_frame, green_count, total


def try_backend(label, *args, **kwargs):
    """Open with given args, probe, release, return (good_frame, green_count, total)."""
    print(f"\n{'='*60}")
    print(f"  BACKEND: {label}")
    print(f"{'='*60}")
    cap = cv2.VideoCapture(*args)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Opened OK  — resolution reported: {w}x{h}")
    good, green_cnt, total = probe(label, cap)
    cap.release()
    return good, green_cnt, total


def try_discard(label, *args, discard=30, warmup=0):
    """Open, discard N frames, then sample."""
    print(f"\n{'='*60}")
    print(f"  BACKEND: {label}  [discard={discard}, warmup={warmup}s]")
    print(f"{'='*60}")
    cap = cv2.VideoCapture(*args)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Opened OK  — resolution reported: {w}x{h}")
    good, green_cnt, total = probe(label, cap, warmup=warmup, discard=discard)
    cap.release()
    return good, green_cnt, total


def try_reopen(label, *args):
    """Open, release immediately, reopen — sometimes fixes virtual cams."""
    print(f"\n{'='*60}")
    print(f"  BACKEND: {label}  [open-release-reopen]")
    print(f"{'='*60}")
    cap = cv2.VideoCapture(*args)
    cap.release()
    time.sleep(0.5)
    cap = cv2.VideoCapture(*args)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Reopened OK — resolution reported: {w}x{h}")
    good, green_cnt, total = probe(label, cap)
    cap.release()
    return good, green_cnt, total


# ── Run all strategies ────────────────────────────────────────────────────────
results = []

# 1. Default backend (MSMF on Windows 11)
g, gc, tot = try_backend("Default (MSMF)", CAM_IDX)
results.append(("Default/MSMF", g, gc, tot))

# 2. CAP_MSMF explicit
g, gc, tot = try_backend("CAP_MSMF", CAM_IDX, cv2.CAP_MSMF)
results.append(("CAP_MSMF", g, gc, tot))

# 3. CAP_DSHOW
g, gc, tot = try_backend("CAP_DSHOW", CAM_IDX, cv2.CAP_DSHOW)
results.append(("CAP_DSHOW", g, gc, tot))

# 4. Default + discard 30 frames first
g, gc, tot = try_discard("Default+discard30", CAM_IDX, discard=30)
results.append(("Default+discard30", g, gc, tot))

# 5. CAP_DSHOW + discard 30 frames first
g, gc, tot = try_discard("DSHOW+discard30", CAM_IDX, cv2.CAP_DSHOW, discard=30)
results.append(("DSHOW+discard30", g, gc, tot))

# 6. Default + 2s warm-up sleep
g, gc, tot = try_discard("Default+2s_warmup", CAM_IDX, discard=0, warmup=2)
results.append(("Default+2s_warmup", g, gc, tot))

# 7. Named device via CAP_DSHOW (try both common DroidCam device names)
for name in ["DroidCam Source 2", "DroidCam Source 3", "DroidCam"]:
    g, gc, tot = try_backend(f'CAP_DSHOW name="{name}"',
                              f"video={name}", cv2.CAP_DSHOW)
    results.append((f'named:"{name}"', g, gc, tot))
    if g is not None:
        break   # found it — no need to try more names

# 8. Open-release-reopen with CAP_DSHOW
g, gc, tot = try_reopen("DSHOW+reopen", CAM_IDX, cv2.CAP_DSHOW)
results.append(("DSHOW+reopen", g, gc, tot))

# 9. Try 320x240 with CAP_DSHOW
print(f"\n{'='*60}")
print(f"  BACKEND: DSHOW 320x240")
print(f"{'='*60}")
cap = cv2.VideoCapture(CAM_IDX, cv2.CAP_DSHOW)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Opened OK  — resolution reported: {w}x{h}")
g, gc, tot = probe("DSHOW 320x240", cap, discard=10)
cap.release()
results.append(("DSHOW 320x240", g, gc, tot))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")

best_label = None
best_frame = None
for label, frame, green_cnt, total in results:
    if total == 0:
        status = "FAILED TO OPEN"
    elif frame is not None:
        status = f"✅  {total - green_cnt}/{total} good frames"
        if best_frame is None:
            best_frame = frame
            best_label = label
    else:
        status = f"❌  ALL {green_cnt}/{total} green"
    print(f"  {label:<30}  {status}")

print()
if best_frame is not None:
    cv2.imwrite(OUT_FILE, best_frame)
    print(f"✅  Saved first good frame → {OUT_FILE}")
    print(f"   Best strategy: {best_label}")
    print(f"\n→  Use this in vision_thread: {best_label}")
else:
    print("❌  ALL strategies returned green frames.")
    print("   Make sure DroidCam app is open and streaming on your phone.")
    print("   Try restarting DroidCam client on Windows, then re-run this script.")
