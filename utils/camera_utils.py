"""
utils/camera_utils.py — Camera auto-detection utility.

Priority:
  1. External / secondary webcam  (by name or higher index)
  2. Built-in laptop camera        (fallback)

Detection strategy:
  - Primary  : PowerShell Get-PnpDevice — always returns real friendly names
  - Fallback : Windows registry KSCATEGORY_VIDEO_CAPTURE enumeration
  - Scan cv2 indices 0-MAX_INDEX; probe each with a real frame read
  - Classify builtin vs external using the friendly name
  - Prefer external; fall back to first working camera

Usage:
    from utils.camera_utils import auto_detect_camera
    idx, name = auto_detect_camera()
    cap = cv2.VideoCapture(idx)
"""

import cv2
import os
import sys
import subprocess

MAX_INDEX = 5   # probe indices 0 … MAX_INDEX

# Keywords that identify BUILT-IN / INTEGRATED cameras.
# Checked as substrings in the lowercased friendly name.
BUILTIN_KEYWORDS = {
    "integrated", "ir camera", "user facing", "user-facing",
    "facetime", "built-in", "infrared",
    # Vendor-specific built-in brands
    "truevision",   # HP TrueVision (laptop webcam)
    "bison",        # Bison Electronics (OEM laptop cam)
    "realtek",      # Realtek IR / depth camera
    "ov2740",       # common OEM laptop sensor
    "hd user",      # "HD User Facing" — front-facing laptop
    "front camera",
    "laptop camera",
}


# ── Name resolvers ────────────────────────────────────────────────────────────

def _get_camera_names_powershell():
    """
    Use PowerShell Get-PnpDevice to list camera friendly names.
    Returns {0: name0, 1: name1, …} in Windows enumeration order
    (which matches cv2 VideoCapture index assignment via DirectShow).
    Returns {} on any failure.
    """
    try:
        result = subprocess.run(
            [
                "powershell", "-NonInteractive", "-Command",
                "(Get-PnpDevice -Class Camera -Status OK).FriendlyName",
            ],
            capture_output=True, text=True, timeout=6,
        )
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        return {i: line.lower() for i, line in enumerate(lines)}
    except Exception:
        return {}


def _get_camera_names_registry():
    """
    Fallback: read camera names from the Windows device class registry key.
    Returns {} on any failure.
    """
    names = {}
    try:
        import winreg
        CAPTURE_KEY = (
            r"SYSTEM\CurrentControlSet\Control\DeviceClasses"
            r"\{65E8773D-8F56-11D0-A3B9-00A0C9223196}"
        )
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, CAPTURE_KEY) as hk:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(hk, i)
                    with winreg.OpenKey(hk, sub) as sk:
                        j = 0
                        while True:
                            try:
                                dev_sub = winreg.EnumKey(sk, j)
                                with winreg.OpenKey(sk, dev_sub) as dk:
                                    try:
                                        name, _ = winreg.QueryValueEx(dk, "DeviceDesc")
                                        if name and ";" in name:
                                            name = name.split(";")[-1].strip()
                                        if name:
                                            names[len(names)] = name.lower()
                                    except FileNotFoundError:
                                        pass
                                j += 1
                            except OSError:
                                break
                    i += 1
                except OSError:
                    break
    except Exception:
        pass
    return names


def _camera_names():
    """
    Best-effort friendly-name lookup.
    Tries PowerShell first (most reliable), then registry.
    Returns {} on non-Windows.
    """
    if sys.platform != "win32":
        return {}
    names = _get_camera_names_powershell()
    if not names:
        names = _get_camera_names_registry()
    return names


def _is_builtin(name: str) -> bool:
    """True if the camera name looks like a built-in / integrated device."""
    n = name.lower()
    return any(kw in n for kw in BUILTIN_KEYWORDS)


# ── Main public function ──────────────────────────────────────────────────────

def auto_detect_camera(fallback_index=0):
    """
    Scans camera indices 0…MAX_INDEX, returns (best_index, description).

    Selection priority:
      1. Named external camera (not in BUILTIN_KEYWORDS)
      2. Any working camera at index > 0 (when names unavailable)
      3. Camera at index 0 (built-in fallback)

    Returns:
        (int, str) — (camera_index, human_readable_description)
    """
    print("[Camera] Scanning for available cameras …")
    names = _camera_names()

    if names:
        print(f"[Camera] Detected devices: "
              + ", ".join(f"[{i}] {n}" for i, n in names.items()))
    else:
        print("[Camera] Could not read device names — using index heuristic.")

    working = []   # list of (index, friendly_name)

    # Suppress OpenCV obsensor / backend error spam during probing
    devnull = open(os.devnull, "w")
    old_stderr_fd = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    try:
        for idx in range(MAX_INDEX + 1):
            cap = cv2.VideoCapture(idx)
            opened = cap.isOpened()
            read_ok = False
            if opened:
                ret, _ = cap.read()
                read_ok = ret
            cap.release()
            if opened and read_ok:
                name = names.get(idx, f"Camera {idx}")
                working.append((idx, name))
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        devnull.close()

    # Print detected working cameras
    for idx, name in working:
        tag = "[BUILTIN]" if _is_builtin(name) else "[EXTERNAL]"
        print(f"  [{idx}] {name}  {tag}  ✓")
    if not working:
        print("  (no cameras found)")

    # ── Selection logic ───────────────────────────────────────────────────────
    if not working:
        print(f"[Camera] No cameras found — defaulting to index {fallback_index}")
        return fallback_index, f"Camera {fallback_index} (unverified)"

    if len(working) == 1:
        idx, name = working[0]
        print(f"[Camera] Only one camera found → using [{idx}] {name}")
        return idx, name

    # Prefer a camera whose name is NOT in the builtin keyword list
    externals = [(i, n) for i, n in working if not _is_builtin(n)]
    if externals:
        # Among external candidates prefer higher index
        # (USB webcams typically get higher indices than built-in)
        idx, name = max(externals, key=lambda x: x[0])
        print(f"[Camera] External camera selected → [{idx}] {name}")
        return idx, name

    # All cameras classified as built-in — try highest index (least likely to
    # be the primary integrated cam)
    idx, name = max(working, key=lambda x: x[0])
    print(f"[Camera] Only built-in cameras found — using [{idx}] {name}")
    return idx, name
