"""
utils/camera_utils.py — Camera auto-detection utility.

Priority:
  1. External / secondary webcam  (index 1, 2, 3 …)
  2. Built-in laptop camera        (index 0)

Detection strategy:
  - Scan indices 0 to MAX_INDEX
  - For each index, try to open with cv2.VideoCapture and read one frame
  - On Windows, also attempt to read the device friendly name via
    Windows.Devices.Enumeration (no extra pip packages needed)
  - Prefer the highest-index working camera that is NOT the built-in
    (built-in cameras usually have keywords like "integrated", "facetime",
    "ir camera", "hd user-facing" in their name)
  - Fall back to index 0 if nothing external found

Usage:
    from utils.camera_utils import auto_detect_camera
    idx, name = auto_detect_camera()
    cap = cv2.VideoCapture(idx)
"""

import cv2
import os
import sys

MAX_INDEX        = 5          # how many indices to probe
BUILTIN_KEYWORDS = {
    "integrated", "ir camera", "user facing", "user-facing",
    "facetime", "front", "built-in", "hd camera", "720p hd",
    "1080p hd", "infrared",
}


# ── Windows camera name resolver ──────────────────────────────────────────────

def _get_windows_camera_names():
    """
    Returns {index: friendly_name} for all cameras visible to DirectShow.
    Requires no extra packages — uses ctypes + comtypes approach via
    cv2's own backend enumeration is not exposed, so we fall back to
    querying Win32 Device Manager through the registry.
    Returns {} on any failure (non-Windows, permission error, etc.).
    """
    names = {}
    try:
        import winreg
        # KSCATEGORY_VIDEO_CAPTURE lists all video-capture devices
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
                        # The device description is inside a nested ##?#... key
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


def _get_camera_names_dshow():
    """
    Alternative: enumerate DirectShow filter names via comtypes.
    Only works if comtypes is installed — silent fallback otherwise.
    """
    names = {}
    try:
        import comtypes.client as cc
        import comtypes
        from comtypes.gen import quartz  # noqa — may not exist
    except Exception:
        return names

    try:
        # Use system device enumerator
        CLSID_SystemDeviceEnum = comtypes.GUID("{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}")
        CLSID_VideoInputDeviceCategory = comtypes.GUID("{860BB310-5D01-11D0-BD3B-00A0C911CE86}")

        dev_enum = cc.CreateObject(CLSID_SystemDeviceEnum)
        enum_moniker, _ = dev_enum.CreateClassEnumerator(
            comtypes.byref(CLSID_VideoInputDeviceCategory), 0)
        i = 0
        while True:
            try:
                moniker, _ = enum_moniker.Next(1)
                prop_bag = moniker.BindToStorage(None, None,
                    comtypes.GUID("{55272A00-42CB-11CE-8135-00AA004BB851}"))
                name = prop_bag.Read("FriendlyName", None)
                names[i] = str(name).lower()
                i += 1
            except Exception:
                break
    except Exception:
        pass
    return names


def _camera_names():
    """Best-effort camera name lookup — tries multiple methods."""
    if sys.platform != "win32":
        return {}
    names = _get_camera_names_dshow()
    if not names:
        names = _get_windows_camera_names()
    return names


def _is_builtin(name: str) -> bool:
    """Returns True if the camera name looks like a built-in/integrated camera."""
    n = name.lower()
    return any(kw in n for kw in BUILTIN_KEYWORDS)


# ── Main public function ──────────────────────────────────────────────────────

def auto_detect_camera(fallback_index=0):
    """
    Scans camera indices 0…MAX_INDEX, returns (best_index, description).

    Priority:
      1. First working external camera  (index ≥ 1, or named as external)
      2. Built-in laptop camera          (index 0)
      3. fallback_index if nothing found

    Returns:
        (int, str) — (camera_index, human_readable_description)
    """
    print("[Camera] Scanning for available cameras …")
    names = _camera_names()

    working = []     # list of (index, name)

    # Temporarily redirect stderr to suppress OpenCV internal backend noise
    # (obsensor/Orbbec depth-camera plugin prints errors for every missing index)
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

    # Report after stderr is restored
    for idx, name in working:
        flag = " [BUILTIN]" if _is_builtin(name) else " [EXTERNAL]"
        print(f"  [{idx}] {name}{flag}  ✓")
    if not working:
        print("  (no cameras found)")

    if not working:
        print(f"[Camera] No cameras found — defaulting to index {fallback_index}")
        return fallback_index, f"Camera {fallback_index} (unverified)"

    if len(working) == 1:
        idx, name = working[0]
        print(f"[Camera] Only one camera found → using [{idx}] {name}")
        return idx, name

    # Multiple cameras — prefer the first external one
    externals = [(i, n) for i, n in working if not _is_builtin(n) and i != 0]
    if externals:
        idx, name = externals[0]
        print(f"[Camera] External camera selected → [{idx}] {name}")
        return idx, name

    # No clearly-external camera by name — prefer index > 0
    non_zero = [(i, n) for i, n in working if i > 0]
    if non_zero:
        idx, name = non_zero[0]
        print(f"[Camera] Secondary camera selected → [{idx}] {name}")
        return idx, name

    # Only built-in found
    idx, name = working[0]
    print(f"[Camera] No external camera detected — using built-in [{idx}] {name}")
    return idx, name
