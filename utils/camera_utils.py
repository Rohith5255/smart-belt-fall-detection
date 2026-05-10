"""
utils/camera_utils.py — Camera auto-detection utility.

Priority (in order):
  1. PREFERRED_CAMERA_INDEX in config.py (if >= 0, used directly — no scan)
  2. DirectShow COM enumeration — same order cv2 uses internally
  3. PowerShell Get-PnpDevice fallback (may differ in order, less reliable)
  4. Index heuristic: prefer highest working index (USB cams register last)

Usage:
    from utils.camera_utils import auto_detect_camera
    idx, name = auto_detect_camera()
    cap = cv2.VideoCapture(idx)

To force a specific camera:
    Set PREFERRED_CAMERA_INDEX = 0  (or 1, 2 …) in config.py.
    Set PREFERRED_CAMERA_INDEX = -1 to re-enable auto-detect.
"""

import cv2
import os
import sys
import subprocess

MAX_INDEX = 5

# Keywords that identify BUILT-IN / INTEGRATED cameras (checked as substrings).
BUILTIN_KEYWORDS = {
    "integrated", "ir camera", "user facing", "user-facing",
    "facetime", "built-in", "infrared",
    # Vendor-specific built-in brands
    "truevision",     # HP TrueVision (most HP laptops)
    "bison",          # Bison Electronics OEM
    "realtek",        # Realtek IR / secondary cam
    "ov2740",         # common OEM image sensor
    "hd user",        # "HD User Facing"
    "front camera",
    "laptop camera",
    "chicony",        # Chicony — OEM built-in brand
    "sunplus",        # Sunplus — OEM built-in
}


# ── Name resolvers (in reliability order) ────────────────────────────────────

def _get_camera_names_directshow():
    """
    Enumerate DirectShow video capture filters via COM in the EXACT order
    that cv2.VideoCapture uses.  Executes a PowerShell C# snippet that calls
    the System Device Enumerator COM API — same code path as OpenCV internally.
    Returns {0: name0, 1: name1, …} or {} on any failure.
    """
    ps_script = r"""
Add-Type @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

[ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"),
 InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface ICreateDevEnum {
    [PreserveSig]
    int CreateClassEnumerator([MarshalAs(UnmanagedType.LPStruct)] Guid pType,
                              out IEnumMoniker ppEnum, int dwFlags);
}

[ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851"),
 InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IPropertyBag {
    [PreserveSig]
    int Read([MarshalAs(UnmanagedType.LPWStr)] string pszPropName,
             ref object pVar,
             IntPtr pErrorLog);
    [PreserveSig]
    int Write([MarshalAs(UnmanagedType.LPWStr)] string pszPropName,
              ref object pVar);
}

public static class DSEnum {
    static readonly Guid CLSID_SysDevEnum =
        new Guid("62BE5D10-60EB-11D0-BD3B-00A0C911CE86");
    static readonly Guid CAT_VideoInput =
        new Guid("860BB310-5D01-11D0-BD3B-00A0C911CE86");
    static readonly Guid IID_IPropertyBag =
        new Guid("55272A00-42CB-11CE-8135-00AA004BB851");

    public static List<string> GetDeviceNames() {
        var names = new List<string>();
        var sde = (ICreateDevEnum)Activator.CreateInstance(
                    Type.GetTypeFromCLSID(CLSID_SysDevEnum));
        IEnumMoniker enumMon;
        sde.CreateClassEnumerator(CAT_VideoInput, out enumMon, 0);
        if (enumMon == null) return names;
        IMoniker[] mon = new IMoniker[1];
        IntPtr fetched = IntPtr.Zero;
        while (enumMon.Next(1, mon, fetched) == 0) {
            try {
                object bag = null;
                mon[0].BindToStorage(null, null, ref IID_IPropertyBag,
                                     out bag);
                var pb = (IPropertyBag)bag;
                object val = "";
                pb.Read("FriendlyName", ref val, IntPtr.Zero);
                names.Add(val.ToString());
            } catch {}
            finally { Marshal.ReleaseComObject(mon[0]); }
        }
        return names;
    }
}
"@ -ErrorAction Stop
[DSEnum]::GetDeviceNames()
"""
    try:
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-Command", ps_script],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l.strip() for l in result.stdout.splitlines()
                 if l.strip() and not l.startswith("WARNING")]
        if lines:
            return {i: line.lower() for i, line in enumerate(lines)}
    except Exception:
        pass
    return {}


def _get_camera_names_powershell():
    """
    Fallback: PowerShell Get-PnpDevice.  Order may differ from cv2 on some
    systems — use only when DirectShow enumeration fails.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-Command",
             "(Get-PnpDevice -Class Camera -Status OK).FriendlyName"],
            capture_output=True, text=True, timeout=6,
        )
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        return {i: line.lower() for i, line in enumerate(lines)}
    except Exception:
        return {}


def _camera_names():
    """Best-effort name lookup in reliability order."""
    if sys.platform != "win32":
        return {}
    names = _get_camera_names_directshow()
    if not names:
        names = _get_camera_names_powershell()
    return names


def _is_builtin(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in BUILTIN_KEYWORDS)


# ── Main public function ──────────────────────────────────────────────────────

def auto_detect_camera(fallback_index=0):
    """
    Returns (camera_index, description).

    Respects PREFERRED_CAMERA_INDEX from config.py:
      >= 0  → use that index directly (no scan)
       -1   → auto-detect (default)
    """
    # ── 0. Check manual override ──────────────────────────────────────────
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import PREFERRED_CAMERA_INDEX
        if PREFERRED_CAMERA_INDEX >= 0:
            print(f"[Camera] Manual override: PREFERRED_CAMERA_INDEX = {PREFERRED_CAMERA_INDEX}")
            return PREFERRED_CAMERA_INDEX, f"Camera {PREFERRED_CAMERA_INDEX} (manual)"
    except (ImportError, AttributeError):
        pass

    # ── 1. Get friendly names (DirectShow order = cv2 order) ─────────────
    print("[Camera] Scanning for available cameras …")
    names = _camera_names()

    if names:
        src = "DirectShow" if _get_camera_names_directshow() else "PnP fallback"
        print(f"[Camera] Device names ({src}): "
              + ", ".join(f"[{i}] {n}" for i, n in names.items()))
    else:
        print("[Camera] Could not read device names — using index heuristic.")

    # ── 2. Probe which cv2 indices actually work ──────────────────────────
    working = []
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

    for idx, name in working:
        tag = "[BUILTIN]" if _is_builtin(name) else "[EXTERNAL]"
        print(f"  [{idx}] {name}  {tag}  ✓")
    if not working:
        print("  (no cameras found)")

    # ── 3. Select best camera ─────────────────────────────────────────────
    if not working:
        print(f"[Camera] No cameras — defaulting to index {fallback_index}")
        return fallback_index, f"Camera {fallback_index} (unverified)"

    if len(working) == 1:
        idx, name = working[0]
        print(f"[Camera] Only one camera found → using [{idx}] {name}")
        return idx, name

    # Prefer the highest-index camera that is NOT classified as builtin
    externals = [(i, n) for i, n in working if not _is_builtin(n)]
    if externals:
        idx, name = max(externals, key=lambda x: x[0])
        print(f"[Camera] External camera selected → [{idx}] {name}")
        return idx, name

    # All look builtin — pick the highest index (USB cams register last)
    idx, name = max(working, key=lambda x: x[0])
    print(f"[Camera] No external camera identified — using highest index [{idx}] {name}")
    print("         Tip: set PREFERRED_CAMERA_INDEX in config.py to force a specific camera.")
    return idx, name
