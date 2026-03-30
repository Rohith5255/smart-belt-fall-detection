"""
preprocessing/sensor_preprocessing.py
Loads UP-Fall Detection Dataset from CompleteDataSet.csv (single file, all 17 subjects).
Uses BELT sensor columns + AUGMENTED pressure cols to match 8-channel real hardware output.

Hardware output: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, pressure_left, pressure_right
Training uses:   belt acc XYZ + belt gyr XYZ + augmented pressure  → 8 channels → 100 features

PRESSURE AUGMENTATION:
  Previous version used dummy zeros (0.0) for pressure during training.
  This caused the SVM to ignore pressure features entirely, so simulated
  and real fall pressure spikes didn't push the SVM above threshold.

  New approach:
    - FALL windows     → pressure_left  ~ N(3200, 200), pressure_right ~ N(800, 150)
                         (high impact on one side, as in a lateral fall)
    - NO-FALL windows  → pressure_left  ~ N(400, 40),   pressure_right ~ N(390, 40)
                         (balanced, moderate belt pressure when upright)
  Values are ADC 0–4095, matching the ESP32 analogRead() output.
  Random noise keeps the augmentation realistic and prevents overfitting.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (UPFALL_DATA_DIR, WINDOW_SIZE, WINDOW_STEP,
                    FALL_ACTIVITIES, NON_FALL_ACTIVITIES)

COMPLETE_CSV = os.path.join(UPFALL_DATA_DIR, "CompleteDataSet.csv")

BELT_ACC_COLS = ["beltaccelerometer", "unnamed:16", "unnamed:17"]
BELT_GYR_COLS = ["beltangularvelocity", "unnamed:19", "unnamed:20"]
BELT_COLS     = BELT_ACC_COLS + BELT_GYR_COLS   # 6 columns


def _augment_pressure(n_samples, label):
    """
    Generate realistic pressure values for n_samples time-steps.
    label=1 (fall)    : impact spike on left sensor, drop on right
    label=0 (no-fall) : balanced moderate pressure (person upright)

    Returns ndarray of shape (n_samples, 2): [pressure_left, pressure_right]
    """
    rng = np.random.default_rng()   # thread-safe RNG

    if label == 1:
        # Fall impact: high pressure spike (lateral slam)
        # Randomise which side is the impact side
        if rng.random() > 0.5:
            p_left  = rng.normal(3200, 200, n_samples)
            p_right = rng.normal( 800, 150, n_samples)
        else:
            p_left  = rng.normal( 800, 150, n_samples)
            p_right = rng.normal(3200, 200, n_samples)
        # First ~20% of window is pre-impact (normal pressure)
        pre = int(n_samples * 0.2)
        p_left[:pre]  = rng.normal(400, 40, pre)
        p_right[:pre] = rng.normal(390, 40, pre)
    else:
        # Normal upright: balanced belt pressure
        p_left  = rng.normal(400, 40, n_samples)
        p_right = rng.normal(390, 40, n_samples)

    p_left  = np.clip(p_left,  0, 4095)
    p_right = np.clip(p_right, 0, 4095)
    return np.stack([p_left, p_right], axis=1).astype(np.float32)


def _extract_features_from_window(window):
    features = []
    n_ch = window.shape[1]
    for ch in range(n_ch):
        sig = window[:, ch].astype(float)
        features += [
            np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
            np.max(sig) - np.min(sig), np.median(sig),
            kurtosis(sig), skew(sig),
            np.sqrt(np.mean(sig ** 2)),
            np.sum(np.abs(np.diff(sig))),
        ]
        freqs, psd = welch(sig, fs=200, nperseg=min(64, len(sig)))
        features += [np.sum(psd), freqs[np.argmax(psd)]]

    if n_ch >= 3:
        def safe_corr(a, b):
            if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                return 0.0
            return np.corrcoef(a, b)[0, 1]

        features += [
            safe_corr(window[:, 0], window[:, 1]),
            safe_corr(window[:, 0], window[:, 2]),
            safe_corr(window[:, 1], window[:, 2]),
            np.sum(np.abs(window[:, :3]), axis=1).mean(),
        ]
    return np.array(features, dtype=np.float32)


def _segment_and_extract(df, label):
    """Slide window over df and extract features. Pressure is augmented per-window."""
    data = df.values.astype(float)   # shape (N, 6) — IMU only
    X_list, y_list = [], []
    for start in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_STEP):
        imu_win = data[start:start + WINDOW_SIZE]          # (200, 6)
        pres_win = _augment_pressure(WINDOW_SIZE, label)   # (200, 2)
        window = np.hstack([imu_win, pres_win])            # (200, 8)
        feat = _extract_features_from_window(window)
        X_list.append(feat)
        y_list.append(label)
    return X_list, y_list


def load_upfall_dataset():
    if not os.path.isdir(UPFALL_DATA_DIR):
        raise FileNotFoundError(
            f"UP-Fall folder not found: {UPFALL_DATA_DIR}\n"
            "Create the folder and place CompleteDataSet.csv inside it."
        )
    if os.path.isfile(COMPLETE_CSV):
        return _load_from_complete_csv()
    all_csvs = glob.glob(os.path.join(UPFALL_DATA_DIR, "**", "*.csv"), recursive=True)
    if all_csvs:
        print("[SensorPrep] CompleteDataSet.csv not found — trying individual CSVs.")
        return _load_from_individual_csvs(all_csvs)
    raise FileNotFoundError(f"No CSV data found in {UPFALL_DATA_DIR}")


def _load_from_complete_csv():
    size_mb = os.path.getsize(COMPLETE_CSV) // 1024 // 1024
    print(f"[SensorPrep] Loading CompleteDataSet.csv ({size_mb} MB) ...")

    # Detect double header
    raw = pd.read_csv(COMPLETE_CSV, header=None, nrows=5)
    row1_vals = raw.iloc[1].astype(str).str.upper().tolist()
    has_double_header = any("axis" in v.lower() or "deg/s" in v.lower() for v in row1_vals)

    if has_double_header:
        print("[SensorPrep] Double header detected — skipping row 1 axis labels.")
        df = pd.read_csv(COMPLETE_CSV, header=0, skiprows=[1], low_memory=False)
    else:
        df = pd.read_csv(COMPLETE_CSV, header=0, low_memory=False)

    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]

    subject_col  = next((c for c in df.columns if c == "subject"), None)
    activity_col = next((c for c in df.columns if c == "activity"), None)

    if activity_col is None:
        raise ValueError(f"[SensorPrep] 'activity' column not found! Columns: {list(df.columns)}")

    df[activity_col] = pd.to_numeric(df[activity_col], errors="coerce")
    df = df.dropna(subset=[activity_col])
    df[activity_col] = df[activity_col].astype(int)

    print(f"[SensorPrep] Rows: {len(df)} | Activities: {sorted(df[activity_col].unique())}")

    # ── Resolve belt sensor columns (6 channels) ──────────────────────
    available_belt = [c for c in BELT_COLS if c in df.columns]

    if len(available_belt) >= 6:
        use_cols = available_belt
    else:
        meta = {"subject", "activity", "trial", "timestamps", "tag",
                "brainsensor", "infrared1", "infrared2", "infrared3",
                "infrared4", "infrared5", "infrared6"}
        numeric_cols = [c for c in df.columns if c not in meta]
        use_cols = numeric_cols[15:21] if len(numeric_cols) >= 21 else numeric_cols[:6]
        print(f"[SensorPrep] Belt cols not found by name — positional fallback: {use_cols}")

    print(f"[SensorPrep] Using {len(use_cols)} IMU channels: {use_cols}")
    print("[SensorPrep] Pressure will be AUGMENTED (realistic values, not zeros)")

    for c in use_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=use_cols)
    print(f"[SensorPrep] Rows after NaN drop: {len(df)}")

    group_cols = [c for c in [subject_col, activity_col] if c is not None]
    groups = list(df.groupby(group_cols))
    print(f"[SensorPrep] Processing {len(groups)} segments...")

    X_all, y_all = [], []
    for group_key, group_df in tqdm(groups, desc="Extracting windows"):
        if isinstance(group_key, tuple):
            activity_num = int(group_key[group_cols.index(activity_col)])
        else:
            activity_num = int(group_key)

        if activity_num in FALL_ACTIVITIES:
            label = 1
        elif activity_num in NON_FALL_ACTIVITIES:
            label = 0
        else:
            continue

        segment = group_df[use_cols].reset_index(drop=True)
        if len(segment) < WINDOW_SIZE:
            continue

        X_win, y_win = _segment_and_extract(segment, label)
        X_all.extend(X_win)
        y_all.extend(y_win)

    if len(X_all) == 0:
        raise ValueError(
            "[SensorPrep] 0 windows extracted!\n"
            f"FALL_ACTIVITIES={FALL_ACTIVITIES} | NON_FALL_ACTIVITIES={NON_FALL_ACTIVITIES}"
        )

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    print(f"\n[SensorPrep] ✅ Ready: {len(y)} windows | "
          f"Falls={int(np.sum(y==1))} | No-falls={int(np.sum(y==0))}")
    return X, y


def _load_from_individual_csvs(all_csvs):
    print(f"[SensorPrep] Found {len(all_csvs)} individual CSV files.")
    X_all, y_all = [], []
    for csv_path in tqdm(all_csvs, desc="Loading CSVs"):
        parts = csv_path.replace("\\", "/").split("/")
        activity_num = None
        for part in parts:
            p = part.upper().strip()
            if re.fullmatch(r'A\d+', p):
                activity_num = int(re.search(r'\d+', p).group()); break
            if re.match(r'ACTIVITY', p):
                d = re.search(r'\d+', p)
                if d:
                    activity_num = int(d.group()); break
        if activity_num is None:
            m = re.search(r'[_\-]A(\d+)[_\-]', os.path.basename(csv_path).upper())
            if m:
                activity_num = int(m.group(1))
        if activity_num is None:
            continue
        label = (1 if activity_num in FALL_ACTIVITIES
                 else 0 if activity_num in NON_FALL_ACTIVITIES else None)
        if label is None:
            continue
        try:
            df = pd.read_csv(csv_path, header=0)
            df.columns = [c.strip().lower() for c in df.columns]
            acc = [c for c in df.columns if "acc" in c][:3]
            gyr = [c for c in df.columns if "gyr" in c or "ang" in c][:3]
            cols = acc + gyr
            if len(cols) < 3 or len(df) < WINDOW_SIZE:
                continue
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=cols)
            X_win, y_win = _segment_and_extract(df[cols], label)
            X_all.extend(X_win)
            y_all.extend(y_win)
        except Exception:
            continue
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    print(f"[SensorPrep] Ready: {len(y)} windows | Falls={int(np.sum(y==1))} | No-falls={int(np.sum(y==0))}")
    return X, y


def extract_realtime_features(sensor_buffer, pressure_buffer=None):
    """
    Extracts features from real-time sensor window.
    sensor_buffer  : shape (>=200, 6)  — acc_xyz + gyr_xyz (raw int16 from ESP32)
    pressure_buffer: shape (>=200, 2)  — left + right pressure (ADC 0–4095)

    Returns 100-feature vector matching SVM training format.
    """
    if sensor_buffer.shape[0] < WINDOW_SIZE:
        raise ValueError(f"Buffer too short: {sensor_buffer.shape[0]} < {WINDOW_SIZE}")

    window = sensor_buffer[-WINDOW_SIZE:]   # (200, 6)

    if pressure_buffer is not None and pressure_buffer.shape[0] >= WINDOW_SIZE:
        pressure_win = pressure_buffer[-WINDOW_SIZE:]   # (200, 2)
    else:
        # No pressure hardware — use normal-upright values (not zeros)
        # Zeros would be out-of-distribution vs. training augmentation
        pressure_win = np.column_stack([
            np.random.normal(400, 40, WINDOW_SIZE),
            np.random.normal(390, 40, WINDOW_SIZE),
        ]).astype(np.float32)

    window = np.hstack([window, pressure_win])   # (200, 8)
    return _extract_features_from_window(window).reshape(1, -1)   # → 100 features