"""
preprocessing/sensor_preprocessing.py
Fall PREDICTION version — re-labels dataset so that windows BEFORE a fall
are the positive class, enabling the model to warn before impact.

Labeling strategy (UP-Fall dataset):
    Fall activities (1-5):  first PRE_FALL_RATIO fraction of windows = 1 (pre-fall)
                            remaining windows (impact + recovery) = EXCLUDED
    ADL activities (6-11):  all windows = 0 (normal)

New prediction features (Phase 2):
    - Jerk RMS + peak      : derivative of acc; spikes before loss of balance
    - ML sway RMS          : low-freq sideways wobble increases pre-fall
    - AP acceleration RMS  : forward/backward instability
    - Resultant acc stats  : magnitude mean, std, max
    - Gyro magnitude stats : trunk rotation increases as balance lost
    - CoP velocity         : fast Centre-of-Pressure shift = losing balance
    - Pressure asymmetry   : uneven weight loading
    - Stability trend      : sub-window variance slope (Lyapunov proxy)
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import welch, butter, filtfilt
from tqdm import tqdm

# Suppress precision-loss warnings from scipy kurtosis/skew on near-constant
# data (e.g. belt-offline mode returns flat neutral pressure/acc windows).
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*catastrophic cancellation.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*Precision loss.*")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (UPFALL_DATA_DIR, WINDOW_SIZE, WINDOW_STEP,
                    FALL_ACTIVITIES, NON_FALL_ACTIVITIES, PRE_FALL_RATIO)

COMPLETE_CSV  = os.path.join(UPFALL_DATA_DIR, "CompleteDataSet.csv")
BELT_ACC_COLS = ["beltaccelerometer", "unnamed:16", "unnamed:17"]
BELT_GYR_COLS = ["beltangularvelocity", "unnamed:19", "unnamed:20"]
BELT_COLS     = BELT_ACC_COLS + BELT_GYR_COLS


def _lowpass(sig, cutoff_hz=2.0, fs=200, order=4):
    nyq  = fs / 2.0
    norm = cutoff_hz / nyq
    if norm >= 1.0:
        return sig
    b, a = butter(order, norm, btype="low")
    return filtfilt(b, a, sig)


def _extract_features_from_window(window):
    """
    window : ndarray (WINDOW_SIZE, 8)
             columns: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, pL, pR]
    Returns 1-D float32 feature vector.
    """
    features = []
    n_ch = window.shape[1]

    # ── 1. Base per-channel stats (12 per channel x 8 channels = 96) ──
    for ch in range(n_ch):
        sig = window[:, ch].astype(float)
        features += [
            np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
            np.max(sig) - np.min(sig), np.median(sig),
            float(kurtosis(sig)), float(skew(sig)),
            np.sqrt(np.mean(sig ** 2)),
            np.sum(np.abs(np.diff(sig))),
        ]
        freqs, psd = welch(sig, fs=200, nperseg=min(64, len(sig)))
        features += [float(np.sum(psd)), float(freqs[np.argmax(psd)])]

    # ── 2. Cross-channel correlations (4) ─────────────────────────────
    if n_ch >= 3:
        def safe_corr(a, b):
            if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])
        features += [
            safe_corr(window[:, 0], window[:, 1]),
            safe_corr(window[:, 0], window[:, 2]),
            safe_corr(window[:, 1], window[:, 2]),
            float(np.sum(np.abs(window[:, :3]), axis=1).mean()),
        ]

    # ── 3. Phase 2 prediction features ────────────────────────────────
    acc = window[:, :3].astype(float)
    gyr = window[:, 3:6].astype(float)
    pL  = window[:, 6].astype(float)
    pR  = window[:, 7].astype(float)

    # Jerk (d/dt acceleration) — sudden spike = loss of balance
    jerk     = np.diff(acc, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    features += [
        float(np.sqrt(np.mean(jerk_mag ** 2))),   # jerk RMS magnitude
        float(np.max(jerk_mag)),                   # jerk peak
        float(np.mean(np.sqrt(np.sum(jerk**2, axis=1)))),  # jerk mean
    ]

    # ML sway RMS (low-freq acc_x oscillation)
    ml_sway = _lowpass(acc[:, 0], cutoff_hz=2.0)
    features.append(float(np.sqrt(np.mean(ml_sway ** 2))))

    # AP acceleration RMS (low-freq acc_z)
    ap_accel = _lowpass(acc[:, 2], cutoff_hz=2.0)
    features.append(float(np.sqrt(np.mean(ap_accel ** 2))))

    # Resultant acceleration magnitude
    acc_mag = np.linalg.norm(acc, axis=1)
    features += [float(np.mean(acc_mag)), float(np.std(acc_mag)),
                 float(np.max(acc_mag))]

    # Gyroscope magnitude (trunk rotation)
    gyr_mag = np.linalg.norm(gyr, axis=1)
    features += [float(np.mean(gyr_mag)), float(np.std(gyr_mag)),
                 float(np.max(gyr_mag))]

    # CoP velocity (fast CoP shift = losing balance)
    cop     = pL - pR
    cop_vel = np.abs(np.diff(cop))
    features += [float(np.mean(cop_vel)), float(np.max(cop_vel))]

    # Pressure asymmetry
    pL_mean = float(np.mean(pL))
    pR_mean = float(np.mean(pR))
    total_p = pL_mean + pR_mean + 1e-6
    features += [abs(pL_mean - pR_mean),
                 abs(pL_mean - pR_mean) / total_p]

    # Stability trend (Lyapunov proxy)
    n_sub    = 5
    sub_wins = np.array_split(acc_mag, n_sub)
    sub_vars = [np.var(s) for s in sub_wins if len(s) > 1]
    trend    = float(np.polyfit(np.arange(len(sub_vars)), sub_vars, 1)[0])
    features += [float(np.std(sub_vars)), trend]

    return np.array(features, dtype=np.float32)


def _augment_pressure(n_samples, label):
    rng = np.random.default_rng()
    if label == 1:
        if rng.random() > 0.5:
            p_left  = rng.normal(3200, 200, n_samples)
            p_right = rng.normal( 800, 150, n_samples)
        else:
            p_left  = rng.normal( 800, 150, n_samples)
            p_right = rng.normal(3200, 200, n_samples)
        pre = int(n_samples * 0.2)
        p_left[:pre]  = rng.normal(400, 40, pre)
        p_right[:pre] = rng.normal(390, 40, pre)
    else:
        p_left  = rng.normal(400, 40, n_samples)
        p_right = rng.normal(390, 40, n_samples)
    return np.stack([np.clip(p_left, 0, 4095),
                     np.clip(p_right, 0, 4095)], axis=1).astype(np.float32)


def _segment_and_extract(df, is_fall_activity):
    """
    Phase 1 — pre-fall labeling.
    is_fall_activity=True  : first PRE_FALL_RATIO windows → label 1 (pre-fall)
                             rest excluded (impact / recovery)
    is_fall_activity=False : all windows → label 0 (normal ADL)
    """
    data   = df.values.astype(float)
    n_wins = max(0, (len(data) - WINDOW_SIZE) // WINDOW_STEP + 1)
    if n_wins == 0:
        return [], []

    X_list, y_list = [], []

    if is_fall_activity:
        cutoff = int(n_wins * PRE_FALL_RATIO)
        for i, start in enumerate(
                range(0, n_wins * WINDOW_STEP, WINDOW_STEP)):
            if i >= cutoff:
                break
            end = start + WINDOW_SIZE
            if end > len(data):
                break
            imu_win  = data[start:end]
            pres_win = _augment_pressure(WINDOW_SIZE, label=1)
            feat     = _extract_features_from_window(
                           np.hstack([imu_win, pres_win]))
            X_list.append(feat)
            y_list.append(1)
    else:
        for start in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_STEP):
            imu_win  = data[start:start + WINDOW_SIZE]
            pres_win = _augment_pressure(WINDOW_SIZE, label=0)
            feat     = _extract_features_from_window(
                           np.hstack([imu_win, pres_win]))
            X_list.append(feat)
            y_list.append(0)

    return X_list, y_list


def load_upfall_dataset():
    if not os.path.isdir(UPFALL_DATA_DIR):
        raise FileNotFoundError(
            f"UP-Fall folder not found: {UPFALL_DATA_DIR}")
    if os.path.isfile(COMPLETE_CSV):
        return _load_from_complete_csv()
    all_csvs = glob.glob(
        os.path.join(UPFALL_DATA_DIR, "**", "*.csv"), recursive=True)
    if all_csvs:
        return _load_from_individual_csvs(all_csvs)
    raise FileNotFoundError(f"No CSV data found in {UPFALL_DATA_DIR}")


def _load_from_complete_csv():
    size_mb = os.path.getsize(COMPLETE_CSV) // 1024 // 1024
    print(f"[SensorPrep] Loading CompleteDataSet.csv ({size_mb} MB) ...")
    print(f"[SensorPrep] PRE-FALL labeling active — "
          f"first {int(PRE_FALL_RATIO*100)}% of fall windows = positive")

    raw = pd.read_csv(COMPLETE_CSV, header=None, nrows=5)
    row1_vals = raw.iloc[1].astype(str).str.upper().tolist()
    has_double = any("axis" in str(v).lower() or "deg/s" in str(v).lower()
                     for v in row1_vals)
    df = (pd.read_csv(COMPLETE_CSV, header=0, skiprows=[1], low_memory=False)
          if has_double
          else pd.read_csv(COMPLETE_CSV, header=0, low_memory=False))

    df.columns    = [c.strip().lower().replace(" ", "") for c in df.columns]
    subject_col   = next((c for c in df.columns if c == "subject"),  None)
    activity_col  = next((c for c in df.columns if c == "activity"), None)

    if activity_col is None:
        raise ValueError(f"'activity' column not found. Columns: {list(df.columns)}")

    df[activity_col] = pd.to_numeric(df[activity_col], errors="coerce")
    df = df.dropna(subset=[activity_col])
    df[activity_col] = df[activity_col].astype(int)
    print(f"[SensorPrep] Rows: {len(df)} | "
          f"Activities: {sorted(df[activity_col].unique())}")

    available = [c for c in BELT_COLS if c in df.columns]
    if len(available) >= 6:
        use_cols = available
    else:
        meta = {"subject","activity","trial","timestamps","tag",
                "brainsensor","infrared1","infrared2","infrared3",
                "infrared4","infrared5","infrared6"}
        num_cols = [c for c in df.columns if c not in meta]
        use_cols = (num_cols[15:21] if len(num_cols) >= 21 else num_cols[:6])
        print(f"[SensorPrep] Positional belt cols: {use_cols}")

    print(f"[SensorPrep] IMU channels: {use_cols}")
    for c in use_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=use_cols)

    group_cols = [c for c in [subject_col, activity_col] if c is not None]
    groups     = list(df.groupby(group_cols))
    print(f"[SensorPrep] Processing {len(groups)} segments ...")

    X_all, y_all = [], []
    for group_key, group_df in tqdm(groups, desc="Extracting windows"):
        act = (int(group_key[group_cols.index(activity_col)])
               if isinstance(group_key, tuple) else int(group_key))
        is_fall = act in FALL_ACTIVITIES
        if not is_fall and act not in NON_FALL_ACTIVITIES:
            continue
        seg = group_df[use_cols].reset_index(drop=True)
        if len(seg) < WINDOW_SIZE:
            continue
        X_win, y_win = _segment_and_extract(seg, is_fall_activity=is_fall)
        X_all.extend(X_win)
        y_all.extend(y_win)

    if not X_all:
        raise ValueError(
            f"0 windows extracted! FALL={FALL_ACTIVITIES} ADL={NON_FALL_ACTIVITIES}")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    print(f"\n[SensorPrep] Ready: {len(y)} windows | "
          f"Pre-fall={int(np.sum(y==1))} | Normal={int(np.sum(y==0))}")
    return X, y


def _load_from_individual_csvs(all_csvs):
    print(f"[SensorPrep] Found {len(all_csvs)} individual CSV files.")
    X_all, y_all = [], []
    for csv_path in tqdm(all_csvs, desc="Loading CSVs"):
        parts = csv_path.replace("\\", "/").split("/")
        act   = None
        for part in parts:
            p = part.upper().strip()
            if re.fullmatch(r'A\d+', p):
                act = int(re.search(r'\d+', p).group()); break
            if re.match(r'ACTIVITY', p):
                d = re.search(r'\d+', p)
                if d:
                    act = int(d.group()); break
        if act is None:
            m = re.search(r'[_\-]A(\d+)[_\-]',
                          os.path.basename(csv_path).upper())
            if m:
                act = int(m.group(1))
        if act is None:
            continue
        is_fall = act in FALL_ACTIVITIES
        if not is_fall and act not in NON_FALL_ACTIVITIES:
            continue
        try:
            df = pd.read_csv(csv_path, header=0)
            df.columns = [c.strip().lower() for c in df.columns]
            acc  = [c for c in df.columns if "acc" in c][:3]
            gyr  = [c for c in df.columns if "gyr" in c or "ang" in c][:3]
            cols = acc + gyr
            if len(cols) < 3 or len(df) < WINDOW_SIZE:
                continue
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=cols)
            X_w, y_w = _segment_and_extract(df[cols], is_fall_activity=is_fall)
            X_all.extend(X_w); y_all.extend(y_w)
        except Exception:
            continue

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    print(f"[SensorPrep] Ready: {len(y)} windows | "
          f"Pre-fall={int(np.sum(y==1))} | Normal={int(np.sum(y==0))}")
    return X, y


def extract_realtime_features(sensor_buffer, pressure_buffer=None):
    """Real-time inference path — same features as training."""
    if sensor_buffer.shape[0] < WINDOW_SIZE:
        raise ValueError(f"Buffer too short: {sensor_buffer.shape[0]} < {WINDOW_SIZE}")
    window = sensor_buffer[-WINDOW_SIZE:]
    if pressure_buffer is not None and pressure_buffer.shape[0] >= WINDOW_SIZE:
        p_win = np.clip(pressure_buffer[-WINDOW_SIZE:], 0, 4095).astype(np.float32)
    else:
        p_win = np.column_stack([
            np.random.normal(400, 40, WINDOW_SIZE),
            np.random.normal(390, 40, WINDOW_SIZE),
        ]).astype(np.float32)
    window = np.hstack([window, p_win])
    return _extract_features_from_window(window).reshape(1, -1)
