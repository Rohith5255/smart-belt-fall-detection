"""
diagnose_live.py — Logs raw sensor values + model scores to CSV with timestamps.
Run: python diagnose_live.py
Output: sensor_log_YYYYMMDD_HHMMSS.csv  (upload this here for analysis)
"""

import serial
import time
import csv
import numpy as np
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

SERIAL_PORT = "COM5"
BAUD = 115200
WINDOW_SIZE = 200
LOG_DURATION = 60  # seconds — increase if you want longer test

def run_diagnostic():
    print("=" * 55)
    print("  SENSOR DIAGNOSTIC — Live Belt Data Logger")
    print("=" * 55)

    # --- Load models ---
    try:
        from models.svm_model import load_svm, svm_predict_proba
        svm = load_svm()
        print("[OK] SVM loaded")
    except Exception as e:
        print(f"[FAIL] SVM load error: {e}")
        svm = None

    # --- Open serial ---
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=2)
        time.sleep(2)
        print(f"[OK] Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"[FAIL] Serial error: {e}")
        return

    # --- Setup log file ---
    fname = f"sensor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    f = open(fname, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "elapsed_s",
        "acc_x_g", "acc_y_g", "acc_z_g",
        "gyr_x_dps", "gyr_y_dps", "gyr_z_dps",
        "pressure_left_adc", "pressure_right_adc",
        "accel_magnitude_g",
        "svm_score", "svm_label",
        "note"
    ])

    buffer = []
    start = time.time()
    sample_count = 0
    last_print = time.time()

    print(f"\nLogging for {LOG_DURATION}s → {fname}")
    print("Move around, walk, sit, do a controlled fall...")
    print("Press Ctrl+C to stop early\n")
    print(f"{'Time':>6}  {'ax':>7}  {'ay':>7}  {'az':>7}  {'|a|':>6}  {'pL':>6}  {'pR':>6}  {'SVM':>6}")
    print("-" * 65)

    try:
        while time.time() - start < LOG_DURATION:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parts = [float(v) for v in line.split(",")]
                if len(parts) < 6:
                    continue
                while len(parts) < 8:
                    parts.append(0.0)

                # Convert raw → physical units
                ax = parts[0] / 16384.0
                ay = parts[1] / 16384.0
                az = parts[2] / 16384.0
                gx = parts[3] / 131.0
                gy = parts[4] / 131.0
                gz = parts[5] / 131.0
                pL = parts[6]
                pR = parts[7]
                amag = np.sqrt(ax**2 + ay**2 + az**2)

                elapsed = time.time() - start
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                sample_count += 1

                # Add to window buffer
                buffer.append([ax, ay, az, gx, gy, gz, pL, pR])
                if len(buffer) > WINDOW_SIZE:
                    buffer.pop(0)

                # SVM inference when buffer full
                svm_score = -1.0
                svm_label = -1
                if svm is not None and len(buffer) >= WINDOW_SIZE:
                    try:
                        from preprocessing.sensor_preprocessing import extract_realtime_features
                        win = np.array(buffer[-WINDOW_SIZE:], dtype=np.float32)
                        X = extract_realtime_features(win[:, :6], win[:, 6:])
                        X = np.nan_to_num(X, nan=0.0)
                        svm_score = float(svm_predict_proba(svm, X)[0])
                        svm_label = 1 if svm_score >= 0.35 else 0
                    except Exception as ex:
                        pass

                # Log to CSV
                writer.writerow([ts, f"{elapsed:.2f}",
                    f"{ax:.4f}", f"{ay:.4f}", f"{az:.4f}",
                    f"{gx:.3f}", f"{gy:.3f}", f"{gz:.3f}",
                    f"{pL:.0f}", f"{pR:.0f}",
                    f"{amag:.4f}",
                    f"{svm_score:.4f}", svm_label, ""])
                f.flush()

                # Print every 0.5s
                if time.time() - last_print >= 0.5:
                    flag = " ⚠️ FALL" if svm_label == 1 else ""
                    print(f"{elapsed:6.1f}s  {ax:7.3f}  {ay:7.3f}  {az:7.3f}  "
                          f"{amag:6.3f}  {pL:6.0f}  {pR:6.0f}  {svm_score:6.3f}{flag}")
                    last_print = time.time()

            except ValueError:
                continue

    except KeyboardInterrupt:
        print("\n[Stopped by user]")

    finally:
        f.close()
        ser.close()
        print(f"\n✅ Logged {sample_count} samples → {fname}")
        print("Upload this CSV file here and I will analyze the sensor values.")
        _quick_summary(fname)


def _quick_summary(fname):
    import pandas as pd
    df = pd.read_csv(fname)
    print("\n--- QUICK SUMMARY ---")
    print(f"Duration    : {df['elapsed_s'].max():.1f}s")
    print(f"Samples     : {len(df)}")
    print(f"acc_y mean  : {df['acc_y_g'].mean():.3f}g  (expect ~+1.0 when upright)")
    print(f"|a| mean    : {df['accel_magnitude_g'].mean():.3f}g  (expect ~1.0 at rest)")
    print(f"pL range    : {df['pressure_left_adc'].min():.0f} – {df['pressure_left_adc'].max():.0f}  (0 = sensor issue)")
    print(f"pR range    : {df['pressure_right_adc'].min():.0f} – {df['pressure_right_adc'].max():.0f}")
    print(f"SVM max     : {df['svm_score'].max():.4f}")
    print(f"FALL triggers: {(df['svm_label'] == 1).sum()}")
    print("---------------------")


if __name__ == "__main__":
    run_diagnostic()