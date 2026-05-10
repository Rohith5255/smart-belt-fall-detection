"""
Quick diagnostic: simulate exactly what sensor_thread does at runtime
and print the raw features + SVM score for both normal and fall windows.
"""
import sys
sys.path.insert(0, r"C:/Users/rohit/OneDrive/Desktop/CAPSTONE")

import numpy as np
import joblib

# ── Reproduce the simulated sensor data exactly as iot_receiver.py does ──

WINDOW_SIZE = 200

def make_normal_window():
    acc = np.column_stack([
        np.random.normal(0.0,  0.05, WINDOW_SIZE),
        np.random.normal(-1.0, 0.08, WINDOW_SIZE),
        np.random.normal(0.0,  0.05, WINDOW_SIZE),
    ])
    gyr = np.random.normal(0.0, 2.0, (WINDOW_SIZE, 3))
    sensor_win = np.hstack([acc, gyr]).astype(np.float32)   # (200, 6)
    p = np.column_stack([
        np.random.normal(400, 20, WINDOW_SIZE),
        np.random.normal(390, 20, WINDOW_SIZE),
    ]).astype(np.float32)                                    # (200, 2)
    return sensor_win, p

def make_fall_window():
    acc = np.column_stack([
        np.random.normal( 2.0, 0.5,   WINDOW_SIZE),
        np.random.normal(-5.0, 1.0,   WINDOW_SIZE),
        np.random.normal( 1.5, 0.5,   WINDOW_SIZE),
    ])
    gyr = np.random.normal(0, 300.0, (WINDOW_SIZE, 3))
    sensor_win = np.hstack([acc, gyr]).astype(np.float32)   # (200, 6)
    p = np.column_stack([
        np.random.normal(3200, 150, WINDOW_SIZE),
        np.random.normal( 800, 100, WINDOW_SIZE),
    ]).astype(np.float32)                                    # (200, 2)
    return sensor_win, p

# ── Load the actual saved SVM ──
pipeline = joblib.load(r"C:/Users/rohit/OneDrive/Desktop/CAPSTONE/saved_models/svm_model.pkl")

# ── Import the actual extract_realtime_features ──
from preprocessing.sensor_preprocessing import extract_realtime_features

print("=== NORMAL WINDOW ===")
for i in range(3):
    s, p = make_normal_window()
    X = extract_realtime_features(s, p)
    score = pipeline.predict_proba(X)[0, 1]
    print(f"  Run {i+1}: SVM fall proba = {score:.4f}  | feature mean={X.mean():.4f}  std={X.std():.4f}")

print()
print("=== FALL WINDOW ===")
for i in range(3):
    s, p = make_fall_window()
    X = extract_realtime_features(s, p)
    score = pipeline.predict_proba(X)[0, 1]
    print(f"  Run {i+1}: SVM fall proba = {score:.4f}  | feature mean={X.mean():.4f}  std={X.std():.4f}")

print()
print("=== TRAINING DATA SAMPLE (what SVM was trained on) ===")
from preprocessing.sensor_preprocessing import load_upfall_dataset
X_train, y_train = load_upfall_dataset()
import numpy as np
fall_idx = np.where(y_train == 1)[0][:3]
nfall_idx = np.where(y_train == 0)[0][:3]
print("Fall training samples feature stats:")
for i in fall_idx:
    print(f"  feature mean={X_train[i].mean():.4f}  std={X_train[i].std():.4f}  -> label=FALL")
print("No-fall training samples feature stats:")
for i in nfall_idx:
    print(f"  feature mean={X_train[i].mean():.4f}  std={X_train[i].std():.4f}  -> label=NORMAL")