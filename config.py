"""
config.py — Central configuration for the Fall Detection System
Adjust paths and hyperparameters here before running any scripts.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset Paths ──────────────────────────────────────────────────────
# URFD: folder that directly contains fall-XX-cam0-rgb/ and adl-XX-cam0-rgb/
UPFALL_DATA_DIR = os.path.join(BASE_DIR, "data", "UP_Fall_Detection_Dataset")

URFD_DATA_DIR   = os.path.join(BASE_DIR, "data", "urfd",
                                "UR_fall_detection_dataset_cam0_rgb")

# Le2i raw videos (your current structure with Videos/ + Annotation_files/)
LE2I_RAW_DIR    = os.path.join(BASE_DIR, "data", "le2i")

# Le2i extracted frames — created by running extract_le2i_frames.py
LE2I_FRAMES_DIR = os.path.join(BASE_DIR, "data", "le2i_frames")

SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ── SVM / Sensor Config ────────────────────────────────────────────────
FALL_ACTIVITIES     = [1, 2, 3, 4, 5]
NON_FALL_ACTIVITIES = [6, 7, 8, 9, 10, 11]
SENSOR_SAMPLING_RATE = 200
WINDOW_SIZE          = 200
WINDOW_STEP          = 100
SENSOR_COLUMNS       = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
PRESSURE_COLUMNS     = ["pressure_left", "pressure_right"]
SVM_MODEL_PATH       = os.path.join(SAVED_MODELS_DIR, "svm_model.pkl")
SCALER_PATH          = os.path.join(SAVED_MODELS_DIR, "svm_scaler.pkl")

# ── ViT Config (legacy — ViT replaced by MediaPipe Pose in Phase 3) ────
VIT_MODEL_NAME  = "vit_base_patch16_224"
VIT_NUM_CLASSES = 2
VIT_IMG_SIZE    = 224
VIT_BATCH_SIZE  = 16
VIT_EPOCHS      = 20
VIT_LR          = 3e-4
VIT_PRETRAINED  = True
VIT_MODEL_PATH  = os.path.join(SAVED_MODELS_DIR, "vit_model.pth")

# ── Fusion Config ──────────────────────────────────────────────────────
FUSION_SVM_WEIGHT = 0.55
FUSION_VIT_WEIGHT = 0.45
FALL_THRESHOLD      = 0.18   # above normal-movement ceiling (SVM noise * amplifier)
SVM_AMPLIFIER       = 5.0    # only applied when SVM spikes above SVM_SPIKE_THRESHOLD
# debug_svm.py shows normal SVM idles at 0.003–0.006, fall reaches 0.017–0.018.
# Old threshold (0.030) was above the fall peak → amplifier NEVER fired.
# New threshold (0.012) sits between normal ceiling (0.006) and fall floor (0.017).
SVM_SPIKE_THRESHOLD = 0.012  # was 0.030 — now catches real fall spikes from belt
FUSION_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "fusion_model.pkl")

# ── IoT / Serial Config ────────────────────────────────────────────────
SERIAL_PORT     = "COM5"       # Windows: COM3, COM4 etc. | Linux: /dev/ttyUSB0
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT  = 1
ESP32_DATA_FORMAT = ["acc_x", "acc_y", "acc_z",
                      "gyr_x", "gyr_y", "gyr_z",
                      "pressure_left", "pressure_right"]
# Set to whichever index is your external webcam.
# Run:  python find_cameras.py   to see which index shows your external webcam.
CAMERA_INDEX    = 0

# ── Alert Config ───────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 10

# ── Fall PREDICTION Config ──────────────────────────────────────────────
# Fraction of each fall-activity segment used as pre-fall positive label.
# First PRE_FALL_RATIO of windows = pre-fall instability (label = 1).
# Remaining windows (impact + recovery) are excluded from training.
PRE_FALL_RATIO        = 0.60   # first 60% of fall sequence = pre-fall

# Rolling risk-trend buffer (number of fusion scores kept for slope calc)
RISK_BUFFER_SIZE      = 8      # ~1.6s at 0.2s per cycle (was 15/3s — too slow to predict)

# Thresholds to fire "FALL PREDICTED" warning
RISK_SLOPE_THRESHOLD  = 0.003  # raised from 0.0015 — ignores SVM noise fluctuations
RISK_SCORE_THRESHOLD  = 0.07   # mean fused score baseline gate (kept for RiskTrend)

# Minimum fused score to even consider a FALL PREDICTED alert.
# Stops noise-driven false positives in belt-offline / camera-only mode
# where SVM idles ~0.015 and brief pose flickers push fused to ~0.10.
# Real pre-fall events produce fused >= 0.20 well before impact.
FALL_PREDICT_THRESHOLD = 0.15

# Prediction horizon printed in alerts
PREDICTION_HORIZON_SEC = 2.0

ROTATE_CAMERA_FRAME = True   # True = phone portrait, False = laptop webcam