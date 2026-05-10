"""
inference/realtime_inference.py — Fall PREDICTION pipeline.

Run from CAPSTONE folder:
    python main.py --mode run              <- live (ESP32 + camera)
    python main.py --mode run --simulate   <- no hardware needed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
import csv
import numpy as np
import torch
from collections import deque
from datetime import datetime

from config import (CAMERA_INDEX,
                    RISK_BUFFER_SIZE, RISK_SLOPE_THRESHOLD,
                    RISK_SCORE_THRESHOLD, PREDICTION_HORIZON_SEC,
                    ALERT_COOLDOWN_SECONDS)
from utils.camera_utils import auto_detect_camera
from models.svm_model import load_svm, svm_predict_proba
from fusion.fusion_model import fuse, load_fusion_model
from preprocessing.sensor_preprocessing import extract_realtime_features
from preprocessing.pose_preprocessing import PoseEstimator
from iot.iot_receiver import ESP32Receiver, SimulatedESP32Receiver
from alerts.alert_system import AlertSystem


# ── Shared state ──────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.svm_proba      = 0.0
        self.pose_score     = 0.0      # replaces vit_proba
        self.fused_proba    = 0.0
        self.label          = 0
        self.fall_predicted = False    # Phase 4: rising risk flag
        self.risk_slope     = 0.0
        self.latest_frame   = None
        self.pose_features  = {}
        self.running        = True
        self.camera_name    = "—"      # set by vision_thread after detection
        self.belt_offline   = False    # True when ESP32 not connected


# ── Phase 4: temporal risk trend ─────────────────────────────────────────────
class RiskTrend:
    """
    Maintains a rolling buffer of fused scores and detects a rising trend.
    A rising trend above a threshold fires the 'FALL PREDICTED' warning
    BEFORE a fall occurs — the core goal of the prediction system.
    """
    def __init__(self, size=RISK_BUFFER_SIZE):
        self._buf = deque(maxlen=size)

    def update(self, score):
        self._buf.append(float(score))

    @property
    def slope(self):
        if len(self._buf) < 5:
            return 0.0
        x = np.arange(len(self._buf))
        return float(np.polyfit(x, list(self._buf), 1)[0])

    @property
    def mean(self):
        return float(np.mean(list(self._buf))) if self._buf else 0.0

    def predict_fall(self):
        return (self.slope > RISK_SLOPE_THRESHOLD and
                self.mean   > RISK_SCORE_THRESHOLD)


# ── Worker threads ────────────────────────────────────────────────────────────
def sensor_thread(receiver, svm_pipeline, state):
    print("[SensorThread] Started.")
    # Reflect belt connection status in shared state
    offline = getattr(receiver, "is_offline", False)
    with state.lock:
        state.belt_offline = offline
    if offline:
        print("[SensorThread] Belt offline — SVM will output 0.0 (camera-only mode).")

    while state.running:
        if not receiver.is_ready():
            time.sleep(0.05)
            continue
        sensor_win, pressure_win = receiver.get_sensor_window()
        try:
            X     = extract_realtime_features(sensor_win, pressure_win)
            X     = np.nan_to_num(X, nan=0.0)
            proba = svm_predict_proba(svm_pipeline, X)[0]
            with state.lock:
                state.svm_proba = float(proba)
        except Exception as e:
            print(f"[SensorThread] Error: {e}")
        time.sleep(0.10)


def vision_thread(pose_estimator, state, simulate=False):
    """
    Phase 3: runs MediaPipe Pose on each camera frame instead of ViT.
    Simulate mode injects synthetic pose scores matching a realistic
    pre-fall → fall progression.
    """
    if simulate:
        print("[VisionThread] Simulate mode — synthetic pose scores.")
        _vision_simulate_pose(state)
        return

    import cv2
    # Auto-detect: prefer external/secondary webcam, fall back to laptop cam (0)
    cam_idx, cam_name = auto_detect_camera(fallback_index=CAMERA_INDEX)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[VisionThread] Cannot open camera [{cam_idx}] {cam_name}.")
        while state.running:
            time.sleep(0.1)
        return

    print(f"[VisionThread] Using camera [{cam_idx}] {cam_name} — MediaPipe Pose active.")
    with state.lock:
        state.camera_name = f"[{cam_idx}] {cam_name}"
    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        features = pose_estimator.process_frame(frame)

        with state.lock:
            state.latest_frame  = frame.copy()
            state.pose_score    = features["pose_score"]
            state.pose_features = features

    cap.release()
    pose_estimator.close()


def _vision_simulate_pose(state):
    """
    Synthetic pose-score timeline for --simulate mode:
        0-5s  : normal posture  → score ~0.05
        5-7s  : balance loss    → score rising 0.2 → 0.65  (PREDICTION fires here)
        7-8s  : fall            → score ~0.90
        8s+   : post-fall       → score ~0.80
    """
    start = time.time()
    while state.running:
        elapsed = time.time() - start
        if elapsed < 5.0:
            score = float(np.clip(np.random.normal(0.05, 0.02), 0, 1))
        elif elapsed < 7.0:
            # Linearly rising score simulating balance deterioration
            t_norm = (elapsed - 5.0) / 2.0
            score  = float(np.clip(
                np.random.normal(0.05 + t_norm * 0.60, 0.03), 0, 1))
        elif elapsed < 8.0:
            score = float(np.clip(np.random.normal(0.90, 0.04), 0, 1))
        else:
            score = float(np.clip(np.random.normal(0.80, 0.05), 0, 1))

        # Synthetic trunk lean that matches the pose timeline
        if elapsed < 5.0:
            lean = 0.0
        elif elapsed < 7.0:
            lean = ((elapsed - 5.0) / 2.0) * 35.0   # 0 → 35° over 2s
        else:
            lean = 45.0                               # fully tipped over

        frame = np.random.randint(
            150 if elapsed >= 7 else 50, 255 if elapsed >= 7 else 120,
            (480, 640, 3), dtype=np.uint8)

        with state.lock:
            state.latest_frame  = frame.copy()
            state.pose_score    = score
            state.pose_features = {
                "pose_detected" : 1,
                "trunk_lean_deg": round(lean, 2),
                "shoulder_roll" : 0.0,
                "knee_asymmetry": 0.0,
                "head_drop_rate": 0.0,
                "pose_score"    : score,
            }
        time.sleep(0.1)


def fusion_thread(state, alert_system, risk_trend, mlp_fusion=None):
    print("[FusionThread] Started.")
    _high_pose_streak = 0   # consecutive cycles with pose_score >= 0.30
    while state.running:
        with state.lock:
            svm_p = state.svm_proba
            pos_p = state.pose_score

        # fuse() uses svm_proba as first arg, pose_score as second
        fused_proba, label, _ = fuse(svm_p, pos_p, mlp_fusion)

        # Phase 4: update risk trend
        risk_trend.update(fused_proba)
        trend_predicted = risk_trend.predict_fall()
        slope           = risk_trend.slope

        # Early-warning: 3 consecutive cycles (0.6s) of elevated pose score
        # catches balance loss before the full trend builds up
        if pos_p >= 0.30:
            _high_pose_streak += 1
        else:
            _high_pose_streak = 0
        pose_early_warn = (_high_pose_streak >= 3)

        predicted = trend_predicted or pose_early_warn

        with state.lock:
            state.fused_proba    = fused_proba
            state.label          = label
            state.fall_predicted = predicted
            state.risk_slope     = slope

        # Alert on rising prediction OR confirmed fall
        if predicted or label == 1:
            alert_system.trigger(fused_proba, svm_p, pos_p,
                                 predicted=predicted)
        time.sleep(0.20)


def csv_logger_thread(state, log_path, interval=0.20):
    """Writes one row every interval seconds — flushed after each write."""
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "elapsed_s",
            "svm_proba", "pose_score", "fused_proba",
            "label", "fall_predicted", "risk_slope",
            "trunk_lean_deg", "knee_asymmetry", "head_drop_rate",
        ])
        f.flush()
        print(f"[CSVLogger] Logging -> {log_path}")
        start = time.time()
        while state.running:
            with state.lock:
                svm_p   = state.svm_proba
                pos_p   = state.pose_score
                fused   = state.fused_proba
                label   = state.label
                pred    = int(state.fall_predicted)
                slope   = state.risk_slope
                pf      = state.pose_features

            ts      = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            elapsed = round(time.time() - start, 2)
            writer.writerow([
                ts, elapsed,
                f"{svm_p:.4f}", f"{pos_p:.4f}", f"{fused:.4f}",
                label, pred, f"{slope:.5f}",
                pf.get("trunk_lean_deg", 0.0),
                pf.get("knee_asymmetry", 0.0),
                pf.get("head_drop_rate", 0.0),
            ])
            f.flush()
            time.sleep(interval)
    print(f"[CSVLogger] Saved -> {log_path}")


def display_thread_gui(state, pose_estimator):
    import cv2
    cv2.namedWindow("Fall Prediction System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fall Prediction System", 800, 600)

    while state.running:
        with state.lock:
            frame        = state.latest_frame
            svm_p        = state.svm_proba
            cam_name     = state.camera_name
            pos_p        = state.pose_score
            fused        = state.fused_proba
            label        = state.label
            pred         = state.fall_predicted
            slope        = state.risk_slope
            pf           = dict(state.pose_features)
            belt_offline = state.belt_offline

        # Draw skeleton on a copy of the frame
        if frame is not None:
            display = pose_estimator.draw_skeleton(frame.copy(), pf)
        else:
            display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(display, "Waiting for camera...", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        h, w = display.shape[:2]

        # Semi-transparent header bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 175), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

        # Camera label (top-right corner)
        cam_label = f"Cam: {cam_name}"
        (lw, lh), _ = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(display, cam_label, (w - lw - 8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Scores
        cv2.putText(display,
                    f"SVM:   {svm_p:.3f}" + ("  [BELT OFFLINE]" if belt_offline else ""),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (100, 100, 255) if belt_offline else (255, 255, 255), 2)
        cv2.putText(display, f"Pose:  {pos_p:.3f}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(display, f"Fused: {fused:.3f}  slope: {slope:+.4f}",
                    (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)

        # Trunk lean indicator (abs because backward lean is also risky)
        lean = pf.get("trunk_lean_deg", 0.0)
        lean_mag = abs(lean)
        lean_color = ((0, 255, 0) if lean_mag < 12
                      else (0, 165, 255) if lean_mag < 25
                      else (0, 0, 255))
        direction = "FWD" if lean > 0 else "BACK" if lean < 0 else "—"
        cv2.putText(display, f"Trunk lean: {lean:+.1f} deg ({direction})",
                    (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lean_color, 2)

        # Status label
        if label == 1:
            status, color = "FALL DETECTED", (0, 0, 255)
        elif pred:
            status, color = (f"FALL PREDICTED (~{PREDICTION_HORIZON_SEC:.0f}s)",
                             (0, 100, 255))
        else:
            status, color = "Normal", (0, 255, 0)

        cv2.putText(display, status, (10, 168),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        # Rising risk bar (bottom of screen)
        if slope > 0:
            bar_w = int(min(slope / 0.02, 1.0) * w)
            cv2.rectangle(display, (0, h - 12), (bar_w, h),
                          (0, 100, 255), -1)
            cv2.putText(display, "Risk rising", (5, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(display, "Q / Esc to quit", (w - 180, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Fall Prediction System", display)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            state.running = False
            break

    cv2.destroyAllWindows()


def display_terminal(state, duration_sec=30):
    print("\n[Display] Terminal mode — updating every 1s. Ctrl+C to stop.")
    print(f"{'Time':>6}  {'SVM':>6}  {'Pose':>6}  {'Fused':>6}  "
          f"{'Slope':>8}  Status")
    print("-" * 58)

    start = time.time()
    try:
        while state.running:
            elapsed = time.time() - start
            if elapsed > duration_sec:
                print(f"\n[Display] Simulation done ({duration_sec}s). Shutting down.")
                state.running = False
                break

            with state.lock:
                svm_p = state.svm_proba
                pos_p = state.pose_score
                fused = state.fused_proba
                label = state.label
                pred  = state.fall_predicted
                slope = state.risk_slope

            if label == 1:
                status = "FALL DETECTED"
            elif pred:
                status = "FALL PREDICTED"
            else:
                status = "Normal"

            print(f"{elapsed:6.1f}s  {svm_p:6.3f}  {pos_p:6.3f}  "
                  f"{fused:6.3f}  {slope:+8.4f}  {status}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Display] Interrupted.")
        state.running = False


# ── Main entry point ──────────────────────────────────────────────────────────
def run_realtime(simulate=False):
    print("\n" + "=" * 60)
    print("  Fall PREDICTION System  —  Real-time Inference")
    print(f"  Mode: {'SIMULATE' if simulate else 'LIVE HARDWARE'}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Init] Device: {device}")

    print("[Init] Loading SVM (pre-fall predictor) ...")
    svm_pipeline  = load_svm()
    mlp_fusion    = load_fusion_model()
    alert_system  = AlertSystem()
    pose_est      = PoseEstimator()
    risk_trend    = RiskTrend()
    state         = SharedState()

    # CSV log file
    mode_tag = "simulate" if simulate else "live"
    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f"predict_log_{mode_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    receiver = SimulatedESP32Receiver() if simulate else ESP32Receiver()
    receiver.start()

    threads = [
        threading.Thread(target=sensor_thread,
                         args=(receiver, svm_pipeline, state), daemon=True),
        threading.Thread(target=vision_thread,
                         args=(pose_est, state, simulate), daemon=True),
        threading.Thread(target=fusion_thread,
                         args=(state, alert_system, risk_trend, mlp_fusion),
                         daemon=True),
        threading.Thread(target=csv_logger_thread,
                         args=(state, log_path), daemon=True),
    ]
    for t in threads:
        t.start()

    print("\n[System] All threads started. Prediction pipeline running ...\n")

    try:
        if simulate:
            display_terminal(state, duration_sec=35)
        else:
            display_thread_gui(state, pose_est)
    except KeyboardInterrupt:
        print("\n[System] Interrupted.")
    finally:
        state.running = False
        receiver.stop()
        for t in threads:
            t.join(timeout=2.0)

        # Summary
        try:
            import pandas as pd
            df = pd.read_csv(log_path)
            print(f"\n[Log] Run summary -> {log_path}")
            print(f"  Duration      : {df['elapsed_s'].max():.1f}s")
            print(f"  Rows logged   : {len(df)}")
            print(f"  SVM max       : {df['svm_proba'].max():.4f}")
            print(f"  Pose max      : {df['pose_score'].max():.4f}")
            print(f"  Fused max     : {df['fused_proba'].max():.4f}")
            print(f"  FALL PREDICTED: {df['fall_predicted'].sum()} rows")
            print(f"  FALL DETECTED : {df['label'].sum()} rows")
        except Exception:
            pass

        print("[System] Shutdown complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    run_realtime(simulate=args.simulate)
