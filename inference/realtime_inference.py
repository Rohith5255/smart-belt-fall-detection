"""
inference/realtime_inference.py — Real-time fall detection pipeline.

Run from CAPSTONE folder:
    python main.py --mode run              ← live system (needs ESP32 + camera)
    python main.py --mode run --simulate   ← no hardware needed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
import numpy as np
import torch

from config import CAMERA_INDEX
from models.svm_model import load_svm, svm_predict_proba
from models.vit_model  import load_vit, vit_predict_proba
from fusion.fusion_model import fuse, load_fusion_model
from preprocessing.sensor_preprocessing import extract_realtime_features
from preprocessing.video_preprocessing  import preprocess_frame_for_inference
from iot.iot_receiver import ESP32Receiver, SimulatedESP32Receiver
from alerts.alert_system import AlertSystem


class SharedState:
    def __init__(self):
        self.lock         = threading.Lock()
        self.svm_proba    = 0.0
        self.vit_proba    = 0.0
        self.fused_proba  = 0.0
        self.label        = 0
        self.latest_frame = None
        self.running      = True


def sensor_thread(receiver, svm_pipeline, state):
    print("[SensorThread] Started.")
    while state.running:
        if not receiver.is_ready():
            time.sleep(0.05)
            continue
        sensor_win, pressure_win = receiver.get_sensor_window()
        try:
            X     = extract_realtime_features(sensor_win, pressure_win)
            proba = svm_predict_proba(svm_pipeline, X)[0]
            with state.lock:
                state.svm_proba = float(proba)
        except Exception as e:
            print(f"[SensorThread] Error: {e}")
        time.sleep(0.10)


def vision_thread(vit_model, device, state, simulate=False):
    """Camera vision thread. In simulate mode, generates random frames."""
    if simulate:
        print("[VisionThread] Simulate mode — generating synthetic frames.")
        _vision_simulate(vit_model, device, state)
        return

    import cv2
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[VisionThread] Cannot open camera {CAMERA_INDEX} — vision disabled.")
        while state.running:
            time.sleep(0.1)
        return

    print(f"[VisionThread] Camera {CAMERA_INDEX} opened.")
    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        with state.lock:
            state.latest_frame = frame.copy()
        tensor = preprocess_frame_for_inference(frame)
        proba  = vit_predict_proba(vit_model, tensor, device)
        with state.lock:
            state.vit_proba = float(proba)
    cap.release()


def _vision_simulate(vit_model, device, state):
    """
    Simulate mode: inject synthetic ViT probabilities directly.

    Why: The ViT is trained on real fall/ADL video frames. Random noise
    frames produce ~0.46 probability for BOTH normal and fall states,
    making the ViT useless as a discriminator in simulation.

    Instead we inject realistic synthetic probabilities that match what
    the ViT would output on real video:
        0 – 6s  : normal ADL  → ViT ~ 0.05  (low fall probability)
        6 – 8s  : fall event  → ViT ~ 0.92  (high fall probability)
        8s+     : post-fall   → ViT ~ 0.85  (still elevated)

    The ViT model is NOT called in simulate mode — this saves GPU cycles.
    In real (non-simulate) mode, the actual ViT runs on camera frames as usual.
    """
    start_time = time.time()
    while state.running:
        elapsed = time.time() - start_time

        if elapsed < 6.0:
            # Normal walking — very low fall probability
            proba = np.random.normal(0.05, 0.02)
        elif elapsed < 8.0:
            # Fall event — high fall probability
            proba = np.random.normal(0.92, 0.04)
        else:
            # Post-fall — still elevated
            proba = np.random.normal(0.85, 0.05)

        proba = float(np.clip(proba, 0.0, 1.0))

        # Generate a placeholder frame for display purposes only
        if elapsed >= 6.0 and elapsed < 8.0:
            frame = np.random.randint(150, 255, (480, 640, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(50, 120, (480, 640, 3), dtype=np.uint8)

        with state.lock:
            state.latest_frame = frame.copy()
            state.vit_proba = proba
        time.sleep(0.1)


def fusion_thread(state, alert_system, mlp_fusion=None):
    print("[FusionThread] Started.")
    while state.running:
        with state.lock:
            svm_p = state.svm_proba
            vit_p = state.vit_proba
        fused_proba, label, _ = fuse(svm_p, vit_p, mlp_fusion)
        with state.lock:
            state.fused_proba = fused_proba
            state.label       = label
        if label == 1:
            alert_system.trigger(fused_proba, svm_p, vit_p)
        time.sleep(0.20)


def display_thread_gui(state):
    """Full OpenCV GUI display — used in real (non-simulate) mode."""
    import cv2
    cv2.namedWindow("Fall Detection System", cv2.WINDOW_NORMAL)
    while state.running:
        with state.lock:
            frame  = state.latest_frame
            fused  = state.fused_proba
            svm_p  = state.svm_proba
            vit_p  = state.vit_proba
            label  = state.label

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        display = frame.copy()
        h, w    = display.shape[:2]

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        status_color = (0, 0, 255) if label == 1 else (0, 255, 0)
        status_text  = "FALL DETECTED" if label == 1 else "Normal"
        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(display,
                    f"Fused:{fused:.2f}  SVM:{svm_p:.2f}  ViT:{vit_p:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        bar_w     = int(fused * (w - 20))
        bar_color = (0, 0, 255) if label == 1 else (0, 200, 0)
        cv2.rectangle(display, (10, h-20), (10+bar_w, h-5), bar_color, -1)
        cv2.rectangle(display, (10, h-20), (w-10,    h-5), (100, 100, 100), 1)

        cv2.imshow("Fall Detection System", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            state.running = False
            break
    cv2.destroyAllWindows()


def display_terminal(state, duration_sec=30):
    """
    Terminal-only display for simulate mode.
    Prints live probabilities every second for `duration_sec` seconds.
    """
    print("\n[Display] Terminal mode — updating every 1s. Ctrl+C to stop.")
    print(f"{'Time':>6}  {'SVM':>6}  {'ViT':>6}  {'Fused':>6}  Status")
    print("-" * 45)

    start = time.time()
    try:
        while state.running:
            elapsed = time.time() - start
            if elapsed > duration_sec:
                print(f"\n[Display] Simulation complete ({duration_sec}s). Shutting down.")
                state.running = False
                break

            with state.lock:
                svm_p  = state.svm_proba
                vit_p  = state.vit_proba
                fused  = state.fused_proba
                label  = state.label

            status = "⚠️  FALL DETECTED" if label == 1 else "✅ Normal"
            print(f"{elapsed:6.1f}s  {svm_p:6.3f}  {vit_p:6.3f}  {fused:6.3f}  {status}")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[Display] Interrupted by user.")
        state.running = False


def run_realtime(simulate=False):
    print("\n" + "=" * 60)
    print("  Fall Detection System  —  Real-time Inference")
    print(f"  Mode: {'SIMULATE' if simulate else 'LIVE HARDWARE'}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Init] Device : {device}")

    print("[Init] Loading SVM ...")
    svm_pipeline = load_svm()
    print("[Init] Loading ViT ...")
    vit_model    = load_vit(device=device)
    mlp_fusion   = load_fusion_model()
    alert_system = AlertSystem()
    state        = SharedState()

    receiver = SimulatedESP32Receiver() if simulate else ESP32Receiver()
    receiver.start()

    threads = [
        threading.Thread(target=sensor_thread,
                         args=(receiver, svm_pipeline, state),
                         daemon=True),
        threading.Thread(target=vision_thread,
                         args=(vit_model, device, state, simulate),
                         daemon=True),
        threading.Thread(target=fusion_thread,
                         args=(state, alert_system, mlp_fusion),
                         daemon=True),
    ]
    for t in threads:
        t.start()

    print("\n[System] All threads started. Pipeline running ...\n")

    try:
        if simulate:
            # Terminal display — no camera/GUI needed
            display_terminal(state, duration_sec=30)
        else:
            # Full GUI display with OpenCV window
            display_thread_gui(state)
    except KeyboardInterrupt:
        print("\n[System] Interrupted.")
    finally:
        state.running = False
        receiver.stop()
        for t in threads:
            t.join(timeout=2.0)
        print("[System] Shutdown complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated sensor data (no ESP32 required)")
    args = parser.parse_args()
    run_realtime(simulate=args.simulate)