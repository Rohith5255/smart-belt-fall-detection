"""
iot/iot_receiver.py
Reads real-time sensor data from ESP32 belt via USB serial.

ESP32 sends CSV at ~200Hz:
    acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,pressure_left,pressure_right

UNITS — must match UP-Fall CompleteDataSet.csv (training data units):
    acc       : g         (standing upright -> ay ~ -1.0)
    gyroscope : deg/s
    pressure  : ADC 0-4095
"""

import serial
import numpy as np
import time
import threading
from collections import deque

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT,
                    WINDOW_SIZE, ESP32_DATA_FORMAT)


class ESP32Receiver:
    def __init__(self, port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT):
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self._ser     = None
        self._buffer  = deque(maxlen=WINDOW_SIZE)
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None
        self._error   = None
        self._offline = False   # True when belt is not connected

    @staticmethod
    def _find_esp32_port():
        """
        Auto-detects the ESP32 COM port by looking for common USB-serial
        bridge chips: CH340 (blue PCBs), CP2102/CP2104 (red/black PCBs).
        Returns the port string, or None if nothing found.
        """
        try:
            from serial.tools import list_ports
            candidates = []
            for p in list_ports.comports():
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                if any(kw in desc or kw in hwid for kw in
                       ("ch340", "cp210", "ftdi", "usb serial", "esp32",
                        "silicon labs", "wch")):
                    candidates.append(p.device)
            if candidates:
                print(f"[ESP32] Auto-detected serial ports: {candidates}")
                return candidates[0]
        except Exception:
            pass
        return None

    def connect(self):
        target_port = self.port
        try:
            self._ser = serial.Serial(port=target_port, baudrate=self.baudrate,
                                       timeout=self.timeout)
        except serial.SerialException:
            # Primary port failed — try auto-detect
            print(f"[ESP32] {target_port} not found — scanning for ESP32 ...")
            detected = self._find_esp32_port()
            if detected is None:
                self._offline = True
                print("[ESP32] ⚠  Belt not found — running in CAMERA-ONLY mode.")
                print("           SVM score will be 0. Pose detection still active.")
                print("           Plug in the belt and restart to enable full fusion.")
                return
            try:
                self._ser = serial.Serial(port=detected, baudrate=self.baudrate,
                                           timeout=self.timeout)
                self.port = detected   # update so logs show the real port
            except serial.SerialException as e2:
                self._offline = True
                print(f"[ESP32] ⚠  Auto-detected {detected} but failed: {e2}")
                print("[ESP32]    Running in CAMERA-ONLY mode.")
                return
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        print(f"[ESP32] Connected on {self.port} @ {self.baudrate} baud")

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._ser and self._ser.is_open:
            self._ser.close()
            print("[ESP32] Disconnected.")

    def start(self):
        if self._ser is None and not self._offline:
            self.connect()
        if self._offline:
            self._running = True
            return   # no read thread needed — is_ready() returns offline data
        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[ESP32] Receiver started.")

    def stop(self):
        self.disconnect()

    def _read_loop(self):
        """
        Reads CSV lines from ESP32 and converts raw MPU6050 units to
        physical units matching the UP-Fall training data:
            acc raw int16  / 16384.0  -> g
            gyr raw int16  / 131.0    -> deg/s
            pressure stays as ADC 0-4095
        """
        while self._running:
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parts = [float(v) for v in line.split(",")]
                if len(parts) < 6:
                    continue
                while len(parts) < len(ESP32_DATA_FORMAT):
                    parts.append(0.0)
                acc = [parts[0] / 16384.0, parts[1] / 16384.0, parts[2] / 16384.0]
                gyr = [parts[3] / 131.0,   parts[4] / 131.0,   parts[5] / 131.0]
                pressure = parts[6:8]
                values = acc + gyr + pressure
                with self._lock:
                    self._buffer.append(np.array(values, dtype=np.float32))
            except (ValueError, serial.SerialException):
                continue

    def is_ready(self):
        if self._offline:
            return True   # always ready; returns neutral data
        with self._lock:
            return len(self._buffer) >= WINDOW_SIZE

    def get_sensor_window(self):
        if self._offline:
            # Return neutral upright-standing data so SVM outputs ~0.0
            neutral_acc = np.tile([0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                  (WINDOW_SIZE, 1)).astype(np.float32)
            neutral_prs = np.zeros((WINDOW_SIZE, 2), dtype=np.float32)
            return neutral_acc, neutral_prs
        with self._lock:
            buf = np.array(list(self._buffer)[-WINDOW_SIZE:], dtype=np.float32)
        return buf[:, :6], buf[:, 6:]

    @property
    def is_offline(self):
        return self._offline

    def get_error(self):
        return self._error

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *_):
        self.stop()


class SimulatedESP32Receiver:
    """
    Generates synthetic sensor data for testing without hardware.

    All values are in physical units matching UP-Fall training data:
        acc  : g       (standing upright -> ay ~ -1.0 g)
        gyr  : deg/s
        pressure: ADC 0-4095

    Simulation timeline:
        0 – fall_after        : normal walking
        fall_after – +2.0s    : FALL EVENT (2 full seconds = 400 samples)
                                Covers >1 complete sliding window so the
                                SVM sees a majority-fall window.
        +2.0s onwards         : post-fall (lying on ground)
    """

    FALL_DURATION = 2.0   # seconds — must be > WINDOW_SIZE/200 = 1.0s

    def __init__(self, fall_after_sec=5.0):
        self._buffer     = deque(maxlen=WINDOW_SIZE)
        self._lock       = threading.Lock()
        self._running    = False
        self._fall_after = fall_after_sec
        self._start_time = None

    def start(self):
        self._running    = True
        self._start_time = time.time()
        self._thread     = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
        print("[SimESP32] Simulated receiver started.")
        print(f"[SimESP32] Normal walking for {self._fall_after}s, "
              f"then FALL for {self.FALL_DURATION}s, then post-fall.")

    def stop(self):
        self._running = False

    def _generate_loop(self):
        while self._running:
            elapsed  = time.time() - self._start_time
            fall_end = self._fall_after + self.FALL_DURATION

            if elapsed < self._fall_after:
                # ── Normal upright walking ───────────────────────
                acc = np.array([
                    np.random.normal(0.0,  0.05),
                    np.random.normal(-1.0, 0.08),   # gravity on Y
                    np.random.normal(0.0,  0.05),
                ], dtype=np.float32)
                gyr = np.random.normal(0.0, 2.0, 3).astype(np.float32)
                p   = np.array([
                    np.random.normal(400, 20),
                    np.random.normal(390, 20),
                ], dtype=np.float32)

            elif elapsed < fall_end:
                # ── Fall event (2 full seconds) ──────────────────
                # Strong acc spike + high gyro + pressure impact
                acc = np.array([
                    np.random.normal( 2.0, 0.5),
                    np.random.normal(-5.0, 1.0),    # downward slam
                    np.random.normal( 1.5, 0.5),
                ], dtype=np.float32)
                gyr = np.random.normal(0, 300.0, 3).astype(np.float32)
                p   = np.array([
                    np.random.normal(3200, 150),
                    np.random.normal( 800, 100),
                ], dtype=np.float32)

            else:
                # ── Post-fall: lying on ground ────────────────────
                acc = np.array([
                    np.random.normal(-1.0, 0.05),   # gravity now on X
                    np.random.normal( 0.0, 0.05),
                    np.random.normal( 0.0, 0.05),
                ], dtype=np.float32)
                gyr = np.random.normal(0.0, 1.0, 3).astype(np.float32)
                p   = np.array([
                    np.random.normal(50, 15),
                    np.random.normal(50, 15),
                ], dtype=np.float32)

            p = np.clip(p, 0, 4095)
            with self._lock:
                self._buffer.append(np.concatenate([acc, gyr, p]))
            time.sleep(1.0 / 200)

    def is_ready(self):
        with self._lock:
            return len(self._buffer) >= WINDOW_SIZE

    def get_sensor_window(self):
        with self._lock:
            buf = np.array(list(self._buffer)[-WINDOW_SIZE:], dtype=np.float32)
        return buf[:, :6], buf[:, 6:]

    def get_error(self):
        return None

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *_):
        self.stop()