"""
live_check.py — Belt sensor diagnostic while wearing the belt.
Run: python live_check.py

Prints a live dashboard every 0.5s showing:
  - Accelerometer in g (should show ~-1.0g on Y when upright)
  - Gyroscope in deg/s (spikes when you move/walk)
  - Pressure left / right ADC (0-4095)
  - Resultant acceleration magnitude (should be ~1.0g at rest)
  - Movement status

Ctrl+C to stop — prints a summary of min/max/mean for each channel.
"""

import serial
import time
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import SERIAL_PORT, SERIAL_BAUDRATE
except ImportError:
    SERIAL_PORT     = "COM5"
    SERIAL_BAUDRATE = 115200


class Stats:
    def __init__(self, name):
        self.name = name
        self.vals = []

    def add(self, v):
        self.vals.append(v)

    def summary(self):
        if not self.vals:
            return f"{self.name}: no data"
        return (f"{self.name:12s}  min={min(self.vals):+8.3f}  "
                f"max={max(self.vals):+8.3f}  "
                f"mean={sum(self.vals)/len(self.vals):+8.3f}  "
                f"({len(self.vals)} samples)")


def main():
    print(f"\n[LiveCheck] Opening {SERIAL_PORT} @ {SERIAL_BAUDRATE} baud ...")
    try:
        s = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=2)
    except Exception as e:
        print(f"[LiveCheck] ERROR: {e}")
        print("  Make sure the belt is plugged in and COM5 is correct.")
        return

    time.sleep(2)
    s.reset_input_buffer()
    print("[LiveCheck] Belt connected. Wear it and move around.")
    print("            Ctrl+C to stop and see summary.\n")

    print(f"{'#':>6}  {'ax(g)':>7}  {'ay(g)':>7}  {'az(g)':>7}  "
          f"{'gx':>7}  {'gy':>7}  {'gz':>7}  "
          f"{'pL':>5}  {'pR':>5}  {'|a|':>6}  status")
    print("-" * 95)

    stats  = {k: Stats(k) for k in
              ["ax", "ay", "az", "gx", "gy", "gz", "pL", "pR", "mag"]}
    count      = 0
    last_print = time.time()
    raw_buf    = []

    try:
        while True:
            line = s.readline().decode("utf-8", errors="ignore").strip()
            if not line or "," not in line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                v = [float(x) for x in parts]
            except ValueError:
                continue
            while len(v) < 8:
                v.append(0.0)

            ax  = v[0] / 16384.0
            ay  = v[1] / 16384.0
            az  = v[2] / 16384.0
            gx  = v[3] / 131.0
            gy  = v[4] / 131.0
            gz  = v[5] / 131.0
            pL  = v[6]
            pR  = v[7]
            mag = math.sqrt(ax**2 + ay**2 + az**2)

            for k, val in zip(
                    ["ax", "ay", "az", "gx", "gy", "gz", "pL", "pR", "mag"],
                    [ax, ay, az, gx, gy, gz, pL, pR, mag]):
                stats[k].add(val)

            raw_buf.append((ax, ay, az, gx, gy, gz, pL, pR, mag))
            count += 1

            now = time.time()
            if now - last_print >= 0.5 and raw_buf:
                n   = len(raw_buf)
                avg = [sum(col) / n for col in zip(*raw_buf)]
                ax_, ay_, az_, gx_, gy_, gz_, pL_, pR_, mag_ = avg

                gyr_mag = math.sqrt(gx_**2 + gy_**2 + gz_**2)
                if mag_ < 0.5:
                    status = "FREE-FALL?"
                elif mag_ > 2.5:
                    status = "IMPACT!"
                elif gyr_mag > 50:
                    status = "ROTATING"
                elif gyr_mag > 15 or abs(1.0 - mag_) > 0.15:
                    status = "WALKING"
                else:
                    status = "STILL"

                print(f"{count:>6}  {ax_:>+7.3f}  {ay_:>+7.3f}  {az_:>+7.3f}  "
                      f"{gx_:>+7.1f}  {gy_:>+7.1f}  {gz_:>+7.1f}  "
                      f"{pL_:>5.0f}  {pR_:>5.0f}  {mag_:>6.3f}  {status}")
                raw_buf    = []
                last_print = now

    except KeyboardInterrupt:
        print(f"\n[LiveCheck] Stopped after {count} raw samples.\n")
    finally:
        s.close()

    print("=" * 65)
    print("SUMMARY  (units: g for acc, deg/s for gyro, ADC 0-4095 for pressure)")
    print("=" * 65)
    for k in ["ax", "ay", "az", "gx", "gy", "gz", "pL", "pR", "mag"]:
        print(stats[k].summary())

    ay_mean  = sum(stats["ay"].vals)  / max(len(stats["ay"].vals),  1)
    mag_mean = sum(stats["mag"].vals) / max(len(stats["mag"].vals), 1)
    pL_mean  = sum(stats["pL"].vals)  / max(len(stats["pL"].vals),  1)
    pR_mean  = sum(stats["pR"].vals)  / max(len(stats["pR"].vals),  1)

    print()
    print("HEALTH CHECK:")
    print(f"  acc_y mean {ay_mean:+.3f}g  ->  "
          f"{'OK (gravity on Y)' if -1.2 < ay_mean < -0.7 else 'WARNING: unexpected orientation'}")
    print(f"  |a|   mean {mag_mean:.3f}g   ->  "
          f"{'OK (~1g at rest)' if 0.85 < mag_mean < 1.15 else 'WARNING: check mount/calibration'}")
    print(f"  pL    mean {pL_mean:.1f}      ->  "
          f"{'OK' if pL_mean > 50 else 'WARNING: left sensor dead (check wiring)'}")
    print(f"  pR    mean {pR_mean:.1f}      ->  "
          f"{'OK' if pR_mean > 50 else 'WARNING: right sensor not reading'}")


if __name__ == "__main__":
    main()
