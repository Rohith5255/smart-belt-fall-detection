"""alerts/alert_system.py — Fall alert handler."""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALERT_COOLDOWN_SECONDS


class AlertSystem:
    def __init__(self, cooldown=ALERT_COOLDOWN_SECONDS):
        self.cooldown    = cooldown
        self._last_alert = 0.0

    def _can_alert(self):
        now = time.time()
        if now - self._last_alert >= self.cooldown:
            self._last_alert = now
            return True
        return False

    def trigger(self, fused_proba, svm_proba, vit_proba, timestamp=None):
        if not self._can_alert():
            return
        ts = time.strftime("%H:%M:%S", time.localtime(timestamp or time.time()))
        print("\n" + "!" * 60)
        print(f"  FALL DETECTED  [{ts}]")
        print(f"  Fused : {fused_proba:.2%}")
        print(f"  SVM   : {svm_proba:.2%}")
        print(f"  ViT   : {vit_proba:.2%}")
        print("!" * 60 + "\n")
        self._play_beep()

    def _play_beep(self):
        try:
            import winsound
            winsound.Beep(1000, 500)
        except ImportError:
            print("\a", end="", flush=True)
