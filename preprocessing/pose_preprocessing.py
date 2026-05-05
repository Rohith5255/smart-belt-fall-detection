"""
preprocessing/pose_preprocessing.py  — Phase 3
Replaces the ViT vision model with MediaPipe Pose estimation.

Why MediaPipe instead of ViT for PREDICTION:
    - ViT classifies "does this frame look like a fall?" (detection)
    - MediaPipe extracts body joint angles in real-time (no large model file)
    - Trunk lean, knee asymmetry, head drop are proven pre-fall predictors
    - Runs at ~30 fps on CPU — no GPU needed

Fall risk features extracted per frame:
    trunk_lean_deg   : angle of torso from vertical (>15 deg = at risk)
    knee_asymmetry   : difference in knee-bend between left and right
    head_drop_rate   : downward velocity of nose landmark (falls forward)
    shoulder_roll    : shoulder tilt (lateral falls)
    pose_confidence  : 0 or 1 (whether pose was detected)
"""

import cv2
import numpy as np
from collections import deque


def _try_import_mediapipe_pose():
    """Returns (mp_pose_module, mp_drawing_module) or (None, None)."""
    try:
        import mediapipe as mp
        # Access via attribute chain — works in mediapipe 0.10.x
        pose_mod = mp.solutions.pose
        draw_mod = mp.solutions.drawing_utils
        return pose_mod, draw_mod
    except Exception:
        return None, None


class PoseEstimator:
    """
    Real-time pose estimator wrapping MediaPipe Pose.
    Falls back gracefully (score=0) if MediaPipe is not installed.
    """

    def __init__(self, history_len=10):
        mp_pose_mod, mp_draw_mod = _try_import_mediapipe_pose()
        self._available  = mp_pose_mod is not None
        self._mp_draw    = mp_draw_mod
        if self._available:
            self._mp_pose = mp_pose_mod
            # static_image_mode=True: each frame processed independently —
            # avoids the "timestamp mismatch" SIGFAULT that occurs in streaming
            # mode when the OS scheduler delivers camera frames with non-strictly-
            # increasing internal timestamps (MediaPipe graph crash, exit 139).
            # We lose temporal landmark smoothing but our own head_y_hist buffer
            # compensates for that.
            self._pose    = mp_pose_mod.Pose(
                static_image_mode=True,
                model_complexity=1,
                smooth_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("[Pose] MediaPipe Pose initialised (static-image mode, no timestamp issues).")
        else:
            print("[Pose] WARNING: mediapipe not installed — pose score = 0.0")

        # Head-position history for drop-rate calculation
        self._head_y_hist    = deque(maxlen=history_len)
        # Cache last landmarks so draw_skeleton never re-calls process()
        self._last_landmarks = None

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame_bgr):
        """
        Args:
            frame_bgr : OpenCV BGR frame
        Returns:
            dict with pose features, and 'pose_score' (0-1 fall risk)
        """
        if not self._available:
            return self._null_result()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            results = self._pose.process(rgb)
        except Exception as exc:
            # MediaPipe internal graph errors (e.g. timestamp edge-cases) should
            # never crash the prediction pipeline — just return a null result.
            print(f"[Pose] process() error (skipping frame): {exc}")
            self._head_y_hist.append(None)
            self._last_landmarks = None
            return self._null_result()

        # Cache landmarks for draw_skeleton — avoids calling process() twice
        self._last_landmarks = results.pose_landmarks

        if not results.pose_landmarks:
            self._head_y_hist.append(None)
            return self._null_result()

        lm = results.pose_landmarks.landmark
        return self._extract(lm)

    def draw_skeleton(self, frame_bgr, features):
        """Overlay skeleton using cached landmarks — never re-runs inference."""
        if (not self._available
                or not features.get("pose_detected")
                or self._last_landmarks is None
                or self._mp_draw is None):
            return frame_bgr
        self._mp_draw.draw_landmarks(
            frame_bgr, self._last_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            self._mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2,
                                      circle_radius=2),
            self._mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2),
        )
        return frame_bgr

    def close(self):
        if self._available:
            self._pose.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract(self, lm):
        h_idx  = 0   # nose
        ls_idx = 11  # left shoulder
        rs_idx = 12  # right shoulder
        lh_idx = 23  # left hip
        rh_idx = 24  # right hip
        lk_idx = 25  # left knee
        rk_idx = 26  # right knee
        la_idx = 27  # left ankle
        ra_idx = 28  # right ankle

        # ── Trunk lean (angle from vertical) ──────────────────────────
        s_mid_x = (lm[ls_idx].x + lm[rs_idx].x) / 2
        s_mid_y = (lm[ls_idx].y + lm[rs_idx].y) / 2
        h_mid_x = (lm[lh_idx].x + lm[rh_idx].x) / 2
        h_mid_y = (lm[lh_idx].y + lm[rh_idx].y) / 2
        dx      = s_mid_x - h_mid_x                 # +ve = forward, -ve = backward
        dy      = h_mid_y - s_mid_y + 1e-6          # positive downward in image
        # Signed angle: + = forward lean, - = backward lean
        trunk_lean = float(np.degrees(np.arctan2(dx, abs(dy))))

        # ── Shoulder roll (lateral tilt) ──────────────────────────────
        shoulder_roll = float(abs(lm[ls_idx].y - lm[rs_idx].y) * 100)

        # ── Knee asymmetry ────────────────────────────────────────────
        left_knee_bend  = abs(lm[lk_idx].y - lm[lh_idx].y)
        right_knee_bend = abs(lm[rk_idx].y - lm[rh_idx].y)
        knee_asymmetry  = float(abs(left_knee_bend - right_knee_bend) * 100)

        # ── Head drop rate ────────────────────────────────────────────
        head_y = float(lm[h_idx].y)
        self._head_y_hist.append(head_y)
        valid_head = [v for v in self._head_y_hist if v is not None]
        if len(valid_head) >= 3:
            # Positive slope = head moving down = falling forward
            slope = float(np.polyfit(
                np.arange(len(valid_head)), valid_head, 1)[0])
            head_drop_rate = max(slope, 0.0)
        else:
            head_drop_rate = 0.0

        # ── Compose risk score ────────────────────────────────────────
        score = 0.0
        # Trunk lean — lowered thresholds to catch 18-30° pre-fall lean
        lean_abs = abs(trunk_lean)
        if lean_abs > 30:
            score += 0.50
        elif lean_abs > 18:
            score += 0.30
        elif lean_abs > 10:
            score += 0.12
        # Shoulder roll
        score += min(shoulder_roll * 0.02, 0.20)
        # Knee asymmetry
        score += min(knee_asymmetry * 0.03, 0.20)
        # Head dropping — boosted multiplier to catch subtle drops sooner
        score += min(head_drop_rate * 7.0, 0.25)

        pose_score = float(min(score, 1.0))

        return {
            "pose_detected"  : 1,
            "trunk_lean_deg" : round(trunk_lean, 2),
            "shoulder_roll"  : round(shoulder_roll, 3),
            "knee_asymmetry" : round(knee_asymmetry, 3),
            "head_drop_rate" : round(head_drop_rate, 4),
            "pose_score"     : round(pose_score, 4),
        }

    @staticmethod
    def _null_result():
        return {
            "pose_detected"  : 0,
            "trunk_lean_deg" : 0.0,
            "shoulder_roll"  : 0.0,
            "knee_asymmetry" : 0.0,
            "head_drop_rate" : 0.0,
            "pose_score"     : 0.0,
        }
