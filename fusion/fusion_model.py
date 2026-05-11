"""
fusion/fusion_model.py — Combines SVM + Pose scores into final fall decision.
"""

import os
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (FUSION_SVM_WEIGHT, FUSION_VIT_WEIGHT, FALL_THRESHOLD,
                    SVM_AMPLIFIER, SVM_SPIKE_THRESHOLD, FUSION_MODEL_PATH)

# Tracks the last 3 pose readings so we can detect when MediaPipe briefly
# re-acquires tracking after dropouts. A single high pose frame right after
# two zeros is almost always a noisy landmark grab (forearm/chair mistaken
# for torso), NOT a real fall — so we ignore it.
_pose_history = []


def _amplify_svm(svm_proba):
    """Spike-gated SVM amplifier — only boosts genuine pre-fall spikes."""
    if svm_proba >= SVM_SPIKE_THRESHOLD:
        return min(svm_proba * SVM_AMPLIFIER, 1.0)
    return svm_proba


def weighted_fusion(svm_proba, pose_score,
                    svm_w=FUSION_SVM_WEIGHT, pose_w=FUSION_VIT_WEIGHT):
    """
    Only amplify SVM when it exceeds the baseline noise ceiling (~0.019).
    Normal walking keeps SVM below SVM_SPIKE_THRESHOLD so the amplifier
    doesn't fire; a real pre-fall spike breaks through and gets boosted.
    When out of camera frame (pose_score≈0), SVM alone carries the signal.

    Pose-dropout guard: if the previous 2 pose readings were 0 and this
    frame suddenly spikes above 0.3, treat it as an untrusted single frame
    (MediaPipe re-acquiring on noise) and fall back to SVM-only.
    """
    # Always compute amplified SVM first — needed by every code path below
    svm_amp = _amplify_svm(svm_proba)

    # ── Pose-dropout guard ───────────────────────────────────────────
    # Track last 3 pose readings
    _pose_history.append(pose_score)
    if len(_pose_history) > 3:
        _pose_history.pop(0)

    # If pose just came back from a 2-frame dropout with a big spike,
    # distrust this frame — use SVM only.
    if (len(_pose_history) == 3
            and _pose_history[0] == 0.0
            and _pose_history[1] == 0.0
            and pose_score > 0.3):
        return svm_amp

    # ── Normal fusion paths ──────────────────────────────────────────
    # Out of frame: SVM alone — never carry a stale fused score
    if pose_score < 0.05:
        return svm_amp

    # Both signals present
    return svm_w * svm_amp + pose_w * pose_score


def decide(fused_proba, threshold=FALL_THRESHOLD):
    if fused_proba >= threshold:
        return 1, f"⚠️  FALL DETECTED  (confidence={fused_proba:.2%})"
    return 0, f"✅  No fall        (confidence={(1-fused_proba):.2%})"


def fuse(svm_proba, pose_score, mlp_pipeline=None):
    """
    Computes fused_proba EVERY call from fresh inputs — never carries
    stale state. Fixes the stuck-fused-score bug where pose dropping to
    0 left fused frozen at the previous value.
    """
    if mlp_pipeline is not None:
        # MLP path: apply same spike-gated amplification
        svm_amp = _amplify_svm(svm_proba)
        X = np.array([[svm_amp, pose_score]])
        fused_proba = float(mlp_pipeline.predict_proba(X)[0, 1])
    else:
        fused_proba = weighted_fusion(svm_proba, pose_score)
    label, message = decide(fused_proba)
    return fused_proba, label, message


def load_fusion_model(path=FUSION_MODEL_PATH):
    if not os.path.exists(path):
        return None
    pipeline = joblib.load(path)
    print(f"[Fusion] Loaded ← {path}")
    return pipeline
