"""
fusion/fusion_model.py — Combines SVM + ViT scores into final fall decision.
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


def weighted_fusion(svm_proba, pose_score,
                     svm_w=FUSION_SVM_WEIGHT, pose_w=FUSION_VIT_WEIGHT):
    """
    Only amplify SVM when it exceeds the baseline noise ceiling (~0.019).
    Normal walking keeps SVM below SVM_SPIKE_THRESHOLD so the amplifier
    doesn't fire; a real pre-fall spike breaks through and gets boosted.
    When out of camera frame (pose_score≈0), SVM alone carries the signal.
    """
    # Amplify only genuine spikes above baseline noise
    if svm_proba >= SVM_SPIKE_THRESHOLD:
        svm_amp = min(svm_proba * SVM_AMPLIFIER, 1.0)
    else:
        svm_amp = svm_proba   # no amplification for low-level noise

    # Out of frame: SVM alone — never carry stale fused score
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
        svm_amp = (min(svm_proba * SVM_AMPLIFIER, 1.0)
                   if svm_proba >= SVM_SPIKE_THRESHOLD else svm_proba)
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
