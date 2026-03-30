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
from config import FUSION_SVM_WEIGHT, FUSION_VIT_WEIGHT, FALL_THRESHOLD, FUSION_MODEL_PATH


def weighted_fusion(svm_proba, vit_proba,
                     svm_w=FUSION_SVM_WEIGHT, vit_w=FUSION_VIT_WEIGHT):
    return (svm_proba * svm_w + vit_proba * vit_w) / (svm_w + vit_w)


def decide(fused_proba, threshold=FALL_THRESHOLD):
    if fused_proba >= threshold:
        return 1, f"⚠️  FALL DETECTED  (confidence={fused_proba:.2%})"
    return 0, f"✅  No fall        (confidence={(1-fused_proba):.2%})"


def fuse(svm_proba, vit_proba, mlp_pipeline=None):
    if mlp_pipeline is not None:
        X = np.array([[svm_proba, vit_proba]])
        fused_proba = float(mlp_pipeline.predict_proba(X)[0, 1])
    else:
        fused_proba = weighted_fusion(svm_proba, vit_proba)
    label, message = decide(fused_proba)
    return fused_proba, label, message


def load_fusion_model(path=FUSION_MODEL_PATH):
    if not os.path.exists(path):
        return None
    pipeline = joblib.load(path)
    print(f"[Fusion] Loaded ← {path}")
    return pipeline
