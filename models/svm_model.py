"""
models/svm_model.py — SVM classifier for sensor pipeline.
"""

import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SVM_MODEL_PATH


def build_svm_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        # C=0.5 (was 10.0): lower confidence → wider probability output range
        # so pre-fall spikes stand out more from normal-movement baseline
        ("svm", SVC(kernel="rbf", C=0.5, gamma="scale",
                    probability=True, class_weight="balanced", random_state=42)),
    ])


def save_svm(pipeline, path=SVM_MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[SVM] Saved → {path}")


def load_svm(path=SVM_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SVM model not found at {path}.\n"
            "Run: python training/train_svm.py")
    pipeline = joblib.load(path)
    print(f"[SVM] Loaded ← {path}")
    return pipeline


def svm_predict_proba(pipeline, X):
    return pipeline.predict_proba(X)[:, 1]

def svm_predict(pipeline, X):
    return pipeline.predict(X)
