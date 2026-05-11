"""
training/train_svm.py — Trains SVM on UP-Fall Detection Dataset.

Run from CAPSTONE folder:
    python training/train_svm.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

from preprocessing.sensor_preprocessing import load_upfall_dataset
from models.svm_model import build_svm_pipeline, save_svm


def main():
    print("=" * 60)
    print("  SVM Training  |  UP-Fall Detection Dataset")
    print("=" * 60)

    X, y = load_upfall_dataset()
    X = np.nan_to_num(X, nan=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=7)

    print(f"\n[Train] Before SMOTE: fall={int(np.sum(y_train==1))}  "
          f"no-fall={int(np.sum(y_train==0))}")
    sm = SMOTE(random_state=7)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    print(f"[Train] After  SMOTE: fall={int(np.sum(y_train_bal==1))}  "
          f"no-fall={int(np.sum(y_train_bal==0))}")

    pipeline = build_svm_pipeline()
    print("\n[Train] Fitting SVM …")
    pipeline.fit(X_train_bal, y_train_bal)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["No-Fall","Fall"]))
    print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
    print(f"5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No-Fall","Fall"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("SVM Confusion Matrix")
    plt.tight_layout()
    os.makedirs("saved_models", exist_ok=True)
    fig.savefig("saved_models/svm_confusion_matrix.png", dpi=150)
    print("[Train] Confusion matrix → saved_models/svm_confusion_matrix.png")

    save_svm(pipeline)
    print("\n[Train] SVM training complete! ✅")


if __name__ == "__main__":
    main()
