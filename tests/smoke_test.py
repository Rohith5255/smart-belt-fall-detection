"""
tests/smoke_test.py — Smoke tests for core pipeline components.

Run: pytest tests/smoke_test.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.svm_model import load_svm
from preprocessing.sensor_preprocessing import extract_realtime_features
from fusion.fusion_model import fuse
from inference.realtime_inference import RiskTrend
from config import WINDOW_SIZE


class TestSVMModel:
    """Test 1: Load SVM model and check predict_proba output shape."""

    def test_svm_predict_proba_shape(self):
        """Assert that SVM predict_proba returns shape (1, 2) for binary classification."""
        print("\n" + "="*60)
        print("TEST 1: SVM Model — predict_proba shape (1, 2)")
        print("="*60)

        try:
            pipeline = load_svm()

            # Create synthetic feature vector matching SVM training input
            # (117 features from sensor preprocessing)
            X = np.random.randn(1, 117).astype(np.float32)

            # Get raw predict_proba output (shape should be (1, 2) for binary SVM)
            proba = pipeline.predict_proba(X)

            assert proba.shape == (1, 2), \
                f"Expected shape (1, 2), got {proba.shape}"

            # Verify probabilities sum to 1.0
            assert np.isclose(proba.sum(axis=1)[0], 1.0), \
                f"Probabilities don't sum to 1.0: {proba.sum(axis=1)[0]}"

            # Verify values in [0, 1]
            assert np.all(proba >= 0) and np.all(proba <= 1), \
                f"Probabilities outside [0, 1]: {proba}"

            print(f"✓ SVM loaded successfully")
            print(f"✓ predict_proba shape: {proba.shape}")
            print(f"✓ Probabilities: {proba[0]}")
            print("PASS\n")

        except Exception as e:
            print(f"✗ Error: {e}")
            print("FAIL\n")
            raise


class TestFeatureExtraction:
    """Test 2: Extract realtime features from synthetic sensor data."""

    def test_extract_realtime_features_shape(self):
        """Assert that extract_realtime_features returns shape (1, 117)."""
        print("="*60)
        print("TEST 2: Feature Extraction — output shape (1, 117)")
        print("="*60)

        try:
            # Create synthetic sensor window: 200 samples x 6 channels
            # (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
            sensor_window = np.random.randn(200, 6).astype(np.float32)

            # Create synthetic pressure window: 200 samples x 2 channels
            # (pressure_left, pressure_right)
            pressure_window = np.random.randint(
                100, 3000, size=(200, 2)
            ).astype(np.float32)

            # Extract features
            features = extract_realtime_features(sensor_window, pressure_window)

            assert features.shape == (1, 117), \
                f"Expected shape (1, 117), got {features.shape}"

            # Verify all features are finite (no NaN/inf)
            assert np.all(np.isfinite(features)), \
                f"Features contain NaN or inf values"

            print(f"✓ Sensor window shape: {sensor_window.shape}")
            print(f"✓ Pressure window shape: {pressure_window.shape}")
            print(f"✓ Features extracted shape: {features.shape}")
            print(f"✓ Feature mean: {features.mean():.4f}")
            print(f"✓ Feature std: {features.std():.4f}")
            print("PASS\n")

        except Exception as e:
            print(f"✗ Error: {e}")
            print("FAIL\n")
            raise


class TestFusion:
    """Test 3: Fusion function returns float in [0, 1]."""

    def test_fuse_returns_valid_probability(self):
        """Assert that fuse() returns floats in [0, 1] for various inputs."""
        print("="*60)
        print("TEST 3: Fusion — output in [0, 1]")
        print("="*60)

        try:
            test_cases = [
                (0.01, 0.5, "low SVM, medium pose"),
                (0.5, 0.0, "medium SVM, no pose"),
                (0.0, 0.0, "no signal"),
                (1.0, 1.0, "maximum signals"),
                (0.1, 0.3, "moderate both"),
            ]

            all_passed = True
            for svm_p, pose_p, label in test_cases:
                fused_proba, pred_label, message = fuse(svm_p, pose_p)

                # Verify output type
                assert isinstance(fused_proba, (float, np.floating)), \
                    f"fused_proba is not float: {type(fused_proba)}"

                # Verify in valid range
                assert 0.0 <= fused_proba <= 1.0, \
                    f"fused_proba outside [0, 1]: {fused_proba}"

                # Verify label is 0 or 1
                assert pred_label in [0, 1], \
                    f"pred_label not in {{0, 1}}: {pred_label}"

                print(f"  ✓ fuse({svm_p}, {pose_p}) = {fused_proba:.4f} "
                      f"[{label}]")

            print("PASS\n")

        except Exception as e:
            print(f"✗ Error: {e}")
            print("FAIL\n")
            raise


class TestRiskTrend:
    """Test 4: RiskTrend detects rising slope."""

    def test_risk_trend_slope_detection(self):
        """Assert that RiskTrend detects positive slope with increasing scores."""
        print("="*60)
        print("TEST 4: RiskTrend — slope detection")
        print("="*60)

        try:
            risk_trend = RiskTrend(size=8)

            # Add 8 increasing scores to create a clear positive trend
            scores = np.linspace(0.1, 0.7, 8)

            print(f"✓ Adding {len(scores)} increasing scores:")
            for i, score in enumerate(scores, 1):
                risk_trend.update(float(score))
                print(f"  Score {i}: {score:.4f}")

            # Get slope
            slope = risk_trend.slope
            mean_score = risk_trend.mean

            print(f"\n✓ Buffer size: {len(risk_trend._buf)}")
            print(f"✓ Mean score: {mean_score:.4f}")
            print(f"✓ Slope: {slope:.6f}")

            # Verify slope is positive (increasing trend)
            assert slope > 0, \
                f"Expected positive slope, got {slope}"

            # Verify slope is reasonable (not too large)
            assert slope < 1.0, \
                f"Slope too large (unrealistic): {slope}"

            # Verify mean is within expected range
            assert 0.0 <= mean_score <= 1.0, \
                f"Mean score outside [0, 1]: {mean_score}"

            print("\nPASS\n")

        except Exception as e:
            print(f"✗ Error: {e}")
            print("FAIL\n")
            raise


# ── Run all tests with clear PASS/FAIL output ─────────────────────────────
if __name__ == "__main__":
    print("\n")
    print("█" * 60)
    print("  SMOKE TEST SUITE — Fall Prediction Pipeline")
    print("█" * 60)

    # Run tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
