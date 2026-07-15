"""
ThreatShield AI | 2026
test_validation.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ml.autoencoder import ThreatAutoencoder
from ml.export_onnx import export_autoencoder, export_dnn
from ml.scaler import FeatureScaler
from ml.train_dnn import ThreatDNN
from ml.validation import ValidationResult, validate_ensemble


@pytest.fixture
def trained_model_dir(tmp_path: Path) -> Path:
    """
    Create a temp directory with trained Deep Learning ONNX models.
    """
    rng = np.random.default_rng(42)
    X_normal = rng.standard_normal((200, 35)).astype(np.float32)

    ae = ThreatAutoencoder(input_dim=35)
    export_autoencoder(ae, tmp_path / "ae.onnx")

    dnn = ThreatDNN(input_dim=35)
    export_dnn(dnn, tmp_path / "dnn.onnx")

    scaler = FeatureScaler()
    scaler.fit(X_normal)
    scaler.save_json(tmp_path / "scaler.json")

    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.05}))
    return tmp_path


@pytest.fixture
def separable_test_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Test data where normal and attack clusters are well-separated.
    """
    rng = np.random.default_rng(99)
    X_normal = rng.standard_normal((50, 35)).astype(np.float32)
    X_attack = (rng.standard_normal((30, 35)).astype(np.float32) + 3.0)
    X = np.vstack([X_normal, X_attack])
    y = np.array([0] * 50 + [1] * 30, dtype=np.int32)
    return X, y


class TestValidateDeepEnsemble:
    """
    Test ensemble validation with metric gates.
    """

    def test_returns_validation_result(
        self,
        trained_model_dir: Path,
        separable_test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X_test, y_test = separable_test_data
        result = validate_ensemble(trained_model_dir, X_test, y_test)
        assert isinstance(result, ValidationResult)

    def test_result_has_all_metrics(
        self,
        trained_model_dir: Path,
        separable_test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X_test, y_test = separable_test_data
        result = validate_ensemble(trained_model_dir, X_test, y_test)

        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0
        assert 0.0 <= result.pr_auc <= 1.0
        assert 0.0 <= result.roc_auc <= 1.0

    def test_confusion_matrix_shape(
        self,
        trained_model_dir: Path,
        separable_test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X_test, y_test = separable_test_data
        result = validate_ensemble(trained_model_dir, X_test, y_test)

        assert len(result.confusion_matrix) == 2
        assert len(result.confusion_matrix[0]) == 2
        assert len(result.confusion_matrix[1]) == 2

    def test_gate_details_contains_both_gates(
        self,
        trained_model_dir: Path,
        separable_test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X_test, y_test = separable_test_data
        result = validate_ensemble(trained_model_dir, X_test, y_test)

        assert "pr_auc" in result.gate_details
        assert "f1" in result.gate_details

    def test_gates_pass_with_low_thresholds(
        self,
        trained_model_dir: Path,
        separable_test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X_test, y_test = separable_test_data
        result = validate_ensemble(
            trained_model_dir,
            X_test,
            y_test,
            pr_auc_gate=0.01,
            f1_gate=0.01,
        )
        assert result.passed_gates is True

    def test_gates_fail_with_high_thresholds(
        self,
        trained_model_dir: Path,
        separable_test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X_test, y_test = separable_test_data
        result = validate_ensemble(
            trained_model_dir,
            X_test,
            y_test,
            pr_auc_gate=1.0,
            f1_gate=1.0,
        )
        assert result.passed_gates is False

    def test_custom_ensemble_weights(
        self,
        trained_model_dir: Path,
        separable_test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X_test, y_test = separable_test_data
        result = validate_ensemble(
            trained_model_dir,
            X_test,
            y_test,
            ensemble_weights={"ae": 0.5, "dnn": 0.5},
        )
        assert isinstance(result, ValidationResult)
