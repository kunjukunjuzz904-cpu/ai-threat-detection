"""
ThreatShield AI | 2026
test_training.py

Tests training pipelines for the autoencoder, random forest, and isolation forest models.
"""

import numpy as np
import pytest

from ml.train_autoencoder import train_autoencoder
from ml.train_dnn import train_neural_network

class TestAutoencoderTraining:

    @pytest.fixture
    def normal_data(self) -> np.ndarray:
        """
        Three-hundred samples of clipped 35-dimensional float32 data representing normal traffic.
        """
        rng = np.random.default_rng(42)
        return (rng.standard_normal(
            (300, 35)) * 0.3 + 0.5).astype(np.float32).clip(0, 1)

    def test_returns_model_and_threshold(self,
                                         normal_data: np.ndarray) -> None:
        """
        Training returns model, threshold, scaler, and history keys.
        """
        result = train_autoencoder(normal_data, epochs=5, batch_size=32)
        assert "model" in result
        assert "threshold" in result
        assert "scaler" in result
        assert "history" in result

    def test_threshold_is_positive(self, normal_data: np.ndarray) -> None:
        """
        Computed reconstruction error threshold is a positive value.
        """
        result = train_autoencoder(normal_data, epochs=5, batch_size=32)
        assert result["threshold"] > 0.0

    def test_history_has_train_loss(self, normal_data: np.ndarray) -> None:
        """
        Training history includes one train_loss entry per epoch.
        """
        result = train_autoencoder(normal_data, epochs=5, batch_size=32)
        assert "train_loss" in result["history"]
        assert len(result["history"]["train_loss"]) == 5

    def test_custom_percentile(self, normal_data: np.ndarray) -> None:
        """
        Higher percentile produces a higher or equal reconstruction threshold.
        """
        result_95 = train_autoencoder(normal_data,
                                      epochs=3,
                                      batch_size=32,
                                      percentile=95.0)
        result_99 = train_autoencoder(normal_data,
                                      epochs=3,
                                      batch_size=32,
                                      percentile=99.0)
        assert result_99["threshold"] >= result_95["threshold"]

    def test_model_is_in_eval_mode(self, normal_data: np.ndarray) -> None:
        """
        Returned model is in eval mode after training completes.
        """
        result = train_autoencoder(normal_data, epochs=3, batch_size=32)
        assert not result["model"].training

class TestDeepNeuralNetworkTraining:

    @pytest.fixture
    def labeled_data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((400, 35)).astype(np.float32)
        y = np.concatenate([
            np.zeros(300, dtype=np.int32),
            np.ones(100, dtype=np.int32)
        ])
        return X, y

    def test_returns_model_and_metrics(self, labeled_data):
        X, y = labeled_data
        result = train_neural_network(X, y, epochs=3)

        assert "model" in result
        assert "metrics" in result

    def test_metrics_have_required_keys(self, labeled_data):
        X, y = labeled_data
        result = train_neural_network(X, y, epochs=3)

        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "pr_auc",
        ):
            assert key in result["metrics"]

    def test_metric_values_in_range(self, labeled_data):
        X, y = labeled_data
        result = train_neural_network(X, y, epochs=3)

        for value in result["metrics"].values():
            assert 0.0 <= value <= 1.0

    def test_model_predicts(self, labeled_data):
        X, y = labeled_data
        result = train_neural_network(X, y, epochs=3)

        preds = result["model"].predict(X[:5])
        assert preds.shape[0] == 5
