"""
ThreatShield AI | 2026
test_inference.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.core.detection.inference import InferenceEngine
from ml.autoencoder import ThreatAutoencoder
from ml.export_onnx import export_autoencoder, export_dnn
from ml.scaler import FeatureScaler
from ml.train_classifiers import train_neural_network


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """
    Create a temp directory with AE + DNN ONNX models, scaler, and threshold.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 35)).astype(np.float32)
    y = np.concatenate([np.zeros(140, dtype=int), np.ones(60, dtype=int)])

    ae = ThreatAutoencoder(input_dim=35)
    export_autoencoder(ae, tmp_path / "ae.onnx")

    dnn = train_neural_network(X, y, epochs=1, batch_size=32)
    export_dnn(dnn["model"], tmp_path / "dnn.onnx")

    scaler = FeatureScaler()
    scaler.fit(X[:140])
    scaler.save_json(tmp_path / "scaler.json")

    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.05}))
    return tmp_path


class TestInferenceEngine:
    def test_loads_all_models(self, model_dir: Path) -> None:
        engine = InferenceEngine(model_dir=str(model_dir))
        assert engine.is_loaded

    def test_returns_none_when_no_models(self) -> None:
        engine = InferenceEngine(model_dir="/nonexistent/path")
        assert not engine.is_loaded

    def test_predict_returns_none_when_not_loaded(self) -> None:
        engine = InferenceEngine(model_dir="/nonexistent/path")
        result = engine.predict(np.zeros((1, 35), dtype=np.float32))
        assert result is None

    def test_predict_returns_scores(self, model_dir: Path) -> None:
        engine = InferenceEngine(model_dir=str(model_dir))
        rng = np.random.default_rng(99)
        result = engine.predict(rng.standard_normal((4, 35)).astype(np.float32))
        assert result is not None
        assert "ae" in result
        assert "dnn" in result

    def test_predict_ae_scores_are_positive(self, model_dir: Path) -> None:
        engine = InferenceEngine(model_dir=str(model_dir))
        rng = np.random.default_rng(99)
        result = engine.predict(rng.standard_normal((4, 35)).astype(np.float32))
        assert result is not None
        assert all(s >= 0.0 for s in result["ae"])

    def test_predict_dnn_probabilities_in_range(self, model_dir: Path) -> None:
        engine = InferenceEngine(model_dir=str(model_dir))
        rng = np.random.default_rng(99)
        result = engine.predict(rng.standard_normal((4, 35)).astype(np.float32))
        assert result is not None
        assert all(0.0 <= p <= 1.0 for p in result["dnn"])

    def test_predict_single_sample(self, model_dir: Path) -> None:
        engine = InferenceEngine(model_dir=str(model_dir))
        rng = np.random.default_rng(99)
        result = engine.predict(rng.standard_normal((1, 35)).astype(np.float32))
        assert result is not None
        assert len(result["ae"]) == 1
        assert len(result["dnn"]) == 1

    def test_threshold_loaded(self, model_dir: Path) -> None:
        engine = InferenceEngine(model_dir=str(model_dir))
        assert engine.threshold == 0.05

    def test_partial_models_not_loaded(self, tmp_path: Path) -> None:
        ae = ThreatAutoencoder(input_dim=35)
        export_autoencoder(ae, tmp_path / "ae.onnx")
        engine = InferenceEngine(model_dir=str(tmp_path))
        assert not engine.is_loaded
