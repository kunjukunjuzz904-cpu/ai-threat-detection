"""
ThreatShield AI | 2026
test_training_e2e.py
"""

from pathlib import Path

from app.core.detection.ensemble import blend_scores, fuse_scores, normalize_ae_score
from app.core.detection.inference import InferenceEngine
from ml.orchestrator import TrainingOrchestrator
from ml.synthetic import generate_mixed_dataset

N_NORMAL = 500
N_ATTACK = 200
N_FEATURES = 35
ENSEMBLE_WEIGHTS = {"ae": 0.45, "dnn": 0.55}


class TestTrainingE2E:
    """
    End-to-end training integration test.
    """

    def test_full_training_produces_loadable_models(self, tmp_path: Path) -> None:
        X, y = generate_mixed_dataset(N_NORMAL, N_ATTACK)
        assert X.shape == (N_NORMAL + N_ATTACK, N_FEATURES)

        model_dir = tmp_path / "models"
        orch = TrainingOrchestrator(output_dir=model_dir, epochs=3)
        result = orch.run(X, y)

        expected_files = ["ae.onnx", "dnn.onnx", "scaler.json", "threshold.json"]
        for filename in expected_files:
            assert (model_dir / filename).exists(), f"Missing {filename}"

        engine = InferenceEngine(str(model_dir))
        assert engine.is_loaded

        predictions = engine.predict(X[:5])
        assert predictions is not None
        assert "ae" in predictions
        assert "dnn" in predictions
        assert len(predictions["ae"]) == 5
        assert len(predictions["dnn"]) == 5

        threshold = engine.threshold
        for i in range(5):
            scores = {
                "ae": normalize_ae_score(predictions["ae"][i], threshold),
                "dnn": predictions["dnn"][i],
            }
            fused = fuse_scores(scores, ENSEMBLE_WEIGHTS)
            assert 0.0 <= fused <= 1.0

            blended = blend_scores(fused, 0.0)
            assert 0.0 <= blended <= 1.0

        assert isinstance(result.passed_gates, bool)
