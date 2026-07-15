"""
ThreatShield AI | 2026
test_export_onnx.py
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.autoencoder import ThreatAutoencoder
from ml.export_onnx import export_autoencoder, export_dnn
from ml.train_classifiers import ThreatDNN


class TestAutoencoderExport:
    def test_creates_onnx_file(self, tmp_path: Path) -> None:
        model = ThreatAutoencoder(input_dim=35)
        path = export_autoencoder(model, tmp_path / "ae.onnx")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_onnx_output_matches_pytorch(self, tmp_path: Path) -> None:
        torch.manual_seed(42)
        model = ThreatAutoencoder(input_dim=35)
        model.eval()
        onnx_path = export_autoencoder(model, tmp_path / "ae.onnx")

        rng = np.random.default_rng(42)
        x = rng.standard_normal((8, 35)).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.from_numpy(x)).numpy()

        session = ort.InferenceSession(str(onnx_path))
        ort_out = session.run(None, {"features": x})[0]

        np.testing.assert_allclose(pt_out, ort_out, atol=1e-5)

    def test_dynamic_batch_dimension(self, tmp_path: Path) -> None:
        model = ThreatAutoencoder(input_dim=35)
        model.eval()
        onnx_path = export_autoencoder(model, tmp_path / "ae.onnx")
        session = ort.InferenceSession(str(onnx_path))

        for batch_size in (1, 16, 64):
            rng = np.random.default_rng(batch_size)
            x = rng.standard_normal((batch_size, 35)).astype(np.float32)
            out = session.run(None, {"features": x})[0]
            assert out.shape == (batch_size, 35)


class TestDeepLearningExport:
    def test_creates_onnx_file(self, tmp_path: Path) -> None:
        model = ThreatDNN(input_dim=35)
        path = export_dnn(model, tmp_path / "dnn.onnx")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_onnx_prediction(self, tmp_path: Path) -> None:
        model = ThreatDNN(input_dim=35)
        model.eval()
        path = export_dnn(model, tmp_path / "dnn.onnx")
        session = ort.InferenceSession(str(path))

        x = np.random.randn(5, 35).astype(np.float32)
        result = session.run(None, {"features": x})

        assert len(result) == 1
        assert result[0].shape == (5, 1)
        assert np.all((0.0 <= result[0]) & (result[0] <= 1.0))
