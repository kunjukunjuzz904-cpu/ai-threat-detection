"""
ThreatShield AI | 2026
inference.py
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)

AE_FILENAME = "ae.onnx"
DNN_FILENAME = "dnn.onnx"
SCALER_FILENAME = "scaler.json"
THRESHOLD_FILENAME = "threshold.json"
SCALER_CLIP_LIMIT = 10.0


class InferenceEngine:
    """
    ONNX-based inference engine for the 2-model Deep learning ensemble.

    Loads autoencoder, random forest, and isolation forest ONNX sessions
    from a model directory. Returns None for predictions when models
    are not available.
    """

    def __init__(self, model_dir: str) -> None:
        self._ae_session: ort.InferenceSession | None = None
        self._dnn_session: ort.InferenceSession | None = None
        self._scaler_center: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self._threshold: float = 0.0
        self._loaded = False

        if ort is None:
            logger.warning("onnxruntime not installed")
            return

        model_path = Path(model_dir)
        ae_path = model_path / AE_FILENAME
        dnn_path = model_path / DNN_FILENAME
        scaler_path = model_path / SCALER_FILENAME
        threshold_path = model_path / THRESHOLD_FILENAME

        required = [ae_path, dnn_path, scaler_path, threshold_path]
        if not all(p.exists() for p in required):
            return

        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1

            self._ae_session = ort.InferenceSession(str(ae_path), opts)
            self._dnn_session = ort.InferenceSession(str(dnn_path), opts)

            scaler_data = json.loads(scaler_path.read_text())
            self._scaler_center = np.array(scaler_data["center"],
                                           dtype=np.float32)
            self._scaler_scale = np.array(scaler_data["scale"],
                                          dtype=np.float32)
            self._scaler_scale = np.where(
                self._scaler_scale == 0,
                1.0,
                self._scaler_scale,
            )

            threshold_data = json.loads(threshold_path.read_text())
            self._threshold = float(threshold_data["threshold"])

            self._loaded = True
            logger.info("Loaded Autoencoder + Deep Learning model from %s", model_dir)
        except Exception:
            logger.exception("Failed to load ONNX models from %s", model_dir)

    @property
    def is_loaded(self) -> bool:
        """
        Whether all 3 models are loaded and ready for inference
        """
        return self._loaded

    @property
    def threshold(self) -> float:
        """
        Autoencoder anomaly detection threshold
        """
        return self._threshold

    def predict(self, batch: np.ndarray) -> dict[str, list[float]] | None:
        """
        Run all 3 models on a batch of feature vectors.

        Returns per-model raw scores for ensemble fusion, or None
        if models are not loaded.
        """
        if not self._loaded:
            return None

        ae_input = self._scale_for_ae(batch)
        ae_reconstructed = self._ae_session.run(  # type: ignore[union-attr]
            None, {"features": ae_input})[0]
        ae_errors = np.mean((ae_input - ae_reconstructed)**2, axis=1)

        dnn_output = self._dnn_session.run(
            None, 
            {"features": batch}
        )[0].flatten()

        result = {
            "ae": ae_errors.tolist(),
            "dnn": dnn_output.tolist(),
        }
        return result  # type: ignore[union-attr]

    def _scale_for_ae(self, batch: np.ndarray) -> np.ndarray:
        """
        Apply RobustScaler transform for autoencoder input
        """
        if self._scaler_center is None or self._scaler_scale is None:
            return batch
        clean = np.nan_to_num(
            batch.astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        compressed = np.sign(clean) * np.log1p(np.abs(clean))
        scaled = (compressed - self._scaler_center) / self._scaler_scale
        return np.clip(scaled, -SCALER_CLIP_LIMIT, SCALER_CLIP_LIMIT).astype(np.float32)  # type: ignore[no-any-return]

    
