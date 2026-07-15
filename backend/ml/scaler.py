"""
ThreatShield AI | 2026
scaler.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler

CLIP_LIMIT = 10.0


class FeatureScaler:
    """
    IQR-based feature scaler persisted as JSON (not pickle).

    Wraps sklearn RobustScaler for outlier-robust normalization.
    Used for preprocessing Autoencoder input features.
    """

    def __init__(self) -> None:
        self._scaler: RobustScaler | None = None
        self._fitted = False

    @property
    def n_features(self) -> int:
        """
        Number of features the scaler was fitted on.
        """
        if not self._fitted or self._scaler is None:
            raise RuntimeError("Scaler has not been fitted")
        return int(self._scaler.n_features_in_)

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """
        Fit the scaler on training data.
        """
        self._scaler = RobustScaler()
        self._scaler.fit(self._preprocess(X))
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted scaler parameters.
        """
        if not self._fitted or self._scaler is None:
            raise RuntimeError("Scaler has not been fitted")
        scaled = self._scaler.transform(self._preprocess(X))
        return np.clip(scaled, -CLIP_LIMIT, CLIP_LIMIT).astype(np.float32)  # type: ignore[no-any-return]

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling transformation.
        """
        if not self._fitted or self._scaler is None:
            raise RuntimeError("Scaler has not been fitted")
        return self._scaler.inverse_transform(X).astype(np.float32)  # type: ignore[no-any-return]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        """
        self.fit(X)
        return self.transform(X)

    def save_json(self, path: Path | str) -> None:
        """
        Serialize scaler parameters to a human-readable JSON file.
        """
        if not self._fitted or self._scaler is None:
            raise RuntimeError("Scaler has not been fitted")
        data = {
            "center": self._scaler.center_.tolist(),
            "scale": self._scaler.scale_.tolist(),
            "n_features": int(self._scaler.n_features_in_),
            "preprocess": "signed_log1p_clip",
            "clip_limit": CLIP_LIMIT,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path | str) -> "FeatureScaler":
        """
        Reconstruct a fitted scaler from a JSON file.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        scaler = cls()
        scaler._scaler = RobustScaler()
        scaler._scaler.center_ = np.array(data["center"], dtype=np.float64)
        scaler._scaler.scale_ = np.array(data["scale"], dtype=np.float64)
        scaler._scaler.n_features_in_ = data["n_features"]
        scaler._fitted = True
        return scaler

    @staticmethod
    def _preprocess(X: np.ndarray) -> np.ndarray:
        """
        Compress high-magnitude traffic counters before robust scaling.
        """
        clean = np.nan_to_num(
            X.astype(np.float64, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return np.sign(clean) * np.log1p(np.abs(clean))
