"""
ThreatShield AI | 2026
test_config_ml.py

Tests ML-related settings defaults: detection mode, ensemble weights, model paths, and MLflow URI.
"""

from app.config import settings


def test_default_detection_mode_is_rules() -> None:
    """
    Default detection_mode is 'rules' before any ML models are loaded.
    """
    assert settings.detection_mode == "rules"


def test_default_ensemble_weights():
    total = (
        settings.ensemble_weight_ae +
        settings.ensemble_weight_dnn
    )
    assert abs(total - 0.95) < 1e-6

def test_default_model_dir() -> None:
    """
    Default model artifact directory is data/models.
    """
    assert settings.model_dir == "data/models"


def test_default_ae_threshold_percentile() -> None:
    """
    Default autoencoder threshold percentile is 99.5.
    """
    assert settings.ae_threshold_percentile == 99.5


def test_default_mlflow_tracking_uri() -> None:
    """
    Default MLflow tracking URI uses local file storage.
    """
    assert settings.mlflow_tracking_uri == "file:./mlruns"
