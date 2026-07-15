"""
ThreatShield AI | 2026
orchestrator.py
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.export_onnx import export_autoencoder, export_dnn
from ml.splitting import prepare_training_data
from ml.train_autoencoder import train_autoencoder
from ml.train_classifiers import train_neural_network
from ml.validation import ValidationResult, validate_ensemble

logger = logging.getLogger(__name__)

AE_FILENAME = "ae.onnx"
DNN_FILENAME = "dnn.onnx"
SCALER_FILENAME = "scaler.json"
THRESHOLD_FILENAME = "threshold.json"


@dataclass
class TrainingResult:
    """
    Aggregated results from a full deep learning training pipeline run.
    """

    ae_metrics: dict[str, float]
    dnn_metrics: dict[str, float]
    ensemble_metrics: ValidationResult | None
    passed_gates: bool
    output_dir: Path
    mlflow_run_id: str | None


class TrainingOrchestrator:
    """
    End-to-end training pipeline that trains the autoencoder and DNN,
    exports both to ONNX, validates the ensemble, and logs to MLflow.
    """

    def __init__(
        self,
        output_dir: Path,
        experiment_name: str = "ThreatShield AI-training",
        epochs: int = 100,
        batch_size: int = 256,
    ) -> None:
        self._output_dir = output_dir
        self._experiment_name = experiment_name
        self._epochs = epochs
        self._batch_size = batch_size

    def run(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """
        Execute the full training pipeline.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        split = prepare_training_data(X, y)

        logger.info(
            "Split: train=%d val=%d test=%d normal_train=%d",
            len(split.X_train),
            len(split.X_val),
            len(split.X_test),
            len(split.X_normal_train),
        )

        with VigilExperiment(self._experiment_name) as experiment:
            experiment.log_params({
                "epochs": self._epochs,
                "batch_size": self._batch_size,
                "n_samples": len(X),
                "n_attack": int(np.sum(y == 1)),
                "n_normal": int(np.sum(y == 0)),
                "n_features": X.shape[1],
            })

            ae_result = self._train_ae(split.X_normal_train)
            ae_metrics = {
                "ae_threshold": float(ae_result["threshold"]),
                "ae_final_train_loss": float(ae_result["history"]["train_loss"][-1]),
                "ae_final_val_loss": float(ae_result["history"]["val_loss"][-1]),
            }

            dnn_result = self._train_dnn(split.X_train, split.y_train)
            dnn_metrics = dnn_result["metrics"]

            self._export_models(ae_result, dnn_result)

            experiment.log_metrics(ae_metrics)
            experiment.log_metrics({f"dnn_{k}": v for k, v in dnn_metrics.items()})

            try:
                ensemble = validate_ensemble(self._output_dir, split.X_test, split.y_test)
                experiment.log_metrics({
                    "ensemble_precision": ensemble.precision,
                    "ensemble_recall": ensemble.recall,
                    "ensemble_f1": ensemble.f1,
                    "ensemble_pr_auc": ensemble.pr_auc,
                    "ensemble_roc_auc": ensemble.roc_auc,
                })
                passed = ensemble.passed_gates
            except Exception:
                logger.exception("Ensemble validation failed")
                ensemble = None
                passed = False

            for name in (AE_FILENAME, DNN_FILENAME, SCALER_FILENAME, THRESHOLD_FILENAME):
                experiment.log_artifact(self._output_dir / name)

            run_id = experiment.run_id

        logger.info("Training complete: passed_gates=%s run_id=%s", passed, run_id)
        return TrainingResult(
            ae_metrics=ae_metrics,
            dnn_metrics=dnn_metrics,
            ensemble_metrics=ensemble,
            passed_gates=passed,
            output_dir=self._output_dir,
            mlflow_run_id=run_id,
        )

    def _train_ae(self, X_normal: np.ndarray) -> dict[str, Any]:
        logger.info("Training autoencoder (%d epochs, %d samples)", self._epochs, len(X_normal))
        return train_autoencoder(X_normal, epochs=self._epochs, batch_size=self._batch_size)

    def _train_dnn(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        logger.info("Training DNN classifier (%d samples)", len(X))
        return train_neural_network(X, y, epochs=self._epochs, batch_size=self._batch_size)

    def _export_models(self, ae_result: dict[str, Any], dnn_result: dict[str, Any]) -> None:
        """
        Export both deep learning models and save scaler/threshold metadata.
        """
        export_autoencoder(ae_result["model"], self._output_dir / AE_FILENAME)
        export_dnn(dnn_result["model"], self._output_dir / DNN_FILENAME)
        ae_result["scaler"].save_json(self._output_dir / SCALER_FILENAME)
        (self._output_dir / THRESHOLD_FILENAME).write_text(
            json.dumps({"threshold": ae_result["threshold"]}, indent=2)
        )
        logger.info("Exported deep learning models to %s", self._output_dir)
