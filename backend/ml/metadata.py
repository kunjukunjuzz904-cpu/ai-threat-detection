"""
ThreatShield AI | 2026
metadata.py
"""

import hashlib
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model_metadata import ModelMetadata

logger = logging.getLogger(__name__)

MODEL_TYPES: dict[str, str] = {
    "ae.onnx": "autoencoder",
    "dnn.onnx": "deep_neural_network",
}

VERSION_HASH_LENGTH = 12


def compute_model_version(artifact_path: Path) -> str:
    """
    Compute a 12-char hex version string from the SHA-256 hash of a file
    """
    sha = hashlib.sha256(artifact_path.read_bytes())
    return sha.hexdigest()[:VERSION_HASH_LENGTH]


async def save_model_metadata(
    session: AsyncSession,
    model_dir: Path,
    training_samples: int,
    metrics_by_model: dict[str, dict[str, object]],
    mlflow_run_id: str | None = None,
    threshold: float | None = None,
) -> list[ModelMetadata]:
    """
    Persist metadata for Deep Learning models
    """
    rows: list[ModelMetadata] = []

    active_result = await session.execute(
        select(ModelMetadata).where(
            ModelMetadata.is_active == True,  # type: ignore[arg-type]  # noqa: E712
        )
    )
    for old in active_result.scalars().all():
        old.is_active = False
    await session.flush()

    for filename, model_type in MODEL_TYPES.items():
        artifact_path = model_dir / filename
        version = compute_model_version(artifact_path)

        row = ModelMetadata(
            model_type=model_type,
            version=version,
            training_samples=training_samples,
            metrics=metrics_by_model[model_type],
            artifact_path=str(artifact_path),
            is_active=True,
            mlflow_run_id=mlflow_run_id,
            threshold=threshold if model_type == "autoencoder" else None,
        )
        session.add(row)
        rows.append(row)

    await session.commit()

    logger.info(
        "Saved metadata for Deep Learning models (samples=%d)",
        training_samples,
    )

    return rows
