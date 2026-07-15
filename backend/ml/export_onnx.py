"""
ThreatShield AI | 2026
export_onnx.py
"""

from pathlib import Path

import torch
from ml.autoencoder import ThreatAutoencoder
from ml.dnn import ThreatDNN

ONNX_OPSET = 17


def export_autoencoder(
    model: ThreatAutoencoder,
    path: Path | str,
    opset: int = ONNX_OPSET,
) -> Path:
    """
    Export a PyTorch autoencoder to ONNX with dynamic batch dimension.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(1, model.input_dim)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version=opset,
        dynamo=False,
        external_data=False,
        export_params=True,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["reconstructed"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "reconstructed": {0: "batch_size"},
        },
    )
    return path


def export_dnn(
    model: ThreatDNN,
    path: Path | str,
    input_dim: int | None = None,
    opset: int = ONNX_OPSET,
) -> Path:
    """
    Export a PyTorch deep neural network classifier to ONNX.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    n_features = input_dim or model.input_dim
    dummy = torch.randn(1, n_features)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        export_params=True,
        opset_version=opset,
        dynamo=False,
        external_data=False,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["prediction"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "prediction": {0: "batch_size"},
        },
    )
    return path


export_classifier = export_dnn
export_neural_network = export_dnn
