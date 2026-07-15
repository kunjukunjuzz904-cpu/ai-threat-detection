"""
ThreatShield AI | 2026
train_dnn.py
"""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


class ThreatDNN(nn.Module):
    """
    Feed-forward deep neural network for binary threat detection.
    """

    def __init__(self, input_dim: int = 35) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Keras-like helper retained for existing callers/tests.
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X.astype(np.float32))
            return self.forward(tensor).cpu().numpy()


def train_neural_network(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 5,
) -> dict[str, Any]:
    """
    Train a PyTorch deep neural network for threat detection.
    """
    X_train, X_eval, y_train, y_eval = train_test_split(
        X.astype(np.float32),
        y.astype(np.float32),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train.reshape(-1, 1)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    model = ThreatDNN(input_dim=X.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCELoss()

    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val.reshape(-1, 1))
    best_state: dict[str, Tensor] | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for _epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for features, labels in train_loader:
            preds = model(features)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(X_val_tensor), y_val_tensor).item())
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_prob = model.predict(X_eval).flatten()
    y_pred = (y_prob >= 0.5).astype(np.int32)
    y_eval_int = y_eval.astype(np.int32)

    metrics = {
        "accuracy": float(accuracy_score(y_eval_int, y_pred)),
        "precision": float(precision_score(y_eval_int, y_pred, zero_division=0)),
        "recall": float(recall_score(y_eval_int, y_pred, zero_division=0)),
        "f1": float(f1_score(y_eval_int, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_eval_int, y_prob)),
    }

    return {"model": model, "metrics": metrics, "history": history}


train_classifier = train_neural_network
