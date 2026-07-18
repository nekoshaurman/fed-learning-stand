from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fl_benchmark.training.metrics import (
    accuracy_score,
    confusion_matrix,
    macro_f1_score,
    per_class_accuracy,
)


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    macro_f1: float
    num_examples: int
    per_class_accuracy: dict[int, float]
    confusion_matrix: np.ndarray
    duration_seconds: float


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str | torch.device) -> EvaluationResult:
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    start = time.perf_counter()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(torch_device)
            labels = labels.to(torch_device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += float(loss.cpu()) * int(labels.numel())
            y_true.extend(labels.cpu().numpy().astype(int).tolist())
            y_pred.extend(preds.cpu().numpy().astype(int).tolist())
    duration = time.perf_counter() - start
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    num_classes = int(max(y_true_arr.max(initial=0), y_pred_arr.max(initial=0))) + 1
    matrix = confusion_matrix(y_true_arr, y_pred_arr, num_classes)
    return EvaluationResult(
        loss=total_loss / max(len(y_true), 1),
        accuracy=accuracy_score(y_true_arr, y_pred_arr),
        macro_f1=macro_f1_score(y_true_arr, y_pred_arr),
        num_examples=len(y_true),
        per_class_accuracy=per_class_accuracy(matrix),
        confusion_matrix=matrix,
        duration_seconds=duration,
    )
