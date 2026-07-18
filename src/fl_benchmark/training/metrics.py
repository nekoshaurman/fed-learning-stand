from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import f1_score


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return sk_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def per_class_accuracy(matrix: np.ndarray) -> dict[int, float]:
    totals = matrix.sum(axis=1)
    return {
        int(cls): float(matrix[cls, cls] / totals[cls]) if totals[cls] else 0.0
        for cls in range(matrix.shape[0])
    }
