from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_benchmark.training.evaluate import evaluate_model
from fl_benchmark.training.metrics import accuracy_score, macro_f1_score


def test_accuracy_and_macro_f1_on_small_example() -> None:
    y_true = np.asarray([0, 0, 1, 1, 2, 2])
    y_pred = np.asarray([0, 1, 1, 1, 2, 0])
    assert accuracy_score(y_true, y_pred) == 4 / 6
    assert round(macro_f1_score(y_true, y_pred), 6) == round((0.5 + 0.8 + (2 / 3)) / 3, 6)


def test_evaluation_reports_cross_entropy_loss_for_each_true_class() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1, 0])
    result = evaluate_model(torch.nn.Identity(), DataLoader(TensorDataset(logits, labels), batch_size=3), "cpu")
    expected = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    assert result.per_class_accuracy == {0: 0.5, 1: 1.0}
    assert result.per_class_loss[0] == pytest.approx(float(expected[[0, 2]].mean()))
    assert result.per_class_loss[1] == pytest.approx(float(expected[1]))
