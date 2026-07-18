from __future__ import annotations

import numpy as np

from fl_benchmark.training.metrics import accuracy_score, macro_f1_score


def test_accuracy_and_macro_f1_on_small_example() -> None:
    y_true = np.asarray([0, 0, 1, 1, 2, 2])
    y_pred = np.asarray([0, 1, 1, 1, 2, 0])
    assert accuracy_score(y_true, y_pred) == 4 / 6
    assert round(macro_f1_score(y_true, y_pred), 6) == round((0.5 + 0.8 + (2 / 3)) / 3, 6)
