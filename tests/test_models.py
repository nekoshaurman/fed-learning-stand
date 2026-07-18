from __future__ import annotations

import numpy as np
import pytest
import torch

from fl_benchmark.config import ModelConfig
from fl_benchmark.models.factory import create_model, get_model_parameters, set_model_parameters


def test_model_output_shapes() -> None:
    mnist = create_model(ModelConfig(name="mnist-cnn", num_classes=10), "mnist")
    assert mnist(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    cifar10 = create_model(ModelConfig(name="cifar-cnn", num_classes=10), "cifar10")
    assert cifar10(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    cifar100 = create_model(ModelConfig(name="cifar-cnn", num_classes=100), "cifar100")
    assert cifar100(torch.zeros(2, 3, 32, 32)).shape == (2, 100)


def test_get_set_parameters_preserves_predictions() -> None:
    model = create_model(ModelConfig(name="mnist-cnn", num_classes=10), "mnist")
    model.eval()
    x = torch.randn(3, 1, 28, 28)
    before = model(x).detach()
    params = get_model_parameters(model)
    set_model_parameters(model, params)
    after = model(x).detach()
    assert torch.allclose(before, after)


def test_set_parameters_rejects_incompatible_shapes() -> None:
    model = create_model(ModelConfig(name="mnist-cnn", num_classes=10), "mnist")
    params = get_model_parameters(model)
    params[0] = np.zeros((1,), dtype=params[0].dtype)
    with pytest.raises(ValueError, match="Shape mismatch"):
        set_model_parameters(model, params)
