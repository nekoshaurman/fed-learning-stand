from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl_benchmark.training.train import proximal_penalty, train_model


def test_proximal_mu_zero_adds_no_proximal_loss() -> None:
    model = nn.Linear(2, 2)
    data = TensorDataset(torch.randn(8, 2), torch.randint(0, 2, (8,)))
    loader = DataLoader(data, batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = train_model(model, loader, optimizer, "cpu", epochs=1, proximal_mu=0.0)
    assert result.proximal_loss == 0.0


def test_positive_mu_adds_non_negative_proximal_component() -> None:
    model = nn.Linear(2, 2)
    global_params = [p.detach().numpy().copy() for p in model.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    tensors = [torch.as_tensor(arr) for arr in global_params]
    assert float(proximal_penalty(model, tensors)) > 0
    data = TensorDataset(torch.randn(8, 2), torch.randint(0, 2, (8,)))
    loader = DataLoader(data, batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = train_model(
        model,
        loader,
        optimizer,
        "cpu",
        epochs=1,
        global_parameters=[np.asarray(arr) for arr in global_params],
        proximal_mu=0.1,
    )
    assert result.proximal_loss >= 0.0
