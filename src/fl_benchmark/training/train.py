from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainResult:
    loss: float
    accuracy: float
    num_examples: int
    duration_seconds: float
    num_batches: int
    proximal_loss: float = 0.0


def _global_tensors(
    model: nn.Module,
    global_parameters: list[np.ndarray] | None,
    device: torch.device,
) -> list[torch.Tensor] | None:
    if global_parameters is None:
        return None
    params = list(model.parameters())
    if len(params) != len(global_parameters):
        raise ValueError("global_parameters length does not match trainable model parameters")
    tensors: list[torch.Tensor] = []
    for param, array in zip(params, global_parameters, strict=True):
        if tuple(param.shape) != tuple(array.shape):
            raise ValueError(f"FedProx global parameter shape mismatch: {param.shape} vs {array.shape}")
        tensors.append(torch.as_tensor(array, dtype=param.dtype, device=device).detach().clone())
    return tensors


def proximal_penalty(model: nn.Module, global_tensors: list[torch.Tensor] | None) -> torch.Tensor:
    if global_tensors is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    penalty = torch.tensor(0.0, device=global_tensors[0].device)
    for local, global_param in zip(model.parameters(), global_tensors, strict=True):
        penalty = penalty + torch.sum((local - global_param) ** 2)
    return penalty


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    epochs: int,
    global_parameters: list[np.ndarray] | None = None,
    proximal_mu: float = 0.0,
) -> TrainResult:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    torch_device = torch.device(device)
    model.to(torch_device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    fixed_global = _global_tensors(model, global_parameters, torch_device) if proximal_mu > 0 else None
    total_loss = 0.0
    total_proximal = 0.0
    correct = 0
    num_examples = 0
    num_batches = 0
    start = time.perf_counter()
    for _ in range(epochs):
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(torch_device)
            labels = labels.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            classification_loss = criterion(outputs, labels)
            prox = proximal_penalty(model, fixed_global)
            loss = classification_loss + (proximal_mu / 2.0) * prox
            loss.backward()
            optimizer.step()
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().cpu()) * batch_size
            total_proximal += float(((proximal_mu / 2.0) * prox).detach().cpu()) * batch_size
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            num_examples += batch_size
            num_batches += 1
    duration = time.perf_counter() - start
    return TrainResult(
        loss=total_loss / max(num_examples, 1),
        accuracy=correct / max(num_examples, 1),
        num_examples=num_examples,
        duration_seconds=duration,
        num_batches=num_batches,
        proximal_loss=total_proximal / max(num_examples, 1),
    )
