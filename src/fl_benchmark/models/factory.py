from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fl_benchmark.config import ModelConfig
from fl_benchmark.models.cifar_cnn import CifarCNN
from fl_benchmark.models.mnist_cnn import MnistCNN


def create_model(config: ModelConfig, dataset_name: str) -> nn.Module:
    if dataset_name == "mnist" or config.name == "mnist-cnn":
        return MnistCNN(config.num_classes)
    if dataset_name in {"cifar10", "cifar100"} or config.name == "cifar-cnn":
        return CifarCNN(config.num_classes)
    raise ValueError(f"Unsupported model '{config.name}' for dataset '{dataset_name}'")


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy().copy() for tensor in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    state = model.state_dict()
    if len(parameters) != len(state):
        raise ValueError(f"Expected {len(state)} tensors, received {len(parameters)}")
    new_state: dict[str, torch.Tensor] = {}
    for (name, current), array in zip(state.items(), parameters, strict=True):
        if tuple(array.shape) != tuple(current.shape):
            raise ValueError(
                f"Shape mismatch for '{name}': expected {tuple(current.shape)}, got {array.shape}"
            )
        tensor = torch.as_tensor(array)
        if tensor.dtype != current.dtype:
            if not (
                tensor.dtype.is_floating_point == current.dtype.is_floating_point
                or tensor.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}
            ):
                raise ValueError(f"Dtype mismatch for '{name}': {tensor.dtype} vs {current.dtype}")
            tensor = tensor.to(dtype=current.dtype)
        new_state[name] = tensor.reshape_as(current)
    model.load_state_dict(new_state, strict=True)
