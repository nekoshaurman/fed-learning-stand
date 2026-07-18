from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from fl_benchmark.config import DatasetConfig, TrainingConfig


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    test: Dataset
    num_classes: int


def load_datasets(config: DatasetConfig) -> DatasetBundle:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError("torchvision is required to load benchmark image datasets") from exc
    data_dir = Path(config.data_dir)
    if config.name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return DatasetBundle(
            train=datasets.MNIST(data_dir, train=True, download=config.download, transform=transform),
            test=datasets.MNIST(data_dir, train=False, download=config.download, transform=transform),
            num_classes=10,
        )
    if config.name == "cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        )
        return DatasetBundle(
            train=datasets.CIFAR10(data_dir, train=True, download=config.download, transform=transform),
            test=datasets.CIFAR10(data_dir, train=False, download=config.download, transform=transform),
            num_classes=10,
        )
    if config.name == "cifar100":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))]
        )
        return DatasetBundle(
            train=datasets.CIFAR100(data_dir, train=True, download=config.download, transform=transform),
            test=datasets.CIFAR100(data_dir, train=False, download=config.download, transform=transform),
            num_classes=100,
        )
    raise ValueError(f"Unsupported dataset: {config.name}")


def split_train_validation(dataset: Dataset, seed: int, val_fraction: float = 0.1) -> tuple[Dataset, Dataset]:
    size = len(cast(Sized, dataset))
    val_size = max(1, int(size * val_fraction))
    train_size = size - val_size
    generator = torch.Generator().manual_seed(seed)
    train, validation = random_split(dataset, [train_size, val_size], generator=generator)
    return train, validation


def make_dataloader(dataset: Dataset, training: TrainingConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(training.seed)
    return DataLoader(
        dataset,
        batch_size=training.batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def make_client_dataloader(
    dataset: Dataset,
    indices: list[int],
    training: TrainingConfig,
    shuffle: bool = True,
) -> DataLoader:
    return make_dataloader(Subset(dataset, indices), training, shuffle=shuffle)
