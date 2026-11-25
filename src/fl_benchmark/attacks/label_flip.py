"""Deterministic label-flipping data-poisoning attack for local FL simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from fl_benchmark.config import AttackConfig


@dataclass(frozen=True)
class LabelFlipPlan:
    """Ground-truth attack plan used only for reproducible simulation and analysis."""

    config: AttackConfig
    malicious_clients: tuple[int, ...]
    poisoned_indices: dict[int, tuple[int, ...]]
    target_labels_by_index: dict[int, dict[int, int]]

    @property
    def total_poisoned_examples(self) -> int:
        """Return the number of local training examples whose labels are replaced."""
        return sum(len(indices) for indices in self.poisoned_indices.values())

    def is_malicious(self, client_id: int) -> bool:
        """Return whether a client is selected as malicious."""
        return client_id in self.malicious_clients

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable record of requested and realised attack settings."""
        label_flips = _label_flips(self.config)
        poisoned_by_flip = {
            f"{source_label}->{target_label}": sum(
                target_label == client_targets.get(index)
                for client_targets in self.target_labels_by_index.values()
                for index in client_targets
            )
            for source_label, target_label in label_flips
        }
        return {
            "enabled": self.config.enabled,
            "type": self.config.type,
            "seed": self.config.seed,
            "malicious_fraction": self.config.malicious_fraction,
            "poison_fraction": self.config.poison_fraction,
            "noise_std": self.config.noise_std if self.config.type == "feature_noise" else None,
            "source_label": self.config.source_label,
            "target_label": self.config.target_label,
            "label_flips": [
                {"source_label": source_label, "target_label": target_label}
                for source_label, target_label in label_flips
            ],
            "malicious_clients": list(self.malicious_clients),
            "num_malicious_clients": len(self.malicious_clients),
            "poisoned_indices": {str(cid): list(indices) for cid, indices in self.poisoned_indices.items()},
            "poisoned_examples_per_client": {
                str(cid): len(indices) for cid, indices in self.poisoned_indices.items()
            },
            "total_poisoned_examples": self.total_poisoned_examples,
            "poisoned_examples_by_flip": poisoned_by_flip,
        }


class LabelFlippedSubset(Dataset[Any]):
    """Dataset view that replaces labels only for selected global sample indices."""

    def __init__(self, dataset: Dataset[Any], indices: list[int], target_labels_by_index: dict[int, int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.target_labels_by_index = dict(target_labels_by_index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        global_index = self.indices[index]
        features, label = self.dataset[global_index]
        if global_index in self.target_labels_by_index:
            return features, self.target_labels_by_index[global_index]
        return features, int(label)


class FeatureNoisySubset(Dataset[Any]):
    """Dataset view that adds deterministic Gaussian noise to selected local images."""

    def __init__(self, dataset: Dataset[Any], indices: list[int], noisy_indices: tuple[int, ...], noise_std: float, seed: int) -> None:
        self.dataset, self.indices = dataset, indices
        self.noisy_indices, self.noise_std, self.seed = set(noisy_indices), noise_std, seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        global_index = self.indices[index]
        features, label = self.dataset[global_index]
        if global_index not in self.noisy_indices or not isinstance(features, torch.Tensor):
            return features, int(label)
        noisy_features = features if torch.is_floating_point(features) else features.to(torch.float32)
        generator = torch.Generator().manual_seed(self.seed + global_index)
        noise = torch.randn(noisy_features.shape, generator=generator, dtype=noisy_features.dtype) * self.noise_std
        return noisy_features + noise, int(label)


def _label_flips(config: AttackConfig) -> tuple[tuple[int, int], ...]:
    if config.type == "label_flip":
        return ((config.source_label, config.target_label),)
    if config.type == "multi_label_flip":
        if len(config.label_flips) < 2:
            raise ValueError("multi_label_flip requires at least two mappings")
        flat_labels = [label for pair in config.label_flips for label in pair]
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in config.label_flips):
            raise ValueError("multi_label_flip mappings must contain distinct source and target labels")
        if len(set(flat_labels)) != len(flat_labels):
            raise ValueError("multi_label_flip mappings must not share source or target classes")
        return config.label_flips
    return ()


def build_label_flip_plan(
    targets: np.ndarray,
    partitions: list[list[int]],
    config: AttackConfig,
) -> LabelFlipPlan:
    """Select malicious clients and source-class samples without mutating the dataset."""
    if not config.enabled:
        return LabelFlipPlan(config, (), {}, {})
    if config.type not in {"label_flip", "multi_label_flip", "feature_noise"}:
        raise ValueError(f"Unsupported enabled attack type: {config.type}")
    if not partitions or config.malicious_fraction == 0 or config.poison_fraction == 0:
        return LabelFlipPlan(config, (), {}, {})

    rng = np.random.default_rng(config.seed)
    num_malicious = min(len(partitions), max(1, math.ceil(len(partitions) * config.malicious_fraction)))
    malicious_clients = tuple(sorted(rng.choice(len(partitions), size=num_malicious, replace=False).astype(int).tolist()))
    poisoned_indices: dict[int, tuple[int, ...]] = {}
    target_labels_by_index: dict[int, dict[int, int]] = {}
    for client_id in malicious_clients:
        client_targets: dict[int, int] = {}
        if config.type == "feature_noise":
            candidates = np.asarray(partitions[client_id], dtype=int)
            num_poisoned = min(len(candidates), max(1, math.ceil(len(candidates) * config.poison_fraction))) if len(candidates) else 0
            chosen = rng.choice(candidates, size=num_poisoned, replace=False).astype(int).tolist() if num_poisoned else []
            poisoned_indices[client_id] = tuple(sorted(chosen))
            target_labels_by_index[client_id] = {}
            continue
        for source_label, target_label in _label_flips(config):
            candidates = np.asarray(
                [index for index in partitions[client_id] if int(targets[index]) == source_label],
                dtype=int,
            )
            if len(candidates) == 0:
                continue
            num_poisoned = min(len(candidates), max(1, math.ceil(len(candidates) * config.poison_fraction)))
            chosen = rng.choice(candidates, size=num_poisoned, replace=False).astype(int).tolist()
            client_targets.update({index: target_label for index in chosen})
        target_labels_by_index[client_id] = client_targets
        poisoned_indices[client_id] = tuple(sorted(client_targets))
    return LabelFlipPlan(config, malicious_clients, poisoned_indices, target_labels_by_index)


def make_client_dataset(
    dataset: Dataset[Any],
    indices: list[int],
    plan: LabelFlipPlan,
    client_id: int,
) -> Dataset[Any]:
    """Return an unmodified or label-flipped view for one client's local partition."""
    if not plan.is_malicious(client_id):
        return LabelFlippedSubset(dataset, indices, {})
    if plan.config.type == "feature_noise":
        return FeatureNoisySubset(dataset, indices, plan.poisoned_indices.get(client_id, ()), plan.config.noise_std, plan.config.seed)
    return LabelFlippedSubset(dataset, indices, plan.target_labels_by_index.get(client_id, {}))
