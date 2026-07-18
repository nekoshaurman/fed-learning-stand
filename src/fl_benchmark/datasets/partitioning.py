from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from fl_benchmark.config import PartitionConfig


class LabelledDataset(Protocol):
    def __len__(self) -> int: ...

    @property
    def targets(self) -> Sequence[int] | np.ndarray: ...


def get_targets(dataset: LabelledDataset) -> np.ndarray:
    if not hasattr(dataset, "targets"):
        raise TypeError("Dataset must expose a 'targets' attribute")
    targets = np.asarray(dataset.targets, dtype=np.int64)
    if targets.ndim != 1 or len(targets) != len(dataset):
        raise ValueError("Dataset targets must be one-dimensional and match dataset length")
    return targets


def create_partitions(dataset: LabelledDataset, config: PartitionConfig) -> list[list[int]]:
    targets = get_targets(dataset)
    if config.type == "iid":
        return _iid(len(targets), config.num_clients, config.seed)
    if config.type == "dirichlet":
        return _dirichlet(targets, config)
    if config.type == "pathological":
        return _pathological(targets, config)
    if config.type == "quantity":
        return _quantity(targets, config)
    raise ValueError(f"Unsupported partition type: {config.type}")


def _iid(num_examples: int, num_clients: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(num_examples)
    return [part.astype(int).tolist() for part in np.array_split(shuffled, num_clients)]


def _dirichlet(targets: np.ndarray, config: PartitionConfig, max_attempts: int = 200) -> list[list[int]]:
    rng = np.random.default_rng(config.seed)
    classes = np.unique(targets)
    for _ in range(max_attempts):
        partitions: list[list[int]] = [[] for _ in range(config.num_clients)]
        for cls in classes:
            cls_indices = np.flatnonzero(targets == cls)
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet(np.full(config.num_clients, config.alpha))
            splits = (np.cumsum(proportions)[:-1] * len(cls_indices)).astype(int)
            for cid, split in enumerate(np.split(cls_indices, splits)):
                partitions[cid].extend(split.astype(int).tolist())
        if min(len(part) for part in partitions) >= config.min_partition_size:
            for part in partitions:
                rng.shuffle(part)
            return partitions
    raise ValueError(
        "Could not create Dirichlet partition satisfying min_partition_size="
        f"{config.min_partition_size} after {max_attempts} attempts"
    )


def _pathological(targets: np.ndarray, config: PartitionConfig) -> list[list[int]]:
    if config.classes_per_client <= 0:
        raise ValueError("classes_per_client must be positive for pathological partitioning")
    rng = np.random.default_rng(config.seed)
    classes = np.unique(targets)
    class_to_clients: dict[int, list[int]] = {int(cls): [] for cls in classes}
    client_classes: list[set[int]] = []
    shuffled_classes = rng.permutation(classes)
    for cid in range(config.num_clients):
        start = (cid * config.classes_per_client) % len(shuffled_classes)
        chosen = np.roll(shuffled_classes, -start)[: min(config.classes_per_client, len(classes))]
        chosen_set = {int(cls) for cls in chosen}
        client_classes.append(chosen_set)
        for cls in chosen_set:
            class_to_clients[int(cls)].append(cid)
    for cls in classes:
        if not class_to_clients[int(cls)]:
            candidates = [cid for cid, values in enumerate(client_classes) if len(values) >= config.classes_per_client]
            cid = int(candidates[int(rng.integers(0, len(candidates)))])
            removed = int(next(iter(client_classes[cid])))
            client_classes[cid].remove(removed)
            class_to_clients[removed].remove(cid)
            client_classes[cid].add(int(cls))
            class_to_clients[int(cls)].append(cid)

    partitions: list[list[int]] = [[] for _ in range(config.num_clients)]
    for cls in classes:
        cls_indices = np.flatnonzero(targets == cls)
        rng.shuffle(cls_indices)
        receivers = class_to_clients[int(cls)]
        for cid, split in zip(receivers, np.array_split(cls_indices, len(receivers)), strict=True):
            partitions[cid].extend(split.astype(int).tolist())
    for part in partitions:
        rng.shuffle(part)
    return partitions


def _quantity(targets: np.ndarray, config: PartitionConfig) -> list[list[int]]:
    rng = np.random.default_rng(config.seed)
    n = len(targets)
    proportions = rng.dirichlet(np.full(config.num_clients, config.quantity_alpha))
    sizes = np.maximum(1, np.floor(proportions * n).astype(int))
    while sizes.sum() < n:
        sizes[int(np.argmax(proportions - sizes / n))] += 1
    while sizes.sum() > n:
        cid = int(np.argmax(sizes))
        sizes[cid] -= 1

    partitions: list[list[int]] = [[] for _ in range(config.num_clients)]
    remaining = sizes.astype(int)
    for cls in np.unique(targets):
        cls_indices = np.flatnonzero(targets == cls)
        rng.shuffle(cls_indices)
        weights = remaining / max(remaining.sum(), 1)
        cls_counts = np.floor(weights * len(cls_indices)).astype(int)
        while cls_counts.sum() < len(cls_indices):
            candidates = np.where(remaining - cls_counts > 0)[0]
            cid = int(candidates[np.argmax(remaining[candidates] - cls_counts[candidates])])
            cls_counts[cid] += 1
        start = 0
        for cid, count in enumerate(cls_counts):
            end = start + int(count)
            partitions[cid].extend(cls_indices[start:end].astype(int).tolist())
            start = end
        remaining = np.maximum(remaining - cls_counts, 0)
    for part in partitions:
        rng.shuffle(part)
    return partitions
