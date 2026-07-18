from __future__ import annotations

import numpy as np

from fl_benchmark.config import PartitionConfig
from fl_benchmark.datasets.partitioning import create_partitions


class DummyDataset:
    def __init__(self, targets: list[int]) -> None:
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)


def _flat(parts: list[list[int]]) -> list[int]:
    return [idx for part in parts for idx in part]


def test_iid_uses_all_indices_once_and_balances_sizes() -> None:
    dataset = DummyDataset([i % 10 for i in range(103)])
    parts = create_partitions(dataset, PartitionConfig(type="iid", num_clients=10, seed=1))
    assert sorted(_flat(parts)) == list(range(103))
    assert len(set(_flat(parts))) == 103
    sizes = [len(part) for part in parts]
    assert max(sizes) - min(sizes) <= 1


def test_dirichlet_reproducible_and_respects_min_size() -> None:
    dataset = DummyDataset([i % 10 for i in range(1000)])
    cfg = PartitionConfig(type="dirichlet", num_clients=8, alpha=0.5, min_partition_size=20, seed=7)
    first = create_partitions(dataset, cfg)
    second = create_partitions(dataset, cfg)
    assert first == second
    assert sorted(_flat(first)) == list(range(1000))
    assert len(set(_flat(first))) == 1000
    assert min(len(part) for part in first) >= 20


def test_pathological_limits_client_classes() -> None:
    dataset = DummyDataset([i % 10 for i in range(1000)])
    cfg = PartitionConfig(type="pathological", num_clients=10, classes_per_client=2, seed=4)
    parts = create_partitions(dataset, cfg)
    targets = np.asarray(dataset.targets)
    assert sorted(_flat(parts)) == list(range(1000))
    for part in parts:
        assert len(set(targets[part].tolist())) <= 2


def test_quantity_skew_changes_sizes_and_preserves_class_distribution() -> None:
    dataset = DummyDataset([i % 5 for i in range(1000)])
    cfg = PartitionConfig(type="quantity", num_clients=8, quantity_alpha=0.2, seed=11)
    parts = create_partitions(dataset, cfg)
    sizes = [len(part) for part in parts]
    assert max(sizes) - min(sizes) > 20
    assert sorted(_flat(parts)) == list(range(1000))
    global_dist = np.bincount(dataset.targets, minlength=5) / len(dataset)
    for part in parts:
        if len(part) > 20:
            local_dist = np.bincount(np.asarray(dataset.targets)[part], minlength=5) / len(part)
            assert np.abs(local_dist - global_dist).max() < 0.2
