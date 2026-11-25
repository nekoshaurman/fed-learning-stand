from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from fl_benchmark.attacks.label_flip import build_label_flip_plan, make_client_dataset
from fl_benchmark.config import AttackConfig


def _dataset() -> TensorDataset:
    return TensorDataset(torch.arange(12).reshape(12, 1), torch.tensor([0, 1, 2] * 4))


def _partitions() -> list[list[int]]:
    return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]


def test_disabled_attack_keeps_all_clients_honest() -> None:
    targets = np.asarray([0, 1, 2] * 4)
    plan = build_label_flip_plan(targets, _partitions(), AttackConfig())
    assert plan.malicious_clients == ()
    assert plan.total_poisoned_examples == 0


def test_label_flip_is_reproducible_and_changes_only_selected_source_labels() -> None:
    targets = np.asarray([0, 1, 2] * 4)
    config = AttackConfig(
        enabled=True,
        type="label_flip",
        malicious_fraction=0.5,
        poison_fraction=1.0,
        source_label=1,
        target_label=2,
        seed=17,
    )
    first = build_label_flip_plan(targets, _partitions(), config)
    second = build_label_flip_plan(targets, _partitions(), config)
    assert first == second
    assert len(first.malicious_clients) == 2
    assert first.total_poisoned_examples == 2

    dataset = _dataset()
    for client_id, indices in enumerate(_partitions()):
        local = make_client_dataset(dataset, indices, first, client_id)
        labels = [int(local[index][1]) for index in range(len(local))]
        original = targets[indices].tolist()
        if first.is_malicious(client_id):
            assert labels == [0, 2, 2]
        else:
            assert labels == original
    assert dataset.tensors[1].tolist() == targets.tolist()


def test_unknown_enabled_attack_is_rejected() -> None:
    config = AttackConfig(enabled=True, type="none")
    with pytest.raises(ValueError, match="Unsupported enabled attack type"):
        build_label_flip_plan(np.asarray([0, 1]), [[0, 1]], config)


def test_multi_label_flip_changes_two_disjoint_source_classes() -> None:
    targets = np.asarray([0, 1, 2] * 4)
    config = AttackConfig(
        enabled=True,
        type="multi_label_flip",
        malicious_fraction=0.5,
        poison_fraction=1.0,
        label_flips=((0, 1), (2, 3)),
        seed=17,
    )
    plan = build_label_flip_plan(targets, _partitions(), config)
    assert len(plan.malicious_clients) == 2
    assert plan.total_poisoned_examples == 4

    dataset = _dataset()
    for client_id, indices in enumerate(_partitions()):
        labels = [int(make_client_dataset(dataset, indices, plan, client_id)[index][1]) for index in range(3)]
        if plan.is_malicious(client_id):
            assert labels == [1, 1, 3]
        else:
            assert labels == targets[indices].tolist()
    manifest = plan.manifest()
    assert manifest["poisoned_examples_by_flip"] == {"0->1": 2, "2->3": 2}


def test_multi_label_flip_rejects_overlapping_classes() -> None:
    config = AttackConfig(
        enabled=True,
        type="multi_label_flip",
        malicious_fraction=0.5,
        poison_fraction=1.0,
        label_flips=((0, 1), (2, 1)),
    )
    with pytest.raises(ValueError, match="must not share"):
        build_label_flip_plan(np.asarray([0, 1, 2]), [[0, 1, 2]], config)


def test_feature_noise_is_reproducible_and_keeps_labels() -> None:
    targets = np.asarray([0, 1, 2] * 4)
    config = AttackConfig(enabled=True, type="feature_noise", malicious_fraction=0.5, poison_fraction=1.0, noise_std=0.5, seed=17)
    first = build_label_flip_plan(targets, _partitions(), config)
    second = build_label_flip_plan(targets, _partitions(), config)
    assert first == second
    dataset = _dataset()
    for client_id, indices in enumerate(_partitions()):
        local = make_client_dataset(dataset, indices, first, client_id)
        assert [int(local[i][1]) for i in range(len(local))] == targets[indices].tolist()
    malicious = first.malicious_clients[0]
    local = make_client_dataset(dataset, _partitions()[malicious], first, malicious)
    assert not torch.equal(local[0][0], dataset[_partitions()[malicious][0]][0])
