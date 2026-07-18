from __future__ import annotations

import math
import time
from collections.abc import Sized
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Subset, TensorDataset

from fl_benchmark.config import ExperimentConfig
from fl_benchmark.datasets.loader import load_datasets, make_client_dataloader, make_dataloader
from fl_benchmark.datasets.partitioning import create_partitions, get_targets
from fl_benchmark.datasets.statistics import compute_partition_statistics
from fl_benchmark.datasets.visualization import save_partition_plots
from fl_benchmark.models.factory import create_model, get_model_parameters, set_model_parameters
from fl_benchmark.tracking.experiment import create_experiment_id
from fl_benchmark.tracking.result_writer import ResultWriter
from fl_benchmark.training.evaluate import evaluate_model
from fl_benchmark.training.train import train_model


def parameters_nbytes(parameters: list[np.ndarray]) -> int:
    return int(sum(array.nbytes for array in parameters))


def weighted_average(updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    total = sum(num_examples for _, num_examples in updates)
    if total <= 0:
        raise ValueError("Cannot aggregate updates with zero total examples")
    averaged: list[np.ndarray] = []
    for tensors in zip(*(params for params, _ in updates), strict=True):
        acc = sum(tensor * (num / total) for tensor, (_, num) in zip(tensors, updates, strict=True))
        averaged.append(np.asarray(acc))
    return averaged


def _sample_clients(config: ExperimentConfig, round_num: int) -> list[int]:
    rng = np.random.default_rng(config.training.seed + round_num)
    num = max(config.federation.min_train_clients, math.ceil(config.partition.num_clients * config.federation.fraction_train))
    num = min(num, config.partition.num_clients)
    return sorted(rng.choice(config.partition.num_clients, size=num, replace=False).astype(int).tolist())


def _summary(config: ExperimentConfig, rounds: pd.DataFrame, partition_stats: pd.DataFrame) -> dict[str, Any]:
    final = rounds.iloc[-1] if len(rounds) else {}
    best_idx = int(rounds["global_accuracy"].idxmax()) if len(rounds) else 0
    best = rounds.loc[best_idx] if len(rounds) else {}
    convergence_round = None
    if config.target_accuracy is not None and len(rounds):
        hit = rounds[rounds["global_accuracy"] >= config.target_accuracy]
        convergence_round = int(hit.iloc[0]["round"]) if len(hit) else None
    client_acc = rounds["train_accuracy"] if len(rounds) and "train_accuracy" in rounds else pd.Series(dtype=float)
    return {
        "dataset": config.dataset.name,
        "strategy": config.strategy.name,
        "partition_type": config.partition.type,
        "alpha": config.partition.alpha,
        "seed": config.training.seed,
        "final_accuracy": float(final.get("global_accuracy", 0.0)),
        "best_accuracy": float(best.get("global_accuracy", 0.0)),
        "best_accuracy_round": int(best.get("round", 0)),
        "final_loss": float(final.get("global_loss", 0.0)),
        "final_macro_f1": float(final.get("macro_f1", 0.0)),
        "total_duration_seconds": float(final.get("cumulative_duration", 0.0)),
        "total_communication_bytes": int(rounds["estimated_total_bytes"].sum()) if len(rounds) else 0,
        "mean_round_duration": float(rounds["round_duration"].mean()) if len(rounds) else 0.0,
        "convergence_round": convergence_round,
        "client_accuracy_mean": float(client_acc.mean()) if len(client_acc) else 0.0,
        "client_accuracy_std": float(client_acc.std(ddof=0)) if len(client_acc) else 0.0,
        "client_accuracy_min": float(client_acc.min()) if len(client_acc) else 0.0,
        "client_accuracy_max": float(client_acc.max()) if len(client_acc) else 0.0,
        "mean_partition_entropy": float(partition_stats["entropy"].mean()),
        "mean_partition_js_divergence": float(partition_stats["js_divergence"].mean()),
    }


def _save_training_curves(rounds: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds["round"], rounds["global_accuracy"], marker="o", label="accuracy")
    ax.plot(rounds["round"], rounds["global_loss"], marker="s", label="loss")
    ax.set_xlabel("Round")
    ax.set_title("Training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def run_local_simulation(
    config: ExperimentConfig,
    experiment_id: str | None = None,
    train_limit: int | None = None,
    test_limit: int | None = None,
    synthetic_smoke: bool = False,
) -> Path:
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    train_dataset: Any
    test_dataset: Any
    if synthetic_smoke:
        train_n = train_limit or 200
        test_n = test_limit or 50
        generator = torch.Generator().manual_seed(config.training.seed)
        train_targets = torch.arange(train_n) % config.model.num_classes
        test_targets = torch.arange(test_n) % config.model.num_classes
        train_dataset = TensorDataset(
            torch.randn(train_n, 1, 28, 28, generator=generator),
            train_targets.long(),
        )
        test_dataset = TensorDataset(
            torch.randn(test_n, 1, 28, 28, generator=generator),
            test_targets.long(),
        )
        train_dataset.targets = train_targets.numpy()
        dataset_bundle = None
    else:
        dataset_bundle = load_datasets(config.dataset)
        train_dataset = dataset_bundle.train
        test_dataset = dataset_bundle.test
    if dataset_bundle is not None and dataset_bundle.num_classes != config.model.num_classes:
        config = replace(config, model=replace(config.model, num_classes=dataset_bundle.num_classes))
    if not synthetic_smoke and train_limit is not None:
        assert dataset_bundle is not None
        train_size = min(train_limit, len(cast(Sized, train_dataset)))
        train_dataset = Subset(train_dataset, list(range(train_size)))
        train_dataset.targets = np.asarray(get_targets(cast(Any, dataset_bundle.train)))[:train_size]
    if not synthetic_smoke and test_limit is not None:
        test_dataset = Subset(test_dataset, list(range(min(test_limit, len(cast(Sized, test_dataset))))))
    partitions = create_partitions(cast(Any, train_dataset), config.partition)
    targets = get_targets(cast(Any, train_dataset))
    partition_stats = compute_partition_statistics(targets, partitions, config.model.num_classes)
    exp_id = experiment_id or create_experiment_id(config.name, config.training.seed)
    writer = ResultWriter(config, exp_id)
    writer.write_partition(partition_stats.statistics, partition_stats.matrix)
    save_partition_plots(partition_stats.statistics, partition_stats.matrix, writer.output_dir)

    global_model = create_model(config.model, config.dataset.name)
    global_parameters = get_model_parameters(global_model)
    test_loader = make_dataloader(test_dataset, config.training, shuffle=False)
    cumulative = 0.0
    velocity: list[np.ndarray] | None = None
    try:
        for round_num in range(1, config.federation.num_rounds + 1):
            round_start = time.perf_counter()
            selected = _sample_clients(config, round_num)
            updates: list[tuple[list[np.ndarray], int]] = []
            client_rows: list[dict[str, Any]] = []
            for cid in selected:
                model = create_model(config.model, config.dataset.name)
                set_model_parameters(model, global_parameters)
                loader = make_client_dataloader(train_dataset, partitions[cid], config.training)
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=config.training.learning_rate,
                    momentum=config.training.momentum,
                    weight_decay=config.training.weight_decay,
                )
                prox_params = [p.copy() for p in global_parameters] if config.strategy.name == "fedprox" else None
                result = train_model(
                    model,
                    loader,
                    optimizer,
                    config.training.device,
                    config.training.local_epochs,
                    global_parameters=prox_params,
                    proximal_mu=config.strategy.proximal_mu if config.strategy.name == "fedprox" else 0.0,
                )
                params = get_model_parameters(model)
                updates.append((params, result.num_examples))
                client_rows.append(
                    {
                        "round": round_num,
                        "partition_id": cid,
                        "train_loss": result.loss,
                        "train_accuracy": result.accuracy,
                        "train_duration": result.duration_seconds,
                        "num_examples": result.num_examples,
                        "proximal_loss": result.proximal_loss,
                    }
                )
            aggregated = weighted_average(updates)
            if config.strategy.name == "fedavgm":
                if velocity is None:
                    velocity = [np.zeros_like(param) for param in aggregated]
                delta = [old - new for old, new in zip(global_parameters, aggregated, strict=True)]
                velocity = [
                    config.strategy.server_momentum * vel + diff
                    for vel, diff in zip(velocity, delta, strict=True)
                ]
                global_parameters = [
                    old - config.strategy.server_learning_rate * vel
                    for old, vel in zip(global_parameters, velocity, strict=True)
                ]
            else:
                global_parameters = aggregated
            set_model_parameters(global_model, global_parameters)
            evaluation = evaluate_model(global_model, test_loader, config.training.device)
            model_bytes = parameters_nbytes(global_parameters)
            round_duration = time.perf_counter() - round_start
            cumulative += round_duration
            writer.append_client_metrics(client_rows)
            writer.append_round_metrics(
                {
                    "round": round_num,
                    "global_loss": evaluation.loss,
                    "global_accuracy": evaluation.accuracy,
                    "macro_f1": evaluation.macro_f1,
                    "train_loss": float(np.mean([row["train_loss"] for row in client_rows])),
                    "train_accuracy": float(np.mean([row["train_accuracy"] for row in client_rows])),
                    "round_duration": round_duration,
                    "cumulative_duration": cumulative,
                    "participating_clients": len(selected),
                    "failed_clients": 0,
                    "estimated_upload_bytes": model_bytes * len(updates),
                    "estimated_download_bytes": model_bytes * len(selected),
                    "estimated_total_bytes": model_bytes * (len(updates) + len(selected)),
                }
            )
        final_eval = evaluate_model(global_model, test_loader, config.training.device)
        writer.write_confusion_matrix(final_eval.confusion_matrix)
        writer.save_model(global_model)
        rounds = pd.read_csv(writer.output_dir / "round_metrics.csv")
        _save_training_curves(rounds, writer.output_dir)
        writer.finish("completed", _summary(config, rounds, partition_stats.statistics))
        return writer.output_dir
    except Exception as exc:
        writer.finish("failed", {"error": str(exc)})
        raise
