from __future__ import annotations

from typing import Any, cast

import torch

from fl_benchmark.attacks.label_flip import build_label_flip_plan, make_client_dataset
from fl_benchmark.config import load_config
from fl_benchmark.datasets.loader import load_datasets, make_dataloader
from fl_benchmark.datasets.partitioning import create_partitions, get_targets
from fl_benchmark.models.factory import create_model, get_model_parameters, set_model_parameters
from fl_benchmark.training.evaluate import evaluate_model
from fl_benchmark.training.train import train_model


def _client_fn(context: Any) -> Any:
    from flwr.client import NumPyClient

    run_config = dict(context.run_config)
    config = load_config(run_config["config"])
    partition_id = int(context.node_config.get("partition-id", context.node_config.get("partition_id", 0)))
    bundle = load_datasets(config.dataset)
    partitions = create_partitions(cast(Any, bundle.train), config.partition)
    attack_plan = build_label_flip_plan(get_targets(cast(Any, bundle.train)), partitions, config.attack)

    class BenchmarkClient(NumPyClient):
        def fit(self, parameters: list[Any], fit_config: dict[str, Any]) -> tuple[list[Any], int, dict[str, Any]]:
            model = create_model(config.model, config.dataset.name)
            set_model_parameters(model, parameters)
            client_dataset = make_client_dataset(bundle.train, partitions[partition_id], attack_plan, partition_id)
            loader = make_dataloader(client_dataset, config.training, shuffle=True)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.training.learning_rate,
                momentum=config.training.momentum,
                weight_decay=config.training.weight_decay,
            )
            prox_mu = float(fit_config.get("proximal_mu", 0.0))
            result = train_model(
                model,
                loader,
                optimizer,
                config.training.device,
                config.training.local_epochs,
                global_parameters=parameters if prox_mu > 0 else None,
                proximal_mu=prox_mu,
            )
            metrics = {
                "train_loss": result.loss,
                "train_accuracy": result.accuracy,
                "train_duration": result.duration_seconds,
                "num_examples": result.num_examples,
                "partition_id": partition_id,
                "is_malicious": attack_plan.is_malicious(partition_id),
                "num_poisoned_examples": len(attack_plan.poisoned_indices.get(partition_id, ())),
            }
            if prox_mu > 0:
                metrics["proximal_loss"] = result.proximal_loss
            return get_model_parameters(model), result.num_examples, metrics

        def evaluate(self, parameters: list[Any], eval_config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
            _ = eval_config
            model = create_model(config.model, config.dataset.name)
            set_model_parameters(model, parameters)
            loader = make_dataloader(bundle.test, config.training, shuffle=False)
            result = evaluate_model(model, loader, config.training.device)
            return result.loss, result.num_examples, {
                "val_loss": result.loss,
                "val_accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "evaluation_duration": result.duration_seconds,
                "num_examples": result.num_examples,
            }

    return BenchmarkClient().to_client()


try:
    from flwr.client import ClientApp

    app: Any = ClientApp(client_fn=_client_fn)
except Exception:
    app = None
