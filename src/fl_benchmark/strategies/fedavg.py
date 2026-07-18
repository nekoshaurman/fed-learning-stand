from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fl_benchmark.config import ExperimentConfig


def build_fedavg(config: ExperimentConfig, evaluation_fn: Callable[..., Any], **kwargs: Any) -> Any:
    from flwr.server.strategy import FedAvg

    return FedAvg(
        fraction_fit=config.federation.fraction_train,
        fraction_evaluate=config.federation.fraction_evaluate,
        min_fit_clients=config.federation.min_train_clients,
        min_evaluate_clients=config.federation.min_evaluate_clients,
        min_available_clients=config.federation.min_available_clients,
        evaluate_fn=evaluation_fn,
        accept_failures=config.strategy.accept_failures,
        **kwargs,
    )
