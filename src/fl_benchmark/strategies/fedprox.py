from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fl_benchmark.config import ExperimentConfig


def build_fedprox(config: ExperimentConfig, evaluation_fn: Callable[..., Any], **kwargs: Any) -> Any:
    from flwr.server.strategy import FedAvg

    def on_fit_config_fn(server_round: int) -> dict[str, bool | bytes | float | int | str]:
        return {"server_round": server_round, "proximal_mu": config.strategy.proximal_mu}

    return FedAvg(
        fraction_fit=config.federation.fraction_train,
        fraction_evaluate=config.federation.fraction_evaluate,
        min_fit_clients=config.federation.min_train_clients,
        min_evaluate_clients=config.federation.min_evaluate_clients,
        min_available_clients=config.federation.min_available_clients,
        evaluate_fn=evaluation_fn,
        on_fit_config_fn=on_fit_config_fn,
        accept_failures=config.strategy.accept_failures,
        **kwargs,
    )
