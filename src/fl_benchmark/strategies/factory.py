from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fl_benchmark.config import ExperimentConfig
from fl_benchmark.strategies.fedavg import build_fedavg
from fl_benchmark.strategies.fedavgm import build_fedavgm
from fl_benchmark.strategies.fedprox import build_fedprox


def create_strategy(
    config: ExperimentConfig,
    evaluation_fn: Callable[..., Any],
    metrics_tracker: Any | None = None,
) -> Any:
    _ = metrics_tracker
    if config.strategy.name == "fedavg":
        return build_fedavg(config, evaluation_fn)
    if config.strategy.name == "fedavgm":
        return build_fedavgm(config, evaluation_fn)
    if config.strategy.name == "fedprox":
        return build_fedprox(config, evaluation_fn)
    raise ValueError(f"Unsupported strategy: {config.strategy.name}")
