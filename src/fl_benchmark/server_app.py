from __future__ import annotations

from typing import Any

from fl_benchmark.config import load_config
from fl_benchmark.datasets.loader import load_datasets, make_dataloader
from fl_benchmark.models.factory import create_model, set_model_parameters
from fl_benchmark.strategies.factory import create_strategy
from fl_benchmark.training.evaluate import evaluate_model


def _server_fn(context: Any) -> Any:
    from flwr.common import ndarrays_to_parameters
    from flwr.server import ServerAppComponents, ServerConfig

    from fl_benchmark.models.factory import get_model_parameters

    config = load_config(dict(context.run_config)["config"])
    bundle = load_datasets(config.dataset)
    model = create_model(config.model, config.dataset.name)
    test_loader = make_dataloader(bundle.test, config.training, shuffle=False)

    def evaluate(server_round: int, parameters: list[Any], eval_config: dict[str, Any]) -> tuple[float, dict[str, float]]:
        _ = server_round, eval_config
        set_model_parameters(model, parameters)
        result = evaluate_model(model, test_loader, config.training.device)
        return result.loss, {"global_accuracy": result.accuracy, "macro_f1": result.macro_f1}

    strategy = create_strategy(
        config,
        evaluation_fn=evaluate,
        metrics_tracker=None,
    )
    strategy.initial_parameters = ndarrays_to_parameters(get_model_parameters(model))
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=config.federation.num_rounds))


try:
    from flwr.server import ServerApp

    app: Any = ServerApp(server_fn=_server_fn)
except Exception:
    app = None
