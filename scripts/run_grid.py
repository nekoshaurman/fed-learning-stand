from __future__ import annotations

import argparse
import copy
import itertools
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from fl_benchmark.config import load_config
from fl_benchmark.task import run_local_simulation
from fl_benchmark.tracking.experiment import deterministic_key


def _base_config(path: Path, base: dict[str, Any]) -> Path:
    template = path.parent / "mnist_fedavg.toml"
    if not template.exists():
        template = path.parent / "mnist_fedprox.toml"
    return template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args(argv)
    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    grid = raw["grid"]
    base = raw.get("base", {})
    config = load_config(_base_config(args.config, base))
    combos = list(
        itertools.product(
            grid["datasets"],
            grid["strategies"],
            grid["partition_types"],
            grid["alphas"],
            grid["local_epochs"],
            grid["num_clients"],
            grid["seeds"],
        )
    )
    rows: list[dict[str, Any]] = []
    for index, combo in enumerate(combos[: args.max_experiments], start=1):
        dataset, strategy, partition_type, alpha, local_epochs, num_clients, seed = combo
        key_values = {
            "dataset": dataset,
            "strategy": strategy,
            "partition_type": partition_type,
            "alpha": alpha,
            "local_epochs": local_epochs,
            "num_clients": num_clients,
            "seed": seed,
        }
        exp_key = deterministic_key(key_values)
        print(f"[{index}] {exp_key} {key_values}")
        if args.dry_run:
            rows.append({**key_values, "experiment_key": exp_key, "status": "dry-run"})
            continue
        run_config = copy.copy(config)
        run_config = replace(
            run_config,
            name=f"{dataset}-{strategy}-{partition_type}-{exp_key}",
            dataset=replace(run_config.dataset, name=dataset),
            partition=replace(
                run_config.partition,
                type=partition_type,
                alpha=float(alpha),
                num_clients=int(num_clients),
                seed=int(seed),
            ),
            strategy=replace(run_config.strategy, name=strategy),
            training=replace(run_config.training, local_epochs=int(local_epochs), seed=int(seed)),
            federation=replace(
                run_config.federation,
                num_rounds=int(base.get("num_rounds", run_config.federation.num_rounds)),
                fraction_train=float(base.get("fraction_train", run_config.federation.fraction_train)),
                min_available_clients=int(num_clients),
                min_train_clients=int(num_clients),
                min_evaluate_clients=int(num_clients),
            ),
        )
        try:
            out = run_local_simulation(run_config, experiment_id=exp_key)
            rows.append({**key_values, "experiment_key": exp_key, "status": "completed", "output_dir": str(out)})
        except Exception as exc:
            rows.append({**key_values, "experiment_key": exp_key, "status": "failed", "error": str(exc)})
    pd.DataFrame(rows).to_csv(config.tracking.output_dir / "grid_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
