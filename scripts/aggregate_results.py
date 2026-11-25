from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    rows = []
    for summary_path in args.results_dir.glob("*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config_path = summary_path.parent / "config.toml"
        rows.append({"experiment_id": summary_path.parent.name, "config_path": str(config_path), **summary})
    experiments = pd.DataFrame(rows)
    experiments.to_csv(args.results_dir / "experiments.csv", index=False)
    if {"final_accuracy", "dataset", "strategy", "partition_type", "alpha"}.issubset(experiments.columns):
        grouped = experiments.groupby(["dataset", "strategy", "partition_type", "alpha"])["final_accuracy"].agg(["mean", "std", "count"])
        grouped["ci95"] = 1.96 * grouped["std"] / grouped["count"].map(lambda n: math.sqrt(max(n, 1)))
        grouped.reset_index().to_csv(args.results_dir / "grouped_results.csv", index=False)
    else:
        experiments.to_csv(args.results_dir / "grouped_results.csv", index=False)
    print(f"Aggregated {len(rows)} summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
