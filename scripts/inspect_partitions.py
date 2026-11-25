from __future__ import annotations

import argparse
from pathlib import Path

from fl_benchmark.config import load_config
from fl_benchmark.datasets.loader import load_datasets
from fl_benchmark.datasets.partitioning import create_partitions, get_targets
from fl_benchmark.datasets.statistics import compute_partition_statistics
from fl_benchmark.datasets.visualization import save_partition_plots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    bundle = load_datasets(config.dataset)
    partitions = create_partitions(bundle.train, config.partition)
    stats = compute_partition_statistics(get_targets(bundle.train), partitions, config.model.num_classes)
    out = args.output_dir or config.tracking.output_dir / "partition_inspection"
    out.mkdir(parents=True, exist_ok=True)
    stats.statistics.to_csv(out / "partition_statistics.csv", index=False)
    stats.matrix.to_csv(out / "partition_matrix.csv", index=False)
    save_partition_plots(stats.statistics, stats.matrix, out)
    print(stats.statistics[["client_id", "num_examples", "num_classes", "entropy", "js_divergence"]].describe())
    print(f"Partition plots saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
