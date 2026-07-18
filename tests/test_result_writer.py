from __future__ import annotations

from pathlib import Path

import pandas as pd

from fl_benchmark.config import (
    DatasetConfig,
    ExperimentConfig,
    FederationConfig,
    ModelConfig,
    PartitionConfig,
    StrategyConfig,
    TrackingConfig,
    TrainingConfig,
)
from fl_benchmark.tracking.result_writer import ResultWriter


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        dataset=DatasetConfig(name="mnist", data_dir=tmp_path / "data"),
        partition=PartitionConfig(type="iid", num_clients=2),
        model=ModelConfig(name="mnist-cnn", num_classes=10),
        training=TrainingConfig(4, 1, 0.1, 0.0, 0.0, "cpu", 1),
        federation=FederationConfig(1, 1.0, 1.0, 2, 2, 2),
        strategy=StrategyConfig(name="fedavg"),
        tracking=TrackingConfig(output_dir=tmp_path),
    )


def test_result_writer_creates_expected_files_and_appends_rounds(tmp_path: Path) -> None:
    writer = ResultWriter(_config(tmp_path), "exp")
    writer.append_round_metrics({"round": 1, "global_accuracy": 0.5})
    writer.append_round_metrics({"round": 2, "global_accuracy": 0.6})
    writer.write_partition(pd.DataFrame({"client_id": [0]}), pd.DataFrame({"client_id": [0]}))
    writer.finish("completed", {"final_accuracy": 0.6})
    out = tmp_path / "exp"
    assert (out / "metadata.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "round_metrics.csv").exists()
    assert (out / "partition_statistics.csv").exists()
    assert len(pd.read_csv(out / "round_metrics.csv")) == 2
