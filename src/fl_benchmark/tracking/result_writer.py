from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from fl_benchmark.config import ExperimentConfig
from fl_benchmark.tracking.experiment import base_metadata


class ResultWriter:
    def __init__(self, config: ExperimentConfig, experiment_id: str) -> None:
        self.config = config
        self.experiment_id = experiment_id
        self.output_dir = config.tracking.output_dir / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_rows: list[dict[str, Any]] = []
        self.client_rows: list[dict[str, Any]] = []
        self.start_time = datetime.now(UTC)
        self.metadata = base_metadata(experiment_id, self.start_time, Path.cwd())
        self.write_config()
        self.write_json("metadata.json", self.metadata)

    def write_config(self) -> None:
        if self.config.source_path and self.config.source_path.exists():
            shutil.copyfile(self.config.source_path, self.output_dir / "config.toml")
        else:
            self.write_json("config.json", self.config.to_dict())

    def write_json(self, filename: str, payload: dict[str, Any]) -> None:
        (self.output_dir / filename).write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    def append_round_metrics(self, row: dict[str, Any]) -> None:
        self.round_rows.append(row)
        pd.DataFrame(self.round_rows).to_csv(self.output_dir / "round_metrics.csv", index=False)

    def append_client_metrics(self, rows: list[dict[str, Any]]) -> None:
        if not self.config.tracking.save_client_metrics:
            return
        self.client_rows.extend(rows)
        pd.DataFrame(self.client_rows).to_csv(self.output_dir / "client_metrics.csv", index=False)

    def write_partition(self, statistics: pd.DataFrame, matrix: pd.DataFrame) -> None:
        statistics.to_csv(self.output_dir / "partition_statistics.csv", index=False)
        matrix.to_csv(self.output_dir / "partition_matrix.csv", index=False)

    def write_attack_manifest(self, manifest: dict[str, Any]) -> None:
        """Persist the simulation-only ground truth for a configured data attack."""
        self.write_json("attack_manifest.json", manifest)

    def write_confusion_matrix(self, matrix: Any) -> None:
        pd.DataFrame(matrix).to_csv(self.output_dir / "final_confusion_matrix.csv", index=False)

    def write_per_class_metrics(
        self,
        per_class_accuracy: dict[int, float],
        per_class_loss: dict[int, float],
        matrix: Any,
    ) -> None:
        """Persist final accuracy and cross-entropy loss for every true class."""
        support = pd.DataFrame(matrix).sum(axis=1)
        classes = sorted(set(per_class_accuracy) | set(per_class_loss))
        pd.DataFrame(
            {
                "class_id": classes,
                "support": [int(support.get(class_id, 0)) for class_id in classes],
                "accuracy": [per_class_accuracy.get(class_id, 0.0) for class_id in classes],
                "loss": [per_class_loss.get(class_id, 0.0) for class_id in classes],
            }
        ).to_csv(self.output_dir / "final_per_class_metrics.csv", index=False)

    def save_model(self, model: torch.nn.Module) -> None:
        if self.config.tracking.save_model:
            torch.save(model.state_dict(), self.output_dir / "final_model.pt")

    def finish(self, status: str, summary: dict[str, Any]) -> None:
        end = datetime.now(UTC)
        self.metadata.update(
            {
                "end_time": end.isoformat(),
                "duration_seconds": (end - self.start_time).total_seconds(),
                "status": status,
            }
        )
        self.write_json("summary.json", summary)
        self.write_json("metadata.json", self.metadata)
