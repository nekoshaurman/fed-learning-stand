from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

DatasetName = Literal["mnist", "cifar10", "cifar100"]
PartitionType = Literal["iid", "dirichlet", "pathological", "quantity"]
StrategyName = Literal["fedavg", "fedavgm", "fedprox"]
AttackName = Literal["none", "label_flip", "multi_label_flip", "feature_noise"]


@dataclass(frozen=True)
class DatasetConfig:
    name: DatasetName
    data_dir: Path
    download: bool = False
    num_workers: int = 0


@dataclass(frozen=True)
class PartitionConfig:
    type: PartitionType
    num_clients: int
    alpha: float = 0.5
    classes_per_client: int = 2
    min_partition_size: int = 1
    quantity_alpha: float = 0.5
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_classes: int
    hidden_dims: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    local_epochs: int
    learning_rate: float
    momentum: float
    weight_decay: float
    device: str
    seed: int


@dataclass(frozen=True)
class FederationConfig:
    num_rounds: int
    fraction_train: float
    fraction_evaluate: float
    min_available_clients: int
    min_train_clients: int
    min_evaluate_clients: int


@dataclass(frozen=True)
class StrategyConfig:
    name: StrategyName
    server_learning_rate: float = 1.0
    server_momentum: float = 0.0
    proximal_mu: float = 0.0
    accept_failures: bool = True


@dataclass(frozen=True)
class AttackConfig:
    """Settings for a data-poisoning attack in a local simulation."""

    enabled: bool = False
    type: AttackName = "none"
    malicious_fraction: float = 0.0
    poison_fraction: float = 0.0
    source_label: int = 0
    target_label: int = 1
    label_flips: tuple[tuple[int, int], ...] = ()
    noise_std: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class TrackingConfig:
    output_dir: Path
    save_client_metrics: bool = True
    save_model: bool = False
    experiment_name: str = "experiment"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    partition: PartitionConfig
    model: ModelConfig
    training: TrainingConfig
    federation: FederationConfig
    strategy: StrategyConfig
    tracking: TrackingConfig
    attack: AttackConfig = field(default_factory=AttackConfig)
    target_accuracy: float | None = None
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset"]["data_dir"] = str(self.dataset.data_dir)
        data["tracking"]["output_dir"] = str(self.tracking.output_dir)
        data["source_path"] = str(self.source_path) if self.source_path else None
        return data


def _require_table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required TOML table [{name}]")
    return value


def load_config(path: Path | str) -> ExperimentConfig:
    config_path = Path(path)
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {config_path}: {exc}") from exc

    exp = _require_table(raw, "experiment")
    dataset_raw = _require_table(raw, "dataset")
    tracking_raw = _require_table(raw, "tracking")
    dataset_raw = {**dataset_raw, "data_dir": Path(dataset_raw["data_dir"])}
    tracking_raw = {**tracking_raw, "output_dir": Path(tracking_raw["output_dir"])}
    dataset = DatasetConfig(**dataset_raw)
    tracking = TrackingConfig(**tracking_raw)
    dataset = DatasetConfig(
        name=dataset.name,
        data_dir=dataset.data_dir.resolve()
        if not dataset.data_dir.is_absolute()
        else dataset.data_dir,
        download=dataset.download,
        num_workers=dataset.num_workers,
    )
    tracking = TrackingConfig(
        output_dir=tracking.output_dir.resolve()
        if not tracking.output_dir.is_absolute()
        else tracking.output_dir,
        save_client_metrics=tracking.save_client_metrics,
        save_model=tracking.save_model,
        experiment_name=tracking.experiment_name,
    )
    attack_raw = dict(raw.get("attack", {}))
    if "label_flips" in attack_raw:
        attack_raw["label_flips"] = tuple(tuple(pair) for pair in attack_raw["label_flips"])
    config = ExperimentConfig(
        name=str(exp["name"]),
        target_accuracy=exp.get("target_accuracy"),
        dataset=dataset,
        partition=PartitionConfig(**_require_table(raw, "partition")),
        model=ModelConfig(**_require_table(raw, "model")),
        training=TrainingConfig(**_require_table(raw, "training")),
        federation=FederationConfig(**_require_table(raw, "federation")),
        strategy=StrategyConfig(**_require_table(raw, "strategy")),
        tracking=tracking,
        attack=AttackConfig(**attack_raw),
        source_path=config_path.resolve(),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.partition.num_clients <= 0:
        raise ValueError("partition.num_clients must be positive")
    if config.partition.min_partition_size < 0:
        raise ValueError("partition.min_partition_size must be non-negative")
    if config.partition.alpha <= 0 or config.partition.quantity_alpha <= 0:
        raise ValueError("partition alpha values must be positive")
    if not 0 < config.federation.fraction_train <= 1:
        raise ValueError("federation.fraction_train must be in (0, 1]")
    if not 0 < config.federation.fraction_evaluate <= 1:
        raise ValueError("federation.fraction_evaluate must be in (0, 1]")
    if config.training.local_epochs <= 0 or config.training.batch_size <= 0:
        raise ValueError("training.local_epochs and batch_size must be positive")
    if config.model.num_classes <= 1:
        raise ValueError("model.num_classes must be greater than one")
    attack = config.attack
    if not 0 <= attack.malicious_fraction <= 1:
        raise ValueError("attack.malicious_fraction must be in [0, 1]")
    if not 0 <= attack.poison_fraction <= 1:
        raise ValueError("attack.poison_fraction must be in [0, 1]")
    if attack.noise_std < 0:
        raise ValueError("attack.noise_std must be non-negative")
    if attack.enabled and attack.type == "none":
        raise ValueError("attack.type must be set when attack.enabled is true")
    if attack.type == "label_flip":
        if not 0 <= attack.source_label < config.model.num_classes:
            raise ValueError("attack.source_label must be a valid model class")
        if not 0 <= attack.target_label < config.model.num_classes:
            raise ValueError("attack.target_label must be a valid model class")
        if attack.source_label == attack.target_label:
            raise ValueError("attack.source_label and attack.target_label must differ")
    if attack.type == "multi_label_flip":
        if len(attack.label_flips) < 2:
            raise ValueError("attack.label_flips must contain at least two mappings for multi_label_flip")
        flat_labels: list[int] = []
        for pair in attack.label_flips:
            if len(pair) != 2:
                raise ValueError("each attack.label_flips item must contain [source_label, target_label]")
            source_label, target_label = pair
            if not isinstance(source_label, int) or not isinstance(target_label, int):
                raise ValueError("attack.label_flips labels must be integers")
            if not 0 <= source_label < config.model.num_classes:
                raise ValueError("attack.label_flips source_label must be a valid model class")
            if not 0 <= target_label < config.model.num_classes:
                raise ValueError("attack.label_flips target_label must be a valid model class")
            if source_label == target_label:
                raise ValueError("attack.label_flips source and target labels must differ")
            flat_labels.extend((source_label, target_label))
        if len(set(flat_labels)) != len(flat_labels):
            raise ValueError("attack.label_flips mappings must not share source or target classes")
    if attack.type == "feature_noise" and attack.enabled and attack.noise_std <= 0:
        raise ValueError("attack.noise_std must be positive for feature_noise")
