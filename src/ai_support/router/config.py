"""Typed router configuration loaded from version-controlled TOML files."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    dataset_path: Path
    evaluation_path: Path
    test_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    ngram_min: int
    ngram_max: int
    min_df: int
    max_features: int
    c: float
    max_iter: int
    top_k: int


@dataclass(frozen=True)
class PromotionConfig:
    min_accuracy: float
    min_macro_f1: float
    min_department_recall: float
    max_calibration_error: float


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    artifact_root: Path
    experiment_name: str
    registered_model_name: str


@dataclass(frozen=True)
class OutputConfig:
    predictions_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class RouterConfig:
    audience: str
    random_seed: int
    data: DataConfig
    model: ModelConfig
    promotion: PromotionConfig
    mlflow: MlflowConfig
    output: OutputConfig
    source_path: Path


def _project_path(root: Path, value: str) -> Path:
    return (root / value).resolve()


def load_router_config(path: Path, project_root: Path | None = None) -> RouterConfig:
    """Load and validate a student or staff router configuration."""

    source_path = path.resolve()
    root = (project_root or Path.cwd()).resolve()
    with source_path.open("rb") as stream:
        raw: dict[str, Any] = tomllib.load(stream)

    audience = str(raw["router"]["audience"])
    if audience not in {"student", "staff"}:
        raise ValueError("router.audience must be 'student' or 'staff'")

    data = raw["data"]
    model = raw["model"]
    promotion = raw["promotion"]
    tracking = raw["mlflow"]
    output = raw["output"]
    test_fraction = float(data["test_fraction"])
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("data.test_fraction must be between 0.05 and 0.5")

    return RouterConfig(
        audience=audience,
        random_seed=int(raw["router"]["random_seed"]),
        data=DataConfig(
            dataset_path=_project_path(root, data["dataset_path"]),
            evaluation_path=_project_path(root, data["evaluation_path"]),
            test_fraction=test_fraction,
        ),
        model=ModelConfig(
            ngram_min=int(model["ngram_min"]),
            ngram_max=int(model["ngram_max"]),
            min_df=int(model["min_df"]),
            max_features=int(model["max_features"]),
            c=float(model["c"]),
            max_iter=int(model["max_iter"]),
            top_k=int(model["top_k"]),
        ),
        promotion=PromotionConfig(**{key: float(value) for key, value in promotion.items()}),
        mlflow=MlflowConfig(
            tracking_uri=os.getenv("AI_SUPPORT_MLFLOW_TRACKING_URI", str(tracking["tracking_uri"])),
            artifact_root=_project_path(root, tracking["artifact_root"]),
            experiment_name=str(tracking["experiment_name"]),
            registered_model_name=str(tracking["registered_model_name"]),
        ),
        output=OutputConfig(
            predictions_path=_project_path(root, output["predictions_path"]),
            manifest_path=_project_path(root, output["manifest_path"]),
        ),
        source_path=source_path,
    )
