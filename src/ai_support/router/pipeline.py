"""Reusable router pipeline stages for direct and Metaflow execution."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from ai_support.router.config import RouterConfig, load_router_config
from ai_support.router.evaluation import (
    EvaluationResult,
    GateDecision,
    apply_promotion_gate,
    evaluate_model,
)
from ai_support.router.model import train_model
from ai_support.router.preprocessing import DatasetSplit, chronological_split
from ai_support.router.schema import Audience, TicketRecord, load_ticket_csv, write_ticket_csv
from ai_support.router.tracking import track_and_register


@dataclass(frozen=True)
class PipelineResult:
    config: RouterConfig
    records: list[TicketRecord]
    split: DatasetSplit
    model: Pipeline
    evaluation: EvaluationResult
    gate: GateDecision
    manifest: dict[str, Any]


def ingest(config: RouterConfig) -> list[TicketRecord]:
    return load_ticket_csv(config.data.dataset_path, config.audience)  # type: ignore[arg-type]


def preprocess(config: RouterConfig, records: list[TicketRecord]) -> DatasetSplit:
    split = chronological_split(records, config.data.test_fraction)
    write_ticket_csv(config.data.evaluation_path, split.evaluation)
    return split


def train(config: RouterConfig, split: DatasetSplit) -> Pipeline:
    return train_model(split.train, config.model, config.random_seed)


def evaluate(config: RouterConfig, model: Pipeline, split: DatasetSplit) -> EvaluationResult:
    return evaluate_model(model, split.evaluation, config.model.top_k)


def validate(config: RouterConfig, evaluation: EvaluationResult) -> GateDecision:
    return apply_promotion_gate(evaluation, config.promotion)


def write_prediction_output(config: RouterConfig, evaluation: EvaluationResult) -> None:
    config.output.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output.predictions_path.open("w", encoding="utf-8") as stream:
        for row in evaluation.predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def register(
    config: RouterConfig,
    model: Pipeline,
    split: DatasetSplit,
    evaluation: EvaluationResult,
    gate: GateDecision,
    metaflow_run_id: str,
    project_root: Path,
) -> dict[str, Any]:
    write_prediction_output(config, evaluation)
    manifest = track_and_register(
        config,
        model,
        split,
        evaluation,
        gate,
        metaflow_run_id,
        project_root,
    )
    config.output.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.output.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def run_pipeline(
    config_path: Path,
    project_root: Path,
    metaflow_run_id: str = "direct",
) -> PipelineResult:
    """Execute all five required stages in order."""

    config = load_router_config(config_path, project_root)
    records = ingest(config)
    split = preprocess(config, records)
    model = train(config, split)
    evaluation = evaluate(config, model, split)
    gate = validate(config, evaluation)
    manifest = register(config, model, split, evaluation, gate, metaflow_run_id, project_root)
    return PipelineResult(config, records, split, model, evaluation, gate, manifest)


def config_path_for_audience(project_root: Path, audience: Audience) -> Path:
    return project_root / "configs" / "router" / f"{audience}.toml"
