"""MLflow tracking, registry aliases, and reproducibility metadata."""

import json
import subprocess
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from ai_support.router.config import RouterConfig
from ai_support.router.evaluation import EvaluationResult, GateDecision
from ai_support.router.preprocessing import DatasetSplit, records_to_frame


def file_sha256(path: Path) -> str:
    """Return a stable file fingerprint."""

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata(project_root: Path) -> tuple[str, bool]:
    """Return commit and dirty state without failing outside Git."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )
    return (commit.stdout.strip() or "uncommitted", bool(dirty.stdout.strip()))


def dvc_hash(data_path: Path) -> str:
    """Read the tracked dataset hash from its DVC pointer when available."""

    pointer = data_path.with_name(f"{data_path.name}.dvc")
    if not pointer.is_file():
        return f"untracked:{file_sha256(data_path)}"
    raw = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    return str(raw["outs"][0]["md5"])


def _ensure_tracking_store(config: RouterConfig) -> str:
    if config.mlflow.tracking_uri.startswith("sqlite:///"):
        database = config.mlflow.tracking_uri.removeprefix("sqlite:///")
        Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    config.mlflow.artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_registry_uri(config.mlflow.tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config.mlflow.experiment_name,
            artifact_location=config.mlflow.artifact_root.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def _flatten_metrics(evaluation: EvaluationResult) -> dict[str, float]:
    metrics = dict(evaluation.metrics)
    for department, values in evaluation.per_department.items():
        slug = department.lower().replace(" ", "_")
        for name, value in values.items():
            metrics[f"department.{slug}.{name}"] = value
    return metrics


def _write_evaluation_artifacts(
    directory: Path, evaluation: EvaluationResult, gate: GateDecision
) -> None:
    (directory / "evaluation.json").write_text(
        json.dumps(evaluation.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (directory / "promotion_gate.json").write_text(
        json.dumps(asdict(gate), indent=2, sort_keys=True), encoding="utf-8"
    )
    matrix = pd.DataFrame(
        evaluation.confusion_matrix,
        index=evaluation.labels,
        columns=evaluation.labels,
    )
    matrix.to_csv(directory / "confusion_matrix.csv", index_label="actual_department")


def track_and_register(
    config: RouterConfig,
    model: Pipeline,
    split: DatasetSplit,
    evaluation: EvaluationResult,
    gate: GateDecision,
    metaflow_run_id: str,
    project_root: Path,
) -> dict[str, Any]:
    """Log one complete experiment and register passing models as candidate."""

    experiment_id = _ensure_tracking_store(config)
    commit, dirty = git_metadata(project_root)
    source_dvc_hash = dvc_hash(config.data.dataset_path)
    evaluation_version = file_sha256(config.data.evaluation_path)
    lock_path = project_root / "uv.lock"
    environment_digest = f"uv-lock:{file_sha256(lock_path)}"
    parameters = {
        "audience": config.audience,
        "random_seed": config.random_seed,
        "test_fraction": config.data.test_fraction,
        "ngram_range": f"{config.model.ngram_min},{config.model.ngram_max}",
        "min_df": config.model.min_df,
        "max_features": config.model.max_features,
        "logistic_regression_c": config.model.c,
        "max_iter": config.model.max_iter,
        "train_rows": len(split.train),
        "evaluation_rows": len(split.evaluation),
        "removed_train_duplicates": split.removed_train_duplicates,
        "chronological_cutoff": split.cutoff,
    }
    tags = {
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
        "dvc_dataset_hash": source_dvc_hash,
        "evaluation_dataset_version": evaluation_version,
        "metaflow_run_id": metaflow_run_id,
        "environment_digest": environment_digest,
        "configuration": str(config.source_path.relative_to(project_root)),
        "promotion_gate_passed": str(gate.passed).lower(),
    }

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{config.audience}-router",
        tags=tags,
    ) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(_flatten_metrics(evaluation))
        with tempfile.TemporaryDirectory(prefix="router-evaluation-") as temp:
            artifact_dir = Path(temp)
            _write_evaluation_artifacts(artifact_dir, evaluation, gate)
            mlflow.log_artifacts(artifact_dir, artifact_path="evaluation")
        mlflow.log_artifact(config.source_path, artifact_path="configuration")
        example = records_to_frame(split.train[:2])
        signature = infer_signature(example, model.predict(example))
        model_info = mlflow.sklearn.log_model(
            model,
            name="router_model",
            input_example=example,
            signature=signature,
            skops_trusted_types=["ai_support.router.preprocessing.TicketTextTransformer"],
            metadata={
                "audience": config.audience,
                "dvc_dataset_hash": source_dvc_hash,
                "evaluation_dataset_version": evaluation_version,
            },
        )
        run_id = run.info.run_id

    model_version: str | None = None
    if gate.passed:
        registered = mlflow.register_model(
            model_info.model_uri,
            config.mlflow.registered_model_name,
            tags={
                "validation_status": "candidate",
                "git_commit": commit,
                "dvc_dataset_hash": source_dvc_hash,
            },
        )
        model_version = str(registered.version)
        MlflowClient().set_registered_model_alias(
            config.mlflow.registered_model_name, "candidate", model_version
        )

    return {
        "experiment_id": experiment_id,
        "mlflow_run_id": run_id,
        "metaflow_run_id": metaflow_run_id,
        "registered_model_name": config.mlflow.registered_model_name,
        "registered_model_version": model_version,
        "candidate_registered": model_version is not None,
        "promotion_gate_passed": gate.passed,
        "promotion_gate_failures": list(gate.failures),
        "git_commit": commit,
        "git_dirty": dirty,
        "dvc_dataset_hash": source_dvc_hash,
        "evaluation_dataset_version": evaluation_version,
        "environment_digest": environment_digest,
        "metrics": evaluation.metrics,
    }


def configure_mlflow(config: RouterConfig) -> None:
    """Point tracking and registry clients at the configured backend."""

    _ensure_tracking_store(config)


def approve_model(config: RouterConfig, version_or_alias: str = "candidate") -> str:
    """Explicitly assign the champion alias to an already registered model."""

    configure_mlflow(config)
    client = MlflowClient()
    if version_or_alias.isdigit():
        version = version_or_alias
        client.get_model_version(config.mlflow.registered_model_name, version)
    else:
        version = client.get_model_version_by_alias(
            config.mlflow.registered_model_name, version_or_alias
        ).version
    client.set_model_version_tag(
        config.mlflow.registered_model_name, str(version), "approval_status", "approved"
    )
    client.set_registered_model_alias(config.mlflow.registered_model_name, "champion", str(version))
    return str(version)


def load_registered_model(config: RouterConfig, version_or_alias: str) -> Pipeline:
    """Load a candidate, champion, or explicit older version without retraining."""

    configure_mlflow(config)
    suffix = f"/{version_or_alias}" if version_or_alias.isdigit() else f"@{version_or_alias}"
    model = mlflow.sklearn.load_model(f"models:/{config.mlflow.registered_model_name}{suffix}")
    return model
