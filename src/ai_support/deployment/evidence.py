"""Export non-sensitive, reviewable MLOps evidence for assessment submission."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ai_support.deployment.catalog import build_experiment_catalog
from ai_support.router.config import load_router_config
from ai_support.router.pipeline import config_path_for_audience


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_value(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=project_root, capture_output=True, check=False, text=True
    )
    return result.stdout.strip() or "unavailable"


def _dvc_pointers(project_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for pointer in sorted((project_root / "data").rglob("*.dvc")):
        raw = yaml.safe_load(pointer.read_text(encoding="utf-8"))
        for output in raw.get("outs", []):
            records.append(
                {
                    "pointer": str(pointer.relative_to(project_root)),
                    "path": str(output.get("path", "")),
                    "md5": str(output.get("md5", "")),
                }
            )
    return records


def _registry_evidence(project_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for audience in ("student", "staff"):
        config = load_router_config(config_path_for_audience(project_root, audience), project_root)
        client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
        aliases: dict[str, str] = {}
        for alias in ("candidate", "champion"):
            try:
                aliases[alias] = str(
                    client.get_model_version_by_alias(
                        config.mlflow.registered_model_name, alias
                    ).version
                )
            except MlflowException:
                aliases[alias] = "unavailable"
        try:
            versions = [
                {
                    "version": str(version.version),
                    "run_id": version.run_id,
                    "status": version.status,
                }
                for version in client.search_model_versions(
                    f"name='{config.mlflow.registered_model_name}'"
                )
            ]
        except MlflowException:
            versions = []
        records.append(
            {
                "audience": audience,
                "registered_model": config.mlflow.registered_model_name,
                "aliases": aliases,
                "versions": sorted(versions, key=lambda item: int(item["version"])),
            }
        )
    return records


def export_evidence(project_root: Path, output_dir: Path) -> list[Path]:
    """Export manifests and registry metadata without models, prompts, or credentials."""

    root = project_root.resolve()
    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    catalog = build_experiment_catalog(root, destination / "experiment-catalog.jsonl")
    registry = destination / "model-registry.json"
    dvc = destination / "dvc-pointers.json"
    provenance = destination / "provenance.json"
    _write_json(registry, _registry_evidence(root))
    _write_json(dvc, _dvc_pointers(root))
    _write_json(
        provenance,
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": _git_value(root, "rev-parse", "HEAD"),
            "git_dirty": _git_value(root, "status", "--short") != "unavailable",
            "source": "generated from local immutable manifests and the MLflow registry",
        },
    )
    return [catalog, registry, dvc, provenance]
