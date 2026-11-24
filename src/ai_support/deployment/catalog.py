"""Build a portable experiment metadata catalog from immutable run manifests."""

import json
from pathlib import Path
from typing import Any


def _record(kind: str, audience: str, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "audience": audience,
        "experiment_id": manifest.get("experiment_id"),
        "git_commit": manifest.get("git_commit"),
        "dvc_dataset_hash": manifest.get("dvc_dataset_hash"),
        "metaflow_run_id": manifest.get("metaflow_run_id"),
        "mlflow_run_id": manifest.get("mlflow_run_id"),
        "environment_digest": manifest.get("environment_digest"),
        "evaluation_dataset_version": manifest.get("evaluation_dataset_version"),
        "configuration": manifest.get("configuration"),
        "metrics": manifest.get("metrics", {}),
        "model_version": manifest.get("registered_model_version"),
        "release_id": manifest.get("release_id"),
        "status": manifest.get("status"),
    }


def build_experiment_catalog(project_root: Path, output_path: Path | None = None) -> Path:
    """Collect router and RAG provenance into deterministic JSON Lines output."""

    records: list[dict[str, Any]] = []
    for audience in ("student", "staff"):
        router_path = project_root / "artifacts" / "router" / audience / "latest_run.json"
        if router_path.is_file():
            manifest = json.loads(router_path.read_text(encoding="utf-8"))
            records.append(_record("router", audience, manifest))
        release_dir = project_root / "artifacts" / "rag" / audience / "releases"
        for release_path in sorted(release_dir.glob("*.json")):
            manifest = json.loads(release_path.read_text(encoding="utf-8"))
            records.append(_record("rag", audience, manifest))

    destination = output_path or project_root / "artifacts" / "catalog" / "experiments.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return destination
