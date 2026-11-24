"""Immutable RAG release manifests and MLflow experiment tracking."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow import MlflowClient

from ai_support.rag.config import RAGConfig
from ai_support.rag.schema import Chunk, GateDecision, RAGMetrics, SourceDocument
from ai_support.router.tracking import file_sha256, git_metadata


def build_release_id(
    config: RAGConfig, documents: list[SourceDocument], chunks: list[Chunk]
) -> str:
    payload = {
        "audience": config.audience,
        "documents": sorted(
            (doc.metadata.document_id, doc.metadata.content_hash) for doc in documents
        ),
        "chunks": sorted((chunk.chunk_id, chunk.content_hash) for chunk in chunks),
        "chunking": [config.chunk_method, config.chunk_size, config.chunk_overlap],
        "embedding": [config.embedding.model_id, config.embedding.revision],
        "retrieval": [
            config.retrieval_mode,
            config.candidate_k,
            config.top_k,
            config.rrf_k,
            config.rerank,
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()[:24]


def dvc_corpus_hash(config: RAGConfig) -> str:
    pointer = config.source_dir.parent.with_suffix(".dvc")
    if not pointer.is_file():
        return f"untracked:{file_sha256(config.benchmark_path)}"
    raw = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    return str(raw["outs"][0]["md5"])


def base_manifest(
    config: RAGConfig,
    release_id: str,
    documents: list[SourceDocument],
    chunks: list[Chunk],
    project_root: Path,
    metaflow_run_id: str,
    index_backend: str,
) -> dict[str, Any]:
    commit, dirty = git_metadata(project_root)
    return {
        "release_id": release_id,
        "audience": config.audience,
        "status": "evaluating",
        "documents": [doc.metadata.model_dump(mode="json") for doc in documents],
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunking": {
            "method": config.chunk_method,
            "size": config.chunk_size,
            "overlap": config.chunk_overlap,
        },
        "embedding": {
            "profile": config.embedding_profile,
            "model_id": config.embedding.model_id,
            "revision": config.embedding.revision,
            "dimension": config.embedding_dimension,
        },
        "retrieval": {
            "mode": config.retrieval_mode,
            "candidate_k": config.candidate_k,
            "top_k": config.top_k,
            "rrf_k": config.rrf_k,
            "rerank": config.rerank,
            "reranker_model_id": config.reranker_model_id if config.rerank else None,
            "reranker_revision": config.reranker_revision if config.rerank else None,
        },
        "index_backend": index_backend,
        "configuration": str(config.source_path.relative_to(project_root)),
        "random_seed": config.random_seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "dvc_dataset_hash": dvc_corpus_hash(config),
        "evaluation_dataset_version": file_sha256(config.benchmark_path),
        "metaflow_run_id": metaflow_run_id,
        "environment_digest": f"uv-lock:{file_sha256(project_root / 'uv.lock')}",
    }


def _ensure_experiment(config: RAGConfig) -> str:
    if config.tracking_uri.startswith("sqlite:///"):
        database = config.tracking_uri.removeprefix("sqlite:///")
        Path(database).resolve().parent.mkdir(parents=True, exist_ok=True)
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment:
        return experiment.experiment_id
    if config.tracking_uri.startswith(("http://", "https://")):
        return mlflow.create_experiment(config.experiment_name)
    return mlflow.create_experiment(
        config.experiment_name, artifact_location=config.artifact_root.as_uri()
    )


def finalize_and_track(
    config: RAGConfig,
    manifest: dict[str, Any],
    metrics: RAGMetrics,
    gate: GateDecision,
    evaluation_path: Path,
) -> dict[str, Any]:
    """Log the experiment and write a content-addressed immutable release file."""

    experiment_id = _ensure_experiment(config)
    status = "candidate" if gate.passed else "rejected"
    final = {
        **manifest,
        "status": status,
        "metrics": metrics.model_dump(),
        "promotion_gate": gate.model_dump(),
        "experiment_id": experiment_id,
    }
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{config.audience}-rag-{manifest['release_id']}",
        tags={
            "audience": config.audience,
            "release_id": manifest["release_id"],
            "git_commit": manifest["git_commit"],
            "dvc_dataset_hash": manifest["dvc_dataset_hash"],
            "metaflow_run_id": manifest["metaflow_run_id"],
            "environment_digest": manifest["environment_digest"],
            "evaluation_dataset_version": manifest["evaluation_dataset_version"],
            "status": status,
        },
    ) as run:
        mlflow.log_params(
            {
                "random_seed": config.random_seed,
                "chunk_method": config.chunk_method,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "embedding_model": config.embedding.model_id,
                "embedding_revision": config.embedding.revision,
                "retrieval_mode": config.retrieval_mode,
                "candidate_k": config.candidate_k,
                "top_k": config.top_k,
                "rrf_k": config.rrf_k,
                "rerank": config.rerank,
            }
        )
        mlflow.log_metrics(metrics.model_dump())
        mlflow.log_artifact(config.source_path, artifact_path="configuration")
        mlflow.log_artifact(evaluation_path, artifact_path="evaluation")
        final["mlflow_run_id"] = run.info.run_id

    config.release_dir.mkdir(parents=True, exist_ok=True)
    release_path = config.release_dir / f"{manifest['release_id']}.json"
    serialized = json.dumps(final, indent=2, sort_keys=True) + "\n"
    if release_path.exists() and release_path.read_text(encoding="utf-8") != serialized:
        raise ValueError(f"immutable release manifest conflict: {release_path}")
    release_path.write_text(serialized, encoding="utf-8")
    MlflowClient().log_artifact(final["mlflow_run_id"], str(release_path), artifact_path="release")
    if gate.passed:
        latest = config.release_dir.parent / "latest.json"
        latest.write_text(
            json.dumps(
                {"audience": config.audience, "release_id": manifest["release_id"]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return final


def select_release(project_root: Path, audience: str, release_id: str) -> Path:
    """Select an existing approved release without rebuilding its index."""

    audience_dir = project_root / "artifacts" / "rag" / audience
    release_path = audience_dir / "releases" / f"{release_id}.json"
    if not release_path.is_file():
        raise FileNotFoundError(f"RAG release does not exist: {release_path}")
    manifest = json.loads(release_path.read_text(encoding="utf-8"))
    if manifest.get("release_id") != release_id or manifest.get("audience") != audience:
        raise ValueError("release manifest identity does not match the requested audience and ID")
    if manifest.get("status") != "candidate":
        raise ValueError(f"release {release_id} did not pass its promotion gate")
    latest_path = audience_dir / "latest.json"
    latest_path.write_text(
        json.dumps({"audience": audience, "release_id": release_id}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return latest_path
