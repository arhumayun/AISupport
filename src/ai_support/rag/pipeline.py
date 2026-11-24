"""Shared student and staff RAG pipeline stages."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_support.rag.chunking import chunk_documents
from ai_support.rag.config import RAGConfig, load_rag_config
from ai_support.rag.embeddings import (
    CrossEncoderReranker,
    Embedder,
    Reranker,
    SentenceTransformerEmbedder,
    attach_embeddings,
)
from ai_support.rag.evaluation import (
    apply_rag_gate,
    evaluate_retrieval,
    load_benchmark,
    write_evaluation,
)
from ai_support.rag.loaders import (
    load_unvalidated_descriptors,
    parse_documents,
    validate_descriptors,
)
from ai_support.rag.repository import InMemoryRAGRepository, PostgresRAGRepository, RAGRepository
from ai_support.rag.retrieval import HybridRetriever
from ai_support.rag.schema import Chunk, GateDecision, RAGMetrics, SourceDocument
from ai_support.rag.tracking import base_manifest, build_release_id, finalize_and_track
from ai_support.router.schema import Audience


@dataclass
class RAGPipelineResult:
    config: RAGConfig
    release_id: str
    documents: list[SourceDocument]
    chunks: list[Chunk]
    metrics: RAGMetrics
    gate: GateDecision
    manifest: dict[str, Any]


def config_path_for_audience(project_root: Path, audience: Audience) -> Path:
    return project_root / "configs" / "rag" / f"{audience}.toml"


def make_repository(database_url: str | None = None) -> RAGRepository:
    if database_url:
        repository = PostgresRAGRepository(database_url)
        repository.initialize()
        return repository
    return InMemoryRAGRepository()


def make_embedder(config: RAGConfig) -> Embedder:
    return SentenceTransformerEmbedder(config.embedding, config.embedding_dimension)


def make_reranker(config: RAGConfig) -> Reranker | None:
    if not config.rerank:
        return None
    return CrossEncoderReranker(config.reranker_model_id, config.reranker_revision)


def run_rag_pipeline(
    config_path: Path,
    project_root: Path,
    metaflow_run_id: str,
    repository: RAGRepository | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> RAGPipelineResult:
    config = load_rag_config(config_path, project_root)
    descriptors = load_unvalidated_descriptors(config)
    validated = validate_descriptors(descriptors, config)
    documents = parse_documents(validated)
    chunks = chunk_documents(documents, config)
    selected_embedder = embedder or make_embedder(config)
    embedded = attach_embeddings(chunks, selected_embedder)
    release_id = build_release_id(config, documents, embedded)
    selected_repository = repository or make_repository(os.getenv("AI_SUPPORT_DATABASE_URL"))
    index_backend = type(selected_repository).__name__
    manifest = base_manifest(
        config,
        release_id,
        documents,
        embedded,
        project_root,
        metaflow_run_id,
        index_backend,
    )
    selected_repository.index_release(release_id, config.audience, documents, embedded, manifest)
    retriever = HybridRetriever(
        selected_repository,
        selected_embedder,
        config,
        release_id,
        reranker if reranker is not None else make_reranker(config),
    )
    metrics, rows = evaluate_retrieval(retriever, load_benchmark(config.benchmark_path), config)
    gate = apply_rag_gate(metrics, config)
    evaluation_path = config.release_dir.parent / "evaluation.json"
    write_evaluation(evaluation_path, metrics, rows)
    final = finalize_and_track(config, manifest, metrics, gate, evaluation_path)
    selected_repository.update_manifest(release_id, final)
    return RAGPipelineResult(config, release_id, documents, embedded, metrics, gate, final)


def load_latest_release(project_root: Path, audience: Audience) -> str:
    path = project_root / "artifacts" / "rag" / audience / "latest.json"
    if not path.is_file():
        raise FileNotFoundError(f"approved RAG release not found: {path}")
    return str(json.loads(path.read_text(encoding="utf-8"))["release_id"])
