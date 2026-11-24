"""Typed RAG experiment and query-time configuration."""

import os
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from ai_support.router.schema import Audience

ChunkMethod = Literal["fixed", "recursive", "heading"]
RetrievalMode = Literal["lexical", "dense", "hybrid"]


@dataclass(frozen=True)
class EmbeddingModelConfig:
    model_id: str
    revision: str


@dataclass(frozen=True)
class RAGConfig:
    audience: Audience
    random_seed: int
    as_of_date: date
    source_dir: Path
    benchmark_path: Path
    release_dir: Path
    experiment_name: str
    chunk_method: ChunkMethod
    chunk_size: int
    chunk_overlap: int
    embedding_profile: str
    embedding_dimension: int
    embedding_models: dict[str, EmbeddingModelConfig]
    retrieval_mode: RetrievalMode
    candidate_k: int
    top_k: int
    rrf_k: int
    rerank: bool
    reranker_model_id: str
    reranker_revision: str
    min_recall_at_k: float
    min_mrr: float
    max_unauthorized_rate: float
    tracking_uri: str
    artifact_root: Path
    source_path: Path

    @property
    def embedding(self) -> EmbeddingModelConfig:
        return self.embedding_models[self.embedding_profile]


def load_rag_config(path: Path, project_root: Path | None = None) -> RAGConfig:
    """Load a version-controlled audience-specific RAG configuration."""

    source = path.resolve()
    root = (project_root or Path.cwd()).resolve()
    with source.open("rb") as stream:
        raw: dict[str, Any] = tomllib.load(stream)
    rag = raw["rag"]
    chunking = raw["chunking"]
    embedding = raw["embedding"]
    retrieval = raw["retrieval"]
    promotion = raw["promotion"]
    tracking = raw["mlflow"]
    audience = rag["audience"]
    if audience not in {"student", "staff"}:
        raise ValueError("rag.audience must be student or staff")
    method = chunking["method"]
    if method not in {"fixed", "recursive", "heading"}:
        raise ValueError("unsupported chunking method")
    mode = retrieval["mode"]
    if mode not in {"lexical", "dense", "hybrid"}:
        raise ValueError("unsupported retrieval mode")
    models = {name: EmbeddingModelConfig(**values) for name, values in embedding["models"].items()}
    profile = embedding["profile"]
    if profile not in models:
        raise ValueError("embedding profile does not exist")
    size = int(chunking["chunk_size"])
    overlap = int(chunking["overlap"])
    if size < 20 or not 0 <= overlap < size:
        raise ValueError("chunk size must be >= 20 and overlap must be smaller")
    return RAGConfig(
        audience=audience,
        random_seed=int(rag["random_seed"]),
        as_of_date=date.fromisoformat(rag["as_of_date"]),
        source_dir=(root / rag["source_dir"]).resolve(),
        benchmark_path=(root / rag["benchmark_path"]).resolve(),
        release_dir=(root / rag["release_dir"]).resolve(),
        experiment_name=rag["experiment_name"],
        chunk_method=method,
        chunk_size=size,
        chunk_overlap=overlap,
        embedding_profile=profile,
        embedding_dimension=int(embedding["dimension"]),
        embedding_models=models,
        retrieval_mode=mode,
        candidate_k=int(retrieval["candidate_k"]),
        top_k=int(retrieval["top_k"]),
        rrf_k=int(retrieval["rrf_k"]),
        rerank=bool(retrieval["rerank"]),
        reranker_model_id=retrieval["reranker_model_id"],
        reranker_revision=retrieval["reranker_revision"],
        min_recall_at_k=float(promotion["min_recall_at_k"]),
        min_mrr=float(promotion["min_mrr"]),
        max_unauthorized_rate=float(promotion["max_unauthorized_rate"]),
        tracking_uri=os.getenv("AI_SUPPORT_MLFLOW_TRACKING_URI", tracking["tracking_uri"]),
        artifact_root=(root / tracking["artifact_root"]).resolve(),
        source_path=source,
    )
