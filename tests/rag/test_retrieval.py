"""Hybrid retrieval, permission filtering, citations, and evaluation tests."""

from dataclasses import replace
from pathlib import Path

import pytest

from ai_support.rag.chunking import chunk_documents
from ai_support.rag.config import load_rag_config
from ai_support.rag.embeddings import attach_embeddings
from ai_support.rag.evaluation import apply_rag_gate, evaluate_retrieval, load_benchmark
from ai_support.rag.loaders import load_descriptors, parse_documents
from ai_support.rag.repository import InMemoryRAGRepository
from ai_support.rag.retrieval import HybridRetriever, reciprocal_rank_fusion
from ai_support.rag.sample_data import generate_sample_corpus

from .helpers import FakeEmbedder, ReverseReranker


def _indexed_retriever(tmp_path: Path, audience: str = "student"):
    config = load_rag_config(Path(f"configs/rag/{audience}.toml"), tmp_path)
    generate_sample_corpus(config)
    documents = parse_documents(load_descriptors(config))
    chunks = attach_embeddings(chunk_documents(documents, config), FakeEmbedder())
    repository = InMemoryRAGRepository()
    repository.index_release("release-1", audience, documents, chunks, {})
    return config, repository, HybridRetriever(repository, FakeEmbedder(), config, "release-1")


def test_hybrid_search_filters_audience_department_and_builds_citations(tmp_path: Path) -> None:
    config, repository, retriever = _indexed_retriever(tmp_path)

    results = retriever.search("student", "missing timetable enrolment", "Registry")

    assert results
    assert {result.department for result in results} == {"Registry"}
    assert {result.audience for result in results} == {"student"}
    assert all(result.citation.startswith("synthetic://student/") for result in results)
    assert retriever.search("staff", "missing timetable", "Registry") == []
    assert repository.lexical_search("release-1", "staff", "timetable", 5) == []
    assert repository.dense_search("release-1", "staff", FakeEmbedder().embed_query("x"), 5) == []


def test_index_rejects_cross_audience_and_is_immutable(tmp_path: Path) -> None:
    config, repository, _ = _indexed_retriever(tmp_path)
    documents = parse_documents(load_descriptors(config))
    chunks = attach_embeddings(chunk_documents(documents, config), FakeEmbedder())
    with pytest.raises(ValueError, match="already exists"):
        repository.index_release("release-1", "student", documents, chunks, {})
    bad = [chunks[0].model_copy(update={"audience": "staff"})]
    with pytest.raises(ValueError, match="cross-audience"):
        repository.index_release("release-2", "student", documents, bad, {})


def test_rrf_and_optional_reranker_change_order(tmp_path: Path) -> None:
    config, repository, retriever = _indexed_retriever(tmp_path)
    lexical = repository.lexical_search("release-1", "student", "portal", 5)
    dense = repository.dense_search("release-1", "student", FakeEmbedder().embed_query("portal"), 5)
    fused = reciprocal_rank_fusion([lexical, dense], 60)
    assert fused
    reranked = HybridRetriever(
        repository,
        FakeEmbedder(),
        replace(config, top_k=5),
        "release-1",
        ReverseReranker(),
    ).search("student", "portal")
    assert reranked[0].score >= reranked[-1].score


def test_benchmark_metrics_and_gate_include_leakage_check(tmp_path: Path) -> None:
    config, _, retriever = _indexed_retriever(tmp_path)
    metrics, rows = evaluate_retrieval(retriever, load_benchmark(config.benchmark_path), config)
    gate = apply_rag_gate(metrics, config)

    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.ndcg == 1.0
    assert metrics.citation_correctness == 1.0
    assert metrics.answer_groundedness == 1.0
    assert metrics.unauthorized_cross_audience_rate == 0.0
    assert metrics.retrieval_latency_ms >= 0.0
    assert gate.passed
    assert len(rows) == 5

    failed = apply_rag_gate(metrics.model_copy(update={"recall_at_k": 0.0}), config)
    assert not failed.passed
