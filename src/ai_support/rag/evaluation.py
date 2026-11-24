"""Independent RAG benchmark evaluation and promotion gate."""

import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

from ai_support.rag.config import RAGConfig
from ai_support.rag.retrieval import HybridRetriever
from ai_support.rag.schema import BenchmarkCase, GateDecision, RAGMetrics


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    if not path.is_file():
        raise FileNotFoundError(f"RAG benchmark not found: {path}")
    cases = [
        BenchmarkCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("RAG benchmark is empty")
    return cases


def evaluate_retrieval(
    retriever: HybridRetriever, cases: list[BenchmarkCase], config: RAGConfig
) -> tuple[RAGMetrics, list[dict]]:
    rows: list[dict] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    citation_scores: list[float] = []
    groundedness_scores: list[float] = []
    latencies: list[float] = []
    unauthorized = 0
    opposite = "staff" if config.audience == "student" else "student"
    for case in cases:
        started = perf_counter()
        results = retriever.search(config.audience, case.question, case.department)
        latencies.append((perf_counter() - started) * 1000)
        retrieved_ids = [result.document_id for result in results]
        expected = set(case.expected_document_ids)
        relevant = [1 if identifier in expected else 0 for identifier in retrieved_ids]
        recalls.append(len(expected & set(retrieved_ids)) / len(expected))
        first = next((index for index, value in enumerate(relevant, start=1) if value), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        dcg = sum(value / math.log2(index + 1) for index, value in enumerate(relevant, start=1))
        ideal_length = min(len(expected), len(results))
        ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_length + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
        citation_scores.append(sum(relevant) / len(results) if results else 0.0)
        evidence = " ".join(result.content.lower() for result in results)
        groundedness_scores.append(
            sum(term.lower() in evidence for term in case.answer_terms) / len(case.answer_terms)
        )
        forbidden = retriever.search(opposite, case.question, case.department)  # type: ignore[arg-type]
        unauthorized += int(bool(forbidden))
        rows.append(
            {
                "question_id": case.question_id,
                "retrieved_document_ids": retrieved_ids,
                "citations": [result.citation for result in results],
                "recall": recalls[-1],
            }
        )
    metrics = RAGMetrics(
        recall_at_k=mean(recalls),
        mrr=mean(reciprocal_ranks),
        ndcg=mean(ndcgs),
        citation_correctness=mean(citation_scores),
        answer_groundedness=mean(groundedness_scores),
        retrieval_latency_ms=mean(latencies),
        unauthorized_cross_audience_rate=unauthorized / len(cases),
    )
    return metrics, rows


def apply_rag_gate(metrics: RAGMetrics, config: RAGConfig) -> GateDecision:
    failures: list[str] = []
    if metrics.recall_at_k < config.min_recall_at_k:
        failures.append("recall_at_k below configured minimum")
    if metrics.mrr < config.min_mrr:
        failures.append("mrr below configured minimum")
    if metrics.unauthorized_cross_audience_rate > config.max_unauthorized_rate:
        failures.append("unauthorized cross-audience retrieval exceeded maximum")
    return GateDecision(passed=not failures, failures=failures)


def write_evaluation(path: Path, metrics: RAGMetrics, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metrics": metrics.model_dump(), "cases": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
