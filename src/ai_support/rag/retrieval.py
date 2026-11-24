"""Lexical, dense, hybrid RRF, and optional reranked retrieval."""

from dataclasses import dataclass

from ai_support.rag.config import RAGConfig
from ai_support.rag.embeddings import Embedder, Reranker
from ai_support.rag.repository import RAGRepository
from ai_support.rag.schema import SearchResult
from ai_support.router.schema import Audience


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], rrf_k: int
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    items: dict[str, SearchResult] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            items[result.chunk_id] = result
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    fused = [
        item.model_copy(update={"score": scores[chunk_id]}) for chunk_id, item in items.items()
    ]
    return sorted(fused, key=lambda item: (-item.score, item.chunk_id))


@dataclass
class HybridRetriever:
    repository: RAGRepository
    embedder: Embedder
    config: RAGConfig
    release_id: str
    reranker: Reranker | None = None

    def search(
        self,
        audience: Audience,
        query: str,
        department: str | None = None,
    ) -> list[SearchResult]:
        if audience != self.config.audience:
            return []
        lexical: list[SearchResult] = []
        dense: list[SearchResult] = []
        if self.config.retrieval_mode in {"lexical", "hybrid"}:
            lexical = self.repository.lexical_search(
                self.release_id,
                audience,
                query,
                self.config.candidate_k,
                department,
            )
        if self.config.retrieval_mode in {"dense", "hybrid"}:
            dense = self.repository.dense_search(
                self.release_id,
                audience,
                self.embedder.embed_query(query),
                self.config.candidate_k,
                department,
            )
        if self.config.retrieval_mode == "lexical":
            results = lexical
        elif self.config.retrieval_mode == "dense":
            results = dense
        else:
            results = reciprocal_rank_fusion([lexical, dense], self.config.rrf_k)
        if self.reranker and results:
            scores = self.reranker.score(query, [result.content for result in results])
            results = [
                result.model_copy(update={"score": score})
                for result, score in zip(results, scores, strict=True)
            ]
            results.sort(key=lambda item: (-item.score, item.chunk_id))
        return results[: self.config.top_k]
