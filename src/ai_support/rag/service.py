"""Runtime search service loading immutable audience-specific releases."""

from pathlib import Path

from ai_support.rag.config import load_rag_config
from ai_support.rag.pipeline import config_path_for_audience, load_latest_release, make_embedder
from ai_support.rag.repository import PostgresRAGRepository
from ai_support.rag.retrieval import HybridRetriever
from ai_support.rag.schema import SearchResult
from ai_support.router.schema import Audience


class RAGService:
    def __init__(self, project_root: Path, database_url: str) -> None:
        self.project_root = project_root.resolve()
        self.database_url = database_url
        self._retrievers: dict[Audience, HybridRetriever] = {}

    def search(
        self,
        audience: Audience,
        query: str,
        department: str | None = None,
    ) -> list[SearchResult]:
        if audience not in self._retrievers:
            config = load_rag_config(
                config_path_for_audience(self.project_root, audience), self.project_root
            )
            repository = PostgresRAGRepository(self.database_url)
            release_id = load_latest_release(self.project_root, audience)
            self._retrievers[audience] = HybridRetriever(
                repository, make_embedder(config), config, release_id
            )
        return self._retrievers[audience].search(audience, query, department)
