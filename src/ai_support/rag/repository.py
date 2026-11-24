"""Permission-filtered in-memory and PostgreSQL/pgvector repositories."""

import re
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from ai_support.rag.schema import Chunk, SearchResult, SourceDocument
from ai_support.router.schema import Audience

_TOKEN = re.compile(r"[a-z0-9]+")


class RAGRepository(Protocol):
    def index_release(
        self,
        release_id: str,
        audience: Audience,
        documents: list[SourceDocument],
        chunks: list[Chunk],
        manifest: dict,
    ) -> None: ...

    def update_manifest(self, release_id: str, manifest: dict) -> None: ...

    def lexical_search(
        self,
        release_id: str,
        audience: Audience,
        query: str,
        limit: int,
        department: str | None = None,
    ) -> list[SearchResult]: ...

    def dense_search(
        self,
        release_id: str,
        audience: Audience,
        embedding: list[float],
        limit: int,
        department: str | None = None,
    ) -> list[SearchResult]: ...


def _result(chunk: Chunk, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        audience=chunk.audience,
        department=chunk.department,
        source=chunk.source,
        content=chunk.content,
        score=float(score),
        citation=f"{chunk.source}#{chunk.chunk_id}",
    )


class InMemoryRAGRepository:
    """Deterministic test and local experiment repository with policy filtering."""

    def __init__(self) -> None:
        self.releases: dict[str, tuple[Audience, list[Chunk]]] = {}
        self.manifests: dict[str, dict] = {}

    def index_release(
        self,
        release_id: str,
        audience: Audience,
        documents: list[SourceDocument],
        chunks: list[Chunk],
        manifest: dict,
    ) -> None:
        if release_id in self.releases:
            raise ValueError(f"immutable release already exists: {release_id}")
        if any(chunk.audience != audience or chunk.access_level != audience for chunk in chunks):
            raise ValueError("cross-audience chunk rejected during indexing")
        self.releases[release_id] = (audience, list(chunks))
        self.manifests[release_id] = dict(manifest)

    def update_manifest(self, release_id: str, manifest: dict) -> None:
        self.manifests[release_id] = dict(manifest)

    def _allowed(self, release_id: str, audience: Audience, department: str | None) -> list[Chunk]:
        release_audience, chunks = self.releases[release_id]
        if release_audience != audience:
            return []
        return [
            chunk
            for chunk in chunks
            if chunk.audience == audience
            and chunk.access_level == audience
            and (department is None or chunk.department == department)
        ]

    def lexical_search(
        self,
        release_id: str,
        audience: Audience,
        query: str,
        limit: int,
        department: str | None = None,
    ) -> list[SearchResult]:
        query_tokens = Counter(_TOKEN.findall(query.lower()))
        scored = []
        for chunk in self._allowed(release_id, audience, department):
            tokens = Counter(_TOKEN.findall(chunk.content.lower()))
            score = sum(min(count, tokens[token]) for token, count in query_tokens.items())
            if score:
                scored.append(_result(chunk, float(score)))
        return sorted(scored, key=lambda item: (-item.score, item.chunk_id))[:limit]

    def dense_search(
        self,
        release_id: str,
        audience: Audience,
        embedding: list[float],
        limit: int,
        department: str | None = None,
    ) -> list[SearchResult]:
        query = np.asarray(embedding, dtype=float)
        scored = []
        for chunk in self._allowed(release_id, audience, department):
            vector = np.asarray(chunk.embedding, dtype=float)
            denominator = np.linalg.norm(query) * np.linalg.norm(vector)
            score = float(np.dot(query, vector) / denominator) if denominator else 0.0
            scored.append(_result(chunk, score))
        return sorted(scored, key=lambda item: (-item.score, item.chunk_id))[:limit]


class PostgresRAGRepository:
    """PostgreSQL full-text and pgvector storage with query-level audience policy."""

    def __init__(self, database_url: str) -> None:
        if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("a PostgreSQL database URL is required")
        self.database_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)

    def initialize(self) -> None:
        migration = Path(__file__).with_name("sql") / "001_rag_schema.sql"
        with psycopg.connect(self.database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.commit()
            register_vector(connection)
            connection.execute(migration.read_text(encoding="utf-8"))

    def index_release(
        self,
        release_id: str,
        audience: Audience,
        documents: list[SourceDocument],
        chunks: list[Chunk],
        manifest: dict,
    ) -> None:
        with psycopg.connect(self.database_url) as connection:
            register_vector(connection)
            exists = connection.execute(
                "SELECT 1 FROM rag_releases WHERE release_id = %s", (release_id,)
            ).fetchone()
            if exists:
                raise ValueError(f"immutable release already exists: {release_id}")
            connection.execute(
                "INSERT INTO rag_releases (release_id, audience, manifest) VALUES (%s, %s, %s)",
                (release_id, audience, psycopg.types.json.Jsonb(manifest)),
            )
            connection.executemany(
                """
                INSERT INTO rag_documents (
                    release_id, document_id, audience, department, source, source_version,
                    content_hash, approval_status, review_date, effective_date, access_level
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        release_id,
                        doc.metadata.document_id,
                        doc.metadata.audience,
                        doc.metadata.department,
                        doc.metadata.source,
                        doc.metadata.source_version,
                        doc.metadata.content_hash,
                        doc.metadata.approval_status,
                        doc.metadata.review_date,
                        doc.metadata.effective_date,
                        doc.metadata.access_level,
                    )
                    for doc in documents
                ],
            )
            connection.executemany(
                """
                INSERT INTO rag_chunks (
                    release_id, chunk_id, document_id, audience, department, source,
                    source_version, access_level, ordinal, content, content_hash, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        release_id,
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.audience,
                        chunk.department,
                        chunk.source,
                        chunk.source_version,
                        chunk.access_level,
                        chunk.ordinal,
                        chunk.content,
                        chunk.content_hash,
                        Vector(chunk.embedding),
                    )
                    for chunk in chunks
                ],
            )

    def update_manifest(self, release_id: str, manifest: dict) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                "UPDATE rag_releases SET manifest = %s WHERE release_id = %s",
                (psycopg.types.json.Jsonb(manifest), release_id),
            )

    @staticmethod
    def _rows_to_results(rows) -> list[SearchResult]:
        return [
            SearchResult(
                **row,
                citation=f"{row['source']}#{row['chunk_id']}",
            )
            for row in rows
        ]

    def lexical_search(
        self,
        release_id: str,
        audience: Audience,
        query: str,
        limit: int,
        department: str | None = None,
    ) -> list[SearchResult]:
        sql = """
            SELECT chunk_id, document_id, audience, department, source, content,
                   ts_rank_cd(textsearch, plainto_tsquery('english', %(query)s)) AS score
              FROM rag_chunks
             WHERE release_id = %(release_id)s
               AND audience = %(audience)s AND access_level = %(audience)s
               AND textsearch @@ plainto_tsquery('english', %(query)s)
               AND (%(department)s IS NULL OR department = %(department)s)
             ORDER BY score DESC, chunk_id LIMIT %(limit)s
        """
        params = dict(
            query=query,
            release_id=release_id,
            audience=audience,
            department=department,
            limit=limit,
        )
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            rows = connection.execute(sql, params).fetchall()
        return self._rows_to_results(rows)

    def dense_search(
        self,
        release_id: str,
        audience: Audience,
        embedding: list[float],
        limit: int,
        department: str | None = None,
    ) -> list[SearchResult]:
        sql = """
            SELECT chunk_id, document_id, audience, department, source, content,
                   1 - (embedding <=> %(embedding)s) AS score
              FROM rag_chunks
             WHERE release_id = %(release_id)s
               AND audience = %(audience)s AND access_level = %(audience)s
               AND (%(department)s IS NULL OR department = %(department)s)
             ORDER BY embedding <=> %(embedding)s, chunk_id LIMIT %(limit)s
        """
        params = dict(
            embedding=Vector(embedding),
            release_id=release_id,
            audience=audience,
            department=department,
            limit=limit,
        )
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            register_vector(connection)
            rows = connection.execute(sql, params).fetchall()
        return self._rows_to_results(rows)
