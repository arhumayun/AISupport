"""Tests for model adapters and the PostgreSQL/pgvector boundary."""

from pathlib import Path

import numpy as np
import pytest

from ai_support.rag.chunking import chunk_documents
from ai_support.rag.config import EmbeddingModelConfig, load_rag_config
from ai_support.rag.embeddings import (
    CrossEncoderReranker,
    SentenceTransformerEmbedder,
    attach_embeddings,
)
from ai_support.rag.loaders import load_descriptors, parse_documents
from ai_support.rag.repository import PostgresRAGRepository
from ai_support.rag.sample_data import generate_sample_corpus

from .helpers import FakeEmbedder


class _EncoderModel:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def encode(self, texts, **_kwargs):
        return np.ones((len(texts), self.dimension), dtype=float)


class _CrossEncoderModel:
    def predict(self, pairs):
        return np.arange(len(pairs), dtype=float)


def test_sentence_transformer_and_cross_encoder_adapters_validate_outputs() -> None:
    config = EmbeddingModelConfig("test/embedding", "immutable-revision")
    embedder = SentenceTransformerEmbedder(config, dimension=3)
    embedder._model = _EncoderModel(3)

    assert embedder.embed_documents(["one", "two"]) == [[1.0, 1.0, 1.0]] * 2
    assert embedder.embed_query("one") == [1.0, 1.0, 1.0]

    embedder._model = _EncoderModel(2)
    with pytest.raises(ValueError, match="unexpected dimension"):
        embedder.embed_documents(["wrong"])

    reranker = CrossEncoderReranker("test/reranker", "immutable-revision")
    reranker._model = _CrossEncoderModel()
    assert reranker.score("query", ["first", "second"]) == [0.0, 1.0]


def test_attach_embeddings_rejects_count_mismatch(tmp_path: Path) -> None:
    config = load_rag_config(Path("configs/rag/student.toml"), tmp_path)
    generate_sample_corpus(config)
    documents = parse_documents(load_descriptors(config))
    chunks = chunk_documents(documents, config)

    class BrokenEmbedder(FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return []

    with pytest.raises(ValueError, match="count does not match"):
        attach_embeddings(chunks, BrokenEmbedder())


class _Cursor:
    def __init__(self, rows=None, one=None, statements=None) -> None:
        self.rows = rows or []
        self.one = one
        self.statements = statements

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.one

    def executemany(self, sql, params) -> None:
        assert self.statements is not None
        self.statements.append((str(sql), list(params)))


class _Connection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def commit(self) -> None:
        return None

    def execute(self, sql, params=None):
        self.statements.append((str(sql), params))
        if "SELECT chunk_id" in str(sql):
            return _Cursor(
                rows=[
                    {
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "audience": "student",
                        "department": "Registry",
                        "source": "synthetic://student/registry",
                        "content": "Timetable guidance",
                        "score": 0.75,
                    }
                ]
            )
        return _Cursor()

    def cursor(self):
        return _Cursor(statements=self.statements)


def test_postgres_repository_enforces_policy_in_queries_and_uses_pgvector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="PostgreSQL"):
        PostgresRAGRepository("sqlite:///not-supported.db")

    config = load_rag_config(Path("configs/rag/student.toml"), tmp_path)
    generate_sample_corpus(config)
    documents = parse_documents(load_descriptors(config))
    chunks = attach_embeddings(chunk_documents(documents, config), FakeEmbedder())
    connection = _Connection()
    monkeypatch.setattr("ai_support.rag.repository.psycopg.connect", lambda *_a, **_k: connection)
    monkeypatch.setattr("ai_support.rag.repository.register_vector", lambda _connection: None)

    repository = PostgresRAGRepository("postgresql+psycopg://localhost/ai_support")
    repository.initialize()
    repository.index_release("release-1", "student", documents, chunks, {"status": "evaluating"})
    repository.update_manifest("release-1", {"status": "candidate"})
    lexical = repository.lexical_search("release-1", "student", "timetable", 3, "Registry")
    dense = repository.dense_search(
        "release-1", "student", FakeEmbedder().embed_query("timetable"), 3, "Registry"
    )

    assert repository.database_url.startswith("postgresql://")
    assert lexical[0].citation == "synthetic://student/registry#chunk-1"
    assert dense[0].score == 0.75
    combined_sql = "\n".join(statement for statement, _ in connection.statements)
    assert "audience = %(audience)s AND access_level = %(audience)s" in combined_sql
    assert "CAST(%(department)s AS text) IS NULL" in combined_sql
    assert "plainto_tsquery" in combined_sql
    assert "embedding <=>" in combined_sql
    assert "INSERT INTO rag_documents" in combined_sql
    assert "INSERT INTO rag_chunks" in combined_sql


def test_postgres_repository_rejects_duplicate_release(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _Connection()

    def execute_with_existing(sql, params=None):
        if "SELECT 1 FROM rag_releases" in str(sql):
            return _Cursor(one=(1,))
        return _Connection.execute(connection, sql, params)

    connection.execute = execute_with_existing  # type: ignore[method-assign]
    monkeypatch.setattr("ai_support.rag.repository.psycopg.connect", lambda *_a, **_k: connection)
    monkeypatch.setattr("ai_support.rag.repository.register_vector", lambda _connection: None)
    repository = PostgresRAGRepository("postgresql://localhost/ai_support")

    with pytest.raises(ValueError, match="already exists"):
        repository.index_release("release-1", "student", [], [], {})
