"""Sentence Transformers embedding and optional CrossEncoder adapters."""

from typing import Protocol

import numpy as np

from ai_support.rag.config import EmbeddingModelConfig


class Embedder(Protocol):
    model_id: str
    revision: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    model_id: str
    revision: str

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Lazy adapter that downloads a pinned checkpoint only when explicitly used."""

    def __init__(self, config: EmbeddingModelConfig, dimension: int) -> None:
        self.model_id = config.model_id
        self.revision = config.revision
        self.dimension = dimension
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_id,
                revision=self.revision,
                trust_remote_code=False,
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        values = self._load().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        result = np.asarray(values, dtype=float)
        if result.ndim != 2 or result.shape[1] != self.dimension:
            raise ValueError("embedding model returned an unexpected dimension")
        return result.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class CrossEncoderReranker:
    def __init__(self, model_id: str, revision: str) -> None:
        self.model_id = model_id
        self.revision = revision
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_id,
                revision=self.revision,
                trust_remote_code=False,
            )
        return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        values = self._load().predict([(query, document) for document in documents])
        return np.asarray(values, dtype=float).reshape(-1).tolist()


def attach_embeddings(chunks, embedder: Embedder):
    vectors = embedder.embed_documents([chunk.content for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("embedding count does not match chunk count")
    pairs = zip(chunks, vectors, strict=True)
    return [chunk.model_copy(update={"embedding": vector}) for chunk, vector in pairs]
