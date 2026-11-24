"""Deterministic RAG test doubles and factories."""

import re
from hashlib import sha256

import numpy as np


class FakeEmbedder:
    model_id = "test/hash-embedder"
    revision = "test-revision"
    dimension = 384

    @staticmethod
    def _embed(text: str) -> list[float]:
        vector = np.zeros(384, dtype=float)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int.from_bytes(sha256(token.encode()).digest()[:2], "big") % 384
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ReverseReranker:
    model_id = "test/reranker"
    revision = "1"

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float(index) for index in range(len(documents))]
