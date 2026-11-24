"""Document, chunk, retrieval, benchmark, and release contracts."""

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_support.router.schema import Audience


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_id: str = Field(min_length=1)
    audience: Audience
    department: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    approval_status: Literal["approved", "draft", "retired"]
    review_date: date
    effective_date: date
    access_level: Literal["student", "staff"]


class SourceDocument(BaseModel):
    metadata: DocumentMetadata
    path: Path
    text: str = ""


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    audience: Audience
    department: str
    source: str
    source_version: str
    access_level: Literal["student", "staff"]
    ordinal: int
    content: str
    content_hash: str
    embedding: list[float] = Field(default_factory=list)


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    audience: Audience
    department: str
    source: str
    content: str
    score: float
    citation: str


class BenchmarkCase(BaseModel):
    question_id: str
    question: str
    expected_document_ids: list[str] = Field(min_length=1)
    answer_terms: list[str] = Field(min_length=1)
    department: str | None = None


class RAGMetrics(BaseModel):
    recall_at_k: float
    mrr: float
    ndcg: float
    citation_correctness: float
    answer_groundedness: float
    retrieval_latency_ms: float
    unauthorized_cross_audience_rate: float


class GateDecision(BaseModel):
    passed: bool
    failures: list[str]
