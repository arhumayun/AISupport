"""Pydantic contracts exposed by the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health status returned to callers and deployment probes."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    service: str
    version: str


class RouterPredictionRequest(BaseModel):
    """Text fields accepted by the internal router inference endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticket_subject: str = Field(min_length=1, max_length=300)
    ticket_description: str = Field(min_length=1, max_length=5000)


class DepartmentProbability(BaseModel):
    department: str
    probability: float = Field(ge=0.0, le=1.0)


class RouterPredictionResponse(BaseModel):
    audience: Literal["student", "staff"]
    department: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_departments: list[DepartmentProbability]
    model_name: str
    model_reference: str


class RAGSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=3000)
    department: str | None = Field(default=None, min_length=1, max_length=100)


class RAGSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    department: str
    source: str
    content: str
    score: float
    citation: str


class RAGSearchResponse(BaseModel):
    audience: Literal["student", "staff"]
    results: list[RAGSearchResult]


class LoginRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=30)
    login_code: str = Field(min_length=1, max_length=200)


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    audience: Literal["student", "staff"]
    user_id: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=3000)


class FeedbackRequest(BaseModel):
    message_id: str = Field(min_length=1, max_length=100)
    helpful: bool


class FeedbackResponse(BaseModel):
    recorded: Literal[True] = True
