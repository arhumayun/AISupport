"""FastAPI application entry point."""

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from mlflow.exceptions import MlflowException
from psycopg import Error as PsycopgError

from ai_support import __version__
from ai_support.api.models import (
    DepartmentProbability,
    HealthResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    RAGSearchResult,
    RouterPredictionRequest,
    RouterPredictionResponse,
)
from ai_support.logging import configure_logging
from ai_support.rag.service import RAGService
from ai_support.router.service import RouterService

SERVICE_NAME = "uowd-ai-support-api"


def create_app() -> FastAPI:
    """Create the API application without starting external services."""

    configure_logging()
    application = FastAPI(
        title="UOWD AI Support Assistant API",
        description="Department routing and permission-filtered retrieval for UOWD support.",
        version=__version__,
    )
    application.state.router_service = RouterService(
        Path.cwd(), os.getenv("AI_SUPPORT_ROUTER_MODEL_REFERENCE", "champion")
    )
    database_url = os.getenv("AI_SUPPORT_DATABASE_URL")
    application.state.rag_service = RAGService(Path.cwd(), database_url) if database_url else None

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:
        """Report that the API process is ready to accept requests."""

        return HealthResponse(service=SERVICE_NAME, version=__version__)

    @application.post(
        "/v1/router/{audience}/predict",
        response_model=RouterPredictionResponse,
        tags=["router"],
    )
    async def predict_department(
        audience: Literal["student", "staff"],
        payload: RouterPredictionRequest,
        request: Request,
    ) -> RouterPredictionResponse:
        """Predict a department using only the selected audience's registered model."""

        try:
            prediction = request.app.state.router_service.predict(
                audience,
                payload.ticket_subject,
                payload.ticket_description,
            )
        except (MlflowException, OSError) as error:
            raise HTTPException(
                status_code=503, detail="approved router model unavailable"
            ) from error
        return RouterPredictionResponse(
            audience=prediction.audience,
            department=prediction.department,
            confidence=prediction.confidence,
            top_departments=[
                DepartmentProbability(department=score.department, probability=score.probability)
                for score in prediction.top_departments
            ],
            model_name=prediction.model_name,
            model_reference=prediction.model_reference,
        )

    @application.post(
        "/v1/rag/{audience}/search",
        response_model=RAGSearchResponse,
        tags=["rag"],
    )
    async def search_knowledge(
        audience: Literal["student", "staff"],
        payload: RAGSearchRequest,
        request: Request,
    ) -> RAGSearchResponse:
        """Search only the selected audience's approved immutable release."""

        service = request.app.state.rag_service
        if service is None:
            raise HTTPException(status_code=503, detail="RAG database is not configured")
        try:
            results = service.search(audience, payload.query, payload.department)
        except (FileNotFoundError, PsycopgError) as error:
            raise HTTPException(
                status_code=503, detail="approved RAG release unavailable"
            ) from error
        return RAGSearchResponse(
            audience=audience,
            results=[
                RAGSearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    department=result.department,
                    source=result.source,
                    content=result.content,
                    score=result.score,
                    citation=result.citation,
                )
                for result in results
            ],
        )

    return application


app = create_app()
