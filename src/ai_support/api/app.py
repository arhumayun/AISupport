"""FastAPI application entry point."""

from fastapi import FastAPI

from ai_support import __version__
from ai_support.api.models import HealthResponse
from ai_support.logging import configure_logging

SERVICE_NAME = "uowd-ai-support-api"


def create_app() -> FastAPI:
    """Create the API application without starting external services."""

    configure_logging()
    application = FastAPI(
        title="UOWD AI Support Assistant API",
        description="Phase 1 foundation service for the UOWD AI Support Assistant.",
        version=__version__,
    )

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:
        """Report that the API process is ready to accept requests."""

        return HealthResponse(service=SERVICE_NAME, version=__version__)

    return application


app = create_app()
