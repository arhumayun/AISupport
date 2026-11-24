from fastapi.testclient import TestClient

from ai_support.api.app import app
from ai_support.monitoring.metrics import (
    EMAIL_FAILURES,
    KNOWLEDGE_GAPS,
    TOOL_FAILURES,
)


def test_metrics_endpoint_exposes_runtime_telemetry() -> None:
    client = TestClient(app)
    TOOL_FAILURES.labels(tool="test_tool", reason="test").inc()
    EMAIL_FAILURES.labels(department="IT").inc()
    KNOWLEDGE_GAPS.labels(audience="student", department="IT", reason="test").inc()

    assert client.get("/health").status_code == 200
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "ai_support_http_requests_total" in response.text
    assert 'tool="test_tool"' in response.text
    assert "ai_support_email_delivery_failures_total" in response.text
    assert "ai_support_knowledge_gaps_total" in response.text
