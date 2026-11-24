"""API health and OpenAPI contract tests."""

from fastapi.testclient import TestClient

from ai_support.api.app import SERVICE_NAME, app

client = TestClient(app)


def test_health_returns_typed_service_metadata() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": "0.1.0",
    }


def test_health_contract_is_in_openapi_schema() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/health" in response.json()["paths"]
