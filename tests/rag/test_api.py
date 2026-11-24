"""RAG API contract and configuration failure tests."""

from fastapi.testclient import TestClient

from ai_support.api.app import create_app
from ai_support.rag.schema import SearchResult


class FakeRAGService:
    def search(self, audience, query, department):
        return [
            SearchResult(
                chunk_id="chunk-1",
                document_id=f"{audience}-registry-001",
                audience=audience,
                department=department or "Registry",
                source=f"synthetic://{audience}/registry",
                content="Approved evidence",
                score=0.9,
                citation=f"synthetic://{audience}/registry#chunk-1",
            )
        ]


def test_rag_api_returns_cited_audience_scoped_results() -> None:
    app = create_app()
    app.state.rag_service = FakeRAGService()
    response = TestClient(app).post(
        "/v1/rag/student/search",
        json={"query": "Where is my timetable?", "department": "Registry"},
    )

    assert response.status_code == 200
    assert response.json()["audience"] == "student"
    assert response.json()["results"][0]["citation"].endswith("#chunk-1")


def test_rag_api_requires_database_configuration_and_valid_input() -> None:
    app = create_app()
    client = TestClient(app)
    assert client.post("/v1/rag/student/search", json={"query": "valid"}).status_code == 503
    assert client.post("/v1/rag/student/search", json={"query": ""}).status_code == 422
    assert client.post("/v1/rag/visitor/search", json={"query": "valid"}).status_code == 422
