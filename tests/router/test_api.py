"""Router prediction API contract and separation tests."""

from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

from ai_support.api.app import create_app
from ai_support.router.service import DepartmentScore, RouterPrediction
from ai_support.security.auth import Principal


class FakeAuthService:
    def verify(self, token: str) -> Principal:
        return Principal("S100001" if token == "student" else "E100001", token, f"{token}-sid")


class FakeRouterService:
    def __init__(self) -> None:
        self.audiences: list[str] = []

    def predict(self, audience, subject, description):
        self.audiences.append(audience)
        department = "Registry" if audience == "student" else "HR"
        return RouterPrediction(
            audience=audience,
            department=department,
            confidence=0.91,
            top_departments=(DepartmentScore(department, 0.91),),
            model_name=f"{audience}-department-router",
            model_reference="champion",
        )


def test_api_keeps_student_and_staff_model_selection_separate() -> None:
    app = create_app()
    fake = FakeRouterService()
    app.state.router_service = fake
    app.state.auth_service = FakeAuthService()
    client = TestClient(app)
    payload = {"ticket_subject": "Help", "ticket_description": "Please assist"}

    student = client.post(
        "/v1/router/student/predict", json=payload, headers={"Authorization": "Bearer student"}
    )
    staff = client.post(
        "/v1/router/staff/predict", json=payload, headers={"Authorization": "Bearer staff"}
    )

    assert student.status_code == 200
    assert student.json()["department"] == "Registry"
    assert staff.json()["department"] == "HR"
    assert fake.audiences == ["student", "staff"]
    assert (
        client.post(
            "/v1/router/visitor/predict",
            json=payload,
            headers={"Authorization": "Bearer student"},
        ).status_code
        == 422
    )
    denied = client.post(
        "/v1/router/staff/predict", json=payload, headers={"Authorization": "Bearer student"}
    )
    assert denied.status_code == 403


def test_api_rejects_malformed_input() -> None:
    app = create_app()
    app.state.auth_service = FakeAuthService()
    client = TestClient(app)
    response = client.post(
        "/v1/router/student/predict",
        json={"ticket_subject": "", "ticket_description": "valid"},
        headers={"Authorization": "Bearer student"},
    )
    assert response.status_code == 422


def test_api_returns_503_when_registry_model_is_unavailable() -> None:
    class UnavailableService:
        def predict(self, audience, subject, description):
            raise MlflowException("not found")

    app = create_app()
    app.state.router_service = UnavailableService()
    app.state.auth_service = FakeAuthService()
    response = TestClient(app).post(
        "/v1/router/student/predict",
        json={"ticket_subject": "Help", "ticket_description": "Please assist"},
        headers={"Authorization": "Bearer student"},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "approved router model unavailable"
