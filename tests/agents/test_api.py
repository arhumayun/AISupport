"""Authenticated chat, confirmation, and feedback API tests."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from ai_support.agents.actions import FeedbackStore, JsonlAuditStore, PendingActionStore
from ai_support.agents.profiles import SyntheticProfileStore
from ai_support.agents.service import ChatService
from ai_support.agents.workflow import AgentDependencies, AgentService
from ai_support.api.app import create_app
from ai_support.security.auth import AuthService

from .helpers import FakeEmailSender, FakeKnowledge, FakeLLM, FakeRouter


def _client(tmp_path: Path):
    profiles = SyntheticProfileStore(
        tmp_path / "profiles.db", Path("data/app/synthetic_profiles.json")
    )
    profiles.initialize()
    auth = AuthService(profiles, "a-test-secret-that-is-long-enough", "demo-code")
    actions = PendingActionStore()
    audit = JsonlAuditStore(tmp_path / "audit.jsonl")
    sender = FakeEmailSender([])
    dependencies = AgentDependencies(
        profiles=profiles,
        router=FakeRouter(),
        knowledge=FakeKnowledge(),
        llm=FakeLLM(),
        actions=actions,
        gap_store=audit,
    )
    chat = ChatService(AgentService(dependencies), actions, sender, FeedbackStore(audit), audit)
    app = create_app()
    app.state.auth_service = auth
    app.state.chat_service = chat
    return TestClient(app), sender


def _login(client: TestClient, user_id: str = "S100001") -> str:
    response = client.post("/v1/auth/login", json={"user_id": user_id, "login_code": "demo-code"})
    assert response.status_code == 200
    return response.json()["access_token"]


def test_login_and_chat_bind_audience_to_bearer_session(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    token = _login(client)
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post("/v1/chat", json={"message": "Show my profile"}, headers=headers)

    assert response.status_code == 200
    assert response.json()["audience"] == "student"
    assert "Amina Rahman" in response.json()["answer"]
    denied = client.post(
        "/v1/router/staff/predict",
        json={"ticket_subject": "Help", "ticket_description": "Help"},
        headers=headers,
    )
    assert denied.status_code == 403
    assert client.post("/v1/chat", json={"message": "Hello"}).status_code == 401
    assert (
        client.post(
            "/v1/auth/login", json={"user_id": "S100001", "login_code": "wrong"}
        ).status_code
        == 401
    )


def test_department_email_requires_explicit_confirmation(tmp_path: Path) -> None:
    client, sender = _client(tmp_path)
    headers = {"Authorization": f"Bearer {_login(client)}"}
    draft_response = client.post(
        "/v1/chat",
        json={"message": "Please email the department for clarification"},
        headers=headers,
    )
    action_id = draft_response.json()["action"]["action_id"]

    assert sender.sent == []
    sent = client.post(f"/v1/actions/{action_id}/confirm", headers=headers)
    assert sent.status_code == 200
    assert sent.json()["delivered"] is True
    assert len(sender.sent) == 1
    assert client.post(f"/v1/actions/{action_id}/confirm", headers=headers).status_code == 404


def test_feedback_is_limited_to_session_owned_messages(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    student_headers = {"Authorization": f"Bearer {_login(client)}"}
    message = client.post(
        "/v1/chat", json={"message": "Show my profile"}, headers=student_headers
    ).json()
    recorded = client.post(
        "/v1/feedback",
        json={"message_id": message["message_id"], "helpful": True},
        headers=student_headers,
    )

    assert recorded.status_code == 200
    rows = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert any(row["event"] == "answer_feedback" for row in rows)

    staff_headers = {"Authorization": f"Bearer {_login(client, 'E100001')}"}
    denied = client.post(
        "/v1/feedback",
        json={"message_id": message["message_id"], "helpful": False},
        headers=staff_headers,
    )
    assert denied.status_code == 404
