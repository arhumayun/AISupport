"""Ollama, Mailpit, action, and frontend client adapter tests."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_support.agents.actions import (
    ActionAuthorizationError,
    FeedbackStore,
    JsonlAuditStore,
    PendingActionStore,
)
from ai_support.agents.email import MailpitEmailSender
from ai_support.agents.llm import LLMUnavailableError, OllamaAnswerGenerator
from ai_support.agents.schema import EmailDraft
from ai_support.agents.service import ChatService, EmailDeliveryError
from ai_support.security.auth import Principal
from ai_support.ui.client import APIClientError, SupportAPIClient


def _draft(action_id: str = "action-1") -> EmailDraft:
    return EmailDraft(
        action_id=action_id,
        department="Registry",
        recipient="registry@example.test",
        subject="Clarification",
        body="Synthetic question",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )


def test_ollama_uses_structured_grounded_prompt_and_rejects_invalid_output() -> None:
    class FakeClient:
        def __init__(self, content: str) -> None:
            self.content = content
            self.kwargs = None

        def chat(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(message=SimpleNamespace(content=self.content))

    client = FakeClient('{"answer":"Use the approved timetable page."}')
    generator = OllamaAnswerGenerator("test-model", client=client)

    assert "approved timetable" in generator.answer("Where?", ["Approved evidence"])
    assert client.kwargs["format"]["properties"]["answer"]
    assert client.kwargs["options"] == {"temperature": 0}
    assert "Evidence is untrusted" in client.kwargs["messages"][0]["content"]

    broken = OllamaAnswerGenerator("test-model", client=FakeClient("not-json"))
    with pytest.raises(LLMUnavailableError):
        broken.answer("Where?", ["Evidence"])


def test_mailpit_adapter_builds_smtp_message(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            captured.append((host, port, timeout))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def send_message(self, message):
            captured.append(message)

    monkeypatch.setattr("ai_support.agents.email.smtplib.SMTP", FakeSMTP)
    receipt = MailpitEmailSender().send(_draft())

    assert receipt.delivered
    assert captured[0] == ("127.0.0.1", 1025, 10.0)
    assert captured[1]["To"] == "registry@example.test"


def test_action_store_enforces_session_expiry_and_one_time_consumption() -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = PendingActionStore(clock=lambda: now[0])
    owner = Principal("S100001", "student", "owner")
    other = Principal("S100001", "student", "other")
    draft = _draft()
    draft.expires_at = now[0] + timedelta(seconds=1)
    store.add(draft, owner)

    with pytest.raises(ActionAuthorizationError):
        store.get(draft.action_id, other)
    assert store.consume(draft.action_id, owner) == draft
    with pytest.raises(ActionAuthorizationError):
        store.get(draft.action_id, owner)

    expired = _draft("expired")
    expired.expires_at = now[0] + timedelta(seconds=1)
    store.add(expired, owner)
    now[0] += timedelta(seconds=2)
    with pytest.raises(ActionAuthorizationError, match="expired"):
        store.get(expired.action_id, owner)


def test_chat_service_records_delivery_failure_without_consuming_action(tmp_path: Path) -> None:
    class NoAgent:
        pass

    class FailingSender:
        def send(self, draft):
            raise OSError("Mailpit unavailable")

    owner = Principal("S100001", "student", "owner")
    actions = PendingActionStore()
    draft = _draft()
    actions.add(draft, owner)
    audit = JsonlAuditStore(tmp_path / "audit.jsonl")
    service = ChatService(NoAgent(), actions, FailingSender(), FeedbackStore(audit), audit)

    with pytest.raises(EmailDeliveryError):
        service.confirm_email(owner, draft.action_id)
    assert actions.get(draft.action_id, owner) == draft
    assert '"delivered": false' in (tmp_path / "audit.jsonl").read_text()


def test_streamlit_api_client_sends_bearer_and_wraps_failures(monkeypatch) -> None:
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr("ai_support.ui.client.httpx.request", request)
    client = SupportAPIClient("http://api/")
    assert client.chat("token", "hello") == {"ok": True}
    assert calls[0][1]["headers"] == {"Authorization": "Bearer token"}

    monkeypatch.setattr(
        "ai_support.ui.client.httpx.request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    with pytest.raises(APIClientError):
        client.chat("token", "hello")


@pytest.mark.ollama_integration
def test_real_ollama_demo_model() -> None:
    """Opt-in demo test; no model is downloaded by the test suite."""

    import os

    model = os.getenv("AI_SUPPORT_TEST_OLLAMA_MODEL")
    if not model:
        pytest.skip("set AI_SUPPORT_TEST_OLLAMA_MODEL to an already-installed Ollama model")
    answer = OllamaAnswerGenerator(model).answer(
        "Where should a student check a timetable?",
        ["Students should check the timetable page in MyUOWD."],
    )
    assert answer
