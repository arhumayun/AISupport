import io
import json
import urllib.error
from typing import Any

import pytest

from ai_support.deployment.verify import main, request_json, verify_deployment


def successful_requester(
    _base_url: str,
    _method: str,
    path: str,
    _payload: dict[str, Any] | None,
    _token: str | None,
) -> dict[str, Any]:
    responses = {
        "/health": {"status": "ok"},
        "/v1/auth/login": {"access_token": "token", "audience": "student"},
        "/v1/router/student/predict": {
            "department": "Registry",
            "model_reference": "champion",
        },
        "/v1/rag/student/search": {"results": [{"citation": "student-registry-001"}]},
        "/v1/chat": {"intent": "knowledge", "citations": [{"document_id": "doc"}]},
    }
    return responses[path]


def test_verify_deployment_covers_authenticated_vertical_slice() -> None:
    result = verify_deployment("http://api:8000", "code", successful_requester)

    assert result == {
        "status": "passed",
        "audience": "student",
        "department": "Registry",
        "model_reference": "champion",
        "rag_results": 1,
        "chat_citations": 1,
    }


def test_verify_deployment_rejects_uncited_rag() -> None:
    def requester(base_url, method, path, payload, token):
        result = successful_requester(base_url, method, path, payload, token)
        return {"results": []} if path == "/v1/rag/student/search" else result

    with pytest.raises(RuntimeError, match="no cited"):
        verify_deployment("http://api:8000", "code", requester)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        ("/health", {"status": "failed"}, "health"),
        ("/v1/auth/login", {"audience": "staff"}, "authentication"),
        ("/v1/router/student/predict", {"department": "IT"}, "router"),
        ("/v1/chat", {"intent": "handoff", "citations": []}, "chatbot"),
    ],
)
def test_verify_deployment_rejects_invalid_stage(path, replacement, message) -> None:
    def requester(base_url, method, requested_path, payload, token):
        if requested_path == path:
            return replacement
        return successful_requester(base_url, method, requested_path, payload, token)

    with pytest.raises(RuntimeError, match=message):
        verify_deployment("http://api:8000", "code", requester)


def test_request_json_builds_authenticated_request(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"ok": True}).encode()

    captured = {}

    def open_request(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("ai_support.deployment.verify.urllib.request.urlopen", open_request)

    result = request_json("http://api:8000/", "POST", "/check", {"value": 1}, "token")

    assert result == {"ok": True}
    assert captured["request"].full_url == "http://api:8000/check"
    assert captured["request"].headers["Authorization"] == "Bearer token"
    assert captured["timeout"] == 120


def test_request_json_reports_http_and_connection_errors(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "http://api:8000/check", 503, "failed", {}, io.BytesIO(b'{"detail":"down"}')
    )
    monkeypatch.setattr(
        "ai_support.deployment.verify.urllib.request.urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(error),
    )
    with pytest.raises(RuntimeError, match="HTTP 503"):
        request_json("http://api:8000", "GET", "/check")

    monkeypatch.setattr(
        "ai_support.deployment.verify.urllib.request.urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("down")),
    )
    with pytest.raises(RuntimeError, match="could not reach"):
        request_json("http://api:8000", "GET", "/check")


def test_verify_main_reads_environment(monkeypatch, capsys) -> None:
    monkeypatch.setenv("AI_SUPPORT_DEMO_LOGIN_CODE", "code")
    monkeypatch.setattr("sys.argv", ["verify", "--base-url", "http://api:8000"])
    monkeypatch.setattr(
        "ai_support.deployment.verify.verify_deployment",
        lambda base_url, login_code: {"base_url": base_url, "login_code": login_code},
    )

    main()

    assert json.loads(capsys.readouterr().out) == {
        "base_url": "http://api:8000",
        "login_code": "code",
    }
