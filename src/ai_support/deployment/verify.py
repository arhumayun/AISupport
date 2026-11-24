"""Authenticated acceptance probe for a bootstrapped assessment deployment."""

import argparse
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

JsonRequest = Callable[[str, str, dict[str, Any] | None, str | None], dict[str, Any]]


def request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} failed with HTTP {error.code}: {detail}") from error
    except OSError as error:
        raise RuntimeError(f"{method} {path} could not reach the API") from error


def verify_deployment(
    base_url: str,
    login_code: str,
    requester: JsonRequest = request_json,
) -> dict[str, Any]:
    """Verify health, authentication, champion routing, RAG, and cited chat output."""

    health = requester(base_url, "GET", "/health", None, None)
    if health.get("status") != "ok":
        raise RuntimeError("API health response was not ok")
    login = requester(
        base_url,
        "POST",
        "/v1/auth/login",
        {"user_id": "S100001", "login_code": login_code},
        None,
    )
    token = str(login.get("access_token", ""))
    if login.get("audience") != "student" or not token:
        raise RuntimeError("synthetic student authentication failed")
    router = requester(
        base_url,
        "POST",
        "/v1/router/student/predict",
        {
            "ticket_subject": "Timetable missing",
            "ticket_description": "I cannot see my class timetable",
        },
        token,
    )
    if router.get("department") != "Registry" or router.get("model_reference") != "champion":
        raise RuntimeError("champion student router did not select Registry")
    rag = requester(
        base_url,
        "POST",
        "/v1/rag/student/search",
        {"query": "I cannot see my timetable", "department": "Registry"},
        token,
    )
    results = rag.get("results", [])
    if not results or not all(result.get("citation") for result in results):
        raise RuntimeError("student RAG returned no cited Registry evidence")
    chat = requester(
        base_url,
        "POST",
        "/v1/chat",
        {"message": "I cannot see my timetable."},
        token,
    )
    if chat.get("intent") != "knowledge" or not chat.get("citations"):
        raise RuntimeError("chatbot did not return a cited knowledge answer")
    return {
        "status": "passed",
        "audience": "student",
        "department": router["department"],
        "model_reference": router["model_reference"],
        "rag_results": len(results),
        "chat_citations": len(chat["citations"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://api:8000")
    args = parser.parse_args()
    login_code = os.environ.get("AI_SUPPORT_DEMO_LOGIN_CODE")
    if not login_code:
        raise RuntimeError("AI_SUPPORT_DEMO_LOGIN_CODE is required")
    print(json.dumps(verify_deployment(args.base_url, login_code), sort_keys=True))


if __name__ == "__main__":
    main()
