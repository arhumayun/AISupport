"""Small HTTP client used by the Streamlit session."""

from typing import Any

import httpx


class APIClientError(RuntimeError):
    """Safe frontend-facing API failure."""


class SupportAPIClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        token: str | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                json=json,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, OSError, ValueError) as error:
            message = "The support service is unavailable or rejected the request."
            raise APIClientError(message) from error

    def login(self, user_id: str, login_code: str) -> dict[str, Any]:
        return self._request(
            "POST", "/v1/auth/login", json={"user_id": user_id, "login_code": login_code}
        )

    def chat(self, token: str, message: str) -> dict[str, Any]:
        return self._request("POST", "/v1/chat", token, {"message": message})

    def confirm(self, token: str, action_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/actions/{action_id}/confirm", token)

    def feedback(self, token: str, message_id: str, helpful: bool) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/feedback",
            token,
            {"message_id": message_id, "helpful": helpful},
        )
