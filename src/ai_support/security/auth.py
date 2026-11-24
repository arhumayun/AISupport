"""Server-verified synthetic identities and signed assessment sessions."""

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

from ai_support.router.schema import Audience


class AuthenticationError(ValueError):
    """Raised when credentials or a session token cannot be verified."""


@dataclass(frozen=True)
class Principal:
    user_id: str
    audience: Audience
    session_id: str


class IdentityLookup:
    def audience_for_user(self, user_id: str) -> Audience | None:
        raise NotImplementedError


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class AuthService:
    """Verify a synthetic login code and issue integrity-protected sessions."""

    def __init__(
        self,
        identities: IdentityLookup,
        session_secret: str,
        login_code: str,
        ttl_seconds: int = 3600,
        clock=time.time,
    ) -> None:
        if len(session_secret) < 24:
            raise ValueError("session secret must contain at least 24 characters")
        if not login_code:
            raise ValueError("synthetic login code is required")
        self.identities = identities
        self.secret = session_secret.encode()
        self.login_code = login_code
        self.ttl_seconds = ttl_seconds
        self.clock = clock

    def login(self, user_id: str, login_code: str) -> tuple[str, Principal, int]:
        normalized = user_id.strip().upper()
        audience = self.identities.audience_for_user(normalized)
        if audience is None or not secrets.compare_digest(login_code, self.login_code):
            raise AuthenticationError("invalid synthetic credentials")
        issued = int(self.clock())
        session_id = secrets.token_urlsafe(18)
        payload = {
            "sub": normalized,
            "aud": audience,
            "sid": session_id,
            "iat": issued,
            "exp": issued + self.ttl_seconds,
        }
        encoded = _encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
        signature = _encode(hmac.new(self.secret, encoded.encode(), hashlib.sha256).digest())
        principal = Principal(normalized, audience, session_id)
        return f"{encoded}.{signature}", principal, self.ttl_seconds

    def verify(self, token: str) -> Principal:
        try:
            encoded, supplied = token.split(".", 1)
            expected = _encode(hmac.new(self.secret, encoded.encode(), hashlib.sha256).digest())
            if not secrets.compare_digest(supplied, expected):
                raise AuthenticationError("invalid session signature")
            payload = json.loads(_decode(encoded))
            if int(payload["exp"]) <= int(self.clock()):
                raise AuthenticationError("session expired")
            user_id = str(payload["sub"])
            audience = payload["aud"]
            current_audience = self.identities.audience_for_user(user_id)
            if audience not in {"student", "staff"} or current_audience != audience:
                raise AuthenticationError("session identity is no longer valid")
            return Principal(user_id, audience, str(payload["sid"]))
        except AuthenticationError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise AuthenticationError("malformed session token") from error
