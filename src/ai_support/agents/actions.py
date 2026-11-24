"""Short-lived confirmation actions and privacy-safe audit records."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from ai_support.agents.schema import EmailDraft, PendingAction
from ai_support.security.auth import Principal


class ActionAuthorizationError(PermissionError):
    """Raised when a pending action does not belong to the authenticated session."""


class PendingActionStore:
    def __init__(self, clock=lambda: datetime.now(UTC)) -> None:
        self._actions: dict[str, PendingAction] = {}
        self._lock = threading.Lock()
        self.clock = clock

    def add(self, draft: EmailDraft, principal: Principal) -> None:
        with self._lock:
            self._actions[draft.action_id] = PendingAction(
                draft=draft,
                session_id=principal.session_id,
                audience=principal.audience,
                created_at=self.clock(),
            )

    def get(self, action_id: str, principal: Principal) -> EmailDraft:
        with self._lock:
            action = self._actions.get(action_id)
        if (
            action is None
            or action.session_id != principal.session_id
            or action.audience != principal.audience
        ):
            raise ActionAuthorizationError("pending action is not available for this session")
        if action.draft.expires_at <= self.clock():
            with self._lock:
                self._actions.pop(action_id, None)
            raise ActionAuthorizationError("pending action has expired")
        return action.draft

    def consume(self, action_id: str, principal: Principal) -> EmailDraft:
        draft = self.get(action_id, principal)
        with self._lock:
            self._actions.pop(action_id, None)
        return draft


class JsonlAuditStore:
    """Append operational metadata without prompts, profile values, or secrets."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, event: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        safe = {**event, "recorded_at": datetime.now(UTC).isoformat()}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(safe, sort_keys=True) + "\n")


class FeedbackAuthorizationError(PermissionError):
    """Raised when feedback targets a response from another session."""


class FeedbackStore:
    def __init__(self, audit: JsonlAuditStore) -> None:
        self.audit = audit
        self._owners: dict[str, str] = {}
        self._lock = threading.Lock()

    def register(self, message_id: str, principal: Principal) -> None:
        with self._lock:
            self._owners[message_id] = principal.session_id

    def record(self, message_id: str, helpful: bool, principal: Principal) -> None:
        with self._lock:
            owner = self._owners.get(message_id)
        if owner != principal.session_id:
            raise FeedbackAuthorizationError("message is not available for this session")
        self.audit.record(
            {
                "event": "answer_feedback",
                "audience": principal.audience,
                "message_id": message_id,
                "helpful": helpful,
            }
        )
