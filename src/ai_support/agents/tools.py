"""Allow-listed agent tools with backend authorization and confirmation controls."""

import hashlib
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Protocol

from ai_support.agents.actions import JsonlAuditStore, PendingActionStore
from ai_support.agents.profiles import SyntheticProfileStore
from ai_support.agents.schema import (
    CourseProgress,
    EmailDraft,
    StaffProfile,
    StudentProfile,
)
from ai_support.monitoring.metrics import KNOWLEDGE_GAPS
from ai_support.rag.schema import SearchResult
from ai_support.security.auth import Principal


class KnowledgeSearch(Protocol):
    def search(
        self, audience: str, query: str, department: str | None = None
    ) -> list[SearchResult]: ...


_DEPARTMENT_RECIPIENTS = {
    "Registry": "registry@example.test",
    "Finance": "finance@example.test",
    "IT": "it-support@example.test",
    "HR": "hr@example.test",
    "Facilities": "facilities@example.test",
    "Careers": "careers@example.test",
}
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+")
_IDENTIFIER = re.compile(r"\b\d{6,}\b")


def search_knowledge(
    principal: Principal,
    query: str,
    department: str | None,
    knowledge: KnowledgeSearch,
) -> list[SearchResult]:
    """Search only the audience derived from the verified session."""

    return knowledge.search(principal.audience, query, department)


def get_student_profile(principal: Principal, profiles: SyntheticProfileStore) -> StudentProfile:
    if principal.audience != "student":
        raise PermissionError("student profile tool is not permitted")
    return profiles.student_profile(principal.user_id)


def get_course_progress(principal: Principal, profiles: SyntheticProfileStore) -> CourseProgress:
    if principal.audience != "student":
        raise PermissionError("course progress tool is not permitted")
    return profiles.course_progress(principal.user_id)


def get_staff_profile(principal: Principal, profiles: SyntheticProfileStore) -> StaffProfile:
    if principal.audience != "staff":
        raise PermissionError("staff profile tool is not permitted")
    return profiles.staff_profile(principal.user_id)


def _minimal_summary(message: str) -> str:
    redacted = _EMAIL.sub("[redacted email]", message)
    redacted = _IDENTIFIER.sub("[redacted identifier]", redacted)
    return " ".join(redacted.split())[:500]


def request_department_input(
    principal: Principal,
    department: str,
    message: str,
    actions: PendingActionStore,
    now: datetime | None = None,
) -> EmailDraft:
    """Prepare, but never send, a synthetic department email draft."""

    recipient = _DEPARTMENT_RECIPIENTS.get(department)
    if recipient is None:
        raise ValueError("department does not have an approved synthetic recipient")
    created = now or datetime.now(UTC)
    draft = EmailDraft(
        action_id=secrets.token_urlsafe(18),
        department=department,
        recipient=recipient,
        subject=f"UOWD support clarification request: {department}",
        body=(
            f"An authenticated {principal.audience} requests clarification.\n\n"
            f"Question: {_minimal_summary(message)}\n\n"
            "This message was prepared in the synthetic assessment environment."
        ),
        expires_at=created + timedelta(minutes=15),
    )
    actions.add(draft, principal)
    return draft


def get_support_handoff(principal: Principal, department: str | None) -> tuple[str, str]:
    label = f"the {department} team" if department else "UOWD support"
    if principal.audience == "student":
        url = "https://www.uowdubai.ac.ae/students"
    else:
        url = "https://www.uowdubai.ac.ae/contact-us"
    return (
        f"I cannot safely resolve this request. Please continue with {label} through the "
        "official support page.",
        url,
    )


def record_knowledge_gap(
    principal: Principal,
    query: str,
    department: str | None,
    reason: str,
    store: JsonlAuditStore,
) -> None:
    KNOWLEDGE_GAPS.labels(principal.audience, department or "unknown", reason).inc()
    store.record(
        {
            "event": "knowledge_gap",
            "audience": principal.audience,
            "department": department,
            "reason": reason,
            "query_hash": hashlib.sha256(query.encode()).hexdigest(),
        }
    )
