"""Typed agent, profile, action, and response contracts."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_support.router.schema import Audience


class StudentProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    name: str
    programme: str
    course: str
    enrolment_status: str


class CourseProgress(BaseModel):
    model_config = ConfigDict(frozen=True)

    completed_subjects: list[str]
    remaining_subjects: list[str]
    progress_percentage: float = Field(ge=0.0, le=100.0)


class StaffProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    name: str
    job_title: str
    department: str
    employment_status: str


class EmailDraft(BaseModel):
    action_id: str
    department: str
    recipient: str
    subject: str
    body: str
    expires_at: datetime


class EmailReceipt(BaseModel):
    action_id: str
    delivery_id: str
    delivered: bool


class Citation(BaseModel):
    document_id: str
    source: str
    citation: str


class AgentReply(BaseModel):
    message_id: str
    audience: Audience
    answer: str
    department: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    action: EmailDraft | None = None
    handoff_url: str | None = None
    intent: Literal[
        "knowledge", "student_profile", "course_progress", "staff_profile", "email", "handoff"
    ]


class PendingAction(BaseModel):
    draft: EmailDraft
    session_id: str
    audience: Audience
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
