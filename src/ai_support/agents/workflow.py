"""Controlled LangGraph workflow for authenticated support conversations."""

import uuid
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from mlflow.exceptions import MlflowException
from psycopg import Error as PsycopgError

from ai_support.agents.actions import JsonlAuditStore, PendingActionStore
from ai_support.agents.llm import AnswerGenerator, LLMUnavailableError
from ai_support.agents.profiles import SyntheticProfileStore
from ai_support.agents.schema import AgentReply, Citation, EmailDraft
from ai_support.agents.tools import (
    KnowledgeSearch,
    get_course_progress,
    get_staff_profile,
    get_student_profile,
    get_support_handoff,
    record_knowledge_gap,
    request_department_input,
    search_knowledge,
)
from ai_support.router.service import RouterPrediction
from ai_support.security.auth import Principal

Intent = Literal[
    "knowledge", "student_profile", "course_progress", "staff_profile", "email", "handoff"
]


class DepartmentRouter(Protocol):
    def predict(
        self, audience: str, subject: str, description: str, top_k: int = 3
    ) -> RouterPrediction: ...


class AgentState(TypedDict, total=False):
    principal: Principal
    message: str
    intent: Intent
    department: str | None
    confidence: float
    evidence: list[str]
    citations: list[Citation]
    answer: str
    action: EmailDraft | None
    handoff_url: str | None


@dataclass(frozen=True)
class AgentDependencies:
    profiles: SyntheticProfileStore
    router: DepartmentRouter
    knowledge: KnowledgeSearch
    llm: AnswerGenerator
    actions: PendingActionStore
    gap_store: JsonlAuditStore
    minimum_router_confidence: float = 0.55


def classify_intent(message: str, principal: Principal) -> Intent:
    normalized = message.lower()
    if any(term in normalized for term in ("human", "real person", "support agent", "handoff")):
        return "handoff"
    if "another student" in normalized or "another staff" in normalized:
        return "handoff"
    email_terms = ("email the department", "ask the department", "clarification email")
    if any(term in normalized for term in email_terms):
        return "email"
    progress_terms = ("course progress", "completed subjects", "remaining subjects")
    if any(term in normalized for term in progress_terms):
        return "course_progress" if principal.audience == "student" else "handoff"
    if any(term in normalized for term in ("my profile", "my programme", "enrolment status")):
        return "student_profile" if principal.audience == "student" else "staff_profile"
    if principal.audience == "staff" and any(
        term in normalized for term in ("job title", "employment status", "my department")
    ):
        return "staff_profile"
    return "knowledge"


def build_agent_graph(dependencies: AgentDependencies):
    def classify(state: AgentState) -> dict:
        return {"intent": classify_intent(state["message"], state["principal"])}

    def needs_route(state: AgentState) -> str:
        return "route" if state["intent"] in {"knowledge", "email", "handoff"} else "tool"

    def route(state: AgentState) -> dict:
        try:
            result = dependencies.router.predict(
                state["principal"].audience,
                state["message"][:300],
                state["message"],
            )
            return {"department": result.department, "confidence": result.confidence}
        except (MlflowException, OSError, ValueError, RuntimeError):
            return {"department": None, "confidence": 0.0}

    def tool(state: AgentState) -> dict:
        principal = state["principal"]
        message = state["message"]
        intent = state["intent"]
        department = state.get("department")
        if intent == "student_profile":
            profile = get_student_profile(principal, dependencies.profiles)
            return {
                "answer": (
                    f"Your profile lists {profile.name}, programme {profile.programme}, "
                    f"course {profile.course}, with enrolment status {profile.enrolment_status}."
                )
            }
        if intent == "course_progress":
            progress = get_course_progress(principal, dependencies.profiles)
            return {
                "answer": (
                    f"Your recorded progress is {progress.progress_percentage:.0f}%. "
                    f"Completed: {', '.join(progress.completed_subjects)}. "
                    f"Remaining: {', '.join(progress.remaining_subjects)}."
                )
            }
        if intent == "staff_profile":
            profile = get_staff_profile(principal, dependencies.profiles)
            return {
                "answer": (
                    f"Your staff profile lists {profile.name}, {profile.job_title} in "
                    f"{profile.department}, with employment status {profile.employment_status}."
                )
            }
        confidence = state.get("confidence", 0.0)
        if intent == "handoff" or confidence < dependencies.minimum_router_confidence:
            reason = "requested_human" if intent == "handoff" else "low_routing_confidence"
            answer, url = get_support_handoff(principal, department)
            record_knowledge_gap(principal, message, department, reason, dependencies.gap_store)
            return {"answer": answer, "handoff_url": url}
        if intent == "email":
            try:
                available = search_knowledge(principal, message, department, dependencies.knowledge)
            except (FileNotFoundError, OSError, PsycopgError):
                available = []
            try:
                draft = request_department_input(
                    principal, str(department), message, dependencies.actions
                )
            except ValueError:
                answer, url = get_support_handoff(principal, department)
                return {"answer": answer, "handoff_url": url}
            return {
                "action": draft,
                "answer": "I prepared a department clarification email. Review it before sending.",
                "citations": [
                    Citation(
                        document_id=result.document_id,
                        source=result.source,
                        citation=result.citation,
                    )
                    for result in available
                ],
            }
        try:
            results = search_knowledge(principal, message, department, dependencies.knowledge)
        except (FileNotFoundError, OSError, PsycopgError):
            answer, url = get_support_handoff(principal, department)
            record_knowledge_gap(
                principal, message, department, "retrieval_failure", dependencies.gap_store
            )
            return {"answer": answer, "handoff_url": url}
        if not results:
            answer, url = get_support_handoff(principal, department)
            record_knowledge_gap(
                principal, message, department, "no_authorized_evidence", dependencies.gap_store
            )
            return {"answer": answer, "handoff_url": url}
        return {
            "evidence": [result.content for result in results],
            "citations": [
                Citation(
                    document_id=result.document_id,
                    source=result.source,
                    citation=result.citation,
                )
                for result in results
            ],
        }

    def compose(state: AgentState) -> dict:
        if state.get("answer"):
            return {}
        try:
            return {"answer": dependencies.llm.answer(state["message"], state.get("evidence", []))}
        except LLMUnavailableError:
            return {
                "answer": (
                    "I found approved guidance and listed its citations, but local answer "
                    "generation is currently unavailable."
                )
            }

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("route", route)
    graph.add_node("tool", tool)
    graph.add_node("compose", compose)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", needs_route, {"route": "route", "tool": "tool"})
    graph.add_edge("route", "tool")
    graph.add_edge("tool", "compose")
    graph.add_edge("compose", END)
    return graph.compile()


class AgentService:
    def __init__(self, dependencies: AgentDependencies) -> None:
        self.graph = build_agent_graph(dependencies)

    def respond(self, principal: Principal, message: str) -> AgentReply:
        state = self.graph.invoke({"principal": principal, "message": message})
        return AgentReply(
            message_id=str(uuid.uuid4()),
            audience=principal.audience,
            answer=state["answer"],
            department=state.get("department"),
            citations=state.get("citations", []),
            action=state.get("action"),
            handoff_url=state.get("handoff_url"),
            intent=state["intent"],
        )
