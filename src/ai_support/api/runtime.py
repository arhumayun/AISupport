"""Construction of local Phase 4 services from environment configuration."""

import os
from pathlib import Path

from ai_support.agents.actions import FeedbackStore, JsonlAuditStore, PendingActionStore
from ai_support.agents.email import MailpitEmailSender
from ai_support.agents.llm import OllamaAnswerGenerator
from ai_support.agents.profiles import SyntheticProfileStore
from ai_support.agents.service import ChatService
from ai_support.agents.tools import KnowledgeSearch
from ai_support.agents.workflow import AgentDependencies, AgentService
from ai_support.monitoring.metrics import KNOWLEDGE_VERSION, MODEL_VERSION
from ai_support.rag.pipeline import load_latest_release
from ai_support.rag.schema import SearchResult
from ai_support.rag.service import RAGService
from ai_support.router.service import RouterService
from ai_support.security.auth import AuthService


class UnavailableKnowledgeSearch:
    def search(
        self, audience: str, query: str, department: str | None = None
    ) -> list[SearchResult]:
        return []


def create_phase4_services(project_root: Path):
    secret = os.getenv("AI_SUPPORT_SESSION_SECRET")
    login_code = os.getenv("AI_SUPPORT_DEMO_LOGIN_CODE")
    if not secret or not login_code:
        return None, None

    profiles = SyntheticProfileStore(
        project_root / "artifacts" / "app" / "profiles.db",
        project_root / "data" / "app" / "synthetic_profiles.json",
    )
    profiles.initialize()
    auth = AuthService(profiles, secret, login_code)
    router = RouterService(project_root, os.getenv("AI_SUPPORT_ROUTER_MODEL_REFERENCE", "champion"))
    for audience in ("student", "staff"):
        MODEL_VERSION.labels(audience, f"{audience}-department-router", router.model_reference).set(
            1
        )
        try:
            release = load_latest_release(project_root, audience)
        except FileNotFoundError:
            continue
        KNOWLEDGE_VERSION.labels(audience, release).set(1)
    database_url = os.getenv("AI_SUPPORT_DATABASE_URL")
    knowledge: KnowledgeSearch = (
        RAGService(project_root, database_url) if database_url else UnavailableKnowledgeSearch()
    )
    actions = PendingActionStore()
    audit = JsonlAuditStore(project_root / "artifacts" / "app" / "audit.jsonl")
    feedback = FeedbackStore(audit)
    dependencies = AgentDependencies(
        profiles=profiles,
        router=router,
        knowledge=knowledge,
        llm=OllamaAnswerGenerator(
            model=os.getenv("AI_SUPPORT_OLLAMA_MODEL", "llama3.1:latest"),
            host=os.getenv("AI_SUPPORT_OLLAMA_HOST", "http://127.0.0.1:11434"),
        ),
        actions=actions,
        gap_store=audit,
    )
    sender = MailpitEmailSender(
        host=os.getenv("AI_SUPPORT_MAILPIT_HOST", "127.0.0.1"),
        port=int(os.getenv("AI_SUPPORT_MAILPIT_PORT", "1025")),
    )
    chat = ChatService(AgentService(dependencies), actions, sender, feedback, audit)
    return auth, chat
