"""Low-cardinality Prometheus metrics for the deployed vertical slice."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

HTTP_REQUESTS = Counter(
    "ai_support_http_requests_total",
    "HTTP requests handled by FastAPI.",
    ("method", "route", "status"),
)
HTTP_DURATION = Histogram(
    "ai_support_http_request_duration_seconds",
    "FastAPI request latency.",
    ("method", "route"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
ROUTER_PREDICTIONS = Counter(
    "ai_support_router_predictions_total",
    "Department router predictions.",
    ("audience", "department", "model_reference"),
)
ROUTER_CONFIDENCE = Histogram(
    "ai_support_router_confidence",
    "Predicted department confidence.",
    ("audience",),
    buckets=(0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
RAG_DURATION = Histogram(
    "ai_support_rag_retrieval_duration_seconds",
    "Permission-filtered retrieval latency.",
    ("audience", "department"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
LLM_DURATION = Histogram(
    "ai_support_llm_response_duration_seconds",
    "Ollama response latency.",
    ("model", "outcome"),
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
CITATION_COVERAGE = Histogram(
    "ai_support_citation_coverage_ratio",
    "Fraction of answer evidence items represented by citations.",
    ("audience",),
    buckets=(0.0, 0.25, 0.5, 0.75, 1.0),
)
KNOWLEDGE_GAPS = Counter(
    "ai_support_knowledge_gaps_total",
    "Privacy-safe knowledge gaps.",
    ("audience", "department", "reason"),
)
TOOL_FAILURES = Counter(
    "ai_support_agent_tool_failures_total",
    "Controlled agent tool failures.",
    ("tool", "reason"),
)
EMAIL_FAILURES = Counter(
    "ai_support_email_delivery_failures_total",
    "Confirmed synthetic email delivery failures.",
    ("department",),
)
MODEL_VERSION = Gauge(
    "ai_support_model_version_info",
    "Active router model reference.",
    ("audience", "model", "reference"),
)
KNOWLEDGE_VERSION = Gauge(
    "ai_support_knowledge_version_info",
    "Active immutable knowledge release.",
    ("audience", "release"),
)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
