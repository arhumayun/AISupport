# Architecture Overview

## Assessment vertical slice

The target system is one chatbot with two authorization-isolated AI systems. A server-verified synthetic identity selects either student or staff router artifacts, RAG indexes, evaluation sets, and permitted tools; an LLM-generated role never grants access.

```text
Streamlit -> FastAPI/auth -> audience policy -> student or staff workflow
                                            |-> department router
                                            |-> permission-filtered hybrid RAG
                                            |-> approved profile/communication/handoff tools

Offline: synthetic data -> DVC -> Metaflow -> MLflow -> approved artifacts
Runtime: FastAPI/agents -> Prometheus -> Grafana
Services: PostgreSQL/pgvector, Ollama, MinIO/local DVC storage, Mailpit
```

## Phase boundaries

1. **Foundation (implemented):** UV/Python project, typed health API, logging, tests, and CI.
2. **Router ML pipelines:** deterministic synthetic tickets, independent student/staff Metaflow flows, DVC, MLflow, validation gates, registration, and prediction API.
3. **Student/staff RAG:** document validation, parsing, configurable chunking and embeddings, hybrid retrieval, evaluation, and immutable release manifests.
4. **Chatbot and agents:** synthetic authentication, Ollama/LangGraph, Streamlit, authorized profile tools, confirmed Mailpit email, and handoff.
5. **Deployment and monitoring:** lockfile-based images, Compose services, Prometheus/Grafana, integration/performance tests, and rollback demonstration.

Only components listed as implemented are currently available.

## Phase 1 runtime

The Phase 1 process is intentionally small: Uvicorn serves FastAPI, the application validates responses through Pydantic, and `/health` provides a deployment probe. It performs no database, model, document, LLM, email, or external network calls.

## Security baseline

- Configuration examples contain no credentials.
- Logs contain operational events, not prompts, raw personal records, or secrets.
- Future audience authorization must be enforced in FastAPI and database queries.
- Future side effects require explicit user confirmation and an audit event.
- Retrieved documents will be treated as untrusted evidence, never as executable instructions.
