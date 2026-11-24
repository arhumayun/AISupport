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
2. **Router ML pipelines (implemented):** deterministic synthetic tickets, independent student/staff Metaflow flows, DVC, MLflow, validation gates, registration, and prediction API.
3. **Student/staff RAG (implemented):** document validation, parsing, configurable chunking and embeddings, hybrid retrieval, evaluation, and immutable release manifests.
4. **Chatbot and agents (implemented):** synthetic authentication, Ollama/LangGraph, Streamlit, authorized profile tools, confirmed Mailpit email, and handoff.
5. **Deployment and monitoring (implemented):** lockfile-based non-root image, private Compose services, Prometheus/Grafana, measured smoke tests, an experiment catalog, and controlled rollback commands.

Only components listed as implemented are currently available.

## Phase 5 deployment architecture

Compose separates the public application edge, a loopback-only admin bridge, and an internal
backend network. FastAPI,
Streamlit, PostgreSQL/pgvector, Ollama, MLflow, MinIO, Mailpit, Prometheus, and Grafana use
pinned release tags or source commits. PostgreSQL, MinIO, and the application use named
volumes; MinIO stores MLflow artifacts while PostgreSQL stores its metadata. Prometheus
scrapes only the API on the private network and Grafana provisions the checked-in dashboard.

The runtime exports bounded-label request, routing, retrieval, LLM, citation, gap, tool,
email, model-version, and knowledge-version series. The JSON Lines experiment catalog joins
the immutable router/RAG manifests for future historical reporting without inventing live
metrics. Rollback remains an approved manual operation: an MLflow model alias can target an
older registered version, and a RAG release pointer can select an existing gate-passing
manifest without rebuilding it.

## Phase 3 RAG architecture

Student and staff collections have independent DVC pointers, benchmarks, configurations, MLflow experiments, release directories, and database audience policies. Shared code implements the mechanics without merging their data.

```text
approved source + metadata sidecar -> permission/date/hash validation
  -> PDF/DOCX/Markdown/TXT/HTML parser -> configurable chunker
  -> revision-pinned Sentence Transformer -> pgvector + PostgreSQL FTS
  -> reciprocal-rank fusion -> optional CrossEncoder reranking
  -> Recall@K/MRR/nDCG/citation/grounding/leakage evaluation
  -> promotion gate -> immutable release manifest + MLflow run
```

The database schema enforces `audience = access_level`, while every lexical and vector query also requires both values to equal the server-selected audience. The in-memory implementation applies the same policy for deterministic tests and local experiments. It is not the production serving store.

## Phase 2 router architecture

Student and staff configurations point to separate source/evaluation datasets, MLflow experiments, registered model names, output paths, and random seeds. Both workflows call shared implementation modules so preprocessing and inference cannot silently diverge.

```text
synthetic generator (fixed seed, 12 months)
  -> DVC source pointer
  -> Metaflow ingestion/schema validation
  -> chronological split + duplicate-content leakage removal
  -> TF-IDF + logistic regression sklearn Pipeline
  -> accuracy/macro-F1/per-class/top-k/calibration evaluation
  -> explicit promotion gate
  -> MLflow artifact + candidate alias (when passing)
  -> manual approval -> champion alias
  -> FastAPI internal prediction endpoint
```

The final resolving department is the supervised target. The original assigned department remains a noisy historical field. Intent is produced by a documented rule taxonomy and is not inferred from department assignment.

## Phase 4 controlled-agent architecture

```text
synthetic ID + login code -> FastAPI verification -> signed expiring session
  -> session audience/user ID -> deterministic LangGraph policy
     |-> audience-specific router + RAG -> Ollama evidence-only answer
     |-> session-bound student/staff profile -> deterministic response
     |-> clarification request -> redacted draft -> explicit confirmation -> Mailpit
     |-> low confidence/restriction/failure -> official handoff + hashed gap
```

Ollama never selects tools. Rules in the graph classify the limited supported intents; each tool independently checks the authenticated principal. Retrieved text is placed in an untrusted-evidence section and cannot alter authorization or trigger an action. Email drafts are stored against a session, expire after 15 minutes, use fixed `example.test` recipients, and are one-time actions after successful delivery.

## Security baseline

- Configuration examples contain no credentials.
- Logs contain operational events, not prompts, raw personal records, or secrets.
- Router, RAG, chat, feedback, and confirmation endpoints require a signed backend session. Audience path parameters must match its audience.
- Side effects require explicit user confirmation and produce privacy-safe audit metadata.
- Retrieved documents are returned as untrusted evidence with citations, never as executable instructions.
- Personal tools derive their record key from the session and never accept a caller-supplied student or staff ID.
