# Implementation Log

This log records material differences between the Assessment 2 proposal and the system built for Assessment 3. Measurements and completion claims will be added only after they are verified.

## 2026-11-24 - Phase 1 foundation

- **Router scope changed:** the Assessment 2 department classifier becomes two independently trained supervised models, one for students and one for staff. Implementation is deferred to Phase 2.
- **RAG isolation strengthened:** the earlier shared-index proposal becomes separate student and staff sources, releases, evaluation datasets, and query policies. Implementation is deferred to Phase 3.
- **Controlled agents added:** profile lookup, course progress, department clarification email, knowledge-gap recording, and human handoff will use allow-listed tools and explicit confirmation for side effects. Implementation is deferred to Phase 4.
- **Visitor scope reduced:** the Assessment 3 vertical slice focuses on authenticated synthetic student and staff identities. Prospective visitors remain a future enhancement.
- **ServiceNow integration reduced:** the prototype will provide a handoff link and privacy-safe gap record, not create or modify real tickets.
- **Foundation implemented:** the default PyCharm script was replaced by a UV-managed Python 3.12 `src/` package, a typed FastAPI health endpoint, JSON logging, automated tests, and GitHub Actions checks.

## Known Phase 1 limitations

- No training, model, data, RAG, agent, user interface, external service, container, or monitoring implementation exists yet.
- CI is defined locally but cannot be claimed as executed until the repository is hosted and the workflow runs.
- Team contribution details and the public demo link remain submission-time inputs.

## 2026-11-24 - Phase 2 router ML pipelines

- Implemented separate `StudentRouterFlow` and `StaffRouterFlow` workflows with explicit ingestion, preprocessing, training, evaluation/gating, registration, and end steps.
- Added deterministic 600-row synthetic datasets per audience spanning calendar year 2025. The generator seeds are version controlled and no real personal data is present.
- The model target is `final_resolving_department`; the initially assigned department includes controlled noise and is not treated as authoritative.
- Added rule-based intents independent of department labels.
- Added chronological 80/20 evaluation with normalized-content duplicate leakage prevention.
- Logged accuracy, macro-F1, top-k accuracy, expected calibration error, per-department precision/recall/F1, confusion matrices, configuration, Git/DVC/Metaflow/environment provenance, evaluation version, and trained artifacts to MLflow.
- Registered passing models independently as `student-department-router` and `staff-department-router`. New passing versions receive `candidate`; only an explicit command changes `champion`.
- Added registered alias and explicit-version loading, allowing an approved older model to be selected without retraining.
- Added an internal FastAPI prediction contract and tests for student/staff separation.

### Measured development result

Both deterministic synthetic chronological holdouts measured accuracy 1.0000, macro-F1 1.0000, and top-3 accuracy 1.0000. Expected calibration error was approximately 0.0308 for students and 0.0297 for staff in the verified Metaflow runs. These results show that the intentionally separable synthetic patterns are learnable; they do not estimate performance on real university tickets.

### Known Phase 2 limitations

- Synthetic templates are less ambiguous and diverse than production support descriptions, so perfect classification is expected and should not be generalized.
- The local DVC remote and local MLflow SQLite database are developer artifacts and are not committed. Shared authenticated storage is Phase 5 work.
- The internal router endpoint accepts an audience path parameter. Server-verified identity binding is intentionally deferred to Phase 4 and must precede chatbot exposure.
- MLflow 3.18 emits two upstream deprecation warnings during the registry integration test (SQLAlchemy loader strategy and a pandas range comparison); neither affects the verified behavior.
- At the Phase 2 checkpoint, RAG, agent tools, Streamlit, Docker Compose, and runtime monitoring were not implemented.

## 2026-11-24 - Phase 3 student/staff RAG

- Added separate DVC-versioned student and staff document collections and benchmark datasets with deterministic generation and no personal data.
- Added required metadata and validation for audience, department, source/version, SHA-256 content hash, approval/review/effective dates, and access level.
- Added PDF, DOCX, Markdown, TXT, and HTML loaders plus fixed-size, recursive, and heading-aware chunkers.
- Configured two revision-pinned embedding choices. The verified runs used `sentence-transformers/all-MiniLM-L6-v2`; the optional CrossEncoder reranker remains configuration-controlled.
- Added PostgreSQL full-text and pgvector retrieval, Reciprocal Rank Fusion, database constraints, query-level audience filters, and a behaviorally equivalent in-memory test repository.
- Implemented independent `StudentRAGFlow` and `StaffRAGFlow` workflows, MLflow experiments, evaluation gates, and immutable release manifests containing Git, DVC, Metaflow, MLflow, configuration, model-revision, and environment provenance.
- Added an internal typed retrieval endpoint returning evidence and source citations.

### Measured development result

Pinned-model Metaflow runs produced student release `ae122094e87aa5d47549352f` and staff release `d1849398e8cabd5f450473f9`. On each five-question synthetic benchmark, Recall@5, MRR, nDCG, citation correctness, and the evidence-term groundedness proxy measured 1.0000; unauthorized cross-audience retrieval measured 0.0000. Average retrieval latency was approximately 792.49 ms for the student run and 366.05 ms for the staff run on this development machine. These tiny, deliberately separable benchmarks demonstrate pipeline behavior, not production retrieval quality.

### Known Phase 3 limitations

- PostgreSQL/pgvector integration is implemented and its SQL/adapter boundary is tested, but no live database integration run was performed because the Compose environment belongs to Phase 5. The measured flows therefore used the in-memory experiment backend.
- The groundedness value is an evidence-term coverage proxy. LLM answer generation and model-based groundedness evaluation begin in Phase 4.
- The second embedding profile and optional CrossEncoder path are configurable and unit-tested with local doubles, but the measured release comparison used only MiniLM to avoid unnecessary downloads.
- Release status is `candidate`; there is no automatic production promotion or rollback.
- The internal audience path is not authentication. Server-verified synthetic identities and the chatbot authorization boundary remain Phase 4 work.
