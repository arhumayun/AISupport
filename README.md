# UOWD AI Support Assistant

The UOWD AI Support Assistant is a reproducible student-and-staff support system combining supervised department routing, permission-filtered RAG, and controlled agent tools. The repository currently implements **Phases 1–3**: a UV-managed FastAPI foundation, independent department routers, and isolated student/staff hybrid-retrieval pipelines orchestrated with Metaflow and tracked in MLflow.

## Demo video

Not available yet. The final submission will include a public, no-sign-in demo link.

## Prerequisites

- [UV](https://docs.astral.sh/uv/) 0.11.23 or compatible
- Git
- macOS or Linux (Windows users should use WSL for the documented shell commands)

UV installs the pinned Python 3.12 runtime when needed. PyCharm can use `.venv/bin/python` after synchronization.

## Set up and run the pipeline

From a clean clone:

```bash
uv sync --locked
```

The checked-in DVC pointers identify the exact synthetic source and evaluation datasets. The development remote is local and intentionally not committed, so a teammate can recreate the same source files from their fixed seeds:

```bash
uv run ai-support data generate --audience all --rows 600
uv run dvc add data/router/student/tickets.csv data/router/staff/tickets.csv
```

Run both router pipelines end to end, including deterministic generation, preprocessing, training, validation, MLflow tracking, candidate registration, and prediction output:

```bash
uv run ai-support routers run-all --generate --rows 600
```

Run the independent Metaflow workflows individually:

```bash
uv run ai-support router train --audience student
uv run ai-support router train --audience staff
```

Equivalent direct Metaflow commands are:

```bash
uv run python -m ai_support.pipelines.student_router_flow run
uv run python -m ai_support.pipelines.staff_router_flow run
```

Passing runs receive the `candidate` alias. Promotion to `champion` is deliberately manual:

```bash
uv run ai-support router approve --audience student --model-reference candidate
uv run ai-support router approve --audience staff --model-reference candidate
```

Start the API after approved champion versions exist:

```bash
uv run uvicorn ai_support.api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` for its OpenAPI interface. Detailed pipeline, reproduction, and rollback instructions are in `docs/router-pipeline.md`.

## Build the RAG releases

The synthetic document collections and independent benchmark sets are deterministic and DVC-versioned:

```bash
uv run ai-support rag generate-sample-data --audience all
uv run dvc add data/rag/student data/rag/staff
```

Run the two audience-specific Metaflow workflows:

```bash
uv run ai-support rag build --audience student
uv run ai-support rag build --audience staff
```

The first real build downloads the configured, revision-pinned Sentence Transformers model; CI tests inject a deterministic local embedder and make no model download. Set `AI_SUPPORT_DATABASE_URL` to a PostgreSQL database with pgvector to persist an index. Without it, the experiment uses the in-memory repository and still evaluates and records a release manifest.

See `docs/rag-pipeline.md` for metadata, experiments, metrics, SQL setup, and release restoration.

## Run tests and quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run dvc status
```

The automated suite trains small deterministic models and uses a temporary real MLflow SQLite registry for registration, alias, and version-loading tests. It does not require Ollama or external model downloads.

## Build and run the container

Container images and Docker Compose are planned for Phase 5. They are deliberately absent at the end of Phase 3.

## Call the service

Health request:

```bash
curl --fail http://127.0.0.1:8000/health
```

Router request:

```bash
curl --fail -X POST http://127.0.0.1:8000/v1/router/student/predict \
  -H 'Content-Type: application/json' \
  -d '{"ticket_subject":"Timetable missing","ticket_description":"I cannot see my class timetable"}'
```

Example response from the measured Phase 2 champion:

```json
{
  "audience": "student",
  "department": "Registry",
  "confidence": 0.851066861553833,
  "top_departments": [
    {"department": "Registry", "probability": 0.851066861553833},
    {"department": "IT", "probability": 0.04904855415621885},
    {"department": "Careers", "probability": 0.04454449506213565}
  ],
  "model_name": "student-department-router",
  "model_reference": "champion"
}
```

This router endpoint is an internal model interface, not end-user authentication. Phase 4 will add server-verified synthetic identities before exposing router selection through the chatbot.

The RAG search endpoint is likewise internal. It returns only stored evidence and citations; answer generation and authenticated chatbot access belong to Phase 4:

```bash
curl --fail -X POST http://127.0.0.1:8000/v1/rag/student/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"I cannot see my timetable","department":"Registry"}'
```

## Repository layout

```text
.
├── .dvc/                         # DVC project and local-remote configuration
├── .github/workflows/ci.yml      # Locked install, lint, and test automation
├── configs/rag/                  # Separate chunking/embedding/retrieval settings
├── configs/router/               # Separate student/staff model and gate settings
├── data/rag/                     # DVC-versioned documents and benchmarks
├── data/router/                  # DVC pointers; CSV contents remain out of Git
├── docs/
│   ├── architecture/overview.md  # Target architecture and phase boundaries
│   ├── rag-pipeline.md           # RAG experiments, releases, and restoration
│   ├── router-pipeline.md        # Execution, tracking, reproduction, rollback
│   ├── implementation-log.md     # Changes from the Assessment 2 design
│   └── reference/                # Supplied assessment PDFs
├── src/ai_support/
│   ├── api/                      # FastAPI contracts and endpoints
│   ├── pipelines/                # Four independent Metaflow workflows
│   ├── rag/                      # Ingestion, chunking, retrieval, evaluation
│   └── router/                   # Data, preprocessing, model, metrics, registry
├── tests/rag/                    # RAG policy, adapter, pipeline, and API tests
├── tests/router/                 # Router unit, API, and registry tests
├── pyproject.toml                # Dependencies and tool configuration
└── uv.lock                       # Reproducible dependency lock
```

## Tools used and why

- **UV** manages Python, environments, dependencies, commands, and the lockfile.
- **DVC** records source and chronological evaluation dataset hashes, with a local development remote.
- **Metaflow** makes ingestion, preprocessing, training, evaluation/gating, and registration visible as independent executable steps.
- **MLflow** records parameters, metrics, artifacts, provenance, model versions, and `candidate`/`champion` aliases.
- **scikit-learn** provides the explainable TF-IDF plus logistic-regression baseline.
- **sentence-transformers** provides revision-pinned dense embeddings and an optional CrossEncoder reranker.
- **PostgreSQL/pgvector** provides persistent dense search alongside PostgreSQL full-text search; RRF combines their ranked lists.
- **FastAPI and Pydantic** expose typed health and router prediction contracts.
- **pytest, HTTPX2, and Ruff** provide automated behavioral, integration, lint, formatting, and coverage checks.
- **GitHub Actions** repeats the locked installation and quality checks on pushes and pull requests.

Ollama, LangGraph, Streamlit, Prometheus/Grafana, Mailpit, and Docker Compose remain later-phase work.

## Team and contributions

- Saima Farooq: Assessment 2 solution architecture and project owner.
- Additional team members: add names and specific code, documentation, testing, demo, and presentation contributions before submission.

The Git history and this section must agree before the final assessment submission.
