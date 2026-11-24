# UOWD AI Support Assistant

The UOWD AI Support Assistant is a reproducible student-and-staff support system that will combine supervised department routing, permission-filtered RAG, and controlled agent tools. The current repository implements **Phase 1 only**: a UV-managed Python foundation and a tested FastAPI health service.

## Demo video

Not available in Phase 1. The final submission will include a public, no-sign-in demo link.

## Prerequisites

- [UV](https://docs.astral.sh/uv/) 0.11 or newer
- Git
- macOS or Linux (Windows should use WSL for the documented shell commands)

UV installs the pinned Python 3.12 runtime when it is not already available. PyCharm can use `.venv/bin/python` after the setup command completes.

## Set up and run

From a clean clone:

```bash
uv sync --locked
uv run uvicorn ai_support.api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the OpenAPI UI or call the health endpoint shown below.

The end-to-end router pipeline command is intentionally deferred to Phase 2. No ML, RAG, agent, or external-service pipeline is claimed in Phase 1.

## Run tests and quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

To apply safe formatting locally:

```bash
uv run ruff format .
```

## Build and run the container

Container images and Docker Compose are planned for Phase 5. They are deliberately absent from the Phase 1 foundation.

## Call the service

Request:

```bash
curl --fail http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "service": "uowd-ai-support-api",
  "version": "0.1.0"
}
```

## Repository layout

```text
.
├── .github/workflows/ci.yml     # Locked install, lint, and test automation
├── docs/
│   ├── architecture/overview.md # Target architecture and phase boundaries
│   ├── implementation-log.md    # Changes from the Assessment 2 design
│   └── reference/               # Supplied assessment PDFs
├── src/ai_support/
│   ├── api/app.py               # FastAPI application and health endpoint
│   ├── api/models.py            # API response contracts
│   └── logging.py               # Privacy-conscious JSON logging
├── tests/                       # Phase 1 automated tests
├── .env.example                 # Non-secret configuration example
├── .python-version              # Python 3.12 pin
├── pyproject.toml               # Dependencies and tool configuration
└── uv.lock                      # Reproducible dependency lock
```

## Tools used and why

- **UV** manages Python, the environment, dependencies, and the lockfile through one reproducible workflow.
- **FastAPI and Pydantic** provide a typed HTTP interface and generated OpenAPI contract.
- **pytest and HTTPX** verify application behavior without running an external server.
- **Ruff** provides fast lint and formatting checks.
- **GitHub Actions** repeats the locked install and quality checks on pushes and pull requests.

Later phases will add DVC, Metaflow, MLflow, scikit-learn, PostgreSQL/pgvector, sentence-transformers, Ollama, LangGraph, Streamlit, Prometheus/Grafana, Mailpit, and Docker Compose only when their working components are implemented.

## Team and contributions

- Saima Farooq: Assessment 2 solution architecture and project owner.
- Additional team members: add names and specific code, documentation, testing, demo, and presentation contributions before submission.

The Git history and this section must agree before the final assessment submission.
