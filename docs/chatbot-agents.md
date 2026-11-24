# Chatbot and Controlled Agent Workflow

Phase 4 provides one Streamlit chatbot backed by FastAPI and two authorization-isolated student/staff systems. The browser never chooses its effective audience: a verified synthetic login produces an expiring, signed session containing the backend-resolved audience and profile key.

## Local configuration

Create `.env` from `.env.example`, generate a unique development session secret of at least 24 characters, and set a synthetic demo login code. Export the file into each service shell. Do not commit it.

Required components for the complete retrieval demo are:

- approved student/staff router versions in local MLflow;
- PostgreSQL with pgvector and the matching Phase 3 release indexed;
- Ollama with the configured model already installed;
- Mailpit SMTP on the configured host/port for confirmed email tests.

No model is pulled automatically. Phase 5 will supply these services through Docker Compose.

## Start the applications

```bash
uv sync --locked
uv run uvicorn ai_support.api.app:app --reload
uv run streamlit run src/ai_support/ui/app.py
```

Open Streamlit, choose a synthetic identity, and enter the configured demo code. The selector only changes the suggested demo ID; authorization comes from the FastAPI login response.

## Supported tools

- `search_knowledge`: uses only the session audience and optional routed department.
- `get_student_profile`: student-only and keyed by the authenticated user.
- `get_course_progress`: student-only authoritative synthetic academic data.
- `get_staff_profile`: staff-only and keyed by the authenticated user.
- `request_department_input`: creates a redacted draft but cannot send it.
- `get_support_handoff`: provides a current official UOWD support page.
- `record_knowledge_gap`: stores audience, department, reason, and a SHA-256 query hash—not raw text.

LangGraph policy code selects from this allow-list. Ollama only receives the question and already-authorized evidence and returns a structured answer. Instructions embedded in retrieved documents are untrusted data.

## Department clarification workflow

1. The verified audience selects the corresponding router.
2. Available permitted knowledge is retrieved.
3. The assistant prepares a minimal draft with email addresses and long numeric identifiers redacted.
4. The UI displays the complete draft and synthetic recipient.
5. The user clicks **Confirm and send via Mailpit**.
6. FastAPI verifies the action belongs to that exact session and is unexpired.
7. SMTP delivery occurs, the action is consumed, and privacy-safe delivery metadata is audited.

No real UOWD address is used, and no ServiceNow ticket is created.

## API outline

`POST /v1/auth/login` returns a bearer token. The token is required by:

- `POST /v1/chat`
- `POST /v1/router/{audience}/predict`
- `POST /v1/rag/{audience}/search`
- `POST /v1/actions/{action_id}/confirm`
- `POST /v1/feedback`

A student token used on a staff route, or vice versa, receives HTTP 403. Missing, invalid, expired, or modified tokens receive HTTP 401.

## Testing

Normal tests use deterministic fake inference and do not need Ollama or Mailpit:

```bash
uv run pytest
```

To verify real Ollama, start the server and select an already-installed model:

```bash
AI_SUPPORT_TEST_OLLAMA_MODEL=llama3.1:latest \
  uv run pytest -q tests/agents/test_adapters.py::test_real_ollama_demo_model --no-cov
```

The integration test never downloads a model. Mailpit SMTP behavior is unit-tested through the real Python email/SMTP adapter with a socket-boundary test double; live Compose integration follows in Phase 5.
