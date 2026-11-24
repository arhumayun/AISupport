FROM ghcr.io/astral-sh/uv:0.11.23 AS uv

FROM python:3.12.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY --from=uv /uv /uvx /usr/local/bin/

RUN groupadd --system --gid 10001 ai-support \
    && useradd --system --uid 10001 --gid ai-support --home-dir /app ai-support

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY data ./data

RUN uv sync --locked --no-dev \
    && mkdir -p /app/artifacts /app/.cache \
    && chown -R ai-support:ai-support /app

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "ai_support.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
