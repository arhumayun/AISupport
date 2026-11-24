# Deployment, Monitoring, and Rollback

## Configure and start

Copy `.env.example` to `.env`, replace all credential and session placeholders, then validate
and start the stack:

```bash
docker compose config --quiet
docker compose build api minio
docker compose up -d
docker compose ps
```

The application is available at Streamlit `127.0.0.1:8501` and FastAPI
`127.0.0.1:8000`. MLflow (`5050`), MinIO (`9000`/`9001`), Prometheus (`9090`), Grafana
(`3000`), Mailpit (`8025`/`1025`), and Ollama (`11434`) bind only to loopback. PostgreSQL is
not published. Compose's `backend` network is internal; services with loopback ports also
join a separate admin bridge because Docker cannot publish a port from an internal-only
network. Replace the assessment passwords; the defaults are intentionally conspicuous and
unsuitable for a shared host.

MLflow uses host port `5050` because macOS AirPlay commonly reserves `5000`; it still listens
on port `5000` inside the private Compose network.

The first build compiles the pinned final MinIO release from its immutable official source
commit because upstream withdrew public images. It also builds the API strictly from
`uv.lock`. The API image runs as non-root UID/GID 10001. Install an Ollama model explicitly:

```bash
docker compose exec ollama ollama pull llama3.1:latest
```

This is intentionally never automatic. Stop without deleting data using `docker compose
down`. Delete named volumes only as an explicit data-destruction operation.

## Verify and measure

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/metrics
uv run ai-support benchmark-api --requests 100 \
  --output docs/performance-baseline.json
uv run ai-support catalog build
```

Prometheus recording rules calculate API p50/p95 latency and error rate plus RAG and LLM p95
latency. Grafana provisions `UOWD AI Support Overview`. Its dashboard shows actual collected
series; no sample or fabricated time series are included. The generated experiment catalog
at `artifacts/catalog/experiments.jsonl` contains Git, DVC, Metaflow, MLflow, environment,
evaluation, configuration, metric, model, and RAG release references for future export.

The checked-in `performance-baseline.json` is a real 100-request sequential measurement of
the containerized `/health` endpoint on the arm64 development host on 2026-11-24. It recorded
0 errors, 0.678 ms p50, and 1.315 ms p95. This is a smoke baseline for that machine, not a
chat/RAG throughput claim or a production capacity estimate.

## Reproduce and roll back

Use the Git commit, `uv.lock` digest, DVC hash, configuration, random seed, evaluation version,
Metaflow ID, and MLflow run ID recorded in a manifest to reproduce an experiment. Restore its
Git revision, run `uv sync --locked`, retrieve the matching DVC data, and rerun the relevant
documented flow. Compare the new run instead of silently replacing the old artifact.

Router rollback is explicit and uses an already registered MLflow version or alias:

```bash
uv run ai-support router approve --audience student --model-reference 3
```

RAG rollback selects an immutable release that already passed its gate:

```bash
uv run ai-support rag select-release --audience student \
  --release-id ae122094e87aa5d47549352f
```

Restart the API after a version selection so its version gauges and cached services match.
Neither path performs automatic rollback.

## Troubleshooting

- `docker compose config` fails: ensure `.env` syntax is valid and Docker Compose v2 is used.
- API waits for MLflow: inspect `docker compose logs postgres minio-init minio mlflow`.
- RAG is unavailable: confirm PostgreSQL is healthy and the selected release was indexed into
  that database; a manifest alone is not an index backup.
- Ollama errors: run `docker compose exec ollama ollama list` and install the configured model.
- Mail is absent: only confirmed drafts are sent, and recipients must end in `example.test`.
- Grafana is empty: call the API, then confirm Prometheus target health at `127.0.0.1:9090`.
