# Deployment verification record

## Isolated clean-bootstrap rehearsal

- Date: 2026-11-24 (Asia/Dubai)
- Command: `COMPOSE_PROJECT_NAME=uowd-ai-support-rehearsal ./scripts/bootstrap_demo.sh`
- Data-plane state: new application-artifact, PostgreSQL, and MinIO volumes
- Model caches: the Ollama and Hugging Face artifacts were explicitly downloaded during this
  rehearsal and retained between diagnostic retries
- Source revision embedded in image: `a012cc57c956796e34ed505c05a0480e8d7fa107`
- Source worktree marker embedded in experiment metadata: `git_dirty=true`
- Application image ID: `sha256:0409431c361a639c1728f1165629ae851ab5fd1d695bda8686c0ce77fa75c052`
- Runtime user: `10001:10001`

The final unchanged corrected image completed the deterministic data generation, both router
Metaflow workflows, explicit candidate-to-champion promotion, both RAG Metaflow workflows,
PostgreSQL/pgvector indexing, MLflow/MinIO logging, and service startup. The authenticated acceptance
probe returned:

```json
{"audience":"student","chat_citations":1,"department":"Registry","model_reference":"champion","rag_results":1,"status":"passed"}
```

### MLOps run evidence

| Pipeline | Metaflow run | MLflow run | Result |
|---|---|---|---|
| Student router | `1795544367888718` | `9a84ef0c20454f18aeeff06c8eb2392c` | Accuracy/Macro-F1/Top-3 `1.0`; version 1 promoted to champion |
| Staff router | `1795544384606780` | `a5cdfacdfd8549a7ba0802cad3ffa011` | Accuracy/Macro-F1/Top-3 `1.0`; version 1 promoted to champion |
| Student RAG | `1795544399255047` | `322aafc8dae04917bd66151e5e9a7167` | Release `ae122094e87aa5d47549352f`; Recall@5/MRR `1.0`; cross-audience rate `0.0` |
| Staff RAG | `1795544422758740` | `5b80c6330ba74e29b68fbce6b54416fe` | Release `d1849398e8cabd5f450473f9`; Recall@5/MRR `1.0`; cross-audience rate `0.0` |

These perfect scores are produced by the deliberately separable synthetic assessment data and tiny
benchmarks; they are pipeline evidence, not a production-performance claim.

### Runtime checks

- FastAPI `/health`: healthy
- Streamlit `/_stcore/health`: healthy
- Prometheus `/-/healthy`: healthy and rules loaded
- Grafana `/api/health`: database `ok`
- PostgreSQL, MinIO, MLflow, Ollama, Mailpit, API, and Streamlit containers: running/healthy
- PostgreSQL: internal network only
- Administrative endpoints: loopback-only alternate rehearsal ports

The rehearsal revealed and verified fixes for configurable host ports, an explicit Metaflow container
identity, non-root Hugging Face cache ownership, psycopg v3 cursor batching, and PostgreSQL optional
parameter typing. Regenerate the JSON evidence bundle and this record after the final reviewed commit
so the submission evidence names the final clean revision.
