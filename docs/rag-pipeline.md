# Student and Staff RAG Pipelines

Phase 3 implements two independently executable retrieval pipelines while sharing parsing, chunking, indexing, evaluation, and tracking code. Student data never enters a staff release, and staff data never enters a student release.

## Reproduce a release

Prepare the locked environment and restore the DVC data when a configured remote is available:

```bash
uv sync --locked
uv run dvc pull data/rag/student.dvc data/rag/staff.dvc
```

The repository's development DVC remote is intentionally local and ignored. When it is unavailable, recreate the exact synthetic assessment corpus and refresh the pointers:

```bash
uv run ai-support rag generate-sample-data --audience all
uv run dvc add data/rag/student data/rag/staff
uv run dvc status
```

Run one audience or both:

```bash
uv run ai-support rag build --audience student
uv run ai-support rag build --audience staff
uv run ai-support rags run-all
```

Equivalent direct workflows are `uv run python -m ai_support.pipelines.student_rag_flow run` and `uv run python -m ai_support.pipelines.staff_rag_flow run`. The first real run may download the configured embedding checkpoint. Model identifiers and immutable revisions live in `configs/rag/*.toml`; automated tests use a local deterministic embedder.

## Source and metadata contract

Each PDF, DOCX, Markdown, TXT, or HTML source has an adjacent `<filename>.metadata.json` sidecar containing:

- `document_id`, `audience`, `department`, `source`, and `source_version`
- SHA-256 `content_hash`
- `approval_status`, `review_date`, and `effective_date`
- `access_level`

Ingestion rejects hash mismatches, unapproved/expired/future documents, and any audience/access mismatch before parsing. Dates are evaluated against the version-controlled `as_of_date`, not the wall clock.

## Configurable experiments

The audience TOML files select fixed, recursive, or heading-aware chunking; chunk size and overlap; one of two pinned embedding profiles; lexical, dense, or hybrid retrieval; candidate/top-K sizes; RRF constant; and optional CrossEncoder reranking. Semantic chunking can be added behind the existing chunker interface.

The PostgreSQL migration at `src/ai_support/rag/sql/001_rag_schema.sql` creates:

- immutable audience-specific releases and document/chunk records;
- a generated English `tsvector` with a GIN index;
- a 384-dimension pgvector column with an HNSW cosine index;
- database checks requiring `audience = access_level`.

Set `AI_SUPPORT_DATABASE_URL` to use persistent PostgreSQL. Every lexical and dense query filters by release, audience, access level, and optional department. Without that variable, flow experiments use the isolated in-memory backend.

## Evaluation and promotion

Student and staff benchmark JSONL files are independent. Each run records:

- Recall@K, MRR, and nDCG;
- citation correctness;
- answer groundedness proxy (expected answer terms present in retrieved evidence);
- average retrieval latency;
- unauthorized cross-audience retrieval rate.

The configurable gate currently requires Recall@K >= 0.80, MRR >= 0.75, and zero unauthorized retrieval. Passing releases are marked `candidate`; nothing automatically deploys them.

## Provenance and restoration

MLflow records parameters, metrics, configuration, evaluation details, and the final manifest. Each content-addressed manifest also records the Git state, DVC directory hash, evaluation hash, Metaflow run ID, MLflow run ID, UV lock digest, chunking configuration, embedding revision, retrieval settings, and index backend.

`artifacts/rag/<audience>/latest.json` points to the newest passing candidate. To restore an older approved release, select its manifest and matching `release_id`; its query-time settings are embedded in the manifest. The corresponding PostgreSQL rows must be retained or re-indexed from the recorded DVC corpus and configuration. Automatic rollback is intentionally excluded.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run dvc status
```

The tests cover schema and metadata rejection, deterministic generation, all parsers and chunkers, embedding/reranking adapters, hybrid retrieval, citations, cross-audience denial, evaluation/gating, immutable manifests, PostgreSQL SQL/pgvector boundaries, and FastAPI contracts. A live PostgreSQL integration run is deferred until the Phase 5 Compose stack.
