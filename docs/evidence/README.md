# Tracked MLOps Evidence

This directory contains non-sensitive exports from the verified local experiments. It intentionally
excludes model binaries, prompts, personal records, credentials, SQLite databases, and MinIO objects.

- `experiment-catalog.jsonl` joins the latest student/staff router runs and immutable RAG release
  manifests. Each row includes the Git commit, DVC hash, Metaflow run ID, MLflow run ID,
  environment digest, evaluation version, configuration reference, and measured metrics.
- `model-registry.json` shows every locally registered router version and the `candidate` and
  `champion` aliases. The aliases deliberately point to different versions, demonstrating that an
  older approved champion can remain live while a newer candidate is evaluated.
- `dvc-pointers.json` records the six tracked source/evaluation data fingerprints.
- `provenance.json` records when and from which commit the export was produced. `git_dirty: true`
  is expected for this checkpoint export because the submission fixes have not yet been committed;
  regenerate the evidence after the final submission commit.

Regenerate the bundle from the local manifests and registry with:

```bash
MLFLOW_DISABLE_AGENT_HINT=1 uv run ai-support evidence export --output docs/evidence
```

The exported values are supporting evidence, not a replacement for demonstrating MLflow, Metaflow,
DVC, the model aliases, and the rollback procedure in the video.
