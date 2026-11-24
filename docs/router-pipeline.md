# Router Pipeline Operations

## Data and labels

Each audience has an independent deterministic dataset containing 600 synthetic tickets over twelve fixed months. Required fields are `ticket_id`, `audience`, `created_date`, subject, description, initially assigned department, rule-based intent, final resolving department, and status.

The final resolving department is the supervised label. Initial assignment has controlled noise to prevent the implementation from assuming that assignment equals resolution. Intent rules operate on keywords and do not derive intent from a department.

Source and derived chronological evaluation CSV files are represented by separate DVC pointer files. The configured development remote is `.dvc-storage`, which is local and ignored by Git.

## Commands

Install exactly the locked environment:

```bash
uv sync --locked
```

Generate both source datasets and refresh their DVC pointers:

```bash
uv run ai-support data generate --audience all --rows 600
uv run dvc add data/router/student/tickets.csv data/router/staff/tickets.csv
```

Run one independently executable Metaflow pipeline:

```bash
uv run ai-support router train --audience student
uv run ai-support router train --audience staff
```

Run both from data generation through prediction output:

```bash
uv run ai-support routers run-all --generate --rows 600
```

After a run, refresh evaluation pointers and push all four datasets to the configured local remote:

```bash
uv run dvc add data/router/student/evaluation.csv data/router/staff/evaluation.csv
uv run dvc push
uv run dvc status
```

## Five pipeline stages

1. Ingestion loads the configured CSV, validates exact columns, Pydantic row types, unique IDs, audience, departments, and class coverage.
2. Preprocessing sorts chronologically, reserves the newest 20%, and removes any normalized duplicate text from training when it appears in evaluation.
3. Training builds one scikit-learn Pipeline containing the shared text transformer, TF-IDF, and logistic regression.
4. Evaluation reports accuracy, macro-F1, per-department precision/recall/F1, top-k accuracy, confusion matrix, and expected calibration error; configured thresholds form the promotion gate.
5. Registration always logs the run and model artifact. A passing model is registered under its audience-specific name and assigned `candidate`; holdout predictions are written as JSON Lines.

## Tracking and inspection

Each MLflow run records the experiment/run IDs, Git commit and dirty flag, source DVC hash, Metaflow execution ID, configuration, seed, UV lock digest, evaluation dataset version, parameters, metrics, and artifacts.

Start the local UI:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db \
  --default-artifact-root ./artifacts/mlflow/files \
  --host 127.0.0.1 --port 5000
```

Do not expose this development server publicly.

## Promotion and rollback

Promotion is a separate approval action:

```bash
uv run ai-support router approve --audience student --model-reference candidate
```

An explicit existing version can be approved instead, which reassigns `champion` without retraining:

```bash
uv run ai-support router approve --audience student --model-reference 1
```

Test an alias or immutable version directly:

```bash
uv run ai-support router predict \
  --audience student \
  --subject 'Timetable missing' \
  --description 'I cannot see my class timetable' \
  --model-reference 1
```

The same commands apply to staff. Automatic production rollback is intentionally not implemented.

## Reproduction record

To reproduce a historical result, check out its Git commit, restore the DVC dataset hash from that commit and remote, run `uv sync --locked`, and execute the matching version-controlled configuration. Compare the resulting MLflow run's seed, configuration, lock digest, DVC hash, evaluation version, and metrics with the saved run. Metaflow retains the local execution graph and artifacts under its ignored datastore for development; durable shared metadata is future deployment work.
