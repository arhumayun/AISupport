import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_support.deployment.catalog import build_experiment_catalog
from ai_support.deployment.minio_init import ensure_bucket
from ai_support.deployment.performance import measure, percentile, write_measurement
from ai_support.rag.tracking import select_release


def test_catalog_collects_router_and_rag_manifests(tmp_path: Path) -> None:
    router = tmp_path / "artifacts/router/student/latest_run.json"
    release = tmp_path / "artifacts/rag/staff/releases/release-1.json"
    router.parent.mkdir(parents=True)
    release.parent.mkdir(parents=True)
    router.write_text(json.dumps({"mlflow_run_id": "router-run", "metrics": {"accuracy": 0.9}}))
    release.write_text(
        json.dumps({"release_id": "release-1", "audience": "staff", "mlflow_run_id": "rag-run"})
    )

    output = build_experiment_catalog(tmp_path)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(record["kind"], record["audience"]) for record in records] == [
        ("router", "student"),
        ("rag", "staff"),
    ]


def test_select_release_validates_gate_and_audience(tmp_path: Path) -> None:
    releases = tmp_path / "artifacts/rag/student/releases"
    releases.mkdir(parents=True)
    (releases / "approved.json").write_text(
        json.dumps({"release_id": "approved", "audience": "student", "status": "candidate"})
    )
    (releases / "rejected.json").write_text(
        json.dumps({"release_id": "rejected", "audience": "student", "status": "rejected"})
    )

    latest = select_release(tmp_path, "student", "approved")
    assert json.loads(latest.read_text())["release_id"] == "approved"
    with pytest.raises(ValueError, match="promotion gate"):
        select_release(tmp_path, "student", "rejected")
    with pytest.raises(FileNotFoundError):
        select_release(tmp_path, "staff", "approved")
    (releases / "mismatch.json").write_text(
        json.dumps({"release_id": "different", "audience": "student", "status": "candidate"})
    )
    with pytest.raises(ValueError, match="identity"):
        select_release(tmp_path, "student", "mismatch")


def test_performance_helpers_are_deterministic(tmp_path: Path) -> None:
    assert percentile([5.0, 1.0, 4.0, 2.0, 3.0], 0.95) == 5.0
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.95)
    with pytest.raises(ValueError, match="positive"):
        measure("http://example.test", requests=0)
    output = write_measurement({"requests": 2}, tmp_path / "result.json")
    assert json.loads(output.read_text()) == {"requests": 2}


def test_measure_records_success_http_error_and_connection_error(monkeypatch) -> None:
    responses = iter([SimpleNamespace(status=200), SimpleNamespace(status=503), OSError("down")])

    class ResponseContext:
        def __init__(self, response):
            self.response = response

        def __enter__(self):
            if isinstance(self.response, OSError):
                raise self.response
            return self.response

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "ai_support.deployment.performance.urllib.request.urlopen",
        lambda *_args, **_kwargs: ResponseContext(next(responses)),
    )

    result = measure("http://example.test/health", requests=3)

    assert result["errors"] == 2
    assert result["error_rate"] == pytest.approx(2 / 3)


def test_minio_bucket_initialization_is_idempotent(monkeypatch) -> None:
    class FakeClient:
        created: list[str] = []

        def list_buckets(self):
            return {"Buckets": []}

        def create_bucket(self, *, Bucket):
            self.created.append(Bucket)

    client = FakeClient()
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setattr("ai_support.deployment.minio_init.boto3.client", lambda *_a, **_k: client)

    ensure_bucket()

    assert client.created == ["mlflow"]


def test_minio_existing_bucket_is_not_recreated(monkeypatch) -> None:
    class FakeClient:
        def list_buckets(self):
            return {"Buckets": [{"Name": "mlflow"}]}

        def create_bucket(self, *, Bucket):  # pragma: no cover - assertion guard
            raise AssertionError(f"unexpected recreation of {Bucket}")

    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setattr(
        "ai_support.deployment.minio_init.boto3.client", lambda *_a, **_k: FakeClient()
    )

    ensure_bucket()


def test_minio_bucket_initialization_retries_then_fails(monkeypatch) -> None:
    class FailedClient:
        def list_buckets(self):
            raise OSError("not ready")

    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setattr(
        "ai_support.deployment.minio_init.boto3.client", lambda *_a, **_k: FailedClient()
    )
    monkeypatch.setattr("ai_support.deployment.minio_init.time.sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="did not become ready"):
        ensure_bucket()
