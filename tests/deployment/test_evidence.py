import json
from pathlib import Path
from types import SimpleNamespace

import yaml
from mlflow.exceptions import MlflowException

from ai_support.deployment.evidence import (
    _dvc_pointers,
    _git_value,
    _registry_evidence,
    export_evidence,
)


def test_dvc_evidence_reads_versioned_pointers(tmp_path: Path) -> None:
    pointer = tmp_path / "data/router/student/tickets.csv.dvc"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        yaml.safe_dump({"outs": [{"md5": "abc123", "path": "tickets.csv"}]}),
        encoding="utf-8",
    )

    assert _dvc_pointers(tmp_path) == [
        {
            "pointer": "data/router/student/tickets.csv.dvc",
            "path": "tickets.csv",
            "md5": "abc123",
        }
    ]


def test_export_evidence_writes_reviewable_bundle(tmp_path: Path, monkeypatch) -> None:
    router = tmp_path / "artifacts/router/student/latest_run.json"
    release = tmp_path / "artifacts/rag/staff/releases/release-1.json"
    router.parent.mkdir(parents=True)
    release.parent.mkdir(parents=True)
    router.write_text(json.dumps({"mlflow_run_id": "router-run"}), encoding="utf-8")
    release.write_text(
        json.dumps({"release_id": "release-1", "audience": "staff"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "ai_support.deployment.evidence._registry_evidence",
        lambda _root: [{"registered_model": "student-department-router"}],
    )
    monkeypatch.setattr("ai_support.deployment.evidence._git_value", lambda _root, *_a: "value")

    paths = export_evidence(tmp_path, tmp_path / "evidence")

    assert {path.name for path in paths} == {
        "experiment-catalog.jsonl",
        "model-registry.json",
        "dvc-pointers.json",
        "provenance.json",
    }
    assert json.loads((tmp_path / "evidence/model-registry.json").read_text()) == [
        {"registered_model": "student-department-router"}
    ]


def test_export_evidence_captures_git_state_before_writing(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "evidence"

    monkeypatch.setattr("ai_support.deployment.evidence._registry_evidence", lambda _root: [])

    def git_value(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "clean-source-commit"
        assert args == ("status", "--short")
        return "unavailable" if not output.exists() else "evidence was written"

    monkeypatch.setattr("ai_support.deployment.evidence._git_value", git_value)

    export_evidence(tmp_path, output)

    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["git_commit"] == "clean-source-commit"
    assert provenance["git_dirty"] is False


def test_registry_evidence_exports_aliases_and_versions(tmp_path: Path, monkeypatch) -> None:
    config = SimpleNamespace(
        mlflow=SimpleNamespace(
            tracking_uri="sqlite:///test.db", registered_model_name="department-router"
        )
    )

    class Client:
        def get_model_version_by_alias(self, _name, alias):
            return SimpleNamespace(version="2" if alias == "champion" else "3")

        def search_model_versions(self, _filter):
            return [SimpleNamespace(version="2", run_id="run-2", status="READY")]

    monkeypatch.setattr("ai_support.deployment.evidence.load_router_config", lambda *_a: config)
    monkeypatch.setattr("ai_support.deployment.evidence.MlflowClient", lambda **_k: Client())

    records = _registry_evidence(tmp_path)

    assert len(records) == 2
    assert records[0]["aliases"] == {"candidate": "3", "champion": "2"}
    assert records[0]["versions"] == [{"version": "2", "run_id": "run-2", "status": "READY"}]


def test_registry_evidence_reports_unavailable_registry(tmp_path: Path, monkeypatch) -> None:
    config = SimpleNamespace(
        mlflow=SimpleNamespace(
            tracking_uri="sqlite:///missing.db", registered_model_name="department-router"
        )
    )

    class Client:
        def get_model_version_by_alias(self, _name, _alias):
            raise MlflowException("missing")

        def search_model_versions(self, _filter):
            raise MlflowException("missing")

    monkeypatch.setattr("ai_support.deployment.evidence.load_router_config", lambda *_a: config)
    monkeypatch.setattr("ai_support.deployment.evidence.MlflowClient", lambda **_k: Client())

    records = _registry_evidence(tmp_path)

    assert records[0]["aliases"] == {"candidate": "unavailable", "champion": "unavailable"}
    assert records[0]["versions"] == []


def test_git_value_returns_command_output_and_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "ai_support.deployment.evidence.subprocess.run",
        lambda *_a, **_k: SimpleNamespace(stdout="abc123\n"),
    )
    assert _git_value(tmp_path, "rev-parse", "HEAD") == "abc123"

    monkeypatch.setattr(
        "ai_support.deployment.evidence.subprocess.run",
        lambda *_a, **_k: SimpleNamespace(stdout=""),
    )
    assert _git_value(tmp_path, "rev-parse", "HEAD") == "unavailable"
