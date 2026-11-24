"""MLflow registry, alias approval, loading, and pipeline output tests."""

from pathlib import Path

from mlflow import MlflowClient

from ai_support.router.evaluation import apply_promotion_gate, evaluate_model
from ai_support.router.model import prediction_frame, train_model
from ai_support.router.pipeline import preprocess, write_prediction_output
from ai_support.router.schema import write_ticket_csv
from ai_support.router.synthetic import generate_tickets
from ai_support.router.tracking import (
    approve_model,
    dvc_hash,
    file_sha256,
    git_metadata,
    load_registered_model,
    track_and_register,
)

from .helpers import router_config


def test_mlflow_registration_approval_and_version_loading(tmp_path: Path) -> None:
    config = router_config(tmp_path)
    config.source_path.write_text("test-config", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("test-lock", encoding="utf-8")
    records = generate_tickets("student", 150, config.random_seed)
    write_ticket_csv(config.data.dataset_path, records)
    config.data.dataset_path.with_name("tickets.csv.dvc").write_text(
        "outs:\n- md5: test-dvc-hash\n  path: tickets.csv\n", encoding="utf-8"
    )
    split = preprocess(config, records)
    model = train_model(split.train, config.model, config.random_seed)
    evaluation = evaluate_model(model, split.evaluation, config.model.top_k)
    gate = apply_promotion_gate(evaluation, config.promotion)
    write_prediction_output(config, evaluation)

    manifest = track_and_register(
        config, model, split, evaluation, gate, "metaflow-test-run", tmp_path
    )

    assert manifest["promotion_gate_passed"]
    assert manifest["dvc_dataset_hash"] == "test-dvc-hash"
    assert manifest["metaflow_run_id"] == "metaflow-test-run"
    assert manifest["registered_model_version"] == "1"
    assert config.output.predictions_path.read_text(encoding="utf-8").count("\n") == len(
        split.evaluation
    )

    version = approve_model(config, "candidate")
    assert version == "1"
    client = MlflowClient()
    champion = client.get_model_version_by_alias(config.mlflow.registered_model_name, "champion")
    assert str(champion.version) == "1"

    loaded_by_alias = load_registered_model(config, "champion")
    loaded_by_version = load_registered_model(config, "1")
    features = prediction_frame("Password reset", "I cannot sign in")
    assert loaded_by_alias.predict(features).tolist() == ["IT"]
    assert loaded_by_version.predict(features).tolist() == ["IT"]


def test_hash_helpers_report_tracked_and_untracked_data(tmp_path: Path) -> None:
    data = tmp_path / "tickets.csv"
    data.write_text("content", encoding="utf-8")
    assert dvc_hash(data).startswith("untracked:")
    assert len(file_sha256(data)) == 64


def test_container_git_metadata_uses_build_provenance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AI_SUPPORT_GIT_COMMIT", "abc123")
    monkeypatch.setenv("AI_SUPPORT_GIT_DIRTY", "false")

    assert git_metadata(tmp_path) == ("abc123", False)
