"""End-to-end in-memory RAG release and MLflow tracking test."""

from pathlib import Path

from ai_support.rag.config import load_rag_config
from ai_support.rag.pipeline import run_rag_pipeline
from ai_support.rag.repository import InMemoryRAGRepository
from ai_support.rag.sample_data import generate_sample_corpus

from .helpers import FakeEmbedder


def _temporary_config(tmp_path: Path) -> Path:
    text = Path("configs/rag/student.toml").read_text(encoding="utf-8")
    replacements = {
        'source_dir = "data/rag/student/documents"': (f'source_dir = "{tmp_path / "documents"}"'),
        'benchmark_path = "data/rag/student/benchmark.jsonl"': (
            f'benchmark_path = "{tmp_path / "benchmark.jsonl"}"'
        ),
        'release_dir = "artifacts/rag/student/releases"': (
            f'release_dir = "{tmp_path / "releases"}"'
        ),
        'tracking_uri = "sqlite:///artifacts/mlflow/mlflow.db"': (
            f'tracking_uri = "sqlite:///{tmp_path / "mlflow.db"}"'
        ),
        'artifact_root = "artifacts/mlflow/files"': (
            f'artifact_root = "{tmp_path / "mlflow-files"}"'
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    path = tmp_path / "student.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_pipeline_registers_immutable_candidate_manifest(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    config = load_rag_config(config_path, tmp_path)
    generate_sample_corpus(config)
    (tmp_path / "uv.lock").write_text("test-lock", encoding="utf-8")
    repository = InMemoryRAGRepository()

    result = run_rag_pipeline(
        config_path,
        tmp_path,
        "metaflow-rag-test",
        repository=repository,
        embedder=FakeEmbedder(),
    )

    assert result.gate.passed
    assert result.manifest["status"] == "candidate"
    assert result.manifest["metaflow_run_id"] == "metaflow-rag-test"
    assert result.manifest["mlflow_run_id"]
    assert result.manifest["embedding"]["revision"]
    assert (config.release_dir / f"{result.release_id}.json").is_file()
    assert (config.release_dir.parent / "latest.json").is_file()
    assert repository.manifests[result.release_id]["status"] == "candidate"
