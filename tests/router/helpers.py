"""Test configuration factories."""

from pathlib import Path

from ai_support.router.config import (
    DataConfig,
    MlflowConfig,
    ModelConfig,
    OutputConfig,
    PromotionConfig,
    RouterConfig,
)


def router_config(root: Path, audience: str = "student") -> RouterConfig:
    return RouterConfig(
        audience=audience,
        random_seed=1234,
        data=DataConfig(
            dataset_path=root / audience / "tickets.csv",
            evaluation_path=root / audience / "evaluation.csv",
            test_fraction=0.2,
        ),
        model=ModelConfig(
            ngram_min=1,
            ngram_max=2,
            min_df=1,
            max_features=2000,
            c=4.0,
            max_iter=300,
            top_k=3,
        ),
        promotion=PromotionConfig(
            min_accuracy=0.9,
            min_macro_f1=0.85,
            min_department_recall=0.75,
            max_calibration_error=0.2,
        ),
        mlflow=MlflowConfig(
            tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
            artifact_root=root / "mlflow-artifacts",
            experiment_name=f"{audience}-test-experiment",
            registered_model_name=f"{audience}-department-router-test",
        ),
        output=OutputConfig(
            predictions_path=root / audience / "predictions.jsonl",
            manifest_path=root / audience / "manifest.json",
        ),
        source_path=root / f"{audience}.toml",
    )
