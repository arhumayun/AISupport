"""Audience-isolated registered-model inference service."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_support.monitoring.metrics import ROUTER_CONFIDENCE, ROUTER_PREDICTIONS
from ai_support.router.config import load_router_config
from ai_support.router.model import prediction_frame
from ai_support.router.schema import Audience
from ai_support.router.tracking import load_registered_model


@dataclass(frozen=True)
class DepartmentScore:
    department: str
    probability: float


@dataclass(frozen=True)
class RouterPrediction:
    audience: Audience
    department: str
    confidence: float
    top_departments: tuple[DepartmentScore, ...]
    model_name: str
    model_reference: str


class RouterService:
    """Load only the model registered for the explicitly selected audience."""

    def __init__(self, project_root: Path, model_reference: str = "champion") -> None:
        self.project_root = project_root.resolve()
        self.model_reference = model_reference
        self._models: dict[Audience, Any] = {}

    def predict(
        self, audience: Audience, subject: str, description: str, top_k: int = 3
    ) -> RouterPrediction:
        config = load_router_config(
            self.project_root / "configs" / "router" / f"{audience}.toml",
            self.project_root,
        )
        if audience not in self._models:
            self._models[audience] = load_registered_model(config, self.model_reference)
        model = self._models[audience]
        features = prediction_frame(subject, description)
        probabilities = np.asarray(model.predict_proba(features))[0]
        classes = np.asarray(model.classes_)
        indices = np.argsort(probabilities)[::-1][: min(top_k, len(classes))]
        scores = tuple(
            DepartmentScore(str(classes[index]), float(probabilities[index])) for index in indices
        )
        prediction = RouterPrediction(
            audience=audience,
            department=scores[0].department,
            confidence=scores[0].probability,
            top_departments=scores,
            model_name=config.mlflow.registered_model_name,
            model_reference=self.model_reference,
        )
        ROUTER_PREDICTIONS.labels(audience, prediction.department, self.model_reference).inc()
        ROUTER_CONFIDENCE.labels(audience).observe(prediction.confidence)
        return prediction
