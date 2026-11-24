"""Router metrics, calibrated confidence reporting, and promotion gate."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from ai_support.router.config import PromotionConfig
from ai_support.router.preprocessing import records_to_frame
from ai_support.router.schema import TicketRecord


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    per_department: dict[str, dict[str, float]]
    labels: list[str]
    confusion_matrix: list[list[int]]
    predictions: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_calibration_error(
    actual: np.ndarray, predicted: np.ndarray, confidence: np.ndarray, bins: int = 10
) -> float:
    """Measure the weighted gap between confidence and observed accuracy."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            bin_accuracy = (actual[mask] == predicted[mask]).mean()
            error += float(mask.mean() * abs(bin_accuracy - confidence[mask].mean()))
    return error


def evaluate_model(model: Pipeline, records: list[TicketRecord], top_k: int) -> EvaluationResult:
    """Evaluate routing quality without fabricating or rounding away results."""

    features = records_to_frame(records)
    actual = np.asarray([record.final_resolving_department for record in records])
    predicted = np.asarray(model.predict(features))
    probabilities = np.asarray(model.predict_proba(features))
    classes = np.asarray(model.classes_)
    confidence = probabilities.max(axis=1)
    k = min(top_k, len(classes))
    top_indices = np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]
    top_labels = classes[top_indices]
    top_k_hits = [truth in labels for truth, labels in zip(actual, top_labels, strict=True)]
    top_k_accuracy = float(np.mean(top_k_hits))

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=classes, zero_division=0
    )
    per_department = {
        str(label): {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(classes)
    }
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(np.mean(f1)),
        "top_k_accuracy": top_k_accuracy,
        "expected_calibration_error": expected_calibration_error(actual, predicted, confidence),
    }
    prediction_rows = [
        {
            "ticket_id": record.ticket_id,
            "actual_department": str(truth),
            "predicted_department": str(prediction),
            "confidence": float(score),
            "top_k_departments": labels.tolist(),
        }
        for record, truth, prediction, score, labels in zip(
            records, actual, predicted, confidence, top_labels, strict=True
        )
    ]
    return EvaluationResult(
        metrics=metrics,
        per_department=per_department,
        labels=classes.tolist(),
        confusion_matrix=confusion_matrix(actual, predicted, labels=classes).tolist(),
        predictions=prediction_rows,
    )


def apply_promotion_gate(evaluation: EvaluationResult, config: PromotionConfig) -> GateDecision:
    """Apply explicit, version-controlled model promotion thresholds."""

    failures: list[str] = []
    metrics = evaluation.metrics
    if metrics["accuracy"] < config.min_accuracy:
        failures.append(f"accuracy {metrics['accuracy']:.4f} < {config.min_accuracy:.4f}")
    if metrics["macro_f1"] < config.min_macro_f1:
        failures.append(f"macro_f1 {metrics['macro_f1']:.4f} < {config.min_macro_f1:.4f}")
    if metrics["expected_calibration_error"] > config.max_calibration_error:
        failures.append(
            "expected_calibration_error "
            f"{metrics['expected_calibration_error']:.4f} > {config.max_calibration_error:.4f}"
        )
    for department, department_metrics in evaluation.per_department.items():
        if department_metrics["recall"] < config.min_department_recall:
            failures.append(
                f"{department} recall {department_metrics['recall']:.4f} "
                f"< {config.min_department_recall:.4f}"
            )
    return GateDecision(passed=not failures, failures=tuple(failures))
