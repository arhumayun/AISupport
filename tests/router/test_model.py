"""Model training, metrics, promotion, and prediction tests."""

from dataclasses import replace

import numpy as np

from ai_support.router.evaluation import (
    apply_promotion_gate,
    evaluate_model,
    expected_calibration_error,
)
from ai_support.router.model import prediction_frame, train_model
from ai_support.router.preprocessing import chronological_split
from ai_support.router.synthetic import generate_tickets

from .helpers import router_config


def test_router_trains_and_exceeds_declared_goal(tmp_path) -> None:
    config = router_config(tmp_path)
    split = chronological_split(generate_tickets("student", 150, 99), 0.2)
    model = train_model(split.train, config.model, config.random_seed)

    evaluation = evaluate_model(model, split.evaluation, top_k=3)
    gate = apply_promotion_gate(evaluation, config.promotion)

    assert evaluation.metrics["accuracy"] >= 0.9
    assert evaluation.metrics["macro_f1"] >= 0.85
    assert evaluation.metrics["top_k_accuracy"] >= evaluation.metrics["accuracy"]
    assert set(evaluation.per_department) == set(model.classes_)
    assert len(evaluation.confusion_matrix) == 5
    assert gate.passed
    assert not gate.failures

    prediction = model.predict(prediction_frame("Timetable missing", "I cannot see my timetable"))
    assert prediction.tolist() == ["Registry"]


def test_promotion_gate_reports_each_failed_rule(tmp_path) -> None:
    config = router_config(tmp_path)
    split = chronological_split(generate_tickets("staff", 100, 15), 0.2)
    model = train_model(split.train, config.model, config.random_seed)
    evaluation = evaluate_model(model, split.evaluation, top_k=3)
    impossible = replace(
        config.promotion,
        min_accuracy=1.01,
        min_macro_f1=1.01,
        min_department_recall=1.01,
        max_calibration_error=-0.01,
    )

    decision = apply_promotion_gate(evaluation, impossible)

    assert not decision.passed
    assert any("accuracy" in failure for failure in decision.failures)
    assert any("macro_f1" in failure for failure in decision.failures)
    assert any("recall" in failure for failure in decision.failures)
    assert any("calibration" in failure for failure in decision.failures)


def test_calibration_error_matches_perfect_and_overconfident_cases() -> None:
    actual = np.asarray(["A", "B"])
    predicted = np.asarray(["A", "B"])
    assert expected_calibration_error(actual, predicted, np.asarray([1.0, 1.0])) == 0.0
    assert expected_calibration_error(actual, np.asarray(["B", "A"]), np.asarray([1.0, 1.0])) == 1.0
