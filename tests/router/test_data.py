"""Synthetic data, schema, configuration, and preprocessing tests."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from ai_support.router.config import load_router_config
from ai_support.router.preprocessing import (
    TicketTextTransformer,
    chronological_split,
    ticket_content_hash,
)
from ai_support.router.schema import load_ticket_csv, write_ticket_csv
from ai_support.router.synthetic import generate_tickets, infer_intent


def test_synthetic_data_is_exactly_reproducible_and_realistically_assigned() -> None:
    first = generate_tickets("student", rows=100, seed=42)
    second = generate_tickets("student", rows=100, seed=42)

    assert first == second
    assert first != generate_tickets("student", rows=100, seed=43)
    assert len({record.created_date.date() for record in first}) > 30
    assert any(record.department != record.final_resolving_department for record in first)
    assert {record.audience for record in first} == {"student"}


def test_rule_based_intent_does_not_require_department() -> None:
    assert infer_intent("Please explain a tuition refund") == "payment_query"
    assert infer_intent("I need general advice") == "general_information"


def test_generator_rejects_too_few_rows() -> None:
    with pytest.raises(ValueError, match="10 tickets per department"):
        generate_tickets("staff", rows=20, seed=1)


def test_csv_round_trip_and_schema_failures(tmp_path: Path) -> None:
    records = generate_tickets("staff", rows=50, seed=7)
    path = tmp_path / "tickets.csv"
    write_ticket_csv(path, records)

    assert load_ticket_csv(path, "staff") == records
    with pytest.raises(FileNotFoundError, match="not found"):
        load_ticket_csv(tmp_path / "missing.csv", "staff")
    with pytest.raises(ValueError, match="wrong audience"):
        load_ticket_csv(path, "student")

    malformed = tmp_path / "malformed.csv"
    malformed.write_text("ticket_id,audience\n1,staff\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns must be exactly"):
        load_ticket_csv(malformed, "staff")


def test_duplicate_identifiers_are_rejected(tmp_path: Path) -> None:
    records = generate_tickets("student", rows=50, seed=8)
    rows = [record.model_dump(mode="json") for record in records]
    rows[1]["ticket_id"] = rows[0]["ticket_id"]
    path = tmp_path / "duplicates.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="must be unique"):
        load_ticket_csv(path, "student")


def test_chronological_split_removes_cross_boundary_duplicate() -> None:
    records = generate_tickets("student", rows=100, seed=11)
    duplicate_source = records[5]
    records[-1] = records[-1].model_copy(
        update={
            "ticket_subject": duplicate_source.ticket_subject,
            "ticket_description": duplicate_source.ticket_description,
            "final_resolving_department": duplicate_source.final_resolving_department,
        }
    )

    split = chronological_split(records, 0.2)

    assert split.removed_train_duplicates == 1
    assert max(record.created_date for record in split.train) < min(
        record.created_date for record in split.evaluation
    )
    train_hashes = {ticket_content_hash(row) for row in split.train}
    evaluation_hashes = {ticket_content_hash(row) for row in split.evaluation}
    assert not train_hashes & evaluation_hashes


def test_transformer_requires_both_serving_fields() -> None:
    transformer = TicketTextTransformer()
    with pytest.raises(ValueError, match="missing columns"):
        transformer.fit(pd.DataFrame({"ticket_subject": ["hello"]}))


def test_checked_in_config_resolves_paths(tmp_path: Path) -> None:
    source = Path("configs/router/student.toml")
    config = load_router_config(source, tmp_path)

    assert config.audience == "student"
    assert config.data.dataset_path == (tmp_path / "data/router/student/tickets.csv").resolve()
    assert config.mlflow.registered_model_name == "student-department-router"
