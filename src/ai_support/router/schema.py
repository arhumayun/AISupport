"""Ticket records, dataset validation, and CSV serialization."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Audience = Literal["student", "staff"]
NonEmpty = Annotated[str, Field(min_length=1)]

DEPARTMENTS: dict[str, tuple[str, ...]] = {
    "student": ("Careers", "Facilities", "Finance", "IT", "Registry"),
    "staff": ("Facilities", "Finance", "HR", "IT", "Registry"),
}
TICKET_FIELDS = (
    "ticket_id",
    "audience",
    "created_date",
    "ticket_subject",
    "ticket_description",
    "department",
    "intent",
    "final_resolving_department",
    "status",
)


class TicketRecord(BaseModel):
    """A de-identified synthetic historical support ticket."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticket_id: NonEmpty
    audience: Audience
    created_date: datetime
    ticket_subject: NonEmpty
    ticket_description: NonEmpty
    department: NonEmpty
    intent: NonEmpty
    final_resolving_department: NonEmpty
    status: Literal["resolved", "closed"]


def validate_records(records: list[TicketRecord], audience: Audience) -> None:
    """Validate cross-row and audience-specific dataset constraints."""

    if not records:
        raise ValueError("ticket dataset is empty")
    identifiers = [record.ticket_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("ticket_id values must be unique")
    allowed = set(DEPARTMENTS[audience])
    for record in records:
        if record.audience != audience:
            raise ValueError(f"ticket {record.ticket_id} has the wrong audience")
        if record.final_resolving_department not in allowed:
            raise ValueError(f"ticket {record.ticket_id} has an unsupported resolving department")

    observed = {record.final_resolving_department for record in records}
    missing = allowed - observed
    if missing:
        raise ValueError(f"dataset is missing departments: {sorted(missing)}")


def load_ticket_csv(path: Path, audience: Audience) -> list[TicketRecord]:
    """Load a CSV with clear failures for missing or malformed inputs."""

    if not path.is_file():
        raise FileNotFoundError(f"ticket dataset not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TICKET_FIELDS:
            raise ValueError(f"ticket dataset columns must be exactly: {', '.join(TICKET_FIELDS)}")
        records = [TicketRecord.model_validate(row) for row in reader]
    validate_records(records, audience)
    return records


def write_ticket_csv(path: Path, records: list[TicketRecord]) -> None:
    """Write records deterministically for stable DVC hashes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TICKET_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = record.model_dump(mode="json")
            writer.writerow(row)
