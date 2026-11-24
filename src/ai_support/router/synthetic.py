"""Deterministic synthetic historical ticket generation."""

import random
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_support.router.schema import DEPARTMENTS, Audience, TicketRecord, write_ticket_csv

_TOPICS: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "student": {
        "Registry": (
            ("Timetable missing", "My class timetable is not visible for the current term"),
            ("Transcript request", "I need an official academic transcript for my application"),
            ("Enrolment issue", "I cannot enrol in a subject shown in my study plan"),
            ("Graduation status", "Please explain my graduation eligibility and ceremony status"),
        ),
        "Finance": (
            ("Tuition payment", "My tuition payment is not reflected on my student account"),
            ("Payment receipt", "I need a receipt for the semester fee payment"),
            ("Refund query", "I want information about the student fee refund process"),
            ("Instalment plan", "Please explain the available tuition instalment schedule"),
        ),
        "IT": (
            ("Password reset", "I cannot sign in and need to reset my university password"),
            ("WiFi access", "The campus wireless network rejects my student credentials"),
            ("Moodle unavailable", "My enrolled subject is missing from the learning portal"),
            ("MFA problem", "The authentication code does not work for my account"),
        ),
        "Facilities": (
            (
                "Lost property",
                "I left personal belongings in a classroom and need lost property help",
            ),
            ("Parking access", "My student parking permit is not opening the campus gate"),
            ("Room maintenance", "The teaching room has a broken desk that needs repair"),
            ("Campus access", "My access card does not unlock the student study area"),
        ),
        "Careers": (
            ("Internship advice", "I need help finding an internship related to my programme"),
            ("CV review", "Can I book a career appointment to review my CV"),
            ("Job portal", "I cannot access the graduate job opportunities portal"),
            ("Career workshop", "Please share the next employability workshop schedule"),
        ),
    },
    "staff": {
        "Registry": (
            ("Grade submission", "I cannot submit final grades for my assigned class"),
            ("Class list", "The current student class list is missing enrolled students"),
            ("Timetable change", "I need a teaching timetable room change reviewed"),
            ("Student record", "An authorised student record needs a registry correction"),
        ),
        "Finance": (
            ("Expense claim", "My approved staff expense claim has not been reimbursed"),
            ("Supplier invoice", "I need help processing a supplier invoice for payment"),
            ("Budget code", "Please confirm the finance budget code for this purchase"),
            ("Purchase approval", "A department purchase request needs finance review"),
        ),
        "IT": (
            ("Software install", "I need approved teaching software installed on my staff laptop"),
            ("Account locked", "My staff account is locked and I cannot sign in"),
            ("Network problem", "The office network connection drops during online meetings"),
            ("Device support", "My university laptop docking station is not detected"),
        ),
        "Facilities": (
            ("Office maintenance", "The air conditioning in the staff office needs maintenance"),
            ("Access card", "My staff card does not open the assigned teaching room"),
            ("Room setup", "I need the event room furniture setup changed"),
            ("Cleaning request", "A shared staff area requires an additional cleaning visit"),
        ),
        "HR": (
            ("Leave balance", "My annual leave balance appears incorrect in the staff system"),
            ("Contract query", "I need clarification about a term in my employment contract"),
            ("Benefits question", "Please explain eligibility for the listed staff benefit"),
            ("Training record", "My completed mandatory training is missing from my HR record"),
        ),
    },
}

_INTENT_RULES = (
    (
        "access_issue",
        re.compile(r"cannot|locked|missing|rejects|does not|not detected|drops", re.I),
    ),
    ("payment_query", re.compile(r"payment|fee|refund|invoice|expense|budget|purchase", re.I)),
    ("schedule_query", re.compile(r"timetable|schedule|appointment|workshop|ceremony", re.I)),
    ("facilities_request", re.compile(r"room|parking|maintenance|cleaning|desk|card", re.I)),
    ("employment_query", re.compile(r"leave|contract|benefit|training", re.I)),
)


def infer_intent(text: str) -> str:
    """Apply a documented keyword taxonomy independent of department labels."""

    for intent, pattern in _INTENT_RULES:
        if pattern.search(text):
            return intent
    return "general_information"


def generate_tickets(audience: Audience, rows: int, seed: int) -> list[TicketRecord]:
    """Generate exactly reproducible tickets spanning a fixed twelve-month period."""

    if rows < len(DEPARTMENTS[audience]) * 10:
        raise ValueError("rows must provide at least 10 tickets per department")
    rng = random.Random(seed)
    departments = DEPARTMENTS[audience]
    start = datetime(2025, 1, 1, tzinfo=UTC)
    records: list[TicketRecord] = []
    prefix = "STU" if audience == "student" else "STF"

    for index in range(rows):
        resolving = departments[index % len(departments)]
        subject, description = rng.choice(_TOPICS[audience][resolving])
        case_number = rng.randint(1000, 9999)
        detailed_description = f"{description}. Reference {case_number}."
        assigned = resolving
        if rng.random() < 0.08:
            assigned = rng.choice([item for item in departments if item != resolving])
        created = start + timedelta(days=rng.randrange(365), seconds=rng.randrange(24 * 60 * 60))
        combined = f"{subject} {detailed_description}"
        records.append(
            TicketRecord(
                ticket_id=f"{prefix}-{index + 1:06d}",
                audience=audience,
                created_date=created,
                ticket_subject=subject,
                ticket_description=detailed_description,
                department=assigned,
                intent=infer_intent(combined),
                final_resolving_department=resolving,
                status=rng.choice(("resolved", "closed")),
            )
        )
    return sorted(records, key=lambda record: (record.created_date, record.ticket_id))


def generate_ticket_file(
    path: Path, audience: Audience, rows: int, seed: int
) -> list[TicketRecord]:
    """Generate and persist a dataset suitable for DVC tracking."""

    records = generate_tickets(audience, rows, seed)
    write_ticket_csv(path, records)
    return records
