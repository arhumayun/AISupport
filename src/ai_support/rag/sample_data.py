"""Deterministic, synthetic approved knowledge and benchmark generation."""

# ruff: noqa: E501 - Knowledge prose stays readable as complete source sentences.

from datetime import date
from hashlib import sha256
from pathlib import Path

from ai_support.rag.config import RAGConfig
from ai_support.rag.loaders import content_hash, write_metadata
from ai_support.rag.schema import BenchmarkCase, DocumentMetadata

_CORPUS = {
    "student": {
        "Registry": (
            "Timetable and enrolment",
            "Students can view the current timetable in the student portal after enrolment is confirmed. If a timetable is missing, first verify subject enrolment and allow one business day for synchronization.",
            "How can I find a missing class timetable?",
            ["student portal", "enrolment", "one business day"],
        ),
        "Finance": (
            "Fee payments",
            "Tuition payment receipts appear in the finance section of the student portal. Card payments may take two business days to be reflected before Finance should be contacted.",
            "When will my tuition payment appear?",
            ["finance section", "two business days"],
        ),
        "IT": (
            "Account access",
            "Students should use the self-service password reset page for account access. Multi-factor authentication must be re-registered after changing a mobile device.",
            "What should I do after changing my phone for MFA?",
            ["multi-factor", "re-registered"],
        ),
        "Facilities": (
            "Lost property",
            "Lost property found on campus is recorded at the main reception desk. Students should provide the building, room, date, and a non-sensitive description of the item.",
            "Where do students ask about lost property?",
            ["main reception", "building", "room"],
        ),
        "Careers": (
            "Career appointments",
            "Students can book a CV review or internship guidance appointment through the careers portal. Appointment availability is released weekly.",
            "How can I book a CV review?",
            ["careers portal", "weekly"],
        ),
    },
    "staff": {
        "Registry": (
            "Grade submission",
            "Teaching staff submit final grades through the approved grade submission workspace. Corrections after the deadline require Registry review and an authorised change form.",
            "How are grades corrected after the deadline?",
            ["Registry review", "authorised change form"],
        ),
        "Finance": (
            "Expense reimbursement",
            "Staff expense claims require an approved cost centre, receipt, and line manager authorization. Complete claims are reviewed by Finance in five business days.",
            "What is required for a staff expense claim?",
            ["cost centre", "receipt", "line manager"],
        ),
        "IT": (
            "Software installation",
            "Staff software installations must use the managed software catalogue. Applications absent from the catalogue require an IT security and licence review.",
            "How can staff request software not in the catalogue?",
            ["IT security", "licence review"],
        ),
        "Facilities": (
            "Room maintenance",
            "Facilities requests for teaching rooms should include the building, room number, fault category, and safety impact. Urgent safety hazards must also be reported by phone.",
            "What details belong in a teaching room maintenance request?",
            ["room number", "fault category", "safety impact"],
        ),
        "HR": (
            "Leave records",
            "Staff should compare leave balances with approved requests in the HR portal. A discrepancy should be sent to HR with the request dates, without attaching medical information.",
            "What should staff provide for an incorrect leave balance?",
            ["request dates", "without attaching medical information"],
        ),
    },
}


def generate_sample_corpus(config: RAGConfig) -> tuple[list[Path], list[BenchmarkCase]]:
    """Write an audience-isolated knowledge corpus and benchmark."""

    config.source_dir.mkdir(parents=True, exist_ok=True)
    cases: list[BenchmarkCase] = []
    paths: list[Path] = []
    for department, (title, body, question, terms) in _CORPUS[config.audience].items():
        document_id = f"{config.audience}-{department.lower()}-001"
        path = config.source_dir / f"{department.lower()}.md"
        path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
        metadata = DocumentMetadata(
            document_id=document_id,
            audience=config.audience,
            department=department,
            source=f"synthetic://{config.audience}/{department.lower()}",
            source_version="2026.11",
            content_hash=content_hash(path),
            approval_status="approved",
            review_date=date(2027, 12, 31),
            effective_date=date(2026, 1, 1),
            access_level=config.audience,
        )
        write_metadata(path, metadata)
        paths.append(path)
        cases.append(
            BenchmarkCase(
                question_id=f"{config.audience}-{department.lower()}-q1",
                question=question,
                expected_document_ids=[document_id],
                answer_terms=terms,
                department=department,
            )
        )
    config.benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    config.benchmark_path.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8"
    )
    return paths, cases


def corpus_fingerprint(config: RAGConfig) -> str:
    digest = sha256()
    for path in sorted(config.source_dir.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(config.source_dir).as_posix().encode())
            digest.update(path.read_bytes())
    digest.update(config.benchmark_path.read_bytes())
    return digest.hexdigest()
