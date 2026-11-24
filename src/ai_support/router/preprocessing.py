"""Leakage-safe chronological splitting and shared text preprocessing."""

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ai_support.router.schema import TicketRecord

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Normalize whitespace and case without removing meaningful tokens."""

    return _WHITESPACE.sub(" ", value.strip().lower())


def ticket_content_hash(record: TicketRecord) -> str:
    """Fingerprint ticket text for exact duplicate leakage prevention."""

    content = (
        f"{normalize_text(record.ticket_subject)}\n{normalize_text(record.ticket_description)}"
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetSplit:
    train: list[TicketRecord]
    evaluation: list[TicketRecord]
    removed_train_duplicates: int
    cutoff: str


def chronological_split(records: list[TicketRecord], test_fraction: float) -> DatasetSplit:
    """Hold out the newest tickets and remove cross-boundary duplicates from training."""

    ordered = sorted(records, key=lambda record: (record.created_date, record.ticket_id))
    split_index = max(1, int(len(ordered) * (1 - test_fraction)))
    if split_index >= len(ordered):
        raise ValueError("dataset is too small for the configured chronological holdout")
    initial_train = ordered[:split_index]
    evaluation = ordered[split_index:]
    evaluation_hashes = {ticket_content_hash(record) for record in evaluation}
    train = [
        record for record in initial_train if ticket_content_hash(record) not in evaluation_hashes
    ]
    if not train or not evaluation:
        raise ValueError("chronological split produced an empty partition")

    train_classes = {record.final_resolving_department for record in train}
    evaluation_classes = {record.final_resolving_department for record in evaluation}
    if train_classes != evaluation_classes:
        raise ValueError("every department must appear in both chronological partitions")
    return DatasetSplit(
        train=train,
        evaluation=evaluation,
        removed_train_duplicates=len(initial_train) - len(train),
        cutoff=evaluation[0].created_date.isoformat(),
    )


def records_to_frame(records: list[TicketRecord]) -> pd.DataFrame:
    """Create the serving-compatible feature frame."""

    return pd.DataFrame(
        {
            "ticket_subject": [record.ticket_subject for record in records],
            "ticket_description": [record.ticket_description for record in records],
        }
    )


class TicketTextTransformer(TransformerMixin, BaseEstimator):
    """Combine and normalize the same fields at training and serving time."""

    def fit(self, features: pd.DataFrame, target: Any = None) -> Self:
        self._validate(features)
        return self

    def transform(self, features: pd.DataFrame) -> list[str]:
        self._validate(features)
        subjects = features["ticket_subject"].astype(str)
        descriptions = features["ticket_description"].astype(str)
        pairs = zip(subjects, descriptions, strict=True)
        return [normalize_text(f"{subject} {description}") for subject, description in pairs]

    @staticmethod
    def _validate(features: pd.DataFrame) -> None:
        required = {"ticket_subject", "ticket_description"}
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"router features are missing columns: {sorted(missing)}")
