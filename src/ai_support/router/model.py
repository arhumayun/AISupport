"""TF-IDF and logistic regression department model."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ai_support.router.config import ModelConfig
from ai_support.router.preprocessing import TicketTextTransformer, records_to_frame
from ai_support.router.schema import TicketRecord


def build_model(config: ModelConfig, random_seed: int) -> Pipeline:
    """Build the explainable baseline model entirely from configuration."""

    return Pipeline(
        steps=[
            ("text", TicketTextTransformer()),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(config.ngram_min, config.ngram_max),
                    min_df=config.min_df,
                    max_features=config.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=config.c,
                    class_weight="balanced",
                    max_iter=config.max_iter,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def train_model(records: list[TicketRecord], config: ModelConfig, random_seed: int) -> Pipeline:
    """Fit the configured model on final resolving departments."""

    model = build_model(config, random_seed)
    labels = [record.final_resolving_department for record in records]
    return model.fit(records_to_frame(records), labels)


def prediction_frame(subject: str, description: str) -> pd.DataFrame:
    """Build a one-row feature frame for online prediction."""

    return pd.DataFrame({"ticket_subject": [subject], "ticket_description": [description]})
