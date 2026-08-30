"""Dataset loading, validation, cleaning, and label construction."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import DEPARTMENTS


REQUIRED_RAW_COLUMNS = {
    "instruction",
    "intent",
    "category",
    "tags",
    "response",
}

# Fine-grained intents are aggregated into the five business departments. The
# intent and source category create the target but must never be model inputs.
INTENT_TO_DEPARTMENT = {
    "pay": "Billing",
    "payment_issue": "Billing",
    "payment_methods": "Billing",
    "technical_issue": "Technical Support",
    "product_issue": "Technical Support",
    "recover_password": "Account Access",
    "refund_policy": "Refund",
    "refund_status": "Refund",
    "request_refund": "Refund",
    "submit_feedback": "Product Feedback",
    "submit_product_feedback": "Product Feedback",
    "submit_product_idea": "Product Feedback",
}

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_LONG_NUMBER_PATTERN = re.compile(r"\b\d{4,}\b")
_ENTITY_PATTERN = re.compile(r"\{\{[^{}]+\}\}")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def validate_raw_data(frame: pd.DataFrame) -> None:
    """Raise a clear error when the downloaded dataset schema has changed."""

    missing = REQUIRED_RAW_COLUMNS.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Raw dataset is missing required columns: {missing_list}")


def clean_ticket_text(value: object) -> str:
    """Normalize ticket text and redact common personal-data patterns."""

    if pd.isna(value):
        return ""

    text = str(value)
    text = _EMAIL_PATTERN.sub("<EMAIL>", text)
    text = _URL_PATTERN.sub("<URL>", text)
    text = _LONG_NUMBER_PATTERN.sub("<NUMBER>", text)
    text = _ENTITY_PATTERN.sub("<ENTITY>", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def build_processed_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Create the leakage-safe modeling table from audited raw records."""

    validate_raw_data(raw)

    processed = pd.DataFrame(
        {
            "source_row": raw.index,
            "text": raw["instruction"].map(clean_ticket_text),
            "department": raw["intent"].map(INTENT_TO_DEPARTMENT),
            "source_intent": raw["intent"],
        }
    )

    processed = processed.dropna(subset=["department"])
    processed = processed.loc[processed["text"].str.len() >= 20]
    processed = processed.drop_duplicates(subset=["text"], keep="first")
    processed = processed.reset_index(drop=True)

    unexpected = set(processed["department"]).difference(DEPARTMENTS)
    if unexpected:
        raise ValueError(f"Unexpected department labels: {sorted(unexpected)}")

    return processed


def load_and_prepare(raw_path: Path) -> pd.DataFrame:
    """Load the source parquet file and return the processed modeling table."""

    raw = pd.read_parquet(raw_path)
    return build_processed_dataset(raw)
