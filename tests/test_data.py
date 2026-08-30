"""Tests for data validation, privacy cleaning, and label construction."""

import pandas as pd
import pytest

from src.config import DEPARTMENTS
from src.data import build_processed_dataset, clean_ticket_text, validate_raw_data


def sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instruction": [
                "I was charged twice for this subscription.",
                "The application crashes whenever I upload a file.",
                "I cannot sign in to my account after resetting my password.",
                "Please return the duplicate charge to my card.",
                "Please add a dark mode option to the product.",
            ],
            "intent": [
                "payment_issue",
                "technical_issue",
                "recover_password",
                "request_refund",
                "submit_product_idea",
            ],
            "category": ["PAYMENT", "APP_WEBSITE", "ACCOUNT", "RETURNS", "FEEDBACK"],
            "tags": [""] * 5,
            "response": ["Example response"] * 5,
        }
    )


def test_clean_ticket_text_redacts_common_personal_data() -> None:
    cleaned = clean_ticket_text(
        "Email me@example.com at https://example.com about card 1234567890123456."
    )
    assert "me@example.com" not in cleaned
    assert "https://example.com" not in cleaned
    assert "1234567890123456" not in cleaned
    assert "<EMAIL>" in cleaned
    assert "<URL>" in cleaned
    assert "<NUMBER>" in cleaned


def test_validate_raw_data_rejects_schema_changes() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_data(pd.DataFrame({"Combined Text": ["hello"]}))


def test_processed_data_contains_all_five_departments() -> None:
    processed = build_processed_dataset(sample_raw_frame())
    assert set(processed["department"]) == set(DEPARTMENTS)


def test_source_labels_and_responses_are_not_model_inputs() -> None:
    processed = build_processed_dataset(sample_raw_frame())
    assert "intent" not in processed.columns
    assert "category" not in processed.columns
    assert "response" not in processed.columns
