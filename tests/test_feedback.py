"""Tests for correction capture."""

import csv

from src.feedback import save_correction


def test_feedback_is_appended_with_redacted_text(tmp_path) -> None:
    path = tmp_path / "feedback.csv"
    save_correction(
        path,
        ticket_text="Email me at person@example.com about my refund.",
        predicted_department="Billing",
        corrected_department="Refund",
        confidence=0.61,
        priority="Medium",
        model_version="test-version",
    )
    with path.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
    assert len(rows) == 1
    assert rows[0]["corrected_department"] == "Refund"
    assert "person@example.com" not in rows[0]["ticket_text"]

