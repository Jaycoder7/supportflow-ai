"""Tests for evaluation and confidence-threshold behavior."""

import numpy as np

from src.modeling import choose_review_threshold, threshold_table


def test_threshold_table_reports_coverage_and_review_rate() -> None:
    probabilities = np.array([[0.8, 0.2], [0.55, 0.45]])
    rows = threshold_table(
        np.array(["A", "B"]),
        probabilities,
        np.array(["A", "B"]),
    )
    row = next(item for item in rows if item["threshold"] == 0.6)
    assert row["coverage"] == 0.5
    assert row["review_rate"] == 0.5
    assert row["selective_accuracy"] == 1.0


def test_threshold_selection_balances_quality_and_review_budget() -> None:
    rows = [
        {
            "threshold": 0.5,
            "coverage": 0.98,
            "review_rate": 0.02,
            "selective_accuracy": 0.996,
        },
        {
            "threshold": 0.6,
            "coverage": 0.95,
            "review_rate": 0.05,
            "selective_accuracy": 0.997,
        },
        {
            "threshold": 0.7,
            "coverage": 0.90,
            "review_rate": 0.10,
            "selective_accuracy": 0.999,
        },
    ]
    assert choose_review_threshold(rows) == 0.6
