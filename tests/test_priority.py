"""Tests for transparent priority assignment."""

from src.priority import assign_priority


def test_security_incident_is_critical() -> None:
    result = assign_priority("My account was hacked in a security breach.")
    assert result.level == "Critical"
    assert "security breach" in result.matched_terms


def test_duplicate_charge_is_high_priority() -> None:
    assert assign_priority("I was charged twice this morning.").level == "High"


def test_normal_product_feedback_is_low_priority() -> None:
    result = assign_priority("Please add dark mode.", "Product Feedback")
    assert result.level == "Low"

