"""Transparent urgency rules for support-ticket prioritization."""

from __future__ import annotations

from dataclasses import dataclass

from src.data import clean_ticket_text


CRITICAL_TERMS = (
    "security breach",
    "account hacked",
    "stolen account",
    "data loss",
    "lost all data",
    "system outage",
    "service outage",
    "system is down",
    "locked out",
    "cannot access my account",
)

HIGH_TERMS = (
    "urgent",
    "asap",
    "charged twice",
    "duplicate charge",
    "payment failed",
    "cannot pay",
    "can't pay",
    "cannot work",
    "can't work",
    "keeps crashing",
    "multiple times",
)


@dataclass(frozen=True)
class PriorityResult:
    level: str
    matched_terms: tuple[str, ...]
    method: str = "rules-v1"


def assign_priority(text: str, department: str | None = None) -> PriorityResult:
    """Assign priority with auditable rules instead of unsupported ML labels."""

    normalized = clean_ticket_text(text).lower()
    critical_matches = tuple(term for term in CRITICAL_TERMS if term in normalized)
    if critical_matches:
        return PriorityResult("Critical", critical_matches)

    high_matches = tuple(term for term in HIGH_TERMS if term in normalized)
    if high_matches:
        return PriorityResult("High", high_matches)

    if department == "Product Feedback":
        return PriorityResult("Low", ())

    return PriorityResult("Medium", ())

