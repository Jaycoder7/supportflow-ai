"""Append-only correction capture for future model improvement."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from src.config import DEPARTMENTS
from src.data import clean_ticket_text


FEEDBACK_FIELDS = (
    "timestamp",
    "ticket_text",
    "predicted_department",
    "corrected_department",
    "confidence",
    "priority",
    "model_version",
)


def save_correction(
    path: Path,
    *,
    ticket_text: str,
    predicted_department: str,
    corrected_department: str,
    confidence: float,
    priority: str,
    model_version: str,
) -> None:
    """Append one validated, redacted correction record to CSV."""

    if predicted_department not in DEPARTMENTS:
        raise ValueError("Unknown predicted department")
    if corrected_department not in DEPARTMENTS:
        raise ValueError("Unknown corrected department")

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    record = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "ticket_text": clean_ticket_text(ticket_text),
        "predicted_department": predicted_department,
        "corrected_department": corrected_department,
        "confidence": f"{confidence:.6f}",
        "priority": priority,
        "model_version": model_version,
    }

    with path.open("a", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=FEEDBACK_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)

