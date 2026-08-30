"""Model loading and complete ticket-routing inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.data import clean_ticket_text
from src.explain import explain_prediction
from src.priority import assign_priority


def load_router(model_path: Path, metadata_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load the trained pipeline and its decision-policy metadata."""

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    return model, metadata


def predict_ticket(
    text: str,
    model: Pipeline,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return route, confidence, review status, priority, and explanation."""

    cleaned = clean_ticket_text(text)
    if len(cleaned) < 8:
        raise ValueError("Please enter at least 8 characters describing the issue.")

    probabilities = model.predict_proba([cleaned])[0]
    classes = model.named_steps["classifier"].classes_
    predicted_index = int(np.argmax(probabilities))
    department = str(classes[predicted_index])
    confidence = float(probabilities[predicted_index])
    review_threshold = float(metadata["review_threshold"])
    priority = assign_priority(cleaned, department)

    return {
        "cleaned_text": cleaned,
        "department": department,
        "confidence": confidence,
        "needs_review": confidence < review_threshold,
        "review_threshold": review_threshold,
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities, strict=True)
        },
        "priority": priority.level,
        "priority_method": priority.method,
        "priority_terms": list(priority.matched_terms),
        "explanation": explain_prediction(model, cleaned, department),
        "model_version": metadata["model_version"],
    }

