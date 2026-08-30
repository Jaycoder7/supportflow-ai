"""Model construction, evaluation, and confidence-threshold selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def build_pipeline(classifier: ClassifierMixin) -> Pipeline:
    """Use identical text features for every classifier comparison."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    """Return the two required classical NLP model candidates."""

    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(
                max_iter=1_000,
                class_weight="balanced",
                random_state=42,
            )
        ),
        "multinomial_naive_bayes": build_pipeline(MultinomialNB(alpha=0.5)),
    }


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Calculate overall and per-class classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def threshold_table(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> list[dict[str, float]]:
    """Measure coverage and accepted-ticket quality over candidate thresholds."""

    confidence = probabilities.max(axis=1)
    predictions = classes[probabilities.argmax(axis=1)]
    truth = np.asarray(y_true)
    rows: list[dict[str, float]] = []

    for threshold in np.arange(0.35, 0.91, 0.05):
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        selective_accuracy = (
            float((predictions[accepted] == truth[accepted]).mean())
            if accepted.any()
            else 0.0
        )
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "coverage": coverage,
                "review_rate": 1.0 - coverage,
                "selective_accuracy": selective_accuracy,
            }
        )

    return rows


def choose_review_threshold(
    rows: list[dict[str, float]],
    target_accuracy: float = 0.995,
    target_review_rate: float = 0.05,
) -> float:
    """Choose a threshold near the review budget that meets the quality target."""

    qualifying = [row for row in rows if row["selective_accuracy"] >= target_accuracy]
    if qualifying:
        best = min(
            qualifying,
            key=lambda row: (
                abs(row["review_rate"] - target_review_rate),
                -row["selective_accuracy"],
            ),
        )
        return float(best["threshold"])

    fallback = max(
        rows,
        key=lambda row: (row["selective_accuracy"], row["coverage"]),
    )
    return float(fallback["threshold"])
