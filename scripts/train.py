"""Train, compare, evaluate, and persist the SupportFlow router."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEPARTMENTS, MODEL_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from src.modeling import (  # noqa: E402
    candidate_models,
    choose_review_threshold,
    evaluate_predictions,
    threshold_table,
)


DATA_PATH = PROCESSED_DATA_DIR / "tickets.csv"
MODEL_PATH = MODEL_DIR / "router.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
SPLITS_PATH = PROCESSED_DATA_DIR / "splits.csv"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "error_analysis.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "figures" / "confusion_matrix.html"


def split_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible 70/15/15 stratified splits."""

    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=frame["department"],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=remainder["department"],
    )
    return train.copy(), validation.copy(), test.copy()


def main() -> int:
    if not DATA_PATH.exists():
        print("Processed data is missing. Run scripts/prepare_data.py first.")
        return 1

    frame = pd.read_csv(DATA_PATH)
    train, validation, test = split_data(frame)

    split_assignments = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ]
    ).sort_values("source_row")
    split_assignments[["source_row", "split"]].to_csv(SPLITS_PATH, index=False)

    labels = list(DEPARTMENTS)
    validation_results: dict[str, dict] = {}
    fitted_models = {}

    for name, pipeline in candidate_models().items():
        pipeline.fit(train["text"], train["department"])
        predictions = pipeline.predict(validation["text"])
        validation_results[name] = evaluate_predictions(
            validation["department"], predictions, labels
        )
        fitted_models[name] = pipeline
        result = validation_results[name]
        print(
            f"{name}: validation accuracy={result['accuracy']:.3f}, "
            f"macro-F1={result['macro_f1']:.3f}"
        )

    selected_name = max(
        validation_results,
        key=lambda name: validation_results[name]["macro_f1"],
    )
    selected_model = fitted_models[selected_name]

    validation_probabilities = selected_model.predict_proba(validation["text"])
    classifier = selected_model.named_steps["classifier"]
    thresholds = threshold_table(
        validation["department"],
        validation_probabilities,
        classifier.classes_,
    )
    review_threshold = choose_review_threshold(thresholds)

    test_predictions = selected_model.predict(test["text"])
    test_probabilities = selected_model.predict_proba(test["text"])
    test_metrics = evaluate_predictions(test["department"], test_predictions, labels)

    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata = {
        "model_version": trained_at,
        "trained_at": trained_at,
        "selected_model": selected_name,
        "review_threshold": review_threshold,
        "departments": labels,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    metrics = {
        "selection_metric": "validation_macro_f1",
        "validation": validation_results,
        "threshold_analysis": thresholds,
        "test": test_metrics,
        "metadata": metadata,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    analysis = test[["source_row", "text", "department"]].copy()
    analysis["predicted_department"] = test_predictions
    analysis["confidence"] = test_probabilities.max(axis=1)
    analysis["review_status"] = analysis["confidence"].map(
        lambda value: "human_review" if value < review_threshold else "auto_route"
    )
    analysis["case_type"] = analysis.apply(
        lambda row: (
            "misclassified"
            if row["department"] != row["predicted_department"]
            else "low_confidence_correct"
        ),
        axis=1,
    )
    misclassified = analysis.loc[analysis["case_type"] == "misclassified"]
    remaining_slots = max(0, 20 - len(misclassified))
    borderline = (
        analysis.loc[analysis["case_type"] == "low_confidence_correct"]
        .sort_values("confidence")
        .head(remaining_slots)
    )
    review_cases = pd.concat([misclassified, borderline]).sort_values(
        ["case_type", "confidence"]
    )
    review_cases["review_note"] = ""
    review_cases.to_csv(ERROR_ANALYSIS_PATH, index=False)

    confusion_frame = pd.DataFrame(
        test_metrics["confusion_matrix"], index=labels, columns=labels
    )
    confusion_figure = px.imshow(
        confusion_frame,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Tickets"},
        color_continuous_scale="Blues",
        aspect="auto",
        title="SupportFlow AI Test Confusion Matrix",
    )
    confusion_figure.write_html(CONFUSION_MATRIX_PATH, include_plotlyjs="cdn")

    print(f"\nSelected model: {selected_name}")
    print(f"Human-review threshold: {review_threshold:.2f}")
    print(
        f"Test accuracy={test_metrics['accuracy']:.3f}, "
        f"macro-F1={test_metrics['macro_f1']:.3f}"
    )
    print(f"Saved model: {MODEL_PATH}")
    print(f"Error-analysis sample: {ERROR_ANALYSIS_PATH}")
    print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
