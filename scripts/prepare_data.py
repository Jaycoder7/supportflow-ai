"""Build the modeling dataset and print an auditable quality summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR  # noqa: E402
from src.data import INTENT_TO_DEPARTMENT, load_and_prepare, validate_raw_data  # noqa: E402


RAW_PATH = RAW_DATA_DIR / "bitext_retail_support.parquet"
PROCESSED_PATH = PROCESSED_DATA_DIR / "tickets.csv"
AUDIT_PATH = REPORTS_DIR / "data_audit.json"


def main() -> int:
    if not RAW_PATH.exists():
        print("Raw dataset is missing. Run scripts/download_data.py first.")
        return 1

    raw = pd.read_parquet(RAW_PATH)
    validate_raw_data(raw)
    processed = load_and_prepare(RAW_PATH)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_PATH, index=False)

    audit = {
        "raw_rows": int(len(raw)),
        "processed_rows": int(len(processed)),
        "raw_exact_duplicates": int(raw.duplicated().sum()),
        "raw_duplicate_texts": int(raw["instruction"].duplicated().sum()),
        "processed_duplicate_texts": int(processed["text"].duplicated().sum()),
        "missing_values_by_raw_column": {
            column: int(count) for column, count in raw.isna().sum().items()
        },
        "department_counts": {
            label: int(count)
            for label, count in processed["department"].value_counts().items()
        },
        "included_intent_mapping": INTENT_TO_DEPARTMENT,
        "source_category_intent_counts": {
            category: {intent: int(count) for intent, count in values.items()}
            for category, values in pd.crosstab(
                raw["category"], raw["intent"]
            ).to_dict(orient="index").items()
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n")

    print(f"Raw rows: {len(raw):,}")
    print(f"Processed rows: {len(processed):,}")
    print("\nDepartment counts:")
    print(processed["department"].value_counts().to_string())
    print(f"\nProcessed data: {PROCESSED_PATH}")
    print(f"Audit report: {AUDIT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
