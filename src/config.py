"""Shared configuration for SupportFlow AI.

Keeping paths and labels here gives training, inference, tests, and the UI a
single source of truth.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEEDBACK_DATA_DIR = DATA_DIR / "feedback"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "router.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

DEPARTMENTS = (
    "Billing",
    "Technical Support",
    "Account Access",
    "Refund",
    "Product Feedback",
)

RANDOM_STATE = 42
DEFAULT_REVIEW_THRESHOLD = 0.65
