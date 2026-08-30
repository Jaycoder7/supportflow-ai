"""Download the versioned public dataset and verify its checksum."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "bitext_retail_support.parquet"
DATASET_REVISION = "12dd624ddcd3057382b2faad661bcda1fa869491"
DATASET_URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-retail-ecommerce-llm-chatbot-training-dataset/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
EXPECTED_SHA256 = "c59ca5d9cd2da954eced59747ac6dbf6b8bc964e4b2fa891a22f09d6956e6e48"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and sha256(OUTPUT_PATH) == EXPECTED_SHA256:
        print(f"Dataset already verified: {OUTPUT_PATH}")
        return 0

    temporary_path = OUTPUT_PATH.with_suffix(".parquet.download")
    print(f"Downloading dataset revision {DATASET_REVISION}...")
    urllib.request.urlretrieve(DATASET_URL, temporary_path)

    actual_hash = sha256(temporary_path)
    if actual_hash != EXPECTED_SHA256:
        temporary_path.unlink(missing_ok=True)
        print(
            f"Checksum mismatch: expected {EXPECTED_SHA256}, got {actual_hash}",
            file=sys.stderr,
        )
        return 1

    temporary_path.replace(OUTPUT_PATH)
    print(f"Verified dataset saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
