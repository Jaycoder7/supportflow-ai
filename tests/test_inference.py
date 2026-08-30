"""Integration tests for saved-model inference and explanations."""

from pathlib import Path

import pytest

from src.config import DEPARTMENTS, METADATA_PATH, MODEL_PATH
from src.inference import load_router, predict_ticket


@pytest.fixture(scope="module")
def router():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        pytest.skip("Train the router before running artifact integration tests")
    return load_router(MODEL_PATH, METADATA_PATH)


def test_prediction_has_valid_probabilities_and_explanation(router) -> None:
    model, metadata = router
    result = predict_ticket("I need a refund for my purchase.", model, metadata)
    assert result["department"] in DEPARTMENTS
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["explanation"]


def test_short_input_is_rejected(router) -> None:
    model, metadata = router
    with pytest.raises(ValueError, match="at least 8 characters"):
        predict_ticket("help", model, metadata)

