"""
Tests for TriageIQ inference / predictor layer.
"""
from __future__ import annotations

import pytest
import pandas as pd

from triage.inference.predictor import TriagePredictor, validate_prediction_frame


# ── validate_prediction_frame ────────────────────────────────────────────────

def test_validate_prediction_frame_accepts_required_columns() -> None:
    validate_prediction_frame(pd.DataFrame([{"title": "A", "body": "B"}]))


def test_validate_prediction_frame_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="body, title"):
        validate_prediction_frame(pd.DataFrame([{"title": "A"}]))


def test_validate_prediction_frame_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError):
        validate_prediction_frame(pd.DataFrame(columns=["title"]))


# ── TriagePredictor.predict_ticket ────────────────────────────────────────────

@pytest.fixture(scope="module")
def predictor() -> TriagePredictor:
    """Load predictor once per module — models must be trained."""
    return TriagePredictor()


def test_predict_ticket_returns_required_keys(
    predictor: TriagePredictor, sample_ticket: dict
) -> None:
    """predict_ticket must return a dict with category, priority, intent."""
    result = predictor.predict_ticket(sample_ticket["title"], sample_ticket["body"])
    assert "category" in result
    assert "priority" in result
    assert "intent" in result


def test_predict_ticket_category_is_list(
    predictor: TriagePredictor, sample_ticket: dict
) -> None:
    result = predictor.predict_ticket(sample_ticket["title"], sample_ticket["body"])
    assert isinstance(result["category"], list)
    assert len(result["category"]) >= 1


def test_predict_ticket_category_items_have_label_and_confidence(
    predictor: TriagePredictor, sample_ticket: dict
) -> None:
    result = predictor.predict_ticket(sample_ticket["title"], sample_ticket["body"])
    for item in result["category"]:
        assert "label" in item
        assert "confidence" in item
        assert isinstance(item["label"], str)
        assert isinstance(item["confidence"], float)


def test_predict_ticket_priority_label_is_valid(
    predictor: TriagePredictor,
    sample_ticket: dict,
    valid_priority_labels: set,
) -> None:
    result = predictor.predict_ticket(sample_ticket["title"], sample_ticket["body"])
    assert result["priority"]["label"] in valid_priority_labels


def test_predict_ticket_intent_label_is_valid(
    predictor: TriagePredictor,
    sample_ticket: dict,
    valid_intent_labels: set,
) -> None:
    result = predictor.predict_ticket(sample_ticket["title"], sample_ticket["body"])
    assert result["intent"]["label"] in valid_intent_labels


def test_predict_ticket_confidence_scores_in_range(
    predictor: TriagePredictor, sample_ticket: dict
) -> None:
    result = predictor.predict_ticket(sample_ticket["title"], sample_ticket["body"])
    assert 0.0 <= result["priority"]["confidence"] <= 1.0
    assert 0.0 <= result["intent"]["confidence"] <= 1.0
    for cat in result["category"]:
        assert 0.0 <= cat["confidence"] <= 1.0


def test_predict_ticket_title_only(predictor: TriagePredictor) -> None:
    """Should succeed with only a title and empty body."""
    result = predictor.predict_ticket("App is crashing", "")
    assert "priority" in result
    assert "intent" in result


# ── Batch prediction ──────────────────────────────────────────────────────────

def test_predict_batch_processes_multiple_tickets(
    predictor: TriagePredictor, sample_batch: list[dict]
) -> None:
    df = pd.DataFrame(sample_batch)
    results = predictor.predict_batch(df)
    assert len(results) == len(sample_batch)


def test_predict_batch_each_result_has_required_keys(
    predictor: TriagePredictor, sample_batch: list[dict]
) -> None:
    df = pd.DataFrame(sample_batch)
    results = predictor.predict_batch(df)
    for result in results:
        assert "category" in result
        assert "priority" in result
        assert "intent" in result


@pytest.mark.parametrize(
    "title,body",
    [
        ("Payment failed", "My payment was declined"),
        ("Cancel account", "Please cancel my account"),
        ("Feature request", "I want dark mode"),
        ("Bug: crash on save", "App crashes when I save a document"),
    ],
)
def test_predict_ticket_parametrized(
    predictor: TriagePredictor,
    title: str,
    body: str,
    valid_priority_labels: set,
    valid_intent_labels: set,
) -> None:
    """Parametrized smoke test across diverse ticket types."""
    result = predictor.predict_ticket(title, body)
    assert result["priority"]["label"] in valid_priority_labels
    assert result["intent"]["label"] in valid_intent_labels
    assert result["category"]
