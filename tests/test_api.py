"""
Tests for the TriageIQ FastAPI REST API endpoints.

These tests use httpx.AsyncClient with ASGITransport (no live server needed).
The models must be trained before running these tests.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient


# ── Health endpoints ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root_returns_200(api_client: AsyncClient) -> None:
    """Root endpoint should always be reachable and return 200."""
    response = await api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert "version" in data
    assert "status" in data


@pytest.mark.asyncio
async def test_health_endpoint_returns_200(api_client: AsyncClient) -> None:
    """Health endpoint must return 200 with expected fields."""
    response = await api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "version" in data
    assert "backend" in data
    assert "app_name" in data


@pytest.mark.asyncio
async def test_health_app_name_is_triageiq(api_client: AsyncClient) -> None:
    response = await api_client.get("/health")
    data = response.json()
    assert data["app_name"] == "TriageIQ"


# ── Authentication ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_without_api_key_returns_401(
    api_client: AsyncClient, sample_ticket: dict
) -> None:
    """POST /predict without an API key must return 401."""
    response = await api_client.post("/predict", json=sample_ticket)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_predict_with_wrong_api_key_returns_403(
    api_client: AsyncClient, sample_ticket: dict
) -> None:
    """POST /predict with a wrong API key must return 403."""
    response = await api_client.post(
        "/predict",
        json=sample_ticket,
        headers={"X-API-Key": "wrong-key-abc"},
    )
    assert response.status_code == 403


# ── Single prediction ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_valid_ticket_returns_200(
    api_client: AsyncClient,
    sample_ticket: dict,
    api_headers: dict,
) -> None:
    """Valid ticket should return 200 with prediction fields."""
    response = await api_client.post("/predict", json=sample_ticket, headers=api_headers)
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "priority" in data
    assert "intent" in data
    assert "latency_ms" in data
    assert "model_backend" in data


@pytest.mark.asyncio
async def test_predict_category_is_list(
    api_client: AsyncClient, sample_ticket: dict, api_headers: dict
) -> None:
    """Category should be a non-empty list of label/confidence objects."""
    response = await api_client.post("/predict", json=sample_ticket, headers=api_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["category"], list)
    assert len(data["category"]) >= 1
    for item in data["category"]:
        assert "label" in item
        assert "confidence" in item


@pytest.mark.asyncio
async def test_predict_confidence_scores_in_range(
    api_client: AsyncClient, sample_ticket: dict, api_headers: dict
) -> None:
    """All confidence scores must be between 0 and 1 inclusive."""
    response = await api_client.post("/predict", json=sample_ticket, headers=api_headers)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["priority"]["confidence"] <= 1.0
    assert 0.0 <= data["intent"]["confidence"] <= 1.0
    for cat in data["category"]:
        assert 0.0 <= cat["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_predict_priority_label_is_valid(
    api_client: AsyncClient,
    sample_ticket: dict,
    api_headers: dict,
    valid_priority_labels: set,
) -> None:
    """Priority label must be one of the known values."""
    response = await api_client.post("/predict", json=sample_ticket, headers=api_headers)
    assert response.status_code == 200
    label = response.json()["priority"]["label"]
    assert label in valid_priority_labels, f"Unexpected priority label: {label!r}"


@pytest.mark.asyncio
async def test_predict_intent_label_is_valid(
    api_client: AsyncClient,
    sample_ticket: dict,
    api_headers: dict,
    valid_intent_labels: set,
) -> None:
    """Intent label must be one of the known values."""
    response = await api_client.post("/predict", json=sample_ticket, headers=api_headers)
    assert response.status_code == 200
    label = response.json()["intent"]["label"]
    assert label in valid_intent_labels, f"Unexpected intent label: {label!r}"


@pytest.mark.asyncio
async def test_predict_latency_is_positive(
    api_client: AsyncClient, sample_ticket: dict, api_headers: dict
) -> None:
    response = await api_client.post("/predict", json=sample_ticket, headers=api_headers)
    assert response.status_code == 200
    assert response.json()["latency_ms"] > 0


# ── Validation ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_missing_title_returns_422(
    api_client: AsyncClient, api_headers: dict
) -> None:
    """Request without 'title' field must return 422 Unprocessable Entity."""
    response = await api_client.post(
        "/predict",
        json={"body": "some description"},
        headers=api_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_blank_title_returns_422(
    api_client: AsyncClient, api_headers: dict
) -> None:
    """A blank/whitespace-only title must be rejected with 422."""
    response = await api_client.post(
        "/predict",
        json={"title": "   ", "body": "some description"},
        headers=api_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_body_is_optional(
    api_client: AsyncClient, api_headers: dict
) -> None:
    """Request with only a title (no body) should succeed."""
    response = await api_client.post(
        "/predict",
        json={"title": "System is down"},
        headers=api_headers,
    )
    assert response.status_code == 200


# ── Batch prediction ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_predict_returns_correct_count(
    api_client: AsyncClient,
    sample_batch: list[dict],
    api_headers: dict,
) -> None:
    """Batch response must contain one prediction per input ticket."""
    payload = {"tickets": sample_batch}
    response = await api_client.post("/predict/batch", json=payload, headers=api_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == len(sample_batch)
    assert len(data["predictions"]) == len(sample_batch)


@pytest.mark.asyncio
async def test_batch_predict_each_has_required_fields(
    api_client: AsyncClient,
    sample_batch: list[dict],
    api_headers: dict,
) -> None:
    """Every prediction in a batch must have category, priority, intent."""
    payload = {"tickets": sample_batch}
    response = await api_client.post("/predict/batch", json=payload, headers=api_headers)
    assert response.status_code == 200
    for pred in response.json()["predictions"]:
        assert "category" in pred
        assert "priority" in pred
        assert "intent" in pred


@pytest.mark.asyncio
async def test_batch_predict_empty_list_returns_422(
    api_client: AsyncClient, api_headers: dict
) -> None:
    """An empty ticket list must be rejected with 422."""
    response = await api_client.post(
        "/predict/batch",
        json={"tickets": []},
        headers=api_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_predict_without_api_key_returns_401(
    api_client: AsyncClient, sample_batch: list[dict]
) -> None:
    response = await api_client.post("/predict/batch", json={"tickets": sample_batch})
    assert response.status_code == 401
