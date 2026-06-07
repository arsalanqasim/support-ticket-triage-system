"""
Pydantic request/response schemas for the TriageIQ REST API.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


# ── Request schemas ───────────────────────────────────────────────────────────

class TicketRequest(BaseModel):
    """A single support ticket to classify."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Short summary / subject line of the ticket.",
        examples=["Login page returns 500 after deploy"],
    )
    body: str = Field(
        default="",
        max_length=10_000,
        description="Full description of the ticket.",
        examples=["Users are unable to log in. The login form returns a 500 Internal Server Error."],
    )

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must not be blank or whitespace-only")
        return v


class BatchTicketRequest(BaseModel):
    """A batch of support tickets to classify in a single call."""

    tickets: list[TicketRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of tickets to classify. Maximum 500 per request.",
    )


# ── Prediction schemas ────────────────────────────────────────────────────────

class PredictionLabel(BaseModel):
    """A predicted label with its confidence score."""

    label: str = Field(..., description="Predicted class label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0–1).")


class TicketPrediction(BaseModel):
    """Full prediction output for a single ticket."""

    category: list[PredictionLabel] = Field(
        ...,
        description="Up to 3 predicted categories with confidence scores.",
    )
    priority: PredictionLabel = Field(
        ...,
        description="Predicted priority level: critical | high | medium | low.",
    )
    intent: PredictionLabel = Field(
        ...,
        description="Predicted intent type (e.g. technical_issue, billing_inquiry).",
    )
    model_backend: str = Field(
        ...,
        description="Which ML backend produced this prediction.",
    )
    latency_ms: float = Field(
        ...,
        description="Time taken to produce the prediction in milliseconds.",
    )


# ── Batch response ────────────────────────────────────────────────────────────

class BatchTicketResponse(BaseModel):
    """Response for a batch prediction request."""

    predictions: list[TicketPrediction]
    total: int = Field(..., description="Number of tickets processed.")
    processed_at: datetime = Field(..., description="UTC timestamp of processing.")
    model_backend: str


# ── Health check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """API health / readiness probe."""

    status: str = Field(..., description="ok | degraded | down")
    models_loaded: bool
    backend: str
    version: str
    app_name: str


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """Standard error response body."""

    detail: str
    code: str | None = None
