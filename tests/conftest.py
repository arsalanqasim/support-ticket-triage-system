"""
Shared pytest fixtures for TriageIQ test suite.
"""
from __future__ import annotations

import os
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


# ── Environment setup ────────────────────────────────────────────────────────
# Ensure tests always have a valid API key set
os.environ.setdefault("TRIAGEIQ_API_KEY", "test-key-pytest")


# ── Sample data fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_ticket() -> dict[str, str]:
    """A standard test ticket with title and body."""
    return {
        "title": "Login page returns 500 error after latest deploy",
        "body": (
            "Since the deployment at 14:00 UTC, users are unable to log in. "
            "The login form submits and immediately returns a 500 Internal Server Error. "
            "Rollback to the previous version resolved the issue temporarily."
        ),
    }


@pytest.fixture
def sample_batch() -> list[dict[str, str]]:
    """A batch of 5 diverse test tickets."""
    return [
        {
            "title": "Payment not processed",
            "body": "My credit card was charged but the order shows as pending.",
        },
        {
            "title": "Feature request: dark mode",
            "body": "Please add a dark mode option to the dashboard.",
        },
        {
            "title": "App crashes on startup",
            "body": "After the latest update the mobile app crashes immediately on launch.",
        },
        {
            "title": "Cancel my subscription",
            "body": "I would like to cancel my subscription effective immediately.",
        },
        {
            "title": "How do I export my data?",
            "body": "I need to export all my project data to CSV. Where is this option?",
        },
    ]


@pytest.fixture
def valid_priority_labels() -> set[str]:
    return {"critical", "high", "medium", "low"}


@pytest.fixture
def valid_intent_labels() -> set[str]:
    return {
        "technical_issue",
        "billing_inquiry",
        "cancellation_request",
        "product_inquiry",
        "refund_request",
    }


# ── FastAPI test client ───────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def api_client():
    """
    Async HTTP test client for the FastAPI app.

    We directly inject a pre-loaded TriagePredictor into the app module so
    tests don't rely on the ASGI lifespan being triggered by ASGITransport.
    Models must be trained before running these tests.
    """
    import triage.api.app as api_module
    from triage.api.app import app
    from triage.inference.predictor import TriagePredictor

    # Inject predictor into the module-level globals (bypasses lifespan)
    api_module._predictor = TriagePredictor()
    api_module._models_loaded = True

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    # Cleanup
    api_module._predictor = None
    api_module._models_loaded = False


@pytest.fixture
def api_key() -> str:
    """The API key configured for tests via environment variable."""
    return os.environ.get("TRIAGEIQ_API_KEY", "test-key-pytest")


@pytest.fixture
def api_headers(api_key: str) -> dict[str, str]:
    """Authorization headers for API requests."""
    return {"X-API-Key": api_key}
