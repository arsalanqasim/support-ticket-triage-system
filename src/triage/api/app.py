"""
TriageIQ — FastAPI REST API

Endpoints:
  GET  /             Health check
  GET  /health       Detailed health + model status
  POST /predict      Single ticket classification (requires X-API-Key)
  POST /predict/batch  Batch classification (requires X-API-Key)

Authentication:
  All POST endpoints require the header:
    X-API-Key: <TRIAGEIQ_API_KEY from .env>

MLflow:
  Every inference call is logged to the configured MLflow tracking server.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

import mlflow
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from triage.config import (
    API_HOST,
    API_PORT,
    APP_NAME,
    LOG_LEVEL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    TRIAGEIQ_API_KEY,
    __version__,
)
from triage.inference.predictor import TriagePredictor
from triage.api.schemas import (
    BatchTicketRequest,
    BatchTicketResponse,
    HealthResponse,
    PredictionLabel,
    TicketPrediction,
    TicketRequest,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("triageiq.api")

# ── State ─────────────────────────────────────────────────────────────────────
_predictor: TriagePredictor | None = None
_models_loaded: bool = False


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and configure MLflow at startup; clean up at shutdown."""
    global _predictor, _models_loaded

    logger.info("🚀 TriageIQ API starting up — loading models…")
    try:
        _predictor = TriagePredictor()
        _models_loaded = True
        logger.info("✅ Models loaded successfully (TF-IDF baseline backend)")
    except FileNotFoundError as exc:
        logger.error("❌ Model artifacts not found: %s", exc)
        logger.error("   Run: python scripts/train_all.py  to train the models first.")
        _models_loaded = False

    # Configure MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info("📊 MLflow tracking: %s | experiment: %s", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow setup failed (inference logging disabled): %s", exc)

    yield

    logger.info("🛑 TriageIQ API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=f"{APP_NAME} — AI Ticket Classifier",
    description=(
        "**TriageIQ** automatically classifies support tickets by **Category**, **Priority**, "
        "and **Intent** using fine-tuned ML models.\n\n"
        "## Authentication\n"
        "All `POST` endpoints require an `X-API-Key` header.\n"
        "Set `TRIAGEIQ_API_KEY` in your `.env` file and send it with every request:\n"
        "```\nX-API-Key: your-secret-key\n```\n\n"
        "## Quick Example\n"
        "```bash\n"
        "curl -X POST http://localhost:8000/predict \\\n"
        '  -H "X-API-Key: your-secret-key" \\\n'
        '  -H "Content-Type: application/json" \\\n'
        "  -d '{\"title\": \"Login fails\", \"body\": \"500 error on the login page\"}'\n"
        "```"
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "TriageIQ", "url": "https://github.com/arsalanqasim/triageiq"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
)

# CORS — allow all origins for demo / portfolio purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Security dependency ───────────────────────────────────────────────────────

async def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> str:
    """Dependency that validates the X-API-Key header."""
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header. Set it to your TRIAGEIQ_API_KEY value.",
        )
    if x_api_key != TRIAGEIQ_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return x_api_key


# ── Helper ────────────────────────────────────────────────────────────────────

def _raw_to_schema(raw: dict[str, Any], latency_ms: float) -> TicketPrediction:
    """Convert raw predictor output dict → TicketPrediction schema."""
    return TicketPrediction(
        category=[PredictionLabel(**c) for c in raw["category"]],
        priority=PredictionLabel(**raw["priority"]),
        intent=PredictionLabel(**raw["intent"]),
        model_backend="TF-IDF + LogisticRegression (Baseline)",
        latency_ms=round(latency_ms, 2),
    )


def _log_to_mlflow(
    request_data: dict[str, Any],
    prediction: TicketPrediction,
    latency_ms: float,
) -> None:
    """Log a single inference call to MLflow asynchronously (best-effort)."""
    try:
        with mlflow.start_run(run_name="api_inference", nested=True):
            mlflow.log_param("title_length", len(request_data.get("title", "")))
            mlflow.log_param("body_length", len(request_data.get("body", "")))
            mlflow.log_param("backend", prediction.model_backend)
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("priority_confidence", prediction.priority.confidence)
            mlflow.log_metric("intent_confidence", prediction.intent.confidence)
            mlflow.log_metric(
                "top_category_confidence",
                prediction.category[0].confidence if prediction.category else 0.0,
            )
            mlflow.log_param("predicted_priority", prediction.priority.label)
            mlflow.log_param("predicted_intent", prediction.intent.label)
    except Exception as exc:  # noqa: BLE001
        logger.debug("MLflow logging skipped: %s", exc)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/",
    summary="Root / ping",
    tags=["Health"],
)
async def root() -> dict[str, str]:
    """Quick ping endpoint — returns app name and version."""
    return {
        "app": APP_NAME,
        "version": __version__,
        "status": "ok" if _models_loaded else "degraded",
        "docs": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & readiness probe",
    tags=["Health"],
)
async def health() -> HealthResponse:
    """
    Returns the health and readiness status of the API.

    Use this endpoint for Docker / Kubernetes liveness and readiness probes.
    """
    return HealthResponse(
        status="ok" if _models_loaded else "degraded",
        models_loaded=_models_loaded,
        backend="TF-IDF + LogisticRegression (Baseline)",
        version=__version__,
        app_name=APP_NAME,
    )


@app.post(
    "/predict",
    response_model=TicketPrediction,
    summary="Classify a single support ticket",
    tags=["Inference"],
    status_code=status.HTTP_200_OK,
)
async def predict(
    ticket: TicketRequest,
    _key: str = Depends(require_api_key),
) -> TicketPrediction:
    """
    Classify a single support ticket.

    Returns predicted **Category** (multi-label), **Priority**, and **Intent**
    with confidence scores for each.

    **Requires** `X-API-Key` header.
    """
    if not _models_loaded or _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Run `python scripts/train_all.py` first.",
        )

    t0 = time.perf_counter()
    raw = _predictor.predict_ticket(ticket.title, ticket.body)
    latency_ms = (time.perf_counter() - t0) * 1000

    result = _raw_to_schema(raw, latency_ms)
    logger.info(
        "predict | priority=%s(%.2f) intent=%s(%.2f) latency=%.1fms",
        result.priority.label,
        result.priority.confidence,
        result.intent.label,
        result.intent.confidence,
        latency_ms,
    )

    _log_to_mlflow({"title": ticket.title, "body": ticket.body}, result, latency_ms)
    return result


@app.post(
    "/predict/batch",
    response_model=BatchTicketResponse,
    summary="Classify a batch of support tickets",
    tags=["Inference"],
    status_code=status.HTTP_200_OK,
)
async def predict_batch(
    payload: BatchTicketRequest,
    _key: str = Depends(require_api_key),
) -> BatchTicketResponse:
    """
    Classify up to **500** support tickets in a single request.

    Each ticket in the list receives its own Category, Priority, and Intent prediction.

    **Requires** `X-API-Key` header.
    """
    if not _models_loaded or _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Run `python scripts/train_all.py` first.",
        )

    import pandas as pd

    df = pd.DataFrame([t.model_dump() for t in payload.tickets])

    t0 = time.perf_counter()
    raw_preds = _predictor.predict_batch(df)
    latency_ms = (time.perf_counter() - t0) * 1000
    per_ticket_ms = latency_ms / len(payload.tickets)

    predictions = [_raw_to_schema(r, per_ticket_ms) for r in raw_preds]
    backend = predictions[0].model_backend if predictions else "unknown"

    logger.info(
        "predict_batch | tickets=%d total_latency=%.1fms",
        len(payload.tickets),
        latency_ms,
    )

    # Log batch summary to MLflow
    try:
        with mlflow.start_run(run_name="api_batch_inference", nested=True):
            mlflow.log_metric("batch_size", len(payload.tickets))
            mlflow.log_metric("total_latency_ms", latency_ms)
            mlflow.log_metric("avg_latency_ms_per_ticket", per_ticket_ms)
            mlflow.log_param("backend", backend)
    except Exception as exc:  # noqa: BLE001
        logger.debug("MLflow batch logging skipped: %s", exc)

    return BatchTicketResponse(
        predictions=predictions,
        total=len(predictions),
        processed_at=datetime.now(timezone.utc),
        model_backend=backend,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Start the TriageIQ API server with uvicorn."""
    uvicorn.run(
        "triage.api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level=LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
