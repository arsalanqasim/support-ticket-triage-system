from __future__ import annotations

import os
from pathlib import Path

# ── Version ──────────────────────────────────────────────────────────────────
__version__ = "1.0.0"
APP_NAME = "TriageIQ"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MODEL_DIR can be overridden via environment variable for Docker / cloud deploys
MODEL_DIR = os.environ.get("MODEL_DIR", None)
ARTIFACT_DIR = Path(MODEL_DIR) if MODEL_DIR else PROJECT_ROOT / "models"

DATASET_DIR = PROJECT_ROOT / "dataset"

# Dataset paths
CATEGORY_DATASET = DATASET_DIR / "dataset_13_labels.csv"
CUSTOMER_SUPPORT_DATASET = DATASET_DIR / "customer_support_tickets.csv"

# Model artifact paths
CATEGORY_MODEL_PATH = ARTIFACT_DIR / "category_model.joblib"
PRIORITY_MODEL_PATH = ARTIFACT_DIR / "priority_model.joblib"
INTENT_MODEL_PATH = ARTIFACT_DIR / "intent_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

# ── API Settings ──────────────────────────────────────────────────────────────
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# API key — required for all POST endpoints
TRIAGEIQ_API_KEY = os.environ.get("TRIAGEIQ_API_KEY", "dev-insecure-key")

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "triageiq_inference")

# ── Streamlit ─────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", f"http://localhost:{API_PORT}")
