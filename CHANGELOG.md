# Changelog

All notable changes to **TriageIQ** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2024-06-07

### Added
- **FastAPI REST API** (`src/triage/api/`) with endpoints:
  - `GET /` — root ping
  - `GET /health` — liveness/readiness probe
  - `POST /predict` — single ticket classification
  - `POST /predict/batch` — batch ticket classification (up to 500)
- **API Key authentication** via `X-API-Key` header on all POST endpoints
- **MLflow inference logging** — every API prediction call logged to the configured experiment
- **Pydantic v2 schemas** (`schemas.py`) for full request/response validation
- **Docker support** — multi-stage `Dockerfile` + `docker-compose.yml` for API + UI
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — lint (ruff) + pytest matrix on Python 3.10/3.11
- **Analytics dashboard tab** in Streamlit UI — session prediction history, priority/intent distribution charts, model performance table
- **REST API guide tab** in Streamlit UI — curl and Python SDK examples
- **`tests/conftest.py`** — shared pytest fixtures (sample tickets, async API client, label sets)
- **`tests/test_api.py`** — 18 API endpoint tests covering health, auth, validation, single + batch predict
- **`tests/test_inference.py`** — expanded with parametrized tests, batch tests, edge cases
- **`.env.example`** — environment variable template for all configurable settings
- **`scripts/run_api.py`** — convenience API launcher with `--host`, `--port`, `--reload` flags
- **`LICENSE`** — MIT license
- **Environment-aware config** (`config.py`) — reads `MODEL_DIR`, `API_PORT`, `TRIAGEIQ_API_KEY`, `MLFLOW_*` from env

### Changed
- Project rebranded from "Support Ticket Triage System" → **TriageIQ**
- `pyproject.toml` — updated name, version, description, classifiers, URLs, and dev/transformer extras
- `src/triage/config.py` — added `__version__`, `APP_NAME`, env var overrides for all settings
- `src/triage/web/app.py` — TriageIQ branding, 4-tab layout, session analytics, API panel in sidebar
- `.gitignore` — added proper ML project patterns (model artifacts, datasets, MLflow DB, logs)

### Removed
- `generate_report.py` — one-off docx report script (not production code)
- `NOTES.md` — personal development notes
- `claude-mapping.json` — AI tool artifact
- `mlflow.db` (root) — binary database files are now gitignored
- `models/streamlit*.log` — log files committed in wrong location
- `models/triage_*.csv` — temporary test run artifacts

---

## [0.1.0] — 2024-05-01

### Added
- Initial project structure with `src/triage/` package layout
- TF-IDF + Logistic Regression baseline models (category, priority, intent)
- `OneVsRestClassifier` for multi-label category classification
- Streamlit web UI with glassmorphism dark theme
- Single ticket prediction + CSV batch upload/download
- MLflow experiment tracking in notebooks
- DistilBERT transformer predictor (notebook + scripts)
- Basic unit tests for preprocessing and inference validation
- `pyproject.toml` package configuration
