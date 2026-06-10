<div align="center">

# 🎯 TriageIQ

### AI-Powered Support Ticket Classifier

*Automatically classify support tickets by **Category**, **Priority**, and **Intent** using Machine Learning*

[![CI](https://github.com/arsalanqasim/triageiq/actions/workflows/ci.yml/badge.svg)](https://github.com/arsalanqasim/triageiq/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?logo=streamlit)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?logo=mlflow)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## ✨ What Is TriageIQ?

**TriageIQ** is an end-to-end ML system that takes raw support ticket text and instantly predicts:

| Output | Example |
|--------|---------|
| 🏷️ **Category** | `bug`, `feature_request`, `build_ci_cd` (multi-label) |
| ⚡ **Priority** | `critical` / `high` / `medium` / `low` |
| 🎯 **Intent** | `technical_issue`, `billing_inquiry`, `refund_request` … |

**Key highlights for recruiters:**
- ✅ **Full REST API** (FastAPI + Swagger UI + API Key auth)
- ✅ **Interactive Web UI** (Streamlit with analytics dashboard)
- ✅ **Dual ML backends** — TF-IDF baseline + DistilBERT transformer
- ✅ **MLflow experiment tracking** in notebooks AND live API inference logging
- ✅ **CI/CD pipeline** (GitHub Actions — lint + test on every push)
- ✅ **Docker support** (multi-stage Dockerfile + docker-compose)
- ✅ **Production-grade test suite** (pytest + httpx async API tests)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       TriageIQ                           │
│                                                          │
│  ┌─────────────┐   ┌──────────────────────────────────┐  │
│  │  Web UI     │   │       REST API (FastAPI)         │  │
│  │ (Streamlit) │   │  POST /predict                   │  │
│  │             │   │  POST /predict/batch             │  │
│  │ • Single    │   │  GET  /health                    │  │
│  │ • Batch CSV │   │  Swagger UI: /docs               │  │
│  │ • Analytics │   │  Auth: X-API-Key header          │  │
│  └──────┬──────┘   └────────────┬─────────────────────┘  │
│         │                       │                        │
│         └───────────┬───────────┘                        │
│                     ▼                                    │
│         ┌──────────────────────┐                         │
│         │  Inference Layer     │                         │
│         │  TriagePredictor     │                         │
│         │  TransformerPredictor│                         │
│         └──────────┬───────────┘                         │
│                    │                                     │
│         ┌──────────┴──────────┐                          │
│         │                     │                          │
│  ┌──────▼──────┐   ┌──────────▼────────┐                 │ 
│  │ TF-IDF +    │   │  DistilBERT       │                 │
│  │ LogReg      │   │  (Fine-tuned)     │                 │
│  │ (Baseline)  │   │  Transformers     │                 │
│  └─────────────┘   └───────────────────┘                 │
│                                                          │
│         ┌──────────────────────────┐                     │
│         │  MLflow Tracking         │                     │
│         │  • Notebook experiments  │                     │
│         │  • Live API inference    │                     │
│         └──────────────────────────┘                     │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

| Target | Model | Dataset | Rows | Metric | Score |
|--------|-------|---------|------|--------|-------|
| Category | TF-IDF + OneVsRest LogReg | GitHub Issues (13 labels) | 106,909 | Micro F1 | **~0.67** |
| Category | TF-IDF + OneVsRest LogReg | GitHub Issues (13 labels) | 106,909 | Macro F1 | **~0.60** |
| Priority | TF-IDF + LogReg | GitHub Issues severity (critical/high/low) | 106,909 | Accuracy | **~0.72** |
| Priority | TF-IDF + LogReg | GitHub Issues severity (critical/high/low) | 106,909 | Macro F1 | **~0.60** |
| Intent | TF-IDF + LogReg | Bitext Customer Support (27 intents) | 26,872 | Accuracy | **~0.92** |
| Intent | TF-IDF + LogReg | Bitext Customer Support (27 intents) | 26,872 | Macro F1 | **~0.92** |

> **Data sources:** GitHub Issues severity labels from `sharjeelyunus/github-issues-dataset` (real human-assigned);  
> Intent utterances from `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (real customer support text, 27 classes).

---

## 🚀 Quick Start

### 1. Clone & Install

```powershell
git clone https://github.com/arsalanqasim/triageiq.git
cd triageiq

python -m venv venv
.\venv\Scripts\Activate.ps1

# Install with all dev dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment

```powershell
# Copy the example config and edit it
copy .env.example .env
```

At minimum, set your API key in `.env`:
```
TRIAGEIQ_API_KEY=my-super-secret-key-123
```

### 3. Train Models

```powershell
# Train all three baseline models (~2–3 min on full dataset)
python scripts/train_all.py

# Quick smoke run (200 rows per dataset, ~10 seconds)
python scripts/train_all.py --limit 200
```

Artifacts written to `models/`:
- `category_model.joblib`
- `priority_model.joblib`
- `intent_model.joblib`
- `metadata.json`

### 4. Run the API

```powershell
python scripts/run_api.py
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

### 5. Run the Web UI

```powershell
streamlit run src/triage/web/app.py
```

Visit **http://localhost:8501**

---

## 🌐 REST API

### Authentication

All `POST` endpoints require the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/health
```

### `POST /predict` — Single Ticket

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Login page returns 500 after deploy",
    "body": "Users cannot log in since the 2pm deployment. Rolling back fixes it."
  }'
```

**Response:**
```json
{
  "category": [
    {"label": "bug", "confidence": 0.82},
    {"label": "build_ci_cd", "confidence": 0.61}
  ],
  "priority": {"label": "high", "confidence": 0.77},
  "intent":   {"label": "technical_issue", "confidence": 0.99},
  "model_backend": "TF-IDF + LogisticRegression (Baseline)",
  "latency_ms": 12.4
}
```

### `POST /predict/batch` — Up to 500 tickets

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "tickets": [
      {"title": "Payment failed",  "body": "Card was declined twice."},
      {"title": "Add dark mode",   "body": "Feature request for the dashboard."}
    ]
  }'
```

### Python Client

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000", headers={"X-API-Key": "your-api-key"})

result = client.post("/predict", json={
    "title": "App crashes on save",
    "body": "Reproducible on every save action since v2.3.1"
}).json()

print(result["priority"]["label"])   # → "high"
print(result["intent"]["label"])     # → "technical_issue"
```

---

## 🖥️ Web Interface

The Streamlit UI has **4 tabs**:

| Tab | Description |
|-----|-------------|
| 🔍 Single Ticket | Paste title + body → instant predictions with confidence bars |
| 📁 CSV Upload | Upload a CSV with `title,body` columns → batch classify + download results |
| 📊 Analytics | Session prediction history, priority/intent distribution charts, model performance table |
| 🌐 REST API | curl and Python code examples, links to Swagger UI |

---

## 🧪 Testing

```powershell
# Full test suite with coverage report
pytest tests/ -v --cov=src/triage --cov-report=term-missing

# Lint with ruff
ruff check .
```

### Test Coverage

| Module | Tests |
|--------|-------|
| `test_preprocessing.py` | Text cleaning, normalization, label parsing |
| `test_inference.py` | Predictor API, confidence ranges, parametrized ticket types |
| `test_training_smoke.py` | Full train → save → load round-trip |
| `test_api.py` | Health probes, auth (401/403), single predict, batch predict, validation (422) |

---

## 🐳 Docker

### Start Everything

```bash
# Copy and configure your .env first
cp .env.example .env

# Build and start API + UI
docker-compose up --build
```

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

### API Only

```bash
# Build
docker build -t triageiq-api .

# Run (mount your trained models)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e TRIAGEIQ_API_KEY=your-api-key \
  triageiq-api
```

---

## 🔭 MLflow Experiment Tracking

```powershell
# View the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Open http://localhost:5000
```

Experiments tracked:
- **`tfidf_baseline`** — TF-IDF model comparisons (LogReg vs SVM) from notebooks
- **`transformer_triage`** — DistilBERT fine-tuning runs with Hugging Face Trainer
- **`triageiq_inference`** — Live API inference calls (latency, confidence scores)

---

## 📁 Project Structure

```
triageiq/
├── src/triage/
│   ├── api/              # FastAPI REST API
│   │   ├── app.py        # Routes, lifespan, auth, MLflow logging
│   │   └── schemas.py    # Pydantic request/response models
│   ├── data/             # Data loading and preprocessing
│   │   ├── loaders.py    # Dataset-specific loaders
│   │   └── preprocessing.py
│   ├── inference/        # Prediction layer
│   │   ├── predictor.py           # TF-IDF baseline predictor
│   │   └── predictor_transformer.py  # DistilBERT predictor
│   ├── models/           # Training and artifact management
│   │   ├── train.py      # Sklearn training pipelines
│   │   └── artifacts.py  # Save/load/metadata
│   ├── web/
│   │   └── app.py        # Streamlit UI (4 tabs)
│   └── config.py         # Centralised config (env-aware)
│
├── scripts/
│   ├── run_api.py        # API launcher
│   ├── train_all.py      # Train all 3 baseline models
│   └── train_transformer.py  # Fine-tune DistilBERT
│
├── tests/
│   ├── conftest.py       # Shared fixtures
│   ├── test_api.py       # FastAPI endpoint tests
│   ├── test_inference.py # Predictor unit + integration tests
│   ├── test_preprocessing.py
│   └── test_training_smoke.py
│
├── notebooks/            # Research & experimentation
│   ├── eda.ipynb
│   ├── tfidf-baseline.ipynb
│   └── transformer_models.ipynb
│
├── .github/workflows/ci.yml  # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── pyproject.toml
└── CHANGELOG.md
```

---

## 🛣️ Roadmap

- [ ] GitHub App integration (auto-triage issues on new issue events)
- [ ] Human feedback loop (label corrections → retraining trigger)
- [ ] Prometheus metrics endpoint (`/metrics`) for production observability
- [ ] Redis caching for repeated predictions
- [ ] Model versioning with MLflow Model Registry
- [ ] Fine-tuned DistilBERT weights published to Hugging Face Hub

---

## 📄 License

[MIT](LICENSE) © 2024 Arsalan Qasim
