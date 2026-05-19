# Support Ticket Triage System

This project builds a local web demo for automatically triaging support and
GitHub-style tickets.

The app predicts:

- Category
- Priority
- Intent

It uses baseline TF-IDF and scikit-learn models for the first end-to-end
version. The notebook transformer experiments remain available for future
model upgrades.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

If editable install is not available in your environment, install the plain
requirements instead:

```powershell
pip install -r requirements.txt
```

## Train Models

Train all baseline models:

```powershell
python scripts/train_all.py
```

For a faster smoke run, limit rows per dataset:

```powershell
python scripts/train_all.py --limit 1000
```

Model artifacts are written to `models/`:

- `category_model.joblib`
- `priority_model.joblib`
- `intent_model.joblib`
- `metadata.json`

## Run The Web App

```powershell
streamlit run src/triage/web/app.py
```

The app supports:

- Single ticket prediction with title and body fields
- CSV upload with required `title` and `body` columns
- Downloadable CSV results with predicted category, priority, intent, and
  confidence scores

## Batch Prediction From CLI

```powershell
python scripts/predict_csv.py input.csv output.csv
```

The input CSV must include:

```csv
title,body
"Login fails","Users see a 500 error after submitting the login form"
```

## Tests

```powershell
pytest
ruff check .
```

## Data Notes

The current datasets are mixed-domain:

- Category uses `dataset/dataset_13_labels.csv` with GitHub issue labels.
- Priority uses `Ticket Priority` from `dataset/customer_support_tickets.csv`.
- Intent uses `Ticket Type` from `dataset/customer_support_tickets.csv`.

This is suitable for a local project demo. A production GitHub bot should later
add GitHub API ingestion, human review feedback, and retraining from labels
created in the target repositories.
