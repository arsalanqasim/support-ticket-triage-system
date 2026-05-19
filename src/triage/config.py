from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACT_DIR = PROJECT_ROOT / "models"

CATEGORY_DATASET = DATASET_DIR / "dataset_13_labels.csv"
CUSTOMER_SUPPORT_DATASET = DATASET_DIR / "customer_support_tickets.csv"

CATEGORY_MODEL_PATH = ARTIFACT_DIR / "category_model.joblib"
PRIORITY_MODEL_PATH = ARTIFACT_DIR / "priority_model.joblib"
INTENT_MODEL_PATH = ARTIFACT_DIR / "intent_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
