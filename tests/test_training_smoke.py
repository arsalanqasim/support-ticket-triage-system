from pathlib import Path
from unittest.mock import MagicMock

from triage.models.artifacts import load_model
from triage.models.train import train_category, train_intent, train_priority


def test_train_scripts_produce_loadable_artifacts(monkeypatch) -> None:
    category_path = Path("models/test_category_model.joblib")
    priority_path = Path("models/test_priority_model.joblib")
    intent_path = Path("models/test_intent_model.joblib")

    # Prevent the smoke test from overwriting the production metadata.json
    monkeypatch.setattr("triage.models.train.write_metadata", MagicMock())

    train_category(limit=300, path=category_path)
    train_priority(limit=300, path=priority_path)
    # Bitext intent CSV is sorted by class — need enough rows to cover all 27 classes
    # (each class has ~1000 samples, so 27 * 35 ≈ 1000 minimum needed to see all classes)
    train_intent(limit=2000, path=intent_path)

    assert load_model(category_path)["target"] == "category"
    assert load_model(priority_path)["target"] == "priority"
    assert load_model(intent_path)["target"] == "intent"

