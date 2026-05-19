import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from triage.config import ARTIFACT_DIR, METADATA_PATH


def ensure_artifact_dir(path: Path = ARTIFACT_DIR) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Any, path: Path) -> None:
    ensure_artifact_dir(path.parent)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return joblib.load(path)


def write_metadata(metadata: dict[str, Any], path: Path = METADATA_PATH) -> None:
    ensure_artifact_dir(path.parent)
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(metadata)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")


def read_metadata(path: Path = METADATA_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
