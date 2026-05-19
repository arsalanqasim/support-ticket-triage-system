import pandas as pd
import pytest

from triage.inference.predictor import validate_prediction_frame


def test_validate_prediction_frame_accepts_required_columns() -> None:
    validate_prediction_frame(pd.DataFrame([{"title": "A", "body": "B"}]))


def test_validate_prediction_frame_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="body, title"):
        validate_prediction_frame(pd.DataFrame([{"title": "A"}]))
