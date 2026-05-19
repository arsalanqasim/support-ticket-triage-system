import pandas as pd

from triage.data.loaders import parse_label_list
from triage.data.preprocessing import add_text_column, clean_text, combine_title_body, normalize_label


def test_clean_text_handles_missing_values() -> None:
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("  Hello\nWorld ") == "hello world"


def test_combine_title_body_normalizes_text() -> None:
    assert combine_title_body(" Bug ", "Login FAILS") == "bug login fails"


def test_add_text_column_drops_empty_rows() -> None:
    df = pd.DataFrame([{"title": "A", "body": ""}, {"title": None, "body": None}])
    result = add_text_column(df)
    assert len(result) == 1
    assert result.loc[0, "text"] == "a"


def test_parse_label_list_from_notebook_dataset_format() -> None:
    assert parse_label_list("['feature_request', 'build-ci-cd']") == [
        "feature_request",
        "build_ci_cd",
    ]


def test_normalize_label_is_deterministic() -> None:
    assert normalize_label("Technical issue") == "technical_issue"
    assert normalize_label("Cancellation-request") == "cancellation_request"
