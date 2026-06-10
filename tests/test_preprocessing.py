"""
Tests for the enhanced preprocessing pipeline.
Covers both basic clean_text() and the new deep_clean_text().
"""
import pandas as pd
import pytest

from triage.data.loaders import parse_label_list
from triage.data.preprocessing import (
    add_text_column,
    clean_text,
    combine_title_body,
    deep_clean_text,
    normalize_label,
)


# ── clean_text ────────────────────────────────────────────────────────────────

def test_clean_text_handles_missing_values() -> None:
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("  Hello\nWorld ") == "hello world"


# ── deep_clean_text ────────────────────────────────────────────────────────────

def test_deep_clean_text_strips_markdown_headers() -> None:
    result = deep_clean_text("## Bug Report\n### Steps to reproduce")
    assert "##" not in result
    assert "bug report" in result

def test_deep_clean_text_strips_code_fences() -> None:
    result = deep_clean_text("Description\n```python\nraise ValueError\n```\nEnd")
    assert "```" not in result
    assert "raise" not in result       # code block removed
    assert "description" in result
    assert "end" in result

def test_deep_clean_text_strips_html() -> None:
    result = deep_clean_text("<div>Click <b>here</b></div>")
    assert "<div>" not in result
    assert "click" in result

def test_deep_clean_text_strips_urls() -> None:
    result = deep_clean_text("See https://github.com/org/repo/issues/42 for details")
    assert "https://" not in result
    assert "see" in result and "details" in result

def test_deep_clean_text_strips_template_placeholders() -> None:
    result = deep_clean_text("Issue with {product_purchased}. Please assist.")
    assert "{product_purchased}" not in result
    assert "issue" in result

def test_deep_clean_text_strips_ansi_codes() -> None:
    result = deep_clean_text("\x1b[31mERROR\x1b[0m: message")
    assert "\x1b" not in result
    assert "error" in result

def test_deep_clean_text_strips_inline_code() -> None:
    result = deep_clean_text("Call `some_function()` with args")
    assert "`" not in result

def test_deep_clean_text_preserves_human_text() -> None:
    raw = "## Bug\n\nThe login page crashes after clicking submit.\n```\nTraceback...\n```"
    result = deep_clean_text(raw)
    assert "login" in result
    assert "crashes" in result
    assert "clicking" in result
    assert "submit" in result

def test_deep_clean_text_handles_none_and_nan() -> None:
    assert deep_clean_text(None) == ""
    assert deep_clean_text(float("nan")) == ""
    assert deep_clean_text("") == ""


# ── combine_title_body ────────────────────────────────────────────────────────

def test_combine_title_body_doubles_title_weight() -> None:
    """Title should appear twice in output for higher TF-IDF weight."""
    result = combine_title_body("Bug", "Description of the bug")
    # "bug" should appear at least twice since title is doubled
    assert result.count("bug") >= 2

def test_combine_title_body_handles_empty_body() -> None:
    result = combine_title_body("Login fails", "")
    assert "login" in result
    assert "fails" in result

def test_combine_title_body_handles_none_values() -> None:
    result = combine_title_body(None, None)
    assert result == ""


# ── add_text_column ────────────────────────────────────────────────────────────

def test_add_text_column_drops_empty_rows() -> None:
    df = pd.DataFrame([{"title": "A", "body": ""}, {"title": None, "body": None}])
    result = add_text_column(df)
    assert len(result) == 1
    assert "a" in result.loc[0, "text"]


# ── normalize_label ────────────────────────────────────────────────────────────

def test_normalize_label_is_deterministic() -> None:
    assert normalize_label("Technical issue") == "technical_issue"
    assert normalize_label("Cancellation-request") == "cancellation_request"
    assert normalize_label("cancel_order") == "cancel_order"
    assert normalize_label("  GET Refund  ") == "get_refund"


# ── parse_label_list ──────────────────────────────────────────────────────────

def test_parse_label_list_from_notebook_dataset_format() -> None:
    assert parse_label_list("['feature_request', 'build-ci-cd']") == [
        "feature_request",
        "build_ci_cd",
    ]

def test_parse_label_list_handles_single_string() -> None:
    result = parse_label_list("bug")
    assert result == ["bug"]

def test_parse_label_list_handles_none() -> None:
    assert parse_label_list(None) == []

def test_parse_label_list_handles_existing_list() -> None:
    result = parse_label_list(["Bug", "Feature Request"])
    assert result == ["bug", "feature_request"]
