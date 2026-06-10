"""
Dataset loaders for TriageIQ training pipeline.

Three loaders — one per model target:

  load_category_training_data()
      Source : dataset/dataset_13_labels.csv  (GitHub Issues, ~107K rows)
      Labels : 13 multi-label categories (bug, feature_request, …)

  load_priority_training_data()
      Source : dataset/priority_dataset.csv  (GitHub Issues severity, ~107K rows)
      Labels : critical | high | low   (mapped from Critical/Major/Minor)

  load_intent_training_data()
      Source : dataset/intent_dataset.csv  (Bitext Customer Support, ~27K rows)
      Labels : 27 fine-grained customer intent classes
               (cancel_order, payment_issue, get_refund, …)

All loaders return a DataFrame with columns ['text', 'label(s)'] ready for
sklearn pipelines.
"""
import ast

import pandas as pd

from triage.config import CATEGORY_DATASET, INTENT_DATASET, PRIORITY_DATASET
from triage.data.preprocessing import add_text_column, clean_text, normalize_label


def parse_label_list(value: object) -> list[str]:
    """Parse a stringified list of labels into a cleaned Python list."""
    if isinstance(value, list):
        return [normalize_label(item) for item in value if normalize_label(item)]
    if value is None or pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = [value]
    if not isinstance(parsed, list):
        parsed = [parsed]
    return [normalize_label(item) for item in parsed if normalize_label(item)]


# ── Category ──────────────────────────────────────────────────────────────────

def load_category_training_data(
    path=CATEGORY_DATASET,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Load GitHub Issues dataset for multi-label category classification.

    Columns: title, body, labels-mapped
    Returns: DataFrame[text, labels]  — labels is a list of str per row.
    """
    df = pd.read_csv(path, nrows=limit)
    required = {"title", "body", "labels-mapped"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Category dataset is missing columns: {sorted(missing)}")

    df = add_text_column(df, "title", "body")
    df["labels"] = df["labels-mapped"].apply(parse_label_list)
    df = df[df["labels"].map(bool)].reset_index(drop=True)
    return df[["text", "labels"]]


# ── Priority ──────────────────────────────────────────────────────────────────

def load_priority_training_data(
    path=PRIORITY_DATASET,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Load GitHub Issues severity dataset for priority classification.

    Source: priority_dataset.csv  (extracted from dataset_clean.csv severity col)
    Labels: critical | high | low  (mapped from GitHub Critical/Major/Minor)
    Columns: title, body, priority
    Returns: DataFrame[text, label]
    """
    df = pd.read_csv(path, nrows=limit)
    required = {"title", "body", "priority"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Priority dataset is missing columns: {sorted(missing)}")

    df = add_text_column(df, "title", "body")
    df["label"] = df["priority"].apply(normalize_label)
    df = df[df["label"].str.len() > 0].reset_index(drop=True)
    return df[["text", "label"]]


# ── Intent ────────────────────────────────────────────────────────────────────

def load_intent_training_data(
    path=INTENT_DATASET,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Load Bitext Customer Support dataset for intent classification.

    Source: intent_dataset.csv
            bitext/Bitext-customer-support-llm-chatbot-training-dataset
    Labels: 27 fine-grained customer support intents
            (cancel_order, payment_issue, get_refund, complaint, …)
    Columns: text, intent
    Returns: DataFrame[text, label]
    """
    df = pd.read_csv(path, nrows=limit)
    required = {"text", "intent"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Intent dataset is missing columns: {sorted(missing)}")

    df["label"] = df["intent"].apply(normalize_label)
    df = df[df["label"].str.len() > 0].reset_index(drop=True)

    # Use the text column directly — it's already clean (single utterances,
    # no Markdown or HTML noise)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df[["text", "label"]]
