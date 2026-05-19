import ast

import pandas as pd

from triage.config import CATEGORY_DATASET, CUSTOMER_SUPPORT_DATASET
from triage.data.preprocessing import add_text_column, normalize_label


def parse_label_list(value: object) -> list[str]:
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


def load_category_training_data(path=CATEGORY_DATASET, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=limit)
    required = {"title", "body", "labels-mapped"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Category dataset is missing columns: {sorted(missing)}")
    df = add_text_column(df, "title", "body")
    df["labels"] = df["labels-mapped"].apply(parse_label_list)
    df = df[df["labels"].map(bool)].reset_index(drop=True)
    return df[["text", "labels"]]


def load_priority_training_data(path= CUSTOMER_SUPPORT_DATASET, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=limit)
    required = {"Ticket Subject", "Ticket Description", "Ticket Priority"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Priority dataset is missing columns: {sorted(missing)}")
    df = add_text_column(df, "Ticket Subject", "Ticket Description")
    df["label"] = df["Ticket Priority"].apply(normalize_label)
    df = df[df["label"].str.len() > 0].reset_index(drop=True)
    return df[["text", "label"]]


def load_intent_training_data(path= CUSTOMER_SUPPORT_DATASET, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=limit)
    required = {"Ticket Subject", "Ticket Description", "Ticket Type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Intent dataset is missing columns: {sorted(missing)}")
    df = add_text_column(df, "Ticket Subject", "Ticket Description")
    df["label"] = df["Ticket Type"].apply(normalize_label)
    df = df[df["label"].str.len() > 0].reset_index(drop=True)
    return df[["text", "label"]]
