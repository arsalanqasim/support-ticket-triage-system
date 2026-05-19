import re
from typing import Any

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower()
    return WHITESPACE_RE.sub(" ", text).strip()


def combine_title_body(title: Any, body: Any) -> str:
    return clean_text(f"{clean_text(title)} {clean_text(body)}")


def add_text_column(
    df: pd.DataFrame,
    title_column: str = "title",
    body_column: str = "body",
    output_column: str = "text",
) -> pd.DataFrame:
    result = df.copy()
    result[output_column] = [
        combine_title_body(title, body)
        for title, body in zip(
            result.get(title_column, pd.Series([""] * len(result))),
            result.get(body_column, pd.Series([""] * len(result))),
        )
    ]
    return result[result[output_column].str.len() > 0].reset_index(drop=True)


def normalize_label(value: Any) -> str:
    return clean_text(value).replace(" ", "_").replace("-", "_")
