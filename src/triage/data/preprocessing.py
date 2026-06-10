"""
Text preprocessing pipeline for TriageIQ.

Provides two cleaning levels:
  - clean_text()      : basic whitespace + lowercasing (used everywhere)
  - deep_clean_text() : removes Markdown, HTML, code fences, URLs, ANSI escapes,
                        and template placeholders (applied on GitHub issue bodies)

Both are deterministic and side-effect-free.
"""
import re
from typing import Any

import pandas as pd


# ── Compiled regexes ──────────────────────────────────────────────────────────
WHITESPACE_RE    = re.compile(r"\s+")
HTML_TAG_RE      = re.compile(r"<[^>]+>")                    # <div>, <br/>, etc.
URL_RE           = re.compile(r"https?://\S+")               # http(s):// URLs
CODE_FENCE_RE    = re.compile(r"```[\s\S]*?```")             # ```…``` blocks
INLINE_CODE_RE   = re.compile(r"`[^`\n]{1,100}`")           # `short code`
PLACEHOLDER_RE   = re.compile(r"\{[^}]+\}")                  # {product_purchased}
ANSI_RE          = re.compile(r"\x1b\[[0-9;]*m")             # ANSI escape codes
MARKDOWN_HDR_RE  = re.compile(r"^#{1,6}\s+", re.MULTILINE)  # ## Heading
MARKDOWN_BOLD_RE = re.compile(r"\*{1,2}([^*]+)\*{1,2}")     # **bold** / *italic*
CHECKMARK_RE     = re.compile(r"-\s*\[[ xX]\]")             # - [ ] / - [x]


def clean_text(value: Any) -> str:
    """Basic clean: lower-case + collapse whitespace. Used for labels too."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).lower()
    return WHITESPACE_RE.sub(" ", text).strip()


def deep_clean_text(value: Any) -> str:
    """
    Full clean for raw GitHub issue / support ticket bodies.
    Order matters: remove blocks first, then inline, then normalise.
    Preserves the human-readable words while discarding markup noise.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = ANSI_RE.sub(" ", text)          # strip ANSI escape codes
    text = HTML_TAG_RE.sub(" ", text)      # strip HTML tags
    text = URL_RE.sub(" ", text)           # strip URLs
    text = CODE_FENCE_RE.sub(" ", text)    # strip ``` code fences ```
    text = INLINE_CODE_RE.sub(" ", text)   # strip `inline code`
    text = PLACEHOLDER_RE.sub(" ", text)   # strip {template_vars}
    text = MARKDOWN_HDR_RE.sub(" ", text)  # strip ## headings
    text = MARKDOWN_BOLD_RE.sub(r"\1", text)   # strip **bold** → bold
    text = CHECKMARK_RE.sub(" ", text)     # strip - [x] checkboxes
    text = text.lower()
    return WHITESPACE_RE.sub(" ", text).strip()


def combine_title_body(title: Any, body: Any) -> str:
    """
    Combine title and body into a single training string.
    Title is prepended twice to give it higher effective weight in TF-IDF,
    since titles are dense with signal compared to long noisy bodies.
    """
    clean_title = deep_clean_text(title)
    clean_body  = deep_clean_text(body)
    # Double the title so TF-IDF scores title terms higher
    parts = [p for p in [clean_title, clean_title, clean_body] if p]
    return WHITESPACE_RE.sub(" ", " ".join(parts)).strip()


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
