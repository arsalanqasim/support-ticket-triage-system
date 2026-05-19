"""
Transformer-based inference predictor.
Drop-in replacement for triage.inference.predictor.TriagePredictor.
Loads fine-tuned HuggingFace models from models/transformer/.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from triage.config import PROJECT_ROOT
from triage.data.preprocessing import combine_title_body

TRANSFORMER_DIR = PROJECT_ROOT / "models" / "transformer"


class TransformerPredictor:
    """
    Inference engine that uses fine-tuned DistilBERT/RoBERTa models.
    API is intentionally identical to TriagePredictor for easy swapping.
    """

    def __init__(self, models_dir: Path = TRANSFORMER_DIR) -> None:
        self.models_dir = models_dir
        self._tokenizers: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Eagerly validate that all three models exist
        for target in ("priority", "intent", "category"):
            p = self.models_dir / f"{target}_final"
            if not p.exists():
                raise FileNotFoundError(
                    f"Transformer model for '{target}' not found at {p}.\n"
                    "Run:  python scripts/train_transformer.py --target all"
                )
            self._load(target, p)

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self, target: str, path: Path) -> None:
        self._tokenizers[target] = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        model.eval()
        model.to(self._device)
        self._models[target] = model

    def _encode(self, target: str, texts: list[str]):
        tok = self._tokenizers[target]
        return tok(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self._device)

    # ── public API (same as TriagePredictor) ─────────────────────────────────

    def predict_ticket(self, title: str, body: str) -> dict[str, Any]:
        return self.predict_batch(pd.DataFrame([{"title": title, "body": body}]))[0]

    def predict_batch(self, rows: pd.DataFrame) -> list[dict[str, Any]]:
        _validate_frame(rows)
        texts = [combine_title_body(r.title, r.body) for r in rows.itertuples()]
        return [
            {"category": cat, "priority": pri, "intent": intent}
            for cat, pri, intent in zip(
                self._predict_multi_label("category", texts),
                self._predict_single_label("priority", texts),
                self._predict_single_label("intent", texts),
            )
        ]

    # ── single-label prediction ───────────────────────────────────────────────

    def _predict_single_label(
        self, target: str, texts: list[str]
    ) -> list[dict[str, Any]]:
        model = self._models[target]
        inputs = self._encode(target, texts)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        results = []
        for row in probs:
            idx = int(np.argmax(row))
            results.append(
                {
                    "label": model.config.id2label[idx],
                    "confidence": round(float(row[idx]), 4),
                }
            )
        return results

    # ── multi-label prediction (category) ────────────────────────────────────

    def _predict_multi_label(
        self, target: str, texts: list[str]
    ) -> list[list[dict[str, Any]]]:
        model = self._models[target]
        inputs = self._encode(target, texts)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.sigmoid(logits).cpu().numpy()
        results = []
        for row in probs:
            scored = sorted(
                [
                    (model.config.id2label[i], float(p))
                    for i, p in enumerate(row)
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            selected = [
                {"label": lbl, "confidence": round(p, 4)}
                for lbl, p in scored
                if p >= 0.5
            ]
            if not selected:
                selected = [{"label": scored[0][0], "confidence": round(scored[0][1], 4)}]
            results.append(selected[:3])
        return results


# ── module-level convenience ─────────────────────────────────────────────────

def _validate_frame(df: pd.DataFrame) -> None:
    required = {"title", "body"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")


@lru_cache(maxsize=1)
def get_transformer_predictor() -> TransformerPredictor:
    return TransformerPredictor()


def predict_ticket_transformer(title: str, body: str) -> dict[str, Any]:
    return get_transformer_predictor().predict_ticket(title, body)
