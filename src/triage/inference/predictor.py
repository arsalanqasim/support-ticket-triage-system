from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from triage.config import CATEGORY_MODEL_PATH, INTENT_MODEL_PATH, PRIORITY_MODEL_PATH
from triage.data.preprocessing import combine_title_body
from triage.models.artifacts import load_model, read_metadata


class TriagePredictor:
    def __init__(
        self,
        category_model_path: Path = CATEGORY_MODEL_PATH,
        priority_model_path: Path = PRIORITY_MODEL_PATH,
        intent_model_path: Path = INTENT_MODEL_PATH,
    ) -> None:
        self.category_artifact = load_model(category_model_path)
        self.priority_artifact = load_model(priority_model_path)
        self.intent_artifact = load_model(intent_model_path)
        self.metadata = read_metadata()

    def predict_ticket(self, title: str, body: str) -> dict[str, Any]:
        return self.predict_batch(pd.DataFrame([{"title": title, "body": body}]))[0]

    def predict_batch(self, rows: pd.DataFrame) -> list[dict[str, Any]]:
        validate_prediction_frame(rows)
        texts = [combine_title_body(row.title, row.body) for row in rows.itertuples()]
        return [
            {
                "category": category,
                "priority": priority,
                "intent": intent,
            }
            for category, priority, intent in zip(
                self._predict_categories(texts),
                self._predict_single(self.priority_artifact, texts),
                self._predict_single(self.intent_artifact, texts),
            )
        ]

    def _predict_categories(self, texts: list[str]) -> list[list[dict[str, float | str]]]:
        pipeline = self.category_artifact["pipeline"]
        mlb = self.category_artifact["label_binarizer"]
        clf = pipeline.named_steps["clf"]

        if hasattr(clf, "predict_proba"):
            scores = pipeline.predict_proba(texts)
        else:
            raw_scores = pipeline.decision_function(texts)
            scores = 1 / (1 + np.exp(-raw_scores))

        predictions = []
        for row in scores:
            ranked = sorted(
                zip(mlb.classes_, row),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            selected = [(label, score) for label, score in ranked if score >= 0.5]
            if not selected and ranked:
                selected = [ranked[0]]
            predictions.append(
                [
                    {"label": str(label), "confidence": round(float(score), 4)}
                    for label, score in selected[:3]
                ]
            )
        return predictions

    def _predict_single(self, artifact: dict[str, Any], texts: list[str]) -> list[dict[str, Any]]:
        pipeline = artifact["pipeline"]
        labels = pipeline.classes_
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(texts)
        else:
            decision = pipeline.decision_function(texts)
            exp = np.exp(decision - decision.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)

        output = []
        for row in probs:
            index = int(np.argmax(row))
            output.append(
                {"label": str(labels[index]), "confidence": round(float(row[index]), 4)}
            )
        return output


def validate_prediction_frame(df: pd.DataFrame) -> None:
    required = {"title", "body"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")


@lru_cache(maxsize=1)
def get_predictor() -> TriagePredictor:
    return TriagePredictor()


def predict_ticket(title: str, body: str) -> dict[str, Any]:
    return get_predictor().predict_ticket(title, body)
