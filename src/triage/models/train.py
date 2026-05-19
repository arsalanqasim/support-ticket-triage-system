from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from triage.config import (
    CATEGORY_MODEL_PATH,
    INTENT_MODEL_PATH,
    PRIORITY_MODEL_PATH,
)
from triage.data.loaders import (
    load_category_training_data,
    load_intent_training_data,
    load_priority_training_data,
)
from triage.models.artifacts import save_model, write_metadata


@dataclass(frozen=True)
class TrainingResult:
    target: str
    path: Path
    rows: int
    labels: list[str]
    metrics: dict[str, float]


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )


def _split_size(rows: int) -> float | int:
    return 0.2 if rows >= 50 else max(1, min(5, rows // 5))


def train_category(limit: int | None = None, path: Path = CATEGORY_MODEL_PATH) -> TrainingResult:
    df = load_category_training_data(limit=limit)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["labels"])

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        y,
        test_size=_split_size(len(df)),
        random_state=42,
    )
    pipe = Pipeline(
        [
            ("tfidf", _vectorizer()),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(max_iter=1_000, class_weight="balanced")
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)

    artifact = {"pipeline": pipe, "label_binarizer": mlb, "target": "category"}
    save_model(artifact, path)

    metrics = {
        "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    result = TrainingResult("category", path, len(df), list(mlb.classes_), metrics)
    write_metadata(
        {
            "category": {
                "model_type": "TF-IDF + OneVsRest LogisticRegression",
                "artifact": str(path),
                "rows": result.rows,
                "labels": result.labels,
                "metrics": metrics,
                "dataset": "dataset/dataset_13_labels.csv",
            }
        }
    )
    return result


def train_single_label(
    target: str,
    loader,
    path: Path,
    limit: int | None = None,
) -> TrainingResult:
    df = loader(limit=limit)
    labels = sorted(df["label"].unique())
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=_split_size(len(df)),
        random_state=42,
        stratify=stratify,
    )
    pipe = Pipeline(
        [
            ("tfidf", _vectorizer()),
            ("clf", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ]
    )
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    save_model({"pipeline": pipe, "labels": labels, "target": target}, path)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    result = TrainingResult(target, path, len(df), labels, metrics)
    write_metadata(
        {
            target: {
                "model_type": "TF-IDF + LogisticRegression",
                "artifact": str(path),
                "rows": result.rows,
                "labels": result.labels,
                "metrics": metrics,
                "dataset": "dataset/customer_support_tickets.csv",
            }
        }
    )
    return result


def train_priority(limit: int | None = None, path: Path = PRIORITY_MODEL_PATH) -> TrainingResult:
    return train_single_label("priority", load_priority_training_data, path, limit)


def train_intent(limit: int | None = None, path: Path = INTENT_MODEL_PATH) -> TrainingResult:
    return train_single_label("intent", load_intent_training_data, path, limit)


def train_all(limit: int | None = None) -> list[TrainingResult]:
    return [
        train_category(limit=limit),
        train_priority(limit=limit),
        train_intent(limit=limit),
    ]


def predict_single_label_with_confidence(pipeline: Pipeline, texts: list[str]) -> list[dict]:
    labels = pipeline.classes_
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(texts)
    else:
        decision = pipeline.decision_function(texts)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        exp = np.exp(decision - decision.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
    output = []
    for row in probs:
        index = int(np.argmax(row))
        output.append({"label": str(labels[index]), "confidence": float(row[index])})
    return output
