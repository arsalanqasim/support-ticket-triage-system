"""
Fine-tune a Transformer model (DistilBERT/RoBERTa) for ticket triage.
Saves models to models/transformer/ — does NOT touch existing scikit-learn artifacts.

Usage:
    python scripts/train_transformer.py --target all
    python scripts/train_transformer.py --target priority --model distilbert-base-uncased
    python scripts/train_transformer.py --target priority --model roberta-base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Resolve project src so 'triage' package can be imported ─────────────────
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch

torch.manual_seed(42)

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from triage.data.loaders import (
    load_category_training_data,
    load_intent_training_data,
    load_priority_training_data,
)
from triage.models.artifacts import write_metadata

TRANSFORMER_DIR = _HERE.parents[1] / "models" / "transformer"


# ── Metrics helpers ──────────────────────────────────────────────────────────

def _metrics_single(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def _metrics_multi(eval_pred):
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


# ── Default training args ────────────────────────────────────────────────────

def _training_args(output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),   # auto-enable half-precision on GPU
        report_to="none",                 # suppress wandb / tensorboard
    )


# ── Single-label fine-tuning ─────────────────────────────────────────────────

def train_single_label(
    target: str,
    model_name: str,
    output_dir: Path,
    limit: int | None = None,
) -> None:
    print(f"\n{'='*60}")
    print(f" Training single-label model: {target.upper()}")
    print(f" Base model : {model_name}")
    print(f"{'='*60}")

    loader = load_intent_training_data if target == "intent" else load_priority_training_data
    df = loader(limit=limit)

    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    df = df.copy()
    df["label"] = df["label"].map(label2id)

    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    df_train, df_test = train_test_split(
        df[["text", "label"]], test_size=0.2, random_state=42, stratify=stratify
    )

    train_ds = Dataset.from_pandas(df_train, preserve_index=False)
    test_ds  = Dataset.from_pandas(df_test,  preserve_index=False)

    print(f" Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds  = test_ds.map(tokenize,  batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    ckpt_dir = output_dir / f"{target}_checkpoints"
    trainer = Trainer(
        model=model,
        args=_training_args(ckpt_dir),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_metrics_single,
    )

    trainer.train()

    final_path = output_dir / f"{target}_final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Evaluate and persist metrics
    metrics = trainer.evaluate(test_ds)
    eval_metrics = {k.replace("eval_", ""): v for k, v in metrics.items() if k.startswith("eval_")}

    meta_file = final_path / "metadata.json"
    meta = {"model_type": model_name, "target": target, "labels": labels, "metrics": eval_metrics}
    meta_file.write_text(json.dumps(meta, indent=2))

    # Merge into the project-wide metadata.json so the sidebar shows it
    write_metadata(
        {
            f"transformer_{target}": {
                "model_type": f"DistilBERT fine-tuned ({model_name})",
                "artifact": str(final_path),
                "rows": len(df),
                "labels": labels,
                "metrics": {k: round(v, 4) for k, v in eval_metrics.items() if isinstance(v, float)},
                "dataset": "dataset/customer_support_tickets.csv",
            }
        }
    )

    print(f"\n✅ Saved {target} model → {final_path}")
    print(f"   Metrics: {eval_metrics}")


# ── Multi-label fine-tuning (category) ──────────────────────────────────────

def train_multi_label(
    target: str,
    model_name: str,
    output_dir: Path,
    limit: int | None = None,
) -> None:
    print(f"\n{'='*60}")
    print(f" Training multi-label model: {target.upper()}")
    print(f" Base model : {model_name}")
    print(f"{'='*60}")

    df = load_category_training_data(limit=limit)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["labels"])     # shape (N, num_labels)
    labels = list(mlb.classes_)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label  = {i: l for i, l in enumerate(labels)}

    texts = df["text"].tolist()
    df_train_x, df_test_x, y_train, y_test = train_test_split(
        texts, y.tolist(), test_size=0.2, random_state=42
    )

    train_ds = Dataset.from_dict({"text": df_train_x, "labels": y_train})
    test_ds  = Dataset.from_dict({"text": df_test_x,  "labels": y_test})

    print(f" Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=256)
        enc["labels"] = [list(map(float, row)) for row in batch["labels"]]
        return enc

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds  = test_ds.map(tokenize,  batched=True, remove_columns=["text"])
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    ckpt_dir = output_dir / f"{target}_checkpoints"
    trainer = Trainer(
        model=model,
        args=_training_args(ckpt_dir),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_metrics_multi,
    )

    trainer.train()

    final_path = output_dir / f"{target}_final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    metrics = trainer.evaluate(test_ds)
    eval_metrics = {k.replace("eval_", ""): v for k, v in metrics.items() if k.startswith("eval_")}

    meta_file = final_path / "metadata.json"
    meta = {"model_type": model_name, "target": target, "labels": labels, "metrics": eval_metrics}
    meta_file.write_text(json.dumps(meta, indent=2))

    write_metadata(
        {
            f"transformer_{target}": {
                "model_type": f"DistilBERT fine-tuned ({model_name})",
                "artifact": str(final_path),
                "rows": len(df),
                "labels": labels,
                "metrics": {k: round(v, 4) for k, v in eval_metrics.items() if isinstance(v, float)},
                "dataset": "dataset/dataset_13_labels.csv",
            }
        }
    )

    print(f"\n✅ Saved {target} model → {final_path}")
    print(f"   Metrics: {eval_metrics}")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a HuggingFace Transformer for ticket triage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        choices=["priority", "intent", "category", "all"],
        default="all",
    )
    parser.add_argument(
        "--model",
        default="distilbert-base-uncased",
        help="HuggingFace model hub ID, e.g. 'distilbert-base-uncased' or 'roberta-base'",
    )
    parser.add_argument(
        "--output_dir",
        default=str(TRANSFORMER_DIR),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Row limit per dataset (useful for quick smoke-tests)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    targets = ["priority", "intent", "category"] if args.target == "all" else [args.target]

    for t in targets:
        if t == "category":
            train_multi_label(t, args.model, out, limit=args.limit)
        else:
            train_single_label(t, args.model, out, limit=args.limit)

    print("\n🎉  All transformer models trained and saved.")


if __name__ == "__main__":
    main()
