"""
Dataset Acquisition Script for TriageIQ
========================================
This script:
1. Extracts Priority dataset from dataset_clean.csv (real GitHub severity labels)
2. Downloads Bitext Customer Support Intent dataset from HuggingFace
3. Deletes the old/synthetic datasets
4. Saves clean, purpose-built CSVs for each model target

Run: python scripts/acquire_datasets.py
"""
import os
import sys
import re
import pandas as pd
from pathlib import Path

# ── Resolve paths ─────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

PYTHONIOENCODING = "utf-8"

print("=" * 60)
print(" TriageIQ Dataset Acquisition")
print("=" * 60)

# ── Regex for cleaning text ────────────────────────────────────────────────────
HTML_TAG_RE    = re.compile(r"<[^>]+>")
URL_RE         = re.compile(r"https?://\S+")
CODE_FENCE_RE  = re.compile(r"```[\s\S]*?```")
PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")
ANSI_RE        = re.compile(r"\x1b\[[0-9;]*m")
WS_RE          = re.compile(r"\s+")

def light_clean(text: str) -> str:
    """Remove Markdown fences, HTML, URLs, ANSI codes, template placeholders."""
    if not isinstance(text, str):
        return ""
    text = ANSI_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = CODE_FENCE_RE.sub(" ", text)
    text = PLACEHOLDER_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Extract Priority dataset from dataset_clean.csv
#           (GitHub Issues severity: critical / major / minor)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 1] Extracting Priority dataset from dataset_clean.csv ...")

clean_path = DATASET_DIR / "dataset_clean.csv"
priority_out = DATASET_DIR / "priority_dataset.csv"

if not clean_path.exists():
    print(f"  SKIP — {clean_path} not found")
else:
    print(f"  Loading {clean_path} ({clean_path.stat().st_size / 1e6:.0f} MB) ...")
    df_clean = pd.read_csv(clean_path, usecols=["title", "body", "severity"])

    print(f"  Loaded {len(df_clean):,} rows")
    print(f"  Severity distribution:\n{df_clean['severity'].value_counts()}")

    # Drop rows with null severity
    df_clean = df_clean.dropna(subset=["severity"]).reset_index(drop=True)
    
    # Normalize severity labels → standard priority names
    severity_map = {
        "critical": "critical",
        "major":    "high",
        "minor":    "low",
    }
    df_clean["severity"] = df_clean["severity"].str.strip().str.lower().map(severity_map)
    df_clean = df_clean.dropna(subset=["severity"]).reset_index(drop=True)

    print(f"  After mapping to priority labels:\n{df_clean['severity'].value_counts()}")

    # Clean text
    print("  Cleaning text fields ...")
    df_clean["title"] = df_clean["title"].apply(light_clean)
    df_clean["body"]  = df_clean["body"].apply(light_clean)

    # Drop rows where both title and body are empty after cleaning
    df_clean = df_clean[(df_clean["title"].str.len() > 0) | (df_clean["body"].str.len() > 0)]
    df_clean = df_clean.rename(columns={"severity": "priority"})

    df_clean[["title", "body", "priority"]].to_csv(priority_out, index=False)
    print(f"  Saved -> {priority_out} ({priority_out.stat().st_size / 1e6:.1f} MB)")
    print(f"  Total rows: {len(df_clean):,}")
    del df_clean

print("[STEP 1] DONE")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Download Bitext Customer Support Intent dataset
#           27 fine-grained customer support intent classes, 26,872 utterances
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2] Downloading Bitext Customer Support Intent dataset from HuggingFace ...")

intent_out = DATASET_DIR / "intent_dataset.csv"

try:
    from datasets import load_dataset

    print("  Pulling bitext/Bitext-customer-support-llm-chatbot-training-dataset ...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        trust_remote_code=True,
    )

    # Combine all splits
    dfs = []
    for split_name in ds.keys():
        split_df = ds[split_name].to_pandas()
        dfs.append(split_df)
        print(f"  Split '{split_name}': {len(split_df):,} rows")

    df_intent = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows: {len(df_intent):,}")
    print(f"  Columns: {df_intent.columns.tolist()}")

    # The dataset has 'instruction' (utterance) and 'intent' columns
    text_col   = "instruction" if "instruction" in df_intent.columns else "utterance"
    intent_col = "intent" if "intent" in df_intent.columns else df_intent.columns[-1]

    print(f"  Intent distribution:\n{df_intent[intent_col].value_counts().to_string()}")

    # Normalize intent labels (snake_case)
    df_intent[intent_col] = (
        df_intent[intent_col]
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )

    # Keep only text + intent
    df_out = df_intent[[text_col, intent_col]].rename(
        columns={text_col: "text", intent_col: "intent"}
    )
    df_out = df_out[df_out["text"].str.len() > 0].reset_index(drop=True)
    df_out.to_csv(intent_out, index=False)
    print(f"  Saved -> {intent_out} ({intent_out.stat().st_size / 1e6:.1f} MB)")
    del df_intent, df_out

except Exception as e:
    print(f"  WARNING: Bitext download failed: {e}")
    print("  Falling back to CLINC150 dataset ...")
    try:
        ds_clinc = load_dataset("clinc_oos", "plus")
        label_names = ds_clinc["train"].features["intent"].names

        dfs = []
        for split_name in ["train", "validation", "test"]:
            split = ds_clinc[split_name].to_pandas()
            split["intent_str"] = split["intent"].map(lambda i: label_names[i])
            dfs.append(split[["text", "intent_str"]].rename(columns={"intent_str": "intent"}))
        
        df_clinc = pd.concat(dfs, ignore_index=True)
        
        # Filter out OOS (out-of-scope) samples that have label "oos"
        df_clinc = df_clinc[df_clinc["intent"] != "oos"].reset_index(drop=True)
        
        print(f"  CLINC150 rows: {len(df_clinc):,} | Unique intents: {df_clinc['intent'].nunique()}")
        df_clinc.to_csv(intent_out, index=False)
        print(f"  Saved -> {intent_out} ({intent_out.stat().st_size / 1e6:.1f} MB)")
        del df_clinc
    except Exception as e2:
        print(f"  ERROR: Both Bitext and CLINC downloads failed: {e2}")

print("[STEP 2] DONE")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Delete old / unwanted datasets
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Cleaning up old datasets ...")

to_delete = [
    DATASET_DIR / "customer_support_tickets.csv",    # Synthetic garbage
    DATASET_DIR / "dataset_clean.csv",               # 296 MB — mined, now redundant
    DATASET_DIR / "confused_minor_major_sample_5k.csv",  # Small exploratory sample
]

for path in to_delete:
    if path.exists():
        size_mb = path.stat().st_size / 1e6
        os.remove(path)
        print(f"  DELETED: {path.name} ({size_mb:.1f} MB freed)")
    else:
        print(f"  SKIP (not found): {path.name}")

print("[STEP 3] DONE")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" FINAL DATASET INVENTORY")
print("=" * 60)
for f in sorted(DATASET_DIR.iterdir()):
    if f.is_file() and f.suffix == ".csv":
        rows = sum(1 for _ in open(f, encoding="utf-8")) - 1
        print(f"  {f.name:<35} {f.stat().st_size/1e6:>7.1f} MB   {rows:>9,} rows")

print("\nAll done! Update config.py and loaders.py to point to the new datasets.")
