"""
Download Bitext Customer Support Intent dataset using pip-installed datasets library.
Falls back to CLINC150 if Bitext is unavailable.
"""
import os
import re
import sys
import pandas as pd
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
intent_out = DATASET_DIR / "intent_dataset.csv"

WS_RE = re.compile(r"\s+")

print("Importing datasets library ...")
try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not found. Run: pip install datasets")
    sys.exit(1)

print("[STEP 2] Downloading Bitext Customer Support Intent dataset ...")

try:
    print("  Pulling bitext/Bitext-customer-support-llm-chatbot-training-dataset ...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        trust_remote_code=True,
    )

    dfs = []
    for split_name in ds.keys():
        split_df = ds[split_name].to_pandas()
        dfs.append(split_df)
        print(f"  Split '{split_name}': {len(split_df):,} rows | cols: {split_df.columns.tolist()}")

    df_intent = pd.concat(dfs, ignore_index=True)
    print(f"  Total combined rows: {len(df_intent):,}")

    # Detect column names
    text_col   = "instruction" if "instruction" in df_intent.columns else "utterance"
    intent_col = "intent"

    print(f"\n  Intent distribution ({df_intent[intent_col].nunique()} classes):")
    print(df_intent[intent_col].value_counts().to_string())

    # Normalize labels to snake_case
    df_intent[intent_col] = (
        df_intent[intent_col]
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )

    # Keep only text + intent, drop empty
    df_out = df_intent[[text_col, intent_col]].rename(
        columns={text_col: "text", intent_col: "intent"}
    )
    df_out = df_out[df_out["text"].str.strip().str.len() > 0].reset_index(drop=True)

    df_out.to_csv(intent_out, index=False)
    print(f"\n  Saved -> {intent_out} ({intent_out.stat().st_size / 1e6:.1f} MB)")
    print(f"  Rows: {len(df_out):,} | Classes: {df_out['intent'].nunique()}")

except Exception as e:
    print(f"  Bitext download failed: {e}")
    print("  Trying CLINC150 fallback ...")
    try:
        ds_clinc = load_dataset("clinc_oos", "plus")
        label_names = ds_clinc["train"].features["intent"].names
        dfs = []
        for split_name in ["train", "validation", "test"]:
            split = ds_clinc[split_name].to_pandas()
            split["intent_str"] = split["intent"].map(lambda i: label_names[i])
            dfs.append(split[["text", "intent_str"]].rename(columns={"intent_str": "intent"}))

        df_clinc = pd.concat(dfs, ignore_index=True)
        df_clinc = df_clinc[df_clinc["intent"] != "oos"].reset_index(drop=True)
        print(f"  CLINC150: {len(df_clinc):,} rows | {df_clinc['intent'].nunique()} intents")
        df_clinc.to_csv(intent_out, index=False)
        print(f"  Saved -> {intent_out}")
    except Exception as e2:
        print(f"  CLINC fallback also failed: {e2}")
        sys.exit(1)

print("\n[DONE] Intent dataset ready.")
