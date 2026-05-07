"""
preprocess_external.py
Prepares external/raw datasets for use in training.
Supports: CLICK-ID raw (46K), Kaggle Indonesian news, custom CSVs.
Output: cleaned CSV with columns [title, label] ready for data_loader.py
"""

import re
import os
import pandas as pd
from data_loader import clean_text

# ─── Output path ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "data/annotated/csv"


# ─── CLICK-ID Raw Processor ───────────────────────────────────────────────────
def process_clickid_raw(raw_csv_path: str,
                        pseudo_label_fn=None,
                        output_name: str = "clickid_raw_processed.csv") -> pd.DataFrame:
    """
    Process the CLICK-ID raw dataset (46K articles from Mendeley).
    Columns expected: title (or headline/judul), optionally label.

    If no label column exists, applies pseudo_label_fn (callable: list[str] -> list[int]).
    """
    df = pd.read_csv(raw_csv_path)
    df.columns = df.columns.str.strip().str.lower()

    # Normalize title column
    for alias in ["headline", "judul", "text", "content"]:
        if alias in df.columns and "title" not in df.columns:
            df.rename(columns={alias: "title"}, inplace=True)
            break

    assert "title" in df.columns, "Cannot find title/headline column in raw dataset"

    df["title"] = df["title"].apply(clean_text)
    df.dropna(subset=["title"], inplace=True)
    df.drop_duplicates(subset=["title"], inplace=True)

    if "label" not in df.columns:
        assert pseudo_label_fn is not None, \
            "Raw dataset has no label column. Provide pseudo_label_fn."
        print("[preprocess_external] No labels found — applying pseudo labeling ...")
        df["label"] = pseudo_label_fn(df["title"].tolist())
    else:
        label_map = {"clickbait": 1, "non-clickbait": 0, "non_clickbait": 0}
        if df["label"].dtype == object:
            df["label"] = df["label"].str.strip().str.lower().map(label_map)
        df["label"] = df["label"].astype(int)

    df = df[["title", "label"]].reset_index(drop=True)
    _save(df, output_name)
    return df


# ─── Generic External CSV Processor ──────────────────────────────────────────
def process_generic(csv_path: str,
                    title_col: str,
                    label_col: str = None,
                    label_map: dict = None,
                    fixed_label: int = None,
                    output_name: str = "external_processed.csv") -> pd.DataFrame:
    """
    Flexible processor for any external CSV.

    Args:
        csv_path   : Path to the external CSV.
        title_col  : Column name containing the headline text.
        label_col  : Column name containing labels (optional).
        label_map  : Dict to map raw label values → 0/1.
        fixed_label: If dataset has no labels, assign this fixed label (0 or 1).
        output_name: Output CSV filename.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    title_col  = title_col.lower()

    assert title_col in df.columns, f"Column '{title_col}' not in CSV"

    df["title"] = df[title_col].apply(clean_text)
    df.dropna(subset=["title"], inplace=True)
    df.drop_duplicates(subset=["title"], inplace=True)

    if label_col and label_col.lower() in df.columns:
        df["label"] = df[label_col.lower()]
        if label_map:
            df["label"] = df["label"].map(label_map)
        df["label"] = df["label"].astype(int)
    elif fixed_label is not None:
        df["label"] = fixed_label
    else:
        raise ValueError("Provide either label_col, label_map, or fixed_label.")

    df = df[["title", "label"]].reset_index(drop=True)
    _save(df, output_name)
    return df


# ─── Merge & Deduplicate ──────────────────────────────────────────────────────
def merge_and_save(dfs: list, output_name: str = "merged_dataset.csv") -> pd.DataFrame:
    """Merge multiple processed DataFrames and deduplicate."""
    merged = pd.concat(dfs, ignore_index=True)
    before = len(merged)
    merged.drop_duplicates(subset=["title"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    print(f"[preprocess_external] Merged: {len(merged)} rows "
          f"(removed {before - len(merged)} duplicates)")
    print(f"  Clickbait    : {merged['label'].sum()}")
    print(f"  Non-clickbait: {(merged['label'] == 0).sum()}")
    _save(merged, output_name)
    return merged


# ─── Helper ───────────────────────────────────────────────────────────────────
def _save(df: pd.DataFrame, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[preprocess_external] Saved {len(df)} rows → {path}")
