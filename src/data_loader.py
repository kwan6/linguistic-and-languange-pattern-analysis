"""
data_loader.py
Loads, cleans, and splits the clickbait dataset.

Dataset structure:
  data/annotated/csv/
    annotated_detikNews.csv
    annotated_fimela.csv
    annotated_kapanlagi.csv
    annotated_kompas.csv
    annotated_liputan6.csv
    annotated_okezone.csv
    annotated_pos_metro.csv
    annotated_republika.csv
    annotated_sindonews.csv
    annotated_tempo.csv
    annotated_tribunnews.csv
    annotated_wowkeren.csv
    pseudo_labeled.csv          <- optional, hasil pseudo labeling
"""

import re
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# Label Mapping
LABEL_MAP     = {"clickbait": 1, "non-clickbait": 0, "non_clickbait": 0}
LABEL_MAP_INV = {1: "clickbait", 0: "non-clickbait"}

DATA_DIR = "data/annotated/csv"


def clean_text(text: str) -> str:
    """Basic cleaning for Indonesian news headlines."""
    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_one(csv_path: str, source_name: str = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    for alias in ["headline", "judul", "text", "content"]:
        if alias in df.columns and "title" not in df.columns:
            df.rename(columns={alias: "title"}, inplace=True)
            break

    if "title" not in df.columns or "label" not in df.columns:
        print(f"[data_loader] SKIP {csv_path} — missing title/label column")
        return pd.DataFrame()

    df["title"] = df["title"].apply(clean_text)
    df.dropna(subset=["title", "label"], inplace=True)

    # Normalize label: string -> int
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map(LABEL_MAP)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    if source_name:
        df["source"] = source_name
    elif "source" not in df.columns:
        df["source"] = os.path.basename(csv_path).replace(".csv", "")

    return df[["title", "label", "source"]]


def load_all(data_dir: str = DATA_DIR,
             extra_paths: list = None,
             include_pseudo: bool = False) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "annotated_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"[data_loader] No annotated_*.csv found in '{data_dir}'"
        )

    frames = []
    print("[data_loader] Loading sources:")
    for f in files:
        source = os.path.basename(f).replace("annotated_", "").replace(".csv", "")
        df     = _load_one(f, source_name=source)
        if not df.empty:
            frames.append(df)
            print(f"  {source:<20} -> {len(df):>5} rows  "
                  f"(clickbait: {df['label'].sum()}, "
                  f"non-clickbait: {(df['label']==0).sum()})")

    if include_pseudo:
        pseudo_path = os.path.join(data_dir, "pseudo_labeled.csv")
        if os.path.exists(pseudo_path):
            df = _load_one(pseudo_path, source_name="pseudo_labeled")
            if not df.empty:
                frames.append(df)
                print(f"  {'pseudo_labeled':<20} -> {len(df):>5} rows")
        else:
            print("[data_loader] WARNING: pseudo_labeled.csv not found, skipping.")

    if extra_paths:
        for path in extra_paths:
            if not os.path.exists(path):
                print(f"[data_loader] WARNING: {path} not found, skipping.")
                continue
            df = _load_one(path)
            if not df.empty:
                frames.append(df)
                print(f"  {path:<20} -> {len(df):>5} rows")

    merged = pd.concat(frames, ignore_index=True)

    before = len(merged)
    merged.drop_duplicates(subset=["title"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    print(f"\n[data_loader] Total: {len(merged)} rows "
          f"(removed {before - len(merged)} duplicates)")
    print(f"  Clickbait    : {merged['label'].sum()}")
    print(f"  Non-clickbait: {(merged['label'] == 0).sum()}")
    print(f"  Sources      : {merged['source'].nunique()} portals\n")
    return merged


def source_stats(df: pd.DataFrame):
    """Print per-source label distribution."""
    print("\n[data_loader] Per-source breakdown:")
    stats = df.groupby("source")["label"].value_counts().unstack(fill_value=0)
    stats.columns = ["non-clickbait", "clickbait"]
    stats["total"] = stats.sum(axis=1)
    print(stats.to_string())
    print()


def split_dataset(df: pd.DataFrame,
                  test_size: float = 0.15,
                  val_size: float = 0.15,
                  random_state: int = 42):
    """Stratified split: 70% train / 15% val / 15% test (default)."""
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size),
        stratify=df["label"], random_state=random_state
    )
    rel_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - rel_val),
        stratify=temp_df["label"], random_state=random_state
    )
    print(f"[data_loader] Split -> Train: {len(train_df)} | "
          f"Val: {len(val_df)} | Test: {len(test_df)}")
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))