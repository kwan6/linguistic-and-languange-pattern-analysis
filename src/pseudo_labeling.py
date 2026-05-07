"""
pseudo_labeling.py
Generate pseudo-labels for unlabeled Indonesian headlines
using a trained IndoBERT model with confidence threshold filtering.
"""

import os
import numpy as np
import pandas as pd
from data_loader import clean_text

# ─── Config ───────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.85   # Only keep predictions above this confidence
OUTPUT_DIR           = "data/annotated/csv"
OUTPUT_FILE          = "pseudo_labeled.csv"


# ─── Pseudo Labeler ───────────────────────────────────────────────────────────
class PseudoLabeler:
    def __init__(self, model_type: str = "indobert"):
        """
        Args:
            model_type: 'indobert' or 'distilbert'
        """
        self.model_type = model_type
        self.model      = None

    def _load_model(self):
        if self.model_type == "indobert":
            from model_indobert import IndoBERTTrainer
            self.model = IndoBERTTrainer()
            self.model._load()
            print("[PseudoLabeler] Loaded IndoBERT model")
        elif self.model_type == "distilbert":
            # Extend here if DistilBERT is added later
            raise NotImplementedError("DistilBERT pseudo labeler not yet implemented.")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def label(self,
              texts: list,
              threshold: float = CONFIDENCE_THRESHOLD) -> pd.DataFrame:
        """
        Predict labels for a list of raw headline strings.
        Returns only high-confidence predictions.

        Returns:
            DataFrame with columns [title, label, confidence]
        """
        if self.model is None:
            self._load_model()

        probs    = self.model.predict_proba(texts)          # (N, 2)
        preds    = np.argmax(probs, axis=1)
        conf     = np.max(probs, axis=1)

        df = pd.DataFrame({
            "title":      texts,
            "label":      preds,
            "confidence": conf,
        })

        before = len(df)
        df = df[df["confidence"] >= threshold].reset_index(drop=True)
        print(f"[PseudoLabeler] {len(df)}/{before} kept "
              f"(threshold={threshold}) | "
              f"Clickbait: {df['label'].sum()} | "
              f"Non-clickbait: {(df['label'] == 0).sum()}")
        return df

    def label_from_csv(self,
                       csv_path: str,
                       title_col: str = "title",
                       threshold: float = CONFIDENCE_THRESHOLD,
                       save: bool = True) -> pd.DataFrame:
        """
        Load an unlabeled CSV, predict labels, filter by confidence, and save.

        Args:
            csv_path : Path to unlabeled CSV.
            title_col: Column name of the headline text.
            threshold: Confidence threshold for keeping a prediction.
            save     : Whether to save the output CSV.
        """
        raw = pd.read_csv(csv_path)
        raw.columns = raw.columns.str.strip().str.lower()

        assert title_col.lower() in raw.columns, \
            f"Column '{title_col}' not found in {csv_path}"

        texts = raw[title_col.lower()].apply(clean_text).dropna().tolist()
        print(f"[PseudoLabeler] Loaded {len(texts)} headlines from {csv_path}")

        result = self.label(texts, threshold=threshold)
        result = result.drop(columns=["confidence"])  # keep only title + label

        if save:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
            result.to_csv(out_path, index=False)
            print(f"[PseudoLabeler] Saved → {out_path}")

        return result


# ─── Convenience function ─────────────────────────────────────────────────────
def get_pseudo_label_fn(model_type: str = "indobert",
                        threshold: float = CONFIDENCE_THRESHOLD):
    """
    Returns a simple callable: list[str] -> list[int]
    Useful for passing directly into preprocess_external.process_clickid_raw().
    """
    labeler = PseudoLabeler(model_type=model_type)

    def fn(texts: list) -> list:
        df = labeler.label(texts, threshold=threshold)
        # For rows below threshold, assign -1 as a sentinel (caller should filter)
        full = pd.DataFrame({"title": texts})
        full = full.merge(df[["title", "label"]], on="title", how="left")
        full["label"] = full["label"].fillna(-1).astype(int)
        return full["label"].tolist()

    return fn
