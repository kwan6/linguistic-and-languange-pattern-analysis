"""
model_ml.py
Traditional ML models: Naive Bayes, SVM, Logistic Regression
Feature extraction: TF-IDF (unigram + bigram)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

# ─── Config ───────────────────────────────────────────────────────────────────
SAVE_DIR  = "models/ml"
TFIDF_CFG = dict(
    ngram_range=(1, 2),
    max_features=50_000,
    sublinear_tf=True,
    min_df=2,
)

MODELS = {
    "naive_bayes":        MultinomialNB(alpha=0.5),
    "svm":                LinearSVC(C=1.0, max_iter=2000),
    "logistic_regression": LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
}


# ─── ML Trainer ───────────────────────────────────────────────────────────────
class MLTrainer:
    def __init__(self):
        self.pipelines: dict[str, Pipeline] = {}

    def _build_pipeline(self, name: str) -> Pipeline:
        steps = [("tfidf", TfidfVectorizer(**TFIDF_CFG))]
        if name == "naive_bayes":
            # MNB requires non-negative features; use_idf off for raw counts
            steps = [("tfidf", TfidfVectorizer(**{**TFIDF_CFG, "use_idf": False}))]
        steps.append(("clf", MODELS[name]))
        return Pipeline(steps)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        """Train all ML models and report validation F1."""
        results = {}
        os.makedirs(SAVE_DIR, exist_ok=True)

        X_train, y_train = train_df["title"].tolist(), train_df["label"].tolist()
        X_val,   y_val   = val_df["title"].tolist(),   val_df["label"].tolist()

        for name in MODELS:
            print(f"\n[ML] Training {name} ...")
            pipe = self._build_pipeline(name)
            pipe.fit(X_train, y_train)

            val_preds = pipe.predict(X_val)
            val_f1    = f1_score(y_val, val_preds, average="macro")
            print(f"  Val F1: {val_f1:.4f}")

            self.pipelines[name] = pipe
            self._save(name, pipe)
            results[name] = {"val_f1": val_f1}

        return results

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Load all saved models and evaluate on test set."""
        self._load_all()
        X_test, y_test = test_df["title"].tolist(), test_df["label"].tolist()
        all_results = {}

        for name, pipe in self.pipelines.items():
            preds  = pipe.predict(X_test)
            report = classification_report(
                y_test, preds,
                target_names=["non-clickbait", "clickbait"],
                output_dict=True,
            )
            print(f"\n[ML] {name.upper()} Test Results:")
            print(classification_report(y_test, preds, target_names=["non-clickbait", "clickbait"]))
            all_results[name] = {"report": report, "preds": preds}

        return all_results

    def predict(self, model_name: str, texts: list) -> list:
        if model_name not in self.pipelines:
            self._load(model_name)
        return self.pipelines[model_name].predict(texts).tolist()

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save(self, name: str, pipe: Pipeline):
        path = os.path.join(SAVE_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(pipe, f)
        print(f"  Saved → {path}")

    def _load(self, name: str):
        path = os.path.join(SAVE_DIR, f"{name}.pkl")
        with open(path, "rb") as f:
            self.pipelines[name] = pickle.load(f)

    def _load_all(self):
        for name in MODELS:
            path = os.path.join(SAVE_DIR, f"{name}.pkl")
            if os.path.exists(path):
                self._load(name)
            else:
                print(f"[ML] WARNING: {path} not found, skipping {name}")
