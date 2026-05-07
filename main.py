"""
main.py
Entry point for the Clickbait Classification project.

Usage:
  python main.py --mode train
  python main.py --mode train --use_pseudo
  python main.py --mode evaluate
  python main.py --mode pseudo_label --unlabeled data/raw/clickid_raw.csv
  python main.py --mode stats
  python main.py --mode predict --text "Kamu Tidak Akan Percaya Apa yang Terjadi!"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_all, split_dataset, source_stats
from model_indobert import IndoBERTTrainer
from model_ml import MLTrainer
from pseudo_labeling import PseudoLabeler

# Dataset directory (sesuaikan jika perlu)
DATA_DIR  = "data/annotated/csv"
TEST_CSV  = os.path.join(DATA_DIR, "test_set.csv")


def run_train(use_pseudo: bool = False):
    print("=" * 55)
    print("Loading dataset")
    print("=" * 55)
    df = load_all(DATA_DIR, include_pseudo=use_pseudo)
    source_stats(df)
    train_df, val_df, test_df = split_dataset(df)

    # Save test set for later evaluation
    test_df.to_csv(TEST_CSV, index=False)
    print(f"[main] Test set saved -> {TEST_CSV}\n")

    # ML Models
    print("=" * 55)
    print("Training ML Models (Naive Bayes, SVM, Logistic Regression)")
    print("=" * 55)
    ml = MLTrainer()
    ml.train(train_df, val_df)

    # IndoBERT
    print("\n" + "=" * 55)
    print("Training IndoBERT")
    print("=" * 55)
    bert = IndoBERTTrainer()
    bert.train(train_df, val_df)

    print("\n[main] Training complete.")


def run_evaluate():
    import pandas as pd
    assert os.path.exists(TEST_CSV), \
        f"Test set not found at {TEST_CSV}. Run --mode train first."

    test_df = pd.read_csv(TEST_CSV)
    print(f"[main] Evaluating on {len(test_df)} test samples\n")

    print("=" * 55)
    print("ML Models Evaluation")
    print("=" * 55)
    ml = MLTrainer()
    ml.evaluate(test_df)

    print("\n" + "=" * 55)
    print("IndoBERT Evaluation")
    print("=" * 55)
    bert = IndoBERTTrainer()
    bert.evaluate(test_df)


def run_pseudo_label(unlabeled_csv: str, model_type: str = "indobert"):
    labeler = PseudoLabeler(model_type=model_type)
    labeler.label_from_csv(unlabeled_csv, title_col="title", save=True)
    print(f"\n[main] Done. Re-run with --mode train --use_pseudo to retrain.")


def run_stats():
    df = load_all(DATA_DIR)
    source_stats(df)


def run_predict(text: str):
    print(f"\n[main] Input : '{text}'\n")

    bert  = IndoBERTTrainer()
    pred  = bert.predict([text])[0]
    print(f"  IndoBERT             -> {'CLICKBAIT' if pred == 1 else 'NON-CLICKBAIT'}")

    ml = MLTrainer()
    ml._load_all()
    for name in ["naive_bayes", "svm", "logistic_regression"]:
        pred  = ml.predict(name, [text])[0]
        print(f"  {name:<24} -> {'CLICKBAIT' if pred == 1 else 'NON-CLICKBAIT'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clickbait classification — Indonesian news headlines"
    )
    parser.add_argument("--mode", required=True,
                        choices=["train", "evaluate", "pseudo_label", "stats", "predict"])
    parser.add_argument("--unlabeled", type=str, default=None,
                        help="Path to unlabeled CSV (for pseudo_label mode)")
    parser.add_argument("--text", type=str, default=None,
                        help="Headline to classify (for predict mode)")
    parser.add_argument("--use_pseudo", action="store_true",
                        help="Include pseudo_labeled.csv during training")
    parser.add_argument("--model", type=str, default="indobert",
                        choices=["indobert"],
                        help="Model for pseudo labeling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        run_train(use_pseudo=args.use_pseudo)
    elif args.mode == "evaluate":
        run_evaluate()
    elif args.mode == "pseudo_label":
        assert args.unlabeled, "Provide --unlabeled <path_to_csv>"
        run_pseudo_label(args.unlabeled, model_type=args.model)
    elif args.mode == "stats":
        run_stats()
    elif args.mode == "predict":
        assert args.text, "Provide --text 'your headline here'"
        run_predict(args.text)
