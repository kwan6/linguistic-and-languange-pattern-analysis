from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train_indobert(df):

    df = df.sample(5000, random_state=42)
    df = df.dropna(subset=["title"])
    df["title"] = df["title"].astype(str)

    texts = df["title"].tolist()
    labels = df["label_score"].tolist()

    # split
    from sklearn.model_selection import train_test_split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    tokenizer(train_texts, truncation=True, padding=True, max_length=64)
    tokenizer(test_texts, truncation=True, padding=True, max_length=64)
    

    train_dataset = Dataset.from_dict({
        **train_encodings,
        "labels": train_labels
    })

    test_dataset = Dataset.from_dict({
        **test_encodings,
        "labels": test_labels
    })

    model = AutoModelForSequenceClassification.from_pretrained(
        "indobenchmark/indobert-base-p1",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        fp16=True  # 
    )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    
    trainer.train()
    df = df.dropna(subset=["title"])
    df["title"] = df["title"].astype(str)
    return trainer.evaluate()