"""
Fine-tunes koheiduck/bert-japanese-finetuned-sentiment on human-labelled comments.

Usage:
    python models/finetune.py --data data/labelled_comments.csv
                              --output models/finetuned
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

BASE_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = (preds == labels).mean()
    return {"macro_f1": f1, "accuracy": acc}


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def run(data_path: str, output_dir: str):
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df = df.dropna(subset=["comment", "human_label"])
    df = df[df["human_label"].isin(LABEL2ID)]
    df["label_id"] = df["human_label"].map(LABEL2ID)

    print(f"Dataset: {len(df)} samples")
    print(df["human_label"].value_counts().to_string())

    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df["label_id"], random_state=42
    )
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}")

    # class weights to handle positive underrepresentation
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(sorted(LABEL2ID.values())),
        y=train_df["label_id"].values,
    )
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"\nClass weights: {dict(zip(ID2LABEL.values(), weights.round(2)))}")

    print(f"\nLoading tokenizer and model from {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_dataset = CommentDataset(
        train_df["comment"].tolist(), train_df["label_id"].tolist(), tokenizer
    )
    val_dataset = CommentDataset(
        val_df["comment"].tolist(), val_df["label_id"].tolist(), tokenizer
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting fine-tuning ...")
    trainer.train()

    print(f"\nSaving best model to {output_dir}/best ...")
    best_path = str(Path(output_dir) / "best")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)

    print("\n=== Validation results ===")
    preds_out = trainer.predict(val_dataset)
    preds = np.argmax(preds_out.predictions, axis=-1)
    print(classification_report(
        val_df["label_id"].tolist(),
        preds,
        target_names=["negative", "neutral", "positive"],
        zero_division=0,
    ))
    print(f"\nFine-tuned model saved to {best_path}")
    print("Update MODEL_NAME in src/sentiment.py to use it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/labelled_comments.csv")
    parser.add_argument("--output", default="models/finetuned")
    args = parser.parse_args()
    run(args.data, args.output)
