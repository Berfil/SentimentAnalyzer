"""
Benchmark two or more HuggingFace models against the hand-labelled sample posts.

Usage:
    python models/benchmark.py --data data/sample_posts.csv

Outputs a per-model accuracy/F1 table and saves a confusion matrix PNG for each model.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sentiment import MODELS, load_model, predict_batch

LABEL_NORMALIZATION = {
    # koheiduck model uses integer string labels
    "0": "negative",
    "1": "positive",
    # cardiffnlp / lxyuan
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


def normalize_label(raw: str) -> str:
    return LABEL_NORMALIZATION.get(raw, raw).lower()


def run_benchmark(data_path: str):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()

    texts = df["text"].tolist()
    true_labels = df["label"].tolist()

    summary_rows = []

    for model_name, config in MODELS.items():
        if config.get("skip"):
            print(f"\nSkipping {model_name} (not fine-tuned for sentiment)")
            continue

        print(f"\nLoading {model_name} ...")
        try:
            classifier = load_model(model_name)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        print("  Running inference ...")
        raw_preds = predict_batch(classifier, texts)
        pred_labels = [normalize_label(p["label"]) for p in raw_preds]

        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

        print(f"\n  === {model_name} ===")
        print(f"  Accuracy : {acc:.3f}")
        print(f"  Macro F1 : {f1:.3f}")
        print(classification_report(true_labels, pred_labels, zero_division=0))

        summary_rows.append(
            {"model": model_name, "accuracy": round(acc, 3), "macro_f1": round(f1, 3)}
        )

        _save_confusion_matrix(true_labels, pred_labels, model_name)

    print("\n\n=== SUMMARY ===")
    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("models/benchmark_results.csv", index=False)
    print("\nResults saved to models/benchmark_results.csv")


def _save_confusion_matrix(true_labels, pred_labels, model_name):
    classes = sorted(set(true_labels) | set(pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    safe_name = model_name.replace("/", "_")
    ax.set_title(safe_name)
    fig.tight_layout()

    out_path = f"models/cm_{safe_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/sample_posts.csv",
        help="Path to labelled CSV (columns: text, label)",
    )
    args = parser.parse_args()
    run_benchmark(args.data)
