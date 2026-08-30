"""
evaluate.py — Evaluate the fine-tuned model on the held-out test split.

Computes accuracy, precision, recall, F1, and a confusion matrix.
Saves metrics to results/metrics.json and the confusion matrix plot
to results/confusion_matrix.png.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_model_and_tokenizer(model_dir: str):
    """Load the saved fine-tuned model and tokenizer.

    Args:
        model_dir: Path to the directory containing the saved model.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict_batch(texts: list, model, tokenizer, max_length: int = 256,
                  batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    """Run inference on a list of texts and return predicted class indices.

    Args:
        texts: List of input strings.
        model: The fine-tuned classification model.
        tokenizer: Corresponding tokenizer.
        max_length: Maximum token length for truncation.
        batch_size: Number of texts per forward pass.
        device: 'cuda' or 'cpu'.

    Returns:
        Numpy array of predicted labels (0 or 1).
    """
    all_preds = []
    model.to(device)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encodings).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)

    return np.array(all_preds)


def compute_and_save_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             results_dir: str) -> dict:
    """Compute classification metrics and save them to JSON.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        results_dir: Directory where metrics.json will be written.

    Returns:
        Dictionary of computed metrics.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          results_dir: str) -> None:
    """Generate and save a confusion matrix plot.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        results_dir: Directory where confusion_matrix.png will be written.
    """
    os.makedirs(results_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Human", "AI-generated"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    save_path = os.path.join(results_dir, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {save_path}")


def print_summary(metrics: dict, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a clean summary table to the console.

    Args:
        metrics: Dictionary of aggregated metrics.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
    """
    print("\n" + "=" * 50)
    print("  Evaluation Results — Test Set")
    print("=" * 50)
    print(f"  {'Metric':<15} {'Value':>10}")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"  {k:<15} {v:>10.4f}")
    print("-" * 30)

    print("\nPer-class report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Human-written", "AI-generated"],
        digits=4,
    ))


def main():
    """Entry-point: load model -> predict on test set -> save metrics + plot."""

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available. Running evaluation on CPU.")

    # Paths
    model_dir = os.path.join(PROJECT_ROOT, "models", "saved_model")
    test_csv = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    results_dir = os.path.join(PROJECT_ROOT, "results")

    # Load
    print(f"Loading model from: {model_dir}")
    model, tokenizer = load_model_and_tokenizer(model_dir)

    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")

    texts = test_df["text"].tolist()
    y_true = test_df["label"].values

    # Predict
    print("Running inference on test set ...")
    y_pred = predict_batch(texts, model, tokenizer, device=device)

    # Metrics
    metrics = compute_and_save_metrics(y_true, y_pred, results_dir)
    plot_confusion_matrix(y_true, y_pred, results_dir)
    print_summary(metrics, y_true, y_pred)

    print("\n[OK] Evaluation complete.")


if __name__ == "__main__":
    main()
