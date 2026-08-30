"""
train.py — Fine-tune a transformer model on the HC3 dataset for fake text detection.

Uses HuggingFace Trainer with fp16 mixed-precision, gradient accumulation,
and early-saving of the best model by validation F1 score.
"""

import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = None) -> dict:
    """Load the YAML configuration file.

    Args:
        config_path: Path to config.yaml. Defaults to PROJECT_ROOT/config.yaml.

    Returns:
        A dictionary of configuration values.
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class TextClassificationDataset(Dataset):
    """PyTorch Dataset for tokenized text classification data.

    Args:
        encodings: Dictionary of tokenized inputs (input_ids, attention_mask, …).
        labels: List or array of integer labels.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        """Return the number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return a single tokenized sample with its label.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing input tensors and the label.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    """Compute accuracy, precision, recall, and F1 (weighted) for the Trainer.

    Args:
        eval_pred: EvalPrediction object with predictions and label_ids.

    Returns:
        Dictionary of metric names to values.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    """Load data, tokenize, train, and save the best model."""

    # --- Device check ---
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("WARNING: CUDA is not available. Training will run on CPU and will be "
              "significantly slower. Install a CUDA-enabled PyTorch build for GPU support.")

    config = load_config()

    # --- Load processed CSVs ---
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    print(f"Train samples: {len(train_df)}  |  Val samples: {len(val_df)}")

    # --- Tokenize ---
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=config["max_length"],
    )
    val_encodings = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=config["max_length"],
    )

    train_dataset = TextClassificationDataset(train_encodings, train_df["label"].tolist())
    val_dataset = TextClassificationDataset(val_encodings, val_df["label"].tolist())

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2,
    )

    # --- Training arguments ---
    output_dir = os.path.join(PROJECT_ROOT, config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Determine fp16 usage -- only on CUDA
    use_fp16 = config.get("fp16", True) and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=float(config["learning_rate"]),
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=config["seed"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # --- Train ---
    try:
        print("\n" + "=" * 60)
        print("  Starting training")
        print("=" * 60 + "\n")
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(
                "\n[ERROR] CUDA out-of-memory error!\n"
                "  Try reducing `batch_size` or `max_length` in config.yaml.\n"
                "  Current batch_size={}, max_length={}".format(
                    config["batch_size"], config["max_length"]
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sys.exit(1)
        else:
            raise

    # --- Save best model + tokenizer ---
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[OK] Best model and tokenizer saved to: {output_dir}")

    # --- Final validation metrics ---
    metrics = trainer.evaluate()
    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
