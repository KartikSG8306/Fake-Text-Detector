"""
predict.py — Run inference on arbitrary text using the fine-tuned model.

Usage:
    python src/predict.py --text "Some sentence to classify."

If the input exceeds max_length tokens, it is split into chunks and
a per-chunk prediction is shown alongside an overall majority verdict.
"""

import os
import sys
import json
import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LABEL_MAP = {0: "Human-written", 1: "AI-generated"}


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


def load_model_and_tokenizer(model_dir: str):
    """Load the saved fine-tuned model and tokenizer.

    Args:
        model_dir: Path to the directory containing the saved model.

    Returns:
        Tuple of (model, tokenizer, device).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available. Running inference on CPU.")
    model.to(device)
    return model, tokenizer, device


def predict_single(text: str, model, tokenizer, device: str,
                   max_length: int = 256) -> dict:
    """Classify a single text chunk.

    Args:
        text: Input text string.
        model: Fine-tuned classification model.
        tokenizer: Corresponding tokenizer.
        device: 'cuda' or 'cpu'.
        max_length: Maximum token length for truncation.

    Returns:
        Dictionary with 'label' and 'confidence' keys.
    """
    encodings = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encodings).logits
        probs = torch.softmax(logits, dim=-1)

    pred_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, pred_class].item()
    return {
        "label": LABEL_MAP[pred_class],
        "confidence": round(confidence, 4),
    }


def chunk_text(text: str, tokenizer, max_length: int) -> list:
    """Split text into token-aligned chunks if it exceeds max_length.

    Args:
        text: Full input text.
        tokenizer: Tokenizer used for encoding.
        max_length: Maximum number of tokens per chunk.

    Returns:
        List of text chunks (strings).
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_length - 2:  # account for [CLS] and [SEP]
        return [text]

    # Split token IDs into chunks of (max_length - 2)
    chunk_size = max_length - 2
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_ids = tokens[i : i + chunk_size]
        chunk_text_str = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text_str)
    return chunks


def predict(text: str, model, tokenizer, device: str,
            max_length: int = 256) -> dict:
    """Predict label for text, handling long inputs via chunking.

    If the input is short enough, a single prediction is returned.
    Otherwise, per-chunk predictions and a majority verdict are returned.

    Args:
        text: Input text string (may be long).
        model: Fine-tuned classification model.
        tokenizer: Corresponding tokenizer.
        device: 'cuda' or 'cpu'.
        max_length: Maximum token length per chunk.

    Returns:
        Dictionary with prediction results.
    """
    chunks = chunk_text(text, tokenizer, max_length)

    if len(chunks) == 1:
        return predict_single(chunks[0], model, tokenizer, device, max_length)

    # Multiple chunks
    chunk_results = []
    for i, chunk in enumerate(chunks):
        result = predict_single(chunk, model, tokenizer, device, max_length)
        result["chunk"] = i + 1
        chunk_results.append(result)

    # Majority vote
    ai_votes = sum(1 for r in chunk_results if r["label"] == "AI-generated")
    human_votes = len(chunk_results) - ai_votes
    majority_label = "AI-generated" if ai_votes >= human_votes else "Human-written"
    avg_conf = sum(r["confidence"] for r in chunk_results) / len(chunk_results)

    return {
        "overall": {
            "label": majority_label,
            "confidence": round(avg_conf, 4),
            "chunks_analyzed": len(chunk_results),
        },
        "per_chunk": chunk_results,
    }


def main():
    """Entry-point: parse CLI args and run prediction."""
    parser = argparse.ArgumentParser(
        description="Predict whether text is AI-generated or human-written."
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="The text to classify.",
    )
    args = parser.parse_args()

    config = load_config()
    model_dir = os.path.join(PROJECT_ROOT, config["output_dir"])
    max_length = config["max_length"]

    model, tokenizer, device = load_model_and_tokenizer(model_dir)
    result = predict(args.text, model, tokenizer, device, max_length)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
