"""
test_predict.py — Smoke test for the prediction pipeline.

Loads the saved model and asserts that a prediction is returned
in the expected format for a single sample sentence.
"""

import os
import sys
import json

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import load_model_and_tokenizer, predict, load_config


def test_prediction_format():
    """Smoke test: load the model and verify that a single prediction
    returns a dict with 'label' and 'confidence' keys, with valid values.
    """
    config = load_config()
    model_dir = os.path.join(PROJECT_ROOT, config["output_dir"])
    max_length = config["max_length"]

    # Verify the saved model directory exists
    assert os.path.isdir(model_dir), (
        f"Saved model directory not found: {model_dir}\n"
        "Run train.py first to produce a saved model."
    )

    model, tokenizer, device = load_model_and_tokenizer(model_dir)

    sample_text = (
        "The mitochondria is the powerhouse of the cell. "
        "It produces ATP through oxidative phosphorylation."
    )
    result = predict(sample_text, model, tokenizer, device, max_length)

    # Validate structure
    assert isinstance(result, dict), "Prediction result must be a dict."
    assert "label" in result, "Result must contain a 'label' key."
    assert "confidence" in result, "Result must contain a 'confidence' key."

    # Validate values
    assert result["label"] in ("Human-written", "AI-generated"), (
        f"Unexpected label: {result['label']}"
    )
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"Confidence out of range: {result['confidence']}"
    )

    print(f"[OK] Smoke test passed: {json.dumps(result)}")


if __name__ == "__main__":
    test_prediction_format()
    print("\nAll tests passed.")
