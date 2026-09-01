"""
streamlit_app.py -- Streamlit dashboard for the Fake Text Detector.

Run locally:
    streamlit run streamlit_app.py

Deploy free on Streamlit Community Cloud:
    1. Push this repo to a public GitHub repo
    2. Go to https://streamlit.io/cloud
    3. Point it to this file
"""

import os
import json
import yaml
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake Text Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
LABEL_MAP = {0: "Human-written", 1: "AI-generated"}


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
@st.cache_data
def load_config():
    """Load the YAML configuration file (cached)."""
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Load model (cached so it only loads once across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the fine-tuned model and tokenizer (cached in memory)."""
    config = load_config()
    model_dir = os.path.join(PROJECT_ROOT, config["output_dir"])

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device, config["max_length"]


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
def predict_single(text, model, tokenizer, device, max_length):
    """Classify a single text chunk."""
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
    human_prob = probs[0, 0].item()
    ai_prob = probs[0, 1].item()

    return {
        "label": LABEL_MAP[pred_class],
        "confidence": round(confidence, 4),
        "human_probability": round(human_prob, 4),
        "ai_probability": round(ai_prob, 4),
    }


def chunk_text(text, tokenizer, max_length):
    """Split text into token-aligned chunks if it exceeds max_length."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_length - 2:
        return [text]

    chunk_size = max_length - 2
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_ids = tokens[i : i + chunk_size]
        chunk_str = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_str)
    return chunks


def predict(text, model, tokenizer, device, max_length):
    """Predict label for text, handling long inputs via chunking."""
    chunks = chunk_text(text, tokenizer, max_length)

    if len(chunks) == 1:
        return predict_single(chunks[0], model, tokenizer, device, max_length)

    chunk_results = []
    for i, chunk in enumerate(chunks):
        result = predict_single(chunk, model, tokenizer, device, max_length)
        result["chunk"] = i + 1
        chunk_results.append(result)

    ai_votes = sum(1 for r in chunk_results if r["label"] == "AI-generated")
    human_votes = len(chunk_results) - ai_votes
    majority_label = "AI-generated" if ai_votes >= human_votes else "Human-written"
    avg_conf = sum(r["confidence"] for r in chunk_results) / len(chunk_results)
    avg_human = sum(r["human_probability"] for r in chunk_results) / len(chunk_results)
    avg_ai = sum(r["ai_probability"] for r in chunk_results) / len(chunk_results)

    return {
        "label": majority_label,
        "confidence": round(avg_conf, 4),
        "human_probability": round(avg_human, 4),
        "ai_probability": round(avg_ai, 4),
        "chunks_analyzed": len(chunk_results),
        "per_chunk": chunk_results,
    }


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
def inject_css():
    """Inject custom CSS for a polished dark UI."""
    st.markdown("""
    <style>
        /* Import font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Main container */
        .stApp { font-family: 'Inter', sans-serif; }

        /* Header */
        .main-title {
            text-align: center;
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6c5ce7, #a29bfe, #74b9ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 4px;
        }
        .main-subtitle {
            text-align: center;
            color: #8b8fa7;
            font-size: 1rem;
            margin-bottom: 30px;
        }

        /* Result badge */
        .result-badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 12px 28px;
            border-radius: 50px;
            font-size: 1.2rem;
            font-weight: 700;
        }
        .badge-human {
            background: rgba(0, 184, 148, 0.15);
            color: #00b894;
            border: 1px solid rgba(0, 184, 148, 0.3);
        }
        .badge-ai {
            background: rgba(225, 112, 85, 0.15);
            color: #e17055;
            border: 1px solid rgba(225, 112, 85, 0.3);
        }

        /* Confidence number */
        .confidence-big {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -2px;
            line-height: 1;
        }
        .confidence-label {
            font-size: 0.8rem;
            color: #8b8fa7;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        /* Probability bar labels */
        .prob-label {
            font-size: 0.88rem;
            font-weight: 500;
            color: #8b8fa7;
            margin-bottom: 4px;
        }
        .prob-value {
            font-size: 0.88rem;
            font-weight: 600;
        }
        .human-color { color: #00b894; }
        .ai-color    { color: #e17055; }

        /* Chunk details */
        .chunk-box {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px;
            padding: 14px 18px;
            font-size: 0.85rem;
            color: #8b8fa7;
            line-height: 1.7;
        }
        .chunk-box strong { color: #e4e6ef; }

        /* Hide Streamlit branding */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* Custom footer */
        .custom-footer {
            text-align: center;
            color: #8b8fa7;
            font-size: 0.8rem;
            padding: 30px 0 10px;
        }
        .custom-footer a { color: #6c5ce7; text-decoration: none; }
        .custom-footer a:hover { text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    """Main Streamlit app."""
    inject_css()

    # Header
    st.markdown('<div class="main-title">🔍 Fake Text Detector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Paste any text below to check if it was written by a human or generated by AI</div>',
        unsafe_allow_html=True,
    )

    # Load model
    with st.spinner("Loading model ... (this only happens once)"):
        model, tokenizer, device, max_length = load_model()

    # Input
    text = st.text_area(
        "Enter text to analyze",
        height=200,
        placeholder="Paste or type the text you want to analyze here ...",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_clicked = st.button("🔍  Analyze Text", use_container_width=True, type="primary")
    with col2:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        st.rerun()

    # Character count
    st.caption(f"{len(text)} characters")

    # Analyze
    if analyze_clicked:
        if not text.strip():
            st.error("Please paste or type some text before analyzing.")
            return

        with st.spinner("Analyzing ..."):
            result = predict(text.strip(), model, tokenizer, device, max_length)

        st.markdown("---")

        is_ai = result["label"] == "AI-generated"

        # ── Verdict + Confidence ──
        col_badge, col_conf = st.columns([2, 1])

        with col_badge:
            badge_class = "badge-ai" if is_ai else "badge-human"
            icon = "⚠️" if is_ai else "✅"
            st.markdown(
                f'<div class="result-badge {badge_class}">'
                f'{icon} {result["label"]}</div>',
                unsafe_allow_html=True,
            )

        with col_conf:
            conf_pct = f'{result["confidence"] * 100:.1f}%'
            st.markdown(
                f'<div style="text-align:right;">'
                f'<div class="confidence-big">{conf_pct}</div>'
                f'<div class="confidence-label">Confidence</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability Bars ──
        human_pct = result["human_probability"]
        ai_pct = result["ai_probability"]

        # Human bar
        st.markdown('<div class="prob-label">Human-written</div>', unsafe_allow_html=True)
        col_bar, col_val = st.columns([5, 1])
        with col_bar:
            st.progress(human_pct)
        with col_val:
            st.markdown(
                f'<div class="prob-value human-color">{human_pct * 100:.1f}%</div>',
                unsafe_allow_html=True,
            )

        # AI bar
        st.markdown('<div class="prob-label">AI-generated</div>', unsafe_allow_html=True)
        col_bar2, col_val2 = st.columns([5, 1])
        with col_bar2:
            st.progress(ai_pct)
        with col_val2:
            st.markdown(
                f'<div class="prob-value ai-color">{ai_pct * 100:.1f}%</div>',
                unsafe_allow_html=True,
            )

        # ── Chunk details (for long texts) ──
        if result.get("chunks_analyzed", 0) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            chunk_html = f'<div class="chunk-box"><strong>{result["chunks_analyzed"]} chunks analyzed</strong> (text exceeded max token length)<br>'
            for c in result["per_chunk"]:
                c_icon = "⚠️" if c["label"] == "AI-generated" else "✅"
                chunk_html += f'Chunk {c["chunk"]}: {c_icon} {c["label"]} ({c["confidence"] * 100:.1f}%)<br>'
            chunk_html += '</div>'
            st.markdown(chunk_html, unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div class="custom-footer">'
        'Powered by <strong>DistilRoBERTa</strong> fine-tuned on the '
        '<a href="https://huggingface.co/datasets/Hello-SimpleAI/HC3" target="_blank">HC3 Dataset</a>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
