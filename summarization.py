# summarization.py
# Upgrades + Legacy T5 option:
# - Legacy T5 mode to mimic older outputs (prefix + fixed lengths)
# - Force CPU device to avoid meta/CPU mismatch
# - Cached model loading, length presets, chunking, extractive mode

import os
import re
from typing import List, Tuple

import streamlit as st

# Always force CPU for safety (prevents meta device issues)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Lazy imports (transformers only needed for abstractive mode)
_TRANSFORMERS_OK = True
try:
    from transformers import pipeline, AutoTokenizer
except Exception:
    _TRANSFORMERS_OK = False

# --------------------------- Utility & Config --------------------------- #

APP_TITLE = "Text in Brief — Summarizer"
APP_TAGLINE = "Abstractive + Extractive summarization with length control"

DEFAULT_MODEL = "t5-small"
MODEL_CHOICES = [
    "t5-small",
    "t5-base",
    "facebook/bart-large-cnn",
    "google/pegasus-xsum",
]

LENGTH_PRESETS = {
    "Short": (30, 80),     # (min_tokens, max_tokens)
    "Medium": (60, 160),
    "Long": (120, 240),
}

# small English stopword set (no extra deps)
STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is","it",
    "its","of","on","that","the","to","was","were","will","with","this","these","those",
    "i","you","we","they","them","their","our","your","or","if","but","not","so","than",
    "then","there","here","when","while","what","which","who","whom","why","how"
}

def clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def split_into_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    if len(parts) == 1:
        parts = re.split(r"\s{2,}", text.strip())
    return [p.strip() for p in parts if p.strip()]

def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\-']+", text.lower())

def score_sentences_frequency(sentences: List[str]) -> List[Tuple[float, str]]:
    freq = {}
    for s in sentences:
        for w in word_tokens(s):
            if w in STOPWORDS or len(w) <= 2:
                continue
            freq[w] = freq.get(w, 0) + 1

    if not freq:
        return [(0.0, s) for s in sentences]

    max_f = max(freq.values())
    for k in list(freq.keys()):
        freq[k] = freq[k] / max_f

    scored = []
    for s in sentences:
        score = 0.0
        for w in word_tokens(s):
            if w in freq:
                score += freq[w]
        scored.append((score, s))
    return scored

def select_top_sentences(sentences: List[str], ratio: float = 0.2, min_sent: int = 3, max_sent: int = 10) -> str:
    if not sentences:
        return ""
    n = max(min_sent, min(max_sent, max(1, int(len(sentences) * ratio))))
    scored = score_sentences_frequency(sentences)
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:n]
    ordered = sorted([s for _, s in top], key=lambda s: sentences.index(s))
    return " ".join(ordered)

def get_length_preset(name: str) -> Tuple[int, int]:
    return LENGTH_PRESETS.get(name, LENGTH_PRESETS["Medium"])

# ------------------------ Abstractive summarization --------------------- #

@st.cache_resource(show_spinner=False)
def load_abstractive_pipeline(model_name: str):
    """
    Load and cache a transformers summarization pipeline for a given model.
    Force CPU with device=-1 to avoid GPU/meta issues on Windows.
    """
    if not _TRANSFORMERS_OK:
        raise RuntimeError(
            "transformers not available. Install dependencies from requirements.txt"
        )
    pipe = pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
        framework="pt",
        device=-1,  # <- force CPU
    )
    return pipe

def chunk_by_tokens(pipe, text: str, max_input_tokens: int) -> List[str]:
    tokenizer: AutoTokenizer = pipe.tokenizer
    words = text.split()
    chunks, cur, cur_tokens = [], [], 0
    for w in words:
        tok_count = len(tokenizer(w, add_special_tokens=False).input_ids) or 1
        if cur_tokens + tok_count >= max_input_tokens - 8:
            if cur:
                chunks.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(w)
        cur_tokens += tok_count
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def summarize_abstractive(text: str, model_name: str, preset: str, beams: int = 4,
                          legacy_t5: bool = False) -> str:
    pipe = load_abstractive_pipeline(model_name)

    # Legacy behavior to mimic older code
    if legacy_t5 and model_name.startswith("t5"):
        min_len, max_len = (20, 100)
    else:
        min_len, max_len = get_length_preset(preset)

    # model's max input length
    model_ctx = getattr(pipe.tokenizer, "model_max_length", 1024)
    if model_ctx is None or model_ctx > 4096:
        model_ctx = 1024

    chunks = chunk_by_tokens(pipe, text, model_ctx)
    results = []
    for ch in chunks:
        # Old scripts often prepended "summarize: " for T5
        if legacy_t5 and model_name.startswith("t5"):
            ch = "summarize: " + ch

        try:
            out = pipe(
                ch,
                do_sample=False,             # deterministic
                num_beams=beams,             # beam search
                early_stopping=True,
                min_length=min_len,
                max_length=max_len,
                no_repeat_ngram_size=2,      # reduce repetition (typical safeguard)
                length_penalty=1.0,
                repetition_penalty=1.0,
            )
            results.append(out[0]["summary_text"])
        except Exception:
            out = pipe(ch, do_sample=False, num_beams=max(1, beams), max_length=max(40, max_len // 2))
            results.append(out[0]["summary_text"])

    # In legacy mode, keep chunk outputs as-is to preserve style
    final_text = " ".join(results).strip()
    if not legacy_t5 and len(results) > 1 and len(final_text.split()) > 120:
        # light compress only in non-legacy
        sentences = split_into_sentences(final_text)
        final_text = select_top_sentences(sentences, ratio=0.35, min_sent=3, max_sent=10)
    return final_text

# ------------------------ Extractive summarization ---------------------- #

def summarize_extractive(text: str, preset: str) -> str:
    sentences = split_into_sentences(text)
    if not sentences:
        return ""
    ratio_map = {"Short": 0.15, "Medium": 0.25, "Long": 0.4}
    ratio = ratio_map.get(preset, 0.25)
    return select_top_sentences(sentences, ratio=ratio, min_sent=3, max_sent=12)

# ------------------------------- UI ------------------------------------ #

def example_text() -> str:
    return (
        "Artificial intelligence (AI) refers to the simulation of human intelligence processes by machines, "
        "especially computer systems. Specific applications of AI include expert systems, natural language processing, "
        "speech recognition, and machine vision. AI programming focuses on three cognitive skills: learning, reasoning, "
        "and self-correction. Learning processes include acquiring data and rules for using it. Reasoning involves "
        "choosing the right algorithm to reach a desired outcome. Self-correction is designed to continually fine-tune "
        "algorithms for increased accuracy."
    )

def main():
    st.set_page_config(page_title="Text in Brief", page_icon="📝", layout="centered")
    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)

    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Summarization mode",
            ["Abstractive (AI generates new text)", "Extractive (pick key sentences)"],
            index=0
        )
        preset = st.selectbox("Summary length", list(LENGTH_PRESETS.keys()), index=1)

        legacy_t5 = st.checkbox(
            "Legacy T5 (match old output)",
            help="Adds 'summarize:' prefix, fixed min/max length (20/100), deterministic decoding."
        )

        if mode.startswith("Abstractive"):
            model_name = st.selectbox("Model", MODEL_CHOICES, index=MODEL_CHOICES.index(DEFAULT_MODEL))
            beams = st.slider("Beam search (quality vs speed)", 1, 8, 4, help="Higher = better quality, slower.")
            st.divider()
            st.markdown("**Tips**")
            st.markdown("- T5 is fastest. BART/Pegasus often give higher-quality news summaries.")
            st.markdown("- Very long inputs are chunked automatically.")
        else:
            model_name = None
            beams = 4

        st.divider()
        st.markdown("**Utilities**")
        use_sample = st.checkbox("Load example text")

    default_text = example_text() if use_sample else ""
    text = st.text_area(
        "Paste your text here",
        value=default_text,
        height=240,
        placeholder="Paste article, report, or notes…",
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        run_btn = st.button("Summarize", type="primary")
    with col_b:
        clear_btn = st.button("Clear")
    with col_c:
        st.metric("Input words", len(text.split()))

    if clear_btn:
        st.experimental_rerun()

    if run_btn:
        text = clean_text(text)
        if not text or len(text.split()) < 25:
            st.warning("Please paste at least a few sentences (≈25+ words).")
            return

        with st.spinner("Summarizing…"):
            try:
                if mode.startswith("Abstractive"):
                    if not _TRANSFORMERS_OK:
                        st.error(
                            "Abstractive mode requires `transformers`, `torch`, and `sentencepiece`.\n"
                            "Install via:  pip install -r requirements.txt"
                        )
                        return
                    summary = summarize_abstractive(
                        text,
                        model_name=model_name,
                        preset=preset,
                        beams=beams,
                        legacy_t5=legacy_t5,
                    )
                else:
                    summary = summarize_extractive(text, preset=preset)
            except Exception as e:
                st.error(f"Error during summarization: {e}")
                return

        st.subheader("Summary")
        st.write(summary)
        st.caption(f"Words: {len(summary.split())}")

        st.download_button(
            "Download summary (.txt)",
            data=summary,
            file_name="summary.txt",
            mime="text/plain",
        )

        st.code(summary, language="markdown")


if __name__ == "__main__":
    main()
