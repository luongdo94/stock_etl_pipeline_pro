"""Grounding checks for language-model answers, and FinBERT sentiment.

Grounding contract (audit F-06): the LLM EXPLAINS the deterministic Decision (core/decision.py); it never issues its own
rating or price targets, and every number in its output is checked against the documents it was given.
"""
import re

import numpy as np
import streamlit as st

_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list:
    out = []
    for m in _NUM.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def ungrounded_numbers(output: str, brief: str) -> list:
    """Numbers in the model output that do not appear (within rounding) in the brief. Small integers (section
    numbers, "3-4 sentences") are ignored. Pure function — tested in tests/test_audit_fixes.py."""
    known = _numbers(brief)
    bad = []
    for x in _numbers(output):
        if float(x).is_integer() and abs(x) <= 10:
            continue
        if not any(abs(x - k) <= max(0.051, 0.006 * abs(k)) for k in known):
            bad.append(x)
    return sorted(set(bad))


_RATINGS = ("STRONG BUY", "BUY", "ACCUMULATE", "HOLD", "WATCH", "REDUCE", "AVOID", "TRIM", "SELL")


def stance_conflicts(text: str, stance) -> list:
    """Rating words in the narrative that are not part of the computed stance (e.g. 'BUY' when the Decision is HOLD)."""
    if not stance or stance == "N/A":
        return []
    allowed = set(re.findall(r"[A-Z]+", str(stance).upper()))
    found = set()
    for w in _RATINGS:
        if re.search(rf"(?<![A-Za-z]){w}(?![A-Za-z])", text or ""):     # upper-case rating words only
            if not set(w.split()) <= allowed:
                found.add(w)
    return sorted(found)


@st.cache_resource(show_spinner="Loading Institutional NLP Engine (FinBERT ~440MB)...")
def get_finbert_pipeline():
    """Loads the ProsusAI/finbert model for financial-specific sentiment analysis."""
    import os
    import warnings
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from transformers import pipeline
        try:
            return pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None


def analyze_sentiment_finbert(headlines):
    """Batch processes headlines using FinBERT and returns an average score (-1 to 1)."""
    pipe = get_finbert_pipeline()
    if not pipe or not headlines:
        return None
    
    results = pipe(headlines)
    scores = []
    for res in results:
        label = res['label'].lower()
        score = res['score']
        # Map: positive -> +score, negative -> -score, neutral -> 0
        if label == 'positive':
            scores.append(score)
        elif label == 'negative':
            scores.append(-score)
        else:
            scores.append(0)
    return np.mean(scores) if scores else None
