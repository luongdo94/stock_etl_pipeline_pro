"""Cached Streamlit wrapper around core.screener.build_screener_table.

Cache key (audit A04): DataFrames prefixed with _ are not hashed by Streamlit, so the key used to be a few row counts
and a changed price or statement with the same row count served stale scores for up to an hour. The wrapper now
computes a content fingerprint of every input frame plus the score / rule versions and passes it as a hashed argument.
"""
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from core.screener import build_screener_table

_CONFIG = Path(__file__).resolve().parent.parent / "config"


def _frame_digest(df) -> str:
    if df is None:
        return "none"
    if not isinstance(df, pd.DataFrame):
        return hashlib.sha1(repr(df).encode()).hexdigest()
    try:
        h = pd.util.hash_pandas_object(df, index=False).to_numpy()
        return f"{df.shape}:{hashlib.sha1(h.tobytes()).hexdigest()}:{','.join(map(str, df.columns))}"
    except TypeError:          # unhashable cells (lists / dicts): fall back to a JSON digest
        return f"{df.shape}:{hashlib.sha1(df.to_json(default_handler=str).encode()).hexdigest()}"


def model_version() -> str:
    """Score definition + every rule file: a change in either must invalidate cached decisions."""
    from core.scoring import SCORE_VERSION
    parts = [SCORE_VERSION]
    for name in ("scoring_rules.yaml", "decision_rules.yaml"):
        try:
            parts.append(hashlib.sha1((_CONFIG / name).read_bytes()).hexdigest()[:12])
        except OSError:
            parts.append("missing")
    return "|".join(parts)


def frame_fingerprint(*frames) -> str:
    return hashlib.sha1("|".join([model_version()] + [_frame_digest(f) for f in frames]).encode()).hexdigest()


@st.cache_data(ttl=3600)
def _cached_screener(_companies_df, _prices_df, _quarterly_fin, _annual_fin, _hist_fcf, risk_free_pct, _estimates,
                     track_record_ok, data_key: str, _earnings_cal=None, today: str = "", _sec_annual_history=None):
    macro = {"US10Y": {"val": risk_free_pct}} if risk_free_pct else {}
    return build_screener_table(_companies_df, _prices_df, _quarterly_fin, _annual_fin, _hist_fcf, macro, _estimates,
                                track_record_ok=track_record_ok, _earnings_cal=_earnings_cal, _sec_annual_history=_sec_annual_history)


def get_master_screener_data(_companies_df, _prices_df, _quarterly_fin, _annual_fin, _hist_fcf=None,
                             risk_free_pct=None, hist_fcf_rows=None, _estimates=None, estimates_rows=None,
                             track_record_ok=False, _earnings_cal=None, _sec_annual_history=None):
    """Same signature as before (hist_fcf_rows / estimates_rows are accepted and ignored). The earnings calendar feeds
    the Decision's pre-earnings blackout; `today` is part of the key because that gate depends on the date."""
    from datetime import date
    key = frame_fingerprint(_companies_df, _prices_df, _quarterly_fin, _annual_fin, _hist_fcf, _estimates, _earnings_cal, _sec_annual_history)
    return _cached_screener(_companies_df, _prices_df, _quarterly_fin, _annual_fin, _hist_fcf, risk_free_pct,
                            _estimates, bool(track_record_ok), key, _earnings_cal=_earnings_cal,
                            today=date.today().isoformat(), _sec_annual_history=_sec_annual_history)


get_master_screener_data.clear = _cached_screener.clear
