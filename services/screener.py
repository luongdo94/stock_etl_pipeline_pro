"""Cached Streamlit wrapper around core.screener.build_screener_table."""
import streamlit as st

from core.screener import build_screener_table


@st.cache_data(ttl=3600)
def get_master_screener_data(_companies_df, _prices_df, _quarterly_fin, _annual_fin, _hist_fcf=None,
                             risk_free_pct=None, hist_fcf_rows=None, _estimates=None, estimates_rows=None,
                             track_record_ok=False):
    """`hist_fcf_rows` / `estimates_rows` only feed the cache key (DataFrames prefixed with _ are not hashed)."""
    macro = {"US10Y": {"val": risk_free_pct}} if risk_free_pct else {}
    return build_screener_table(_companies_df, _prices_df, _quarterly_fin, _annual_fin, _hist_fcf, macro, _estimates,
                                  track_record_ok=track_record_ok)
