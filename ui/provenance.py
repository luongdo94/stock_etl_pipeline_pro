"""
Per-ticker data-confidence badges next to the numbers they qualify (Stock Analysis). The aggregate view stays in the
🩺 Data Health tab; here the reader sees, while reading a number, whether it is observed, kept from an earlier
extraction, contradicted by an independent source, or affected by an unexplained price jump.
"""
import pandas as pd
import streamlit as st

from core.valuation import EXPLICIT_YEARS

TERMINAL_SHARE_WARN = 0.75
_FIELD_LABELS = {"market_cap": "market cap", "revenue_ttm": "revenue (TTM)", "total_debt": "debt", "ebitda": "EBITDA",
                 "free_cashflow": "free cash flow", "trailing_eps": "EPS", "forward_eps": "forward EPS",
                 "price_to_book": "P/B", "roe": "ROE", "beta": "beta", "dividend_yield": "dividend yield"}


def _first(df: pd.DataFrame, col: str):
    if df is None or df.empty or col not in df.columns:
        return None
    v = df.iloc[0][col]
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def stale_fields(prov: dict) -> list:
    """[(label, as_of)] of company fields kept from an earlier extraction (the provider returned nothing)."""
    raw = _first(prov.get("stale"), "stale_fields")
    out = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if "@" in part:
            f, d = part.split("@", 1)
            out.append((_FIELD_LABELS.get(f, f.replace("_", " ")), d))
    return out


def payout_status(prov: dict) -> tuple:
    """(status, note) of the payout evidence (E5); (None, None) when unknown to this warehouse."""
    return _first(prov.get("payout"), "payout_status"), _first(prov.get("payout"), "payout_note")


def source_mismatches(prov: dict) -> pd.DataFrame:
    s = prov.get("sources")
    return s[s["status"] == "MISMATCH"] if s is not None and not s.empty else pd.DataFrame()


def history_jumps(prov: dict) -> pd.DataFrame:
    e = prov.get("exceptions")
    return e[e["stage"].isin(["history_audit", "price_integrity"])] if e is not None and not e.empty else pd.DataFrame()


def confidence_items(prov: dict, vin: dict = None) -> list:
    """[(severity, text)] — severity 'warn' or 'info'. Empty = nothing to flag."""
    items = []
    stale = stale_fields(prov)
    if stale:
        items.append(("warn", "kept from an earlier extraction: " + ", ".join(f"{f} ({d})" for f, d in stale)))
    st_, _ = payout_status(prov)
    if st_ and st_ not in ("OBSERVED",):
        items.append(("warn", f"payout evidence {st_}"))
    mm = source_mismatches(prov)
    for r in mm.head(3).itertuples():
        items.append(("warn", f"{r.source} disagrees on {r.check_name.replace('_', ' ')} ({r.diff_pct:+.1f}%)"))
    if not history_jumps(prov).empty:
        items.append(("warn", "unexplained price jump in the history (see the chart section)"))
    ts = (vin or {}).get("terminal_share")
    if ts is not None and ts > TERMINAL_SHARE_WARN and (vin or {}).get("model") == "dcf":
        items.append(("warn", f"{ts:.0%} of the DCF value comes after year {EXPLICIT_YEARS} (perpetuity assumption)"))
    src = prov.get("sources")
    if src is not None and not src.empty and mm.empty and (src["status"] == "MATCH").any():
        items.append(("info", "matches " + " & ".join(sorted(set(src.loc[src["status"] == "MATCH", "source"])))))
    return items


def render_confidence_line(prov: dict, vin: dict = None) -> None:
    """One compact line under the Decision Summary."""
    items = confidence_items(prov, vin)
    if not items:
        return
    warn = [t for s, t in items if s == "warn"]
    info = [t for s, t in items if s == "info"]
    color = "#f39c12" if warn else "#2ecc71"
    text = " · ".join(([f"⚠️ {t}" for t in warn]) + ([f"✓ {t}" for t in info]))
    st.markdown(f"<div style='font-size:0.78rem;color:{color};margin:-4px 0 10px 0;'><b>Data confidence:</b> {text}</div>",
                unsafe_allow_html=True)


def pit_caption(prov: dict, kind: str = "annual") -> str:
    """'Latest annual statement (period 2025-12-31) public since 2026-02-12 (SEC filing).'"""
    df = prov.get(f"{kind}_pit")
    d, a, s = _first(df, "date"), _first(df, "available_date"), _first(df, "available_source")
    if d is None or a is None:
        return ""
    how = {"SEC_FILING": "SEC filing", "EARNINGS_RELEASE": "earnings release",
           "LAG_ESTIMATE": "estimated: period end + typical reporting lag"}.get(str(s), str(s))
    return f"Latest {kind} statement (period ending {pd.Timestamp(d):%Y-%m-%d}) public since {pd.Timestamp(a):%Y-%m-%d} ({how})."
