"""Market Scanner presets — pure filters over the screener table (no Streamlit).

Kept deliberately short: each preset answers a distinct question. Presets that overlapped another one
almost entirely, returned (nearly) nothing on the real universe, or rested only on the volume-flow
heuristic were removed; RSI / Z-Score / Smart Money remain available as Custom Refinement sliders.
"""
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from core.rating import QUALITY_TIERS

ELITE, SOLID, FAIR = (t[0] for t in QUALITY_TIERS)      # Quality tier cut-offs (core/rating.py)
UPTREND = {"STRONG BULL", "BULLISH"}       # golden cross (MA50 > MA200)
DOWNTREND = {"STRONG BEAR", "BEARISH"}     # death cross (MA50 < MA200)

ALL = "🔍 All Stock Universe"


@dataclass(frozen=True)
class Preset:
    label: str
    group: str                                  # "opportunity" | "risk"
    rule: Callable[[pd.DataFrame], pd.Series]
    note: str
    level: str = "success"                      # st.success / st.info / st.warning / st.error


def _up(d):
    return d["Trend"].isin(UPTREND)


def _down(d):
    return d["Trend"].isin(DOWNTREND)


def _flags(d):
    return d["Flags"].fillna("")


PRESETS = [
    # ── opportunity ────────────────────────────────────────────────────────────────────────
    Preset(f"🏆 Institutional Pulse (Quality ≥ {ELITE} & Uptrend)", "opportunity",
           lambda d: (d["Quality"] >= ELITE) & _up(d),
           f"Quality ≥ {ELITE} (ELITE tier) in an uptrend (MA50 > MA200)."),
    Preset(f"💎 Quality at a Fair Price (Quality ≥ {ELITE} & Value ≥ 50)", "opportunity",
           lambda d: (d["Quality"] >= ELITE) & (d["Value"] >= 50),
           f"Quality ≥ {ELITE} (ELITE) and Value ≥ 50 vs sector peers — a strong business that is not expensive. "
           "Momentum is deliberately not required."),
    Preset(f"🏷️ Deep Value (Value ≥ 70 & Quality ≥ {SOLID})", "opportunity",
           lambda d: (d["Value"] >= 70) & (d["Quality"] >= SOLID),
           f"Value ≥ 70 (cheap on FCF yield / EV-EBITDA / earnings yield vs peers) with Quality ≥ {SOLID} — cheap but not broken."),
    Preset(f"📈 Rising Estimates (Revisions ≥ 65 & Quality ≥ {SOLID})", "opportunity",
           lambda d: (d["Revisions"] >= 65) & (d["Quality"] >= SOLID),
           f"Analysts raised EPS estimates over the last 30 days (Revisions ≥ 65) for a business with Quality ≥ {SOLID}. "
           "Revisions are a flow with documented drift; the level of the consensus is not used."),
    Preset(f"🌱 GARP (PEG < 1.0 & Quality ≥ {SOLID})", "opportunity",
           lambda d: (d["PEG"] > 0) & (d["PEG"] < 1.0) & (d["Quality"] >= SOLID),
           f"Growth at a reasonable price: PEG below 1.0 (P/E lower than the growth rate) for a business with Quality ≥ {SOLID}."),
    Preset("⚙️ Both Accelerating (EPS + Revenue QoQ, 2 qtrs > +10%)", "opportunity",
           lambda d: (d["EPS Momentum"] == "Accelerating") & (d["Rev Momentum"] == "Accelerating"),
           "EPS and revenue both grew more than 10% quarter-on-quarter for two consecutive quarters."),
    Preset("🚀 Buy on Dip (Uptrend + RSI < 40)", "opportunity",
           lambda d: _up(d) & (d["RSI (14)"] < 40),
           "Uptrend (MA50 > MA200) with RSI cooling below 40: a pullback inside a rising trend.", "info"),
    Preset("⚡ Strong Breakout (Uptrend + Momentum ≥ 70, RSI 50-70)", "opportunity",
           lambda d: _up(d) & (d["vs MA200 (%)"] > 5) & d["RSI (14)"].between(50, 70) & (d["Momentum"] >= 70),
           "Uptrend, price more than 5% above MA200, top-30% Momentum score and RSI 50-70 (strong but not yet overbought)."),
    Preset(f"💰 Quality Dividend (Yield > 2.5%, Quality ≥ {SOLID}, covered)", "opportunity",
           lambda d: (d["Yield (%)"] > 2.5) & (d["Quality"] >= SOLID)
                     & ~_flags(d).str.contains("Dividend not covered") & ~(d["Debt/EBITDA"] >= 3),
           f"Yield above 2.5%, Quality ≥ {SOLID}, the dividend is covered by free cash flow and net debt is below 3x EBITDA "
           "(banks and insurers, which have no EBITDA, are kept). No trend requirement: this is an income screen."),
    # ── risk / warning ─────────────────────────────────────────────────────────────────────
    Preset(f"🪤 Value Trap Risk (Value ≥ 65 & Quality < {FAIR})", "risk",
           lambda d: (d["Value"] >= 65) & (d["Quality"] < FAIR),
           f"Looks cheap (Value ≥ 65) but Quality < {FAIR}. Cheap and weak is the classic value trap — "
           "the Decision will not call these a BUY.", "error"),
    Preset("🚩 Red Flags (any quality penalty)", "risk",
           lambda d: _flags(d) != "",
           "Loss-making, debt without EBITDA, high net debt/EBITDA, dividend not covered by FCF, negative book equity. "
           "See the Flags column.", "error"),
    Preset(f"📉 Downtrend (MA50 < MA200 and RSI < 50 or Quality < {FAIR})", "risk",
           lambda d: _down(d) & ((d["RSI (14)"] < 50) | (d["Quality"] < FAIR)),
           f"Death cross (MA50 < MA200) that is either still falling (RSI < 50) or a weak business (Quality < {FAIR}). "
           "Avoid catching these too early.", "error"),
    Preset("⚠️ Earnings Deterioration (EPS QoQ, 2 qtrs < -10%)", "risk",
           lambda d: (d["EPS Momentum"] == "Decelerating") & (d["Rev Momentum"] != "Accelerating"),
           "EPS fell more than 10% quarter-on-quarter for two consecutive quarters and revenue is not accelerating.", "error"),
    Preset("🎈 Overextended (RSI > 70 & Z-Score > +2)", "risk",
           lambda d: (d["RSI (14)"] > 70) & (d["Z-Score"] > 2.0),
           "Overbought (RSI > 70) and more than 2 std dev above the 5-year mean price. A statistic about price, "
           "not a valuation — elevated pullback risk, not a sell signal.", "warning"),
]

_BY_LABEL = {p.label: p for p in PRESETS}
SEPARATORS = {"opportunity": "──────────── 📈 OPPORTUNITY ────────────",
              "risk": "──────────── ⛔ RISK / WARNING ────────────"}


def options() -> list:
    """Selectbox options: All, then each group under its separator."""
    out = [ALL]
    for group in ("opportunity", "risk"):
        out.append(SEPARATORS[group])
        out += [p.label for p in PRESETS if p.group == group]
    return out


def get(label: str):
    return _BY_LABEL.get(label)


def apply(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Rows of `df` matching the preset; All / a separator / an unknown label return `df` unchanged.
    Unknown values (NaN) never match."""
    p = get(label)
    if p is None:
        return df
    return df[p.rule(df).fillna(False).astype(bool)]
