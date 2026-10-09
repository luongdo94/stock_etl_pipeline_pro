"""Market Scanner presets: definitions, NaN handling and the selectbox options."""
import numpy as np
import pandas as pd

from core import scan_presets as sp


def row(**kw):
    base = {"Ticker": "X", "Quality": 50.0, "Value": 50.0, "Momentum": 50.0, "Revisions": 50.0, "Trend": "NEUTRAL",
            "RSI (14)": 50.0, "Z-Score": 0.0, "vs MA200 (%)": 0.0, "PEG": np.nan, "Yield (%)": 0.0, "Debt/EBITDA": 1.0,
            "Flags": "", "EPS Momentum": "Neutral", "Rev Momentum": "Neutral"}
    base.update(kw)
    return base


def hits(label, *rows):
    df = pd.DataFrame([dict(r, Ticker=f"T{i}") for i, r in enumerate(rows)])
    return list(sp.apply(df, label)["Ticker"])


def label(prefix):
    return next(p.label for p in sp.PRESETS if p.label.startswith(prefix))


def test_options_have_all_both_groups_and_no_duplicates():
    opts = sp.options()
    assert opts[0] == sp.ALL and len(opts) == len(set(opts)) == len(sp.PRESETS) + 3
    assert opts.index(sp.SEPARATORS["opportunity"]) < opts.index(sp.SEPARATORS["risk"])
    assert all(p.group in ("opportunity", "risk") and p.level in ("success", "info", "warning", "error") for p in sp.PRESETS)


def test_all_separator_and_unknown_label_leave_the_table_unchanged():
    for lab in (sp.ALL, sp.SEPARATORS["risk"], "🔥 Short Squeeze Watch (removed)"):
        assert hits(lab, row(), row()) == ["T0", "T1"]


def test_institutional_pulse_counts_both_bull_states():
    assert hits(label("🏆"), row(Quality=80, Trend="STRONG BULL"), row(Quality=80, Trend="BULLISH"),
                row(Quality=80, Trend="BEARISH"), row(Quality=70, Trend="BULLISH")) == ["T0", "T1"]


def test_garp_needs_a_positive_peg_below_one_and_solid_quality():
    assert hits(label("🌱"), row(PEG=0.8, Quality=65), row(PEG=-0.5, Quality=65), row(PEG=1.2, Quality=65),
                row(PEG=0.8, Quality=55), row(PEG=np.nan, Quality=90)) == ["T0"]


def test_dividend_requires_cover_and_keeps_unknown_leverage():
    assert hits(label("💰"), row(**{"Yield (%)": 4, "Quality": 65}),
                row(**{"Yield (%)": 4, "Quality": 65, "Flags": "Dividend not covered by FCF"}),
                row(**{"Yield (%)": 4, "Quality": 65, "Debt/EBITDA": 4.0}),
                row(**{"Yield (%)": 4, "Quality": 65, "Debt/EBITDA": np.nan}),       # bank: no EBITDA
                row(**{"Yield (%)": 2, "Quality": 65})) == ["T0", "T3"]


def test_strong_breakout_is_narrower_than_any_uptrend():
    base = {"Trend": "BULLISH", "vs MA200 (%)": 10, "RSI (14)": 60}
    assert hits(label("⚡"), row(**base, Momentum=80), row(**base, Momentum=60),
                row(**dict(base, **{"RSI (14)": 75}), Momentum=80)) == ["T0"]


def test_downtrend_merges_falling_and_weak():
    assert hits(label("📉"), row(Trend="STRONG BEAR", **{"RSI (14)": 40}), row(Trend="BEARISH", Quality=30, **{"RSI (14)": 60}),
                row(Trend="BEARISH", Quality=70, **{"RSI (14)": 60}), row(Trend="BULLISH", **{"RSI (14)": 30})) == ["T0", "T1"]


def test_earnings_deterioration_no_longer_requires_revenue_to_fall_too():
    assert hits(label("⚠️ Earnings"), row(**{"EPS Momentum": "Decelerating"}),
                row(**{"EPS Momentum": "Decelerating", "Rev Momentum": "Decelerating"}),
                row(**{"EPS Momentum": "Decelerating", "Rev Momentum": "Accelerating"})) == ["T0", "T1"]


def test_overextended_needs_both_rsi_and_z_score():
    assert hits(label("🎈"), row(**{"RSI (14)": 75, "Z-Score": 2.5}), row(**{"RSI (14)": 75, "Z-Score": 1.0}),
                row(**{"RSI (14)": 60, "Z-Score": 3.0})) == ["T0"]


def test_value_trap_and_red_flags():
    assert hits(label("🪤"), row(Value=70, Quality=40), row(Value=70, Quality=60), row(Value=np.nan, Quality=10)) == ["T0"]
    assert hits(label("🚩"), row(Flags="Loss-making"), row(Flags=np.nan), row()) == ["T0"]

def test_scanner_table_has_a_slim_fixed_default_and_unique_columns():
    from views import scanner
    assert len(scanner.DEFAULT_COLS) == 12
    assert len(set(scanner.DEFAULT_COLS + scanner.EXTRA_COLS)) == len(scanner.DEFAULT_COLS + scanner.EXTRA_COLS)
    assert "Upside (%)" not in scanner.DEFAULT_COLS + scanner.EXTRA_COLS           # analyst target: not used by Value / Decision
    assert {"Decision", "MoS (%)", "Quality", "Value", "Momentum", "Flags"} <= set(scanner.DEFAULT_COLS)
