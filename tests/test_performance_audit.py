"""PERFORMANCE_SCALABILITY_AUDIT fixes (2026-10-16): watermark buckets (PERF-02), config caches (PERF-05/15) and the
array pivot test of the swing zones (PERF-04) — speed changes that must not change any result."""
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core import levels


# ── PERF-02: one stale ticker no longer widens every download ─────────────────────────────────────────────────────
def test_watermark_buckets_keep_each_ticker_overlap_but_stop_widening():
    from etl.extract import watermark_buckets
    wm = {"A": date(2026, 10, 15), "B": date(2026, 10, 14), "C": date(2026, 10, 9), "EA": date(2026, 8, 4)}
    got = watermark_buckets(wm, overlap_days=2)
    assert got == [(datetime(2026, 8, 2), ["EA"]), (datetime(2026, 10, 7), ["C", "B", "A"])]
    for start, ts in got:                                   # every ticker starts ≥ 2 days before its own watermark
        assert all(start <= datetime.combine(wm[t], datetime.min.time()) - timedelta(days=2) for t in ts)


def test_incremental_extract_downloads_each_bucket_from_its_own_start(monkeypatch):
    from etl import extract
    calls = []
    monkeypatch.setattr(extract.yf, "download", lambda batch, start, **k: calls.append((sorted(batch), start)) or pd.DataFrame())

    class _T:
        def __init__(self, t):
            pass

        def history(self, **k):
            return pd.DataFrame()
    monkeypatch.setattr(extract.yf, "Ticker", _T)
    tickers = {"A": {}, "B": {}, "EA": {}}
    wm = {"A": date(2026, 10, 15), "B": date(2026, 10, 14), "EA": date(2026, 8, 4), "GONE": date(2020, 1, 1)}
    out = extract.extract_stock_prices(tickers=tickers, watermarks=wm)
    assert out.empty
    starts = {tuple(b): s.date() for b, s in calls}
    od = extract._overlap_days()
    assert starts == {("EA",): date(2026, 8, 4) - timedelta(days=od),                # a ticker outside the universe
                      ("A", "B"): date(2026, 10, 14) - timedelta(days=od)}          # (GONE) no longer counts


# ── PERF-05 / PERF-15: parsed once per file version, re-read after an edit ────────────────────────────────────────
def test_holding_horizon_uses_the_versioned_config_cache(monkeypatch):
    import yaml
    from core import valuation as v
    v._CFG_CACHE.clear()
    n = {"parse": 0}
    real = yaml.safe_load
    monkeypatch.setattr(yaml, "safe_load", lambda *a, **k: n.__setitem__("parse", n["parse"] + 1) or real(*a, **k))
    hs = {v.holding_horizon() for _ in range(50)}
    assert hs == {int(v._valuation_config().get("horizon_years", 4))} and n["parse"] <= 1


def test_scoring_rules_follow_an_edit_in_the_same_process(tmp_path, monkeypatch):
    from core import scoring as sc
    p = tmp_path / "scoring_rules.yaml"
    p.write_text("version: one\n", encoding="utf-8")
    monkeypatch.setattr(sc, "_RULES_PATH", p)
    assert sc.load_rules() == {"version": "one"}
    assert sc.load_rules() is sc.load_rules()                    # cached while the file is unchanged
    p.write_text("version: two, edited\n", encoding="utf-8")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # a new modification time, as an editor writes
    assert sc.load_rules() == {"version": "two, edited"}


# ── PERF-04: array pivot test == the old per-bar pandas loop ─────────────────────────────────────────────────────
def _reference_masks(df, half):
    n = len(df)
    lo, hi = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(half, n - half):
        w = df.iloc[i - half: i + half + 1]
        lo[i] = df.iloc[i]["price_low"] == w["price_low"].min()
        hi[i] = df.iloc[i]["price_high"] == w["price_high"].max()
    return lo, hi


@pytest.mark.parametrize("seed", range(6))
def test_pivot_masks_match_the_per_bar_loop_including_ties_and_nans(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 300))
    close = 100 + np.round(np.cumsum(rng.normal(0, 1, n)), 0)        # rounded → many exact ties
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n), "price_close": close,
                       "price_low": close - rng.integers(0, 3, n), "price_high": close + rng.integers(0, 3, n),
                       "volume": rng.integers(1, 1e6, n)})
    df.loc[rng.random(n) < 0.05, "price_low"] = np.nan
    df.loc[rng.random(n) < 0.05, "price_high"] = np.nan
    if seed % 2:
        df = df.astype({"price_low": "float32", "price_high": "float32"})
    for half in (1, 2, 3, 4, 5):
        got, ref = levels._pivot_masks(df, half), _reference_masks(df, half)
        assert (got[0] == ref[0]).all() and (got[1] == ref[1]).all()



# ── PERF-05 / PERF-03: prepared issuer inputs and per-ticker partitions give the same valuation ──────────────────
def _hist(ticker, n=5, base=1e9):
    return pd.DataFrame({"ticker": ticker, "year": list(range(2021, 2021 + n)), "free_cash_flow": [base * (1 + .05 * i) for i in range(n)],
                         "stock_based_comp": [-1e7] * n, "interest_paid_cff": [-2e7] * n,
                         "lease_liability_current": [3e7] * n, "lease_liability_total": [9e7] * n})


def _meta(ticker):
    return pd.Series({"ticker": ticker, "sector": "Industrials", "industry": "Machinery", "currency": "EUR",
                      "country": "Germany", "market_cap": 2e10, "beta": 1.0, "revenue_growth": .04,
                      "earnings_growth": .05, "free_cashflow": 9e8})


def test_valuation_memo_returns_exactly_the_unmemoised_result(monkeypatch):
    from core import valuation as v
    hist = pd.concat([_hist("A.DE"), _hist("B.DE", base=2e9)], ignore_index=True)
    memoised = v.valuation_inputs(_meta("A.DE"), 40.0, hist, "A.DE", {})
    monkeypatch.setattr(v, "_memo", lambda key, build: build())                  # every helper recomputes
    plain = v.valuation_inputs(_meta("A.DE"), 40.0, hist, "A.DE", {})
    assert repr(memoised) == repr(plain)
    assert v._MEMO.get() is None                                                    # the memo ends with the call


def test_valuation_memo_hands_out_independent_copies():
    from core import valuation as v
    hist = _hist("A.DE")
    token = v._MEMO.set({})
    try:
        first = v.basis_rows(hist, "A.DE")
        first.loc[first.index[0], "free_cash_flow"] = -1.0                          # a caller edits its copy
        st = v.fcfe_deduction_status(hist, "A.DE")
        st["stock_based_comp"]["status"] = "EDITED"
        assert v.basis_rows(hist, "A.DE")["free_cash_flow"].min() > 0
        assert v.fcfe_deduction_status(hist, "A.DE")["stock_based_comp"]["status"] != "EDITED"
    finally:
        v._MEMO.reset(token)


def test_issuer_partition_matches_the_full_frame_mask():
    from core.screener import _by_ticker
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"ticker": rng.choice(["A", "B", "C"], 200), "date": rng.integers(0, 50, 200), "x": rng.random(200)})
    parts = _by_ticker(df)
    for t in ("A", "B", "C"):
        assert parts[t].equals(df[df["ticker"] == t]) and list(parts[t].index) == list(df.index[df["ticker"] == t])
    assert _by_ticker(None) == {} and _by_ticker(df.iloc[:0]) == {}
