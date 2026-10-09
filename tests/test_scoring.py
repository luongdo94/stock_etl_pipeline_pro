"""
Quality / Value / Momentum (core/scoring.py): behaviour that makes the scores trustworthy.
  - peers: margins and multiples are judged against the sector, not a universal yardstick
  - unknown inputs are excluded (never scored as zero) and thin data is pulled toward 50
  - banks are scored on ROE, not on FCF / EBITDA / leverage
  - analyst ratings and price momentum never change Quality or Value
  - red flags subtract points and are named
"""
import numpy as np
import pandas as pd
import pytest

from core import scoring as sc

BASE = dict(
    sector="Industrial Machinery", industry="Industrial Machinery", market_cap=100e9, pe_ratio=20.0,
    forward_pe=18.0, trailing_eps=5.0, ebitda=10e9, total_debt=20e9, free_cashflow=6e9, fcf_margin=12.0,
    ev_to_ebitda=12.0, gross_margin=0.45, operating_margin=0.20, roe=0.18, price_to_sales=3.0,
    price_to_book=4.0, peg_ratio=1.4, current_ratio=1.5, revenue_ttm=50e9, dividend_yield_pct=1.5,
    dividends_paid_yield_pct=1.5, net_payout_yield_pct=2.0, beta=1.0,
    target_mean_price=100.0, recommendation_key="hold",
)


def universe(n=10, subject=None, **subject_over):
    """n peers varying around BASE plus one 'SUBJ' row (overrides applied). Returns (companies, annual)."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n):
        r = dict(BASE, ticker=f"P{i}")
        r["operating_margin"] = 0.10 + 0.02 * i
        r["fcf_margin"] = 6 + 1.5 * i
        r["ev_to_ebitda"] = 8 + 1.2 * i
        r["pe_ratio"], r["forward_pe"] = 12 + 2 * i, 11 + 2 * i
        r["free_cashflow"] = r["fcf_margin"] / 100 * r["revenue_ttm"]
        rows.append(r)
    subj = dict(BASE, ticker="SUBJ", **(subject or {}), **subject_over)
    rows.append(subj)
    c = pd.DataFrame(rows)
    ann = []
    for t in c["ticker"]:
        for k, y in enumerate(range(2022, 2026)):
            rev = 40e9 * 1.05 ** k
            ann.append(dict(ticker=t, year=y, revenue=rev, net_income=rev * 0.10, total_equity=50e9))
    return c, pd.DataFrame(ann)


def score(c, ann=None, prices=None):
    return sc.score_universe(c, ann, prices)


# ── helpers ───────────────────────────────────────────────────────────────────────────────────
def test_band_interpolates_clamps_and_keeps_nan():
    spec = {"x": [0, 10, 20], "y": [0, 50, 100]}
    out = sc.band(pd.Series([-5, 5, 15, 99, np.nan]), spec)
    assert out.tolist()[:4] == [0, 25, 75, 100] and np.isnan(out.iloc[4])


def test_peer_percentile_falls_back_industry_sector_universe():
    v = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0], index=list("abcdefghij"))
    industry = pd.Series(["x"] * 3 + ["y"] * 7, index=v.index)       # x has only 3 members
    sector = pd.Series(["s"] * 10, index=v.index)
    p = sc.peer_percentile(v, industry, sector, min_size=5)
    assert p["j"] == pytest.approx((7 - 0.5) / 7 * 100)               # inside industry y (7 peers)
    assert p["a"] == pytest.approx((1 - 0.5) / 10 * 100)              # industry x too small → sector (10)
    assert sc.peer_percentile(v, industry, sector, 5, higher_is_better=False)["j"] == pytest.approx(100 - p["j"])


def test_peer_percentile_group_smaller_than_min_size_is_unknown():
    v = pd.Series([1.0, 2.0, 3.0])
    blank = pd.Series("", index=v.index)
    assert sc.peer_percentile(v, blank, blank, min_size=5).isna().all()


# ── quality ───────────────────────────────────────────────────────────────────────────────────
def test_strong_business_outscores_weak_business():
    good, ann = universe(subject=dict(roe=0.30, operating_margin=0.35, fcf_margin=25, free_cashflow=12e9,
                                      ebitda=15e9, total_debt=5e9, ev_to_ebitda=11.0))
    weak, _ = universe(subject=dict(roe=0.02, operating_margin=0.04, fcf_margin=1, free_cashflow=0.5e9,
                                    ebitda=3e9, total_debt=40e9, ev_to_ebitda=14.0))
    assert score(good, ann).at["SUBJ", "quality"] > score(weak, ann).at["SUBJ", "quality"] + 20


def test_margin_is_judged_against_peers_not_in_absolute_terms():
    """A 15% operating margin is top-of-class among 8%-margin peers, bottom-of-class among 30% peers."""
    def run(peer_margin):
        c, ann = universe(subject=dict(operating_margin=0.15))
        c.loc[c["ticker"] != "SUBJ", "operating_margin"] = peer_margin + np.linspace(-0.02, 0.02, 10)
        return sc.quality_components(sc.build_features(c, ann), sc.load_rules()).at["SUBJ", "op_margin"]
    assert run(0.08) > 90 and run(0.30) < 10


def test_unknown_inputs_are_not_scored_as_zero():
    full, ann = universe()
    gap, _ = universe(subject=dict(ebitda=np.nan, total_debt=np.nan, ev_to_ebitda=np.nan, current_ratio=np.nan,
                                   free_cashflow=np.nan, fcf_margin=np.nan, gross_margin=np.nan))
    f, g = score(full, ann).loc["SUBJ"], score(gap, ann).loc["SUBJ"]
    assert g["quality_coverage"] < f["quality_coverage"]
    assert abs(g["quality"] - f["quality"]) < 25           # not collapsed to ~0 by the gaps
    assert "net debt / EBITDA" in g["missing"]              # reported so the Decision can lower its confidence


def test_thin_data_is_pulled_toward_neutral():
    c, ann = universe(subject=dict(roe=0.40, operating_margin=0.45))
    rich = score(c, ann).at["SUBJ", "quality"]
    thin = c.copy()
    keep = ["ticker", "sector", "industry", "market_cap", "operating_margin", "gross_margin"]
    thin.loc[thin["ticker"] == "SUBJ", [k for k in thin.columns if k not in keep]] = np.nan
    t = score(thin).at["SUBJ", "quality"]
    assert abs(t - 50) < abs(rich - 50)


def test_banks_are_scored_on_roe_without_cash_flow_or_leverage_tests():
    rows = [dict(BASE, ticker=f"B{i}", sector="Banks", industry="Banks", roe=0.08 + 0.01 * i, total_debt=400e9,
                 ebitda=np.nan, free_cashflow=np.nan, fcf_margin=np.nan, ev_to_ebitda=np.nan) for i in range(8)]
    out = score(pd.DataFrame(rows))
    assert out["quality"].notna().all() and out["quality_coverage"].min() >= 60
    assert not out["flags"].str.contains("Net debt|Debt with no").any()    # leverage rules are for non-financials
    assert out.loc["B7", "quality"] > out.loc["B0", "quality"]               # higher ROE → higher quality


def test_utilities_get_relaxed_leverage_bands():
    util, ann = universe(subject=dict(sector="Regulated Utilities", industry="Regulated Utilities",
                                      ebitda=10e9, total_debt=45e9, ev_to_ebitda=14.5))
    ind, _ = universe(subject=dict(ebitda=10e9, total_debt=45e9, ev_to_ebitda=14.5))
    fu = sc.build_features(util, ann)
    fi = sc.build_features(ind, ann)
    bu = sc.quality_components(fu, sc.load_rules()).at["SUBJ", "balance_sheet"]
    bi = sc.quality_components(fi, sc.load_rules()).at["SUBJ", "balance_sheet"]
    assert bu > bi


@pytest.mark.parametrize("over, flag", [
    (dict(pe_ratio=-5.0, trailing_eps=-1.0, forward_pe=np.nan, free_cashflow=-1e9), "Loss-making"),
    (dict(ebitda=-1e9), "Debt with no positive EBITDA"),
    (dict(total_debt=90e9, ebitda=10e9, ev_to_ebitda=18.0), "Net debt/EBITDA"),
    (dict(dividends_paid_yield_pct=8.0, free_cashflow=2e9), "Dividend not covered"),
])
def test_red_flags_are_named_and_cost_points(over, flag):
    c, ann = universe(subject=over)
    clean = score(universe()[0], ann).at["SUBJ", "quality"]
    flagged = score(c, ann).loc["SUBJ"]
    assert flag in flagged["flags"]
    assert flagged["quality"] < clean


def test_penalties_are_capped():
    c, ann = universe(subject=dict(pe_ratio=-5.0, trailing_eps=-1.0, forward_pe=np.nan, ebitda=-1e9,
                                   dividends_paid_yield_pct=8.0, free_cashflow=-2e9))
    ann.loc[ann["ticker"] == "SUBJ", "total_equity"] = -10e9          # + negative equity: 8+10+5+5 = 28 > cap
    f = sc.build_features(c, ann)
    _, pts = sc.quality_penalties(f, sc.load_rules())
    assert pts["SUBJ"] == sc.load_rules()["quality"]["penalties"]["max_total"]


def test_negative_equity_is_flagged_and_has_no_return_on_capital():
    c, ann = universe()
    ann.loc[ann["ticker"] == "SUBJ", "total_equity"] = [-60e9] * 4
    f = sc.build_features(c, ann)
    assert bool(f.at["SUBJ", "negative_equity"]) and np.isnan(f.at["SUBJ", "roc"])


# ── value ─────────────────────────────────────────────────────────────────────────────────────
def test_cheap_beats_expensive_on_value():
    cheap, ann = universe(subject=dict(free_cashflow=9e9, fcf_margin=18, ev_to_ebitda=7.0, forward_pe=9.0,
                                       pe_ratio=10.0, peg_ratio=0.7, price_to_sales=1.2))
    dear, _ = universe(subject=dict(free_cashflow=1.5e9, fcf_margin=3, ev_to_ebitda=30.0, forward_pe=45.0,
                                    pe_ratio=50.0, peg_ratio=3.5, price_to_sales=15.0))
    assert score(cheap, ann).at["SUBJ", "value"] > score(dear, ann).at["SUBJ", "value"] + 40


def test_loss_makers_are_expensive_not_unknown():
    c, ann = universe(subject=dict(pe_ratio=-5.0, trailing_eps=-1.0, forward_pe=np.nan, ebitda=-1e9,
                                   ev_to_ebitda=np.nan))
    comps = sc.value_components(sc.build_features(c, ann), sc.load_rules()).loc["SUBJ"]
    assert comps["earn_yield"] < 5 and comps["ev_ebitda"] == 0      # bottom of the range, not NaN


def test_uncovered_dividend_halves_shareholder_yield_credit():
    ok, ann = universe(subject=dict(net_payout_yield_pct=6.0))
    bad, _ = universe(subject=dict(net_payout_yield_pct=6.0, dividends_paid_yield_pct=6.0, free_cashflow=1e9))
    rules = sc.load_rules()
    a = sc.value_components(sc.build_features(ok, ann), rules).at["SUBJ", "shareholder_yield"]
    b = sc.value_components(sc.build_features(bad, ann), rules).at["SUBJ", "shareholder_yield"]
    assert b == pytest.approx(a * rules["value"]["uncovered_dividend_factor"])


# ── independence of the three scores ─────────────────────────────────────────────────────────
def test_analyst_ratings_and_targets_do_not_move_any_score():
    c, ann = universe()
    base = score(c, ann)
    c2 = c.assign(target_mean_price=c["target_mean_price"] * 3, recommendation_key="strong_buy")
    pd.testing.assert_frame_equal(base.drop(columns=["components"]), score(c2, ann).drop(columns=["components"]))


def _prices(c, drift_by_ticker, days=300):
    rows = []
    for t in c["ticker"]:
        close = 100 * np.cumprod(np.full(days, 1 + drift_by_ticker.get(t, 0.0005)))
        ma200 = pd.Series(close).rolling(200, min_periods=1).mean().to_numpy()
        rows.append(pd.DataFrame({"ticker": t, "date": pd.bdate_range("2025-01-01", periods=days), "price_close": close,
                                  "ma_signal": "BULLISH" if drift_by_ticker.get(t, 0.0005) > 0 else "BEARISH",
                                  "pct_from_ma200": (close / ma200 - 1) * 100}))
    return pd.concat(rows, ignore_index=True)


def test_price_momentum_never_changes_quality_or_value():
    c, ann = universe()
    up = score(c, ann, _prices(c, {"SUBJ": 0.004}))
    down = score(c, ann, _prices(c, {"SUBJ": -0.004}))
    assert up.at["SUBJ", "momentum"] > down.at["SUBJ", "momentum"] + 40
    for col in ("quality", "value"):
        assert up.at["SUBJ", col] == down.at["SUBJ", col]


def test_momentum_skips_the_last_month_and_needs_a_year_of_history():
    c, ann = universe()
    p = _prices(c, {}, days=300)
    f = sc.build_features(c, ann, p)
    assert f["ret_12_1"].notna().all()
    short = _prices(c, {}, days=120)
    assert sc.build_features(c, ann, short)["ret_12_1"].isna().all()
    # a crash in the final 21 days does not touch 12-1 momentum
    crash = p.copy()
    last = crash[crash["ticker"] == "SUBJ"].index[-15:]
    crash.loc[last, "price_close"] *= 0.5
    assert sc.build_features(c, ann, crash).at["SUBJ", "ret_12_1"] == pytest.approx(f.at["SUBJ", "ret_12_1"])


def test_indices_and_benchmarks_are_not_scored():
    c, ann = universe()
    extra = pd.DataFrame([dict(BASE, ticker="SPY", sector="Benchmark"), dict(BASE, ticker="^VIX", sector="Index")])
    out = score(pd.concat([c, extra], ignore_index=True), ann)
    assert "SPY" not in out.index and "^VIX" not in out.index and "SUBJ" in out.index


def test_scores_are_bounded():
    c, ann = universe(subject=dict(roe=5.0, operating_margin=3.0, fcf_margin=900, ev_to_ebitda=0.5))
    out = score(c, ann)
    for col in ("quality", "value"):
        assert out[col].between(0, 100).all()


# ── look-through rules learned from real data ────────────────────────────────────────────────
def test_gaap_loss_with_positive_cash_is_a_small_flag_not_a_loss_maker():
    """An impairment-driven net loss (General Mills, Kraft Heinz, Bayer) must not read as an economic loss."""
    gaap, ann = universe(subject=dict(pe_ratio=-5.0, trailing_eps=-1.0, forward_pe=np.nan))       # EBITDA, FCF > 0
    econ, _ = universe(subject=dict(pe_ratio=-5.0, trailing_eps=-1.0, forward_pe=np.nan, free_cashflow=-1e9))
    g, e = score(gaap, ann).loc["SUBJ"], score(econ, ann).loc["SUBJ"]
    assert "GAAP loss but cash-positive" in g["flags"] and "Loss-making" not in g["flags"]
    assert "Loss-making" in e["flags"]
    assert g["quality"] > e["quality"] + 3


def test_utilities_are_not_punished_for_negative_free_cash_flow():
    kw = dict(free_cashflow=-3e9, fcf_margin=-6.0, ebitda=10e9, total_debt=45e9, ev_to_ebitda=14.5)
    util, ann = universe(subject=dict(sector="Regulated Utilities", industry="Regulated Utilities", **kw))
    ind, _ = universe(subject=kw)
    u, i = score(util, ann).loc["SUBJ"], score(ind, ann).loc["SUBJ"]
    assert u["quality"] > i["quality"] + 10
    assert "free cash flow" not in " ".join(u["missing"])          # FCF tests are not part of the utility weights


def test_utility_dividend_is_judged_on_earnings_not_free_cash_flow():
    f = sc.build_features(*universe(subject=dict(sector="Regulated Utilities", industry="Regulated Utilities",
                                                 free_cashflow=-3e9, dividends_paid_yield_pct=1.5)))
    assert not bool(f.at["SUBJ", "dividend_uncovered"])            # earnings = mcap/PE = 5bn vs dividend 1.5bn


def test_reits_are_exempt_from_gaap_loss_and_dividend_flags():
    f = sc.build_features(*universe(subject=dict(sector="REITs", industry="REIT - Residential", pe_ratio=-30.0,
                                                 trailing_eps=-1.0, forward_pe=np.nan, free_cashflow=-1e9,
                                                 dividends_paid_yield_pct=4.0)))
    assert not f.at["SUBJ", "loss_making"] and not f.at["SUBJ", "dividend_uncovered"]


def test_negative_ebitda_with_net_cash_is_not_debt_without_ebitda():
    """Snowflake-style: EBITDA < 0 but EV ~ market cap (cash covers the convertible debt)."""
    cash, ann = universe(subject=dict(ebitda=-1e9, ev_to_ebitda=-100.0, total_debt=3e9, free_cashflow=1e9))  # EV = 100bn = mcap
    lev, _ = universe(subject=dict(ebitda=-1e9, ev_to_ebitda=-130.0, total_debt=3e9, free_cashflow=1e9))     # EV = 130bn → 30bn net debt
    fc, fl = sc.build_features(cash, ann), sc.build_features(lev, ann)
    assert not fc.at["SUBJ", "debt_without_ebitda"] and fc.at["SUBJ", "net_debt_ebitda"] == 0
    assert fl.at["SUBJ", "debt_without_ebitda"] and fl.at["SUBJ", "net_debt_ebitda"] == 99


def test_nullable_integer_columns_do_not_poison_the_flags():
    """DuckDB BIGINT columns arrive as Int64 with pd.NA: a stock with no balance-sheet data was flagged 'critical'."""
    c, ann = universe()
    c = c.copy()
    for col in ("market_cap", "ebitda", "total_debt"):
        c[col] = c[col].astype("Int64")
    c.loc[c["ticker"] == "SUBJ", ["ebitda", "total_debt"]] = pd.NA
    f = sc.build_features(c, ann)
    assert np.isnan(f.at["SUBJ", "net_debt_ebitda"]) and not f.at["SUBJ", "debt_without_ebitda"]
    assert "Net debt" not in score(c, ann).at["SUBJ", "flags"]


def test_peer_group_uses_all_members_not_only_unassigned_rows():
    """Regression: a 3-member industry fell back to a sector group counted over only the 3 leftovers."""
    v = pd.Series(range(1, 11), index=list("abcdefghij"), dtype=float)
    industry = pd.Series(["x"] * 3 + ["y"] * 7, index=v.index)
    sector = pd.Series("s", index=v.index)
    assert sc.peer_percentile(v, industry, sector, 5).notna().all()
