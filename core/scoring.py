"""
core/scoring.py — three independent scores per stock (pure pandas, no Streamlit).

    QUALITY  how good is the business?  returns on capital, margins, stability, balance sheet, cash conversion
    VALUE    how cheap is the price?    FCF yield, EV/EBITDA, earnings yield, PEG, shareholder yield
    MOMENTUM what has price been doing? 12-1 month return rank + trend  (timing only)

Design rules (thresholds live in config/scoring_rules.yaml):
  * Scores are computed for the whole universe at once, because margins and multiples only mean
    something relative to peers: percentile within industry -> sector -> universe (min peer size).
  * Absolute bands are blended in for value (a whole sector can be expensive) and used alone where an
    absolute standard exists (return on capital, leverage, cash conversion).
  * An unknown input is excluded and the remaining weights renormalised - never scored as 0 or as the
    worst case. With little observable weight the score is shrunk toward 50 (little evidence = neutral).
  * Momentum never leaks into Quality or Value; analyst ratings are not scored at all.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCORE_VERSION = "v5"   # bump when the meaning of a score changes; snapshots are stamped with it
_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "scoring_rules.yaml"
_NON_EQUITY_SECTORS = {"benchmark", "volatility", "index"}
_UPTREND = {"STRONG BULL", "BULLISH"}

COMPONENT_LABELS = {
    "roc": "return on capital", "op_margin": "operating margin", "gross_margin": "gross margin",
    "fcf_margin": "FCF margin", "growth_stability": "multi-year growth history",
    "balance_sheet": "net debt / EBITDA", "earnings_quality": "FCF conversion", "roe": "ROE",
    "net_margin": "net margin", "fcf_yield": "FCF yield", "ev_ebitda": "EV/EBITDA",
    "earn_yield": "earnings yield (P/E)", "peg": "PEG", "shareholder_yield": "shareholder yield",
    "ps": "P/S", "pb": "P/B",
}
FLAG_LABELS = {
    "loss_making": "Loss-making", "debt_without_ebitda": "Debt with no positive EBITDA",
    "net_debt_ebitda_high": "Net debt/EBITDA high", "net_debt_ebitda_critical": "Net debt/EBITDA critical",
    "dividend_uncovered": "Dividend not covered by free cash flow", "negative_equity": "Negative book equity",
    "gaap_loss_cash_positive": "GAAP loss but cash-positive (impairment?)",
}


@lru_cache(maxsize=1)
def load_rules() -> dict:
    with open(_RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── helpers ───────────────────────────────────────────────────────────────────────────────────
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        # float64 + NaN: DuckDB's nullable Int64 columns carry pd.NA, which poisons comparisons and masks
        return pd.to_numeric(df[name], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=df.index, dtype=float)


def band(x, spec: dict) -> pd.Series:
    """Piecewise-linear points for x (NaN stays NaN; values outside the range clamp to the end points)."""
    x = pd.to_numeric(x, errors="coerce").astype("float64")
    out = np.interp(x.fillna(spec["x"][0]).to_numpy(float), spec["x"], spec["y"])
    return pd.Series(out, index=x.index).where(x.notna())


def peer_percentile(values, industry: pd.Series, sector: pd.Series, min_size: int,
                    higher_is_better: bool = True) -> pd.Series:
    """0-100 mid-rank percentile within the smallest peer group (industry, sector, universe) with >= min_size."""
    v = pd.to_numeric(values, errors="coerce").astype("float64")
    out = pd.Series(np.nan, index=v.index, dtype=float)
    levels = (industry, sector, pd.Series("universe", index=v.index))
    for key in levels:
        k = key.fillna("").astype(str)
        valid = v.notna() & (k != "")
        if not (valid & out.isna()).any():
            continue
        # group statistics use EVERY member with a value; only still-unassigned rows take the result
        g = v[valid].groupby(k[valid])
        n, r = g.transform("count"), g.rank(method="average")
        pct = (r - 0.5) / n * 100
        ok = (n >= min_size) & out.reindex(pct.index).isna()
        out.loc[pct.index[ok]] = pct[ok]
    return out if higher_is_better else 100 - out


def _blend(a: pd.Series, b: pd.Series, w_a: float) -> pd.Series:
    """w_a*a + (1-w_a)*b; falls back to whichever side is known."""
    both = w_a * a + (1 - w_a) * b
    return both.where(a.notna() & b.notna(), a.fillna(b))


def _weighted(comps: pd.DataFrame, weights: dict):
    cols = [c for c in weights if c in comps.columns]
    w = pd.Series({c: weights[c] for c in cols}, dtype=float)
    known = comps[cols].notna()
    wsum = (known * w).sum(axis=1)
    raw = (comps[cols].fillna(0) * w).sum(axis=1) / wsum.where(wsum > 0)
    return raw, wsum / sum(weights.values())


def _by_group(comps: pd.DataFrame, tables: dict, group: pd.Series):
    """Weighted score + observable-weight share, each row using the weight table of its group."""
    raw = pd.Series(np.nan, index=comps.index, dtype=float)
    cov = pd.Series(np.nan, index=comps.index, dtype=float)
    for g, weights in tables.items():
        r, c = _weighted(comps, weights)
        m = group == g
        raw[m], cov[m] = r[m], c[m]
    return raw, cov


def _shrink(raw: pd.Series, coverage: pd.Series, full_at: float) -> pd.Series:
    return 50 + (raw - 50) * np.minimum(1.0, coverage / full_at)


# ── features ──────────────────────────────────────────────────────────────────────────────────
def _history_features(annual: pd.DataFrame) -> pd.DataFrame:
    a = annual.sort_values(["ticker", "year"])

    def agg(df):
        df = df.tail(5)
        rev, ni = df["revenue"].to_numpy(float), df["net_income"].to_numpy(float)
        out = {"rev_cagr": np.nan, "profit_years": np.nan, "margin_vol": np.nan, "ni_median": np.nan}
        n = len(df)
        if n >= 3 and rev[0] > 0 and rev[-1] > 0:
            out["rev_cagr"] = ((rev[-1] / rev[0]) ** (1 / (n - 1)) - 1) * 100
        ok = ~np.isnan(ni)
        if ok.sum() >= 2:
            out["ni_median"] = float(np.median(ni[ok][-3:]))      # through-the-cycle: one impairment year is not the story
        if ok.sum() >= 3:
            out["profit_years"] = float((ni[ok] > 0).mean())
        m_ok = ok & (rev > 0)
        if m_ok.sum() >= 3:
            out["margin_vol"] = float(np.std(ni[m_ok] / rev[m_ok]) * 100)
        return pd.Series(out)

    hist = a.groupby("ticker")[["revenue", "net_income"]].apply(agg)
    last = a.groupby("ticker").tail(1).set_index("ticker")
    hist["last_equity"] = last["total_equity"]
    return hist


def _momentum_features(prices: pd.DataFrame, rules: dict) -> pd.DataFrame:
    m = rules["momentum"]
    p = prices.sort_values(["ticker", "date"])
    need = m["lookback_days"] + 1

    def ret(s):
        c = s.to_numpy(float)
        return c[-1 - m["skip_days"]] / c[-need] - 1 if len(c) >= need else np.nan

    out = pd.DataFrame({"ret_12_1": p.groupby("ticker")["price_close"].apply(ret) * 100})
    last = p.groupby("ticker").tail(1).set_index("ticker")
    out["pct_from_ma200"] = _col(last, "pct_from_ma200")
    out["ma_signal"] = last["ma_signal"].astype(str) if "ma_signal" in last.columns else np.nan
    return out


def build_features(companies: pd.DataFrame, annual_fin: pd.DataFrame = None, prices: pd.DataFrame = None,
                   rules: dict = None) -> pd.DataFrame:
    """Derived inputs per investable ticker (index = ticker). Amounts are EUR, ratios currency-free."""
    rules = rules or load_rules()
    c = companies.drop_duplicates("ticker", keep="last").set_index("ticker")
    sector = c["sector"].fillna("").astype(str).str.strip().str.lower() if "sector" in c else pd.Series("", index=c.index)
    caret = pd.Series(c.index.astype(str), index=c.index).str.startswith("^")
    keep = ~sector.isin(_NON_EQUITY_SECTORS) & ~caret
    c, sector = c[keep], sector[keep]
    industry = (c["industry"].fillna("").astype(str).str.strip().str.lower()
                if "industry" in c else pd.Series("", index=c.index))
    f = pd.DataFrame(index=c.index)
    f["sector"], f["industry"] = sector, industry
    grp = rules["sector_groups"]
    in_group = lambda name: sector.isin(set(grp[name])) | industry.isin(set(grp[name]))   # noqa: E731
    f["is_financial"], f["is_relaxed"] = in_group("financial"), in_group("leveraged_ok")
    f["is_reit"] = in_group("reit")
    f["group"] = np.where(f["is_financial"], "financial", np.where(in_group("capex_heavy"), "capex_heavy", "standard"))

    mcap = _col(c, "market_cap").where(lambda s: s > 0)
    pe, fpe, eps = _col(c, "pe_ratio"), _col(c, "forward_pe"), _col(c, "trailing_eps")
    ebitda, debt, fcf = _col(c, "ebitda"), _col(c, "total_debt"), _col(c, "free_cashflow")
    ev = _col(c, "ev_to_ebitda").where(lambda s: s > 0)

    gaap_loss = (pe < 0) | (eps < 0)
    # positive free cash flow means the business earns cash even when GAAP / EBITDA (stock comp, impairments) is
    # negative; with FCF unknown, positive EBITDA stands in. Unknown on both = not ok (conservative).
    cash_ok = (fcf > 0) | (fcf.isna() & (ebitda > 0))
    f["loss_making"] = gaap_loss & ~cash_ok & ~f["is_reit"]
    f["gaap_loss_cash_positive"] = gaap_loss & cash_ok & ~f["is_reit"]
    f["gaap_loss"] = gaap_loss
    earn_ttm = (mcap / pe).where(pe > 0)
    # earnings yield (%): forward P/E preferred; negative earnings = 0 (a fact, not a gap)
    base = fpe.where(fpe.notna(), pe)
    ey = (100 / base).where(base > 0, 0.0).where(base.notna())
    f["earn_yield"] = ey.where(ey.notna() | ~f["gaap_loss"], 0.0)
    f["fcf_yield"] = fcf / mcap * 100
    f["fcf_margin"] = _col(c, "fcf_margin")
    f["ev_ebitda"] = ev.where(ebitda > 0)
    f["ebitda_pos"] = ebitda.gt(0).where(ebitda.notna())          # NaN = unknown
    f["peg"] = _col(c, "peg_ratio").where(lambda s: s > 0)
    f["ps"] = _col(c, "price_to_sales").where(lambda s: s > 0)
    f["pb"] = _col(c, "price_to_book").where(lambda s: s > 0)
    f["op_margin"], f["gross_margin"] = _col(c, "operating_margin"), _col(c, "gross_margin")
    f["roe"] = _col(c, "roe") * 100
    f["net_margin"] = (earn_ttm / _col(c, "revenue_ttm").where(lambda s: s > 0) * 100)
    f["current_ratio"] = _col(c, "current_ratio")
    f["fcf_conversion"] = (fcf / earn_ttm).where(earn_ttm > 0)

    # Net debt = Yahoo's EV (which already nets cash) - market cap; EV = EV/EBITDA x EBITDA also works when
    # both are negative. Gross debt is the fallback (cash unknown -> conservative).
    ev_ratio = _col(c, "ev_to_ebitda")
    ev_abs = (ev_ratio * ebitda).where(lambda s: s > 0)
    net_debt = (ev_abs - mcap).where(ev_abs.notna() & mcap.notna(), debt)
    nd = (net_debt / ebitda).where(ebitda > 0)
    nd = nd.where(~(debt <= 0), 0.0)                                           # debt-free
    material = (net_debt > rules["quality"]["penalties"]["material_net_debt_pct_of_mcap"] * mcap).fillna(False)
    no_ebitda = (ebitda <= 0).fillna(False)
    nd = nd.where(~(no_ebitda & material), 99.0).where(~(no_ebitda & ~material & net_debt.notna()), 0.0)
    f["net_debt_ebitda"] = nd.clip(-5, 99)
    f["debt_without_ebitda"] = no_ebitda & material

    # dividend cover: FCF over cash dividends; earnings for financials and capex-heavy sectors (their FCF is
    # structurally negative); not assessed for REITs (GAAP earnings understate distributable cash)
    divy = _col(c, "dividends_paid_yield_pct")
    divy = divy.where(divy.notna(), _col(c, "dividend_yield_pct"))
    div_cash = (divy / 100 * mcap).where(lambda s: s > 0)
    cover = pd.Series(np.where(f["group"] == "standard", fcf, earn_ttm), index=c.index) / div_cash
    f["dividend_uncovered"] = (cover < 1).fillna(False) & ~f["is_reit"]
    sy = _col(c, "net_payout_yield_pct")
    f["shareholder_yield"] = sy.where(sy.notna(), _col(c, "dividend_yield_pct"))

    for col in ("rev_cagr", "profit_years", "margin_vol", "ni_median", "last_equity"):
        f[col] = np.nan
    if annual_fin is not None and not annual_fin.empty:
        h = _history_features(annual_fin).reindex(f.index)
        for col in h.columns:
            f[col] = h[col]
    denom = (f["last_equity"] + debt).where(lambda s: s > 0)
    f["roc"] = f["ni_median"] / denom * 100        # median of the last 3 reported net incomes
    f["negative_equity"] = (f["last_equity"] < 0).fillna(False)

    f["ret_12_1"], f["pct_from_ma200"], f["ma_signal"] = np.nan, np.nan, np.nan
    if prices is not None and not prices.empty:
        mo = _momentum_features(prices, rules).reindex(f.index)
        for col in mo.columns:
            f[col] = mo[col]
    return f


# ── scores ────────────────────────────────────────────────────────────────────────────────────
def quality_components(f: pd.DataFrame, rules: dict) -> pd.DataFrame:
    q, mp = rules["quality"], rules["peers"]["min_size"]
    b = q["bands"]
    pct = lambda s, hib=True: peer_percentile(s, f["industry"], f["sector"], mp, hib)  # noqa: E731
    c = pd.DataFrame(index=f.index)
    c["roc"] = band(f["roc"], b["roc"])
    c["roe"] = band(f["roe"], b["roe"])
    c["op_margin"], c["gross_margin"] = pct(f["op_margin"]), pct(f["gross_margin"])
    c["fcf_margin"] = _blend(pct(f["fcf_margin"]), band(f["fcf_margin"], b["fcf_margin"]), 0.5)
    c["net_margin"] = pct(f["net_margin"])

    gw = q["growth_stability_weights"]
    parts = pd.DataFrame({"rev_cagr": band(f["rev_cagr"], b["rev_cagr"]),
                          "profit_years": band(f["profit_years"], b["profit_years"]),
                          "margin_vol": band(f["margin_vol"], b["margin_vol"])})
    raw, cov = _weighted(parts, gw)
    c["growth_stability"] = raw.where(cov >= 0.5)       # need at least revenue trend or profit history

    nd_spec = np.where(f["is_relaxed"], 1, 0)
    nd_std = band(f["net_debt_ebitda"], b["net_debt_ebitda_standard"])
    nd_rel = band(f["net_debt_ebitda"], b["net_debt_ebitda_relaxed"])
    nd = pd.Series(np.where(nd_spec == 1, nd_rel, nd_std), index=f.index).where(f["net_debt_ebitda"].notna())
    w = q["net_debt_weight_in_balance_sheet"]
    c["balance_sheet"] = _blend(nd, band(f["current_ratio"], b["current_ratio"]), w).where(nd.notna())
    c["earnings_quality"] = band(f["fcf_conversion"], b["fcf_conversion"])
    return c


def quality_penalties(f: pd.DataFrame, rules: dict) -> pd.DataFrame:
    pen = rules["quality"]["penalties"]
    th = rules["quality"]["net_debt_thresholds"]
    fin = f["is_financial"]
    hi = np.where(f["is_relaxed"], th["relaxed"]["high"], th["standard"]["high"])
    cr = np.where(f["is_relaxed"], th["relaxed"]["critical"], th["standard"]["critical"])
    nd = f["net_debt_ebitda"]
    flags = pd.DataFrame(index=f.index)
    flags["loss_making"] = f["loss_making"]
    flags["gaap_loss_cash_positive"] = f["gaap_loss_cash_positive"]
    flags["debt_without_ebitda"] = f["debt_without_ebitda"] & ~fin
    crit = (nd > cr) & ~fin & ~f["debt_without_ebitda"]
    flags["net_debt_ebitda_critical"] = crit
    flags["net_debt_ebitda_high"] = (nd > hi) & ~crit & ~fin & ~f["debt_without_ebitda"]
    flags["dividend_uncovered"] = f["dividend_uncovered"]
    flags["negative_equity"] = f["negative_equity"] & ~fin
    pts = sum(flags[k].astype(int) * pen[k] for k in flags.columns)
    return flags, pts.clip(upper=pen["max_total"])


def value_components(f: pd.DataFrame, rules: dict) -> pd.DataFrame:
    v, mp = rules["value"], rules["peers"]["min_size"]
    b, wa = v["bands"], v["peer_blend"]
    pct = lambda s, hib=True: peer_percentile(s, f["industry"], f["sector"], mp, hib)  # noqa: E731
    c = pd.DataFrame(index=f.index)
    c["fcf_yield"] = _blend(pct(f["fcf_yield"]), band(f["fcf_yield"], b["fcf_yield"]), wa)
    ev = _blend(pct(f["ev_ebitda"], False), band(f["ev_ebitda"], b["ev_ebitda"]), wa)
    c["ev_ebitda"] = ev.where(f["ebitda_pos"] != False, 0.0)          # noqa: E712  (negative EBITDA = 0, unknown stays NaN)
    c["earn_yield"] = _blend(pct(f["earn_yield"]), band(f["earn_yield"], b["earn_yield"]), wa)
    c["peg"] = _blend(pct(f["peg"], False), band(f["peg"], b["peg"]), wa)
    c["ps"] = _blend(pct(f["ps"], False), band(f["ps"], b["ps"]), wa)
    c["pb"] = pct(f["pb"], False)
    sy = band(f["shareholder_yield"], b["shareholder_yield"])
    c["shareholder_yield"] = sy.where(~f["dividend_uncovered"], sy * v["uncovered_dividend_factor"])
    return c


def momentum_score(f: pd.DataFrame, rules: dict):
    m = rules["momentum"]["weights"]
    rank = peer_percentile(f["ret_12_1"], pd.Series("", index=f.index), pd.Series("", index=f.index),
                           rules["peers"]["min_size"])
    trend = (50 * (f["pct_from_ma200"] > 0).astype(float) + 50 * f["ma_signal"].isin(_UPTREND).astype(float))
    trend = trend.where(f["pct_from_ma200"].notna() | f["ma_signal"].notna())
    comps = pd.DataFrame({"return_rank": rank, "trend": trend})
    raw, cov = _weighted(comps, m)
    return _shrink(raw, cov, rules["coverage"]["full_at"]), comps


def score_universe(companies: pd.DataFrame, annual_fin: pd.DataFrame = None, prices: pd.DataFrame = None,
                   rules: dict = None) -> pd.DataFrame:
    """
    Quality / Value / Momentum for every investable ticker (index = ticker).

    Columns: quality, value, momentum (0-100, NaN when nothing is observable), quality_coverage,
    value_coverage (% of weight observable), flags (red-flag text), missing (list of unknown inputs
    that matter), components (dict of sub-scores, None = unknown).
    """
    rules = rules or load_rules()
    f = build_features(companies, annual_fin, prices, rules)
    if f.empty:
        return pd.DataFrame(columns=["quality", "value", "momentum", "quality_coverage", "value_coverage",
                                     "flags", "missing", "components"])
    full_at = rules["coverage"]["full_at"]
    group = f["group"]

    qc = quality_components(f, rules)
    qw, vw = rules["quality"]["weights"], rules["value"]["weights"]
    q_raw, q_cov = _by_group(qc, qw, group)
    flags, penalty = quality_penalties(f, rules)
    quality = (_shrink(q_raw, q_cov, full_at) - penalty).clip(0, 100)

    vc = value_components(f, rules)
    v_raw, v_cov = _by_group(vc, vw, group)
    value = _shrink(v_raw, v_cov, full_at).clip(0, 100)

    momentum, mc = momentum_score(f, rules)

    min_w = rules["missing_report_min_weight"]
    all_comps = pd.concat([qc.add_prefix("q_"), vc.add_prefix("v_"), mc.add_prefix("m_")], axis=1)
    rows_missing, rows_flags, rows_comps = [], [], []
    for t in f.index:
        g = group[t]
        miss = [COMPONENT_LABELS[k] for k, w in qw[g].items() if w >= min_w and pd.isna(qc.at[t, k])]
        miss += [COMPONENT_LABELS[k] for k, w in vw[g].items() if w >= min_w and pd.isna(vc.at[t, k])]
        rows_missing.append(miss)
        rows_flags.append("; ".join(FLAG_LABELS[k] for k in flags.columns if flags.at[t, k]))
        rows_comps.append({k: (None if pd.isna(v) else round(float(v), 1)) for k, v in all_comps.loc[t].items()})

    out = pd.DataFrame({
        "quality": quality.round(), "value": value.round(), "momentum": momentum.round(),
        "quality_coverage": (q_cov * 100).round(), "value_coverage": (v_cov * 100).round(),
        "flags": rows_flags, "missing": rows_missing, "components": rows_comps,
    }, index=f.index)
    return out
