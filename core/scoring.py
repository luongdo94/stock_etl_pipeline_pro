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
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCORE_VERSION = "v8"   # v8: ROIC on adjacent fiscal years only, estimated ROIC flagged; payout unknown ≠ 0 (E5/E8)
# v7: real ROIC, loss-maker leverage n/a, EPS-based PEG, cyclical normalisation, reference peers   # bump when the meaning of a score changes; snapshots are stamped with it
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
    "illiquid": "Illiquid (median daily turnover under EUR 1M)", "micro_cap": "Micro-cap (market cap under EUR 300M)",
    "adv_unknown": "Liquidity unknown (no traded-value history)",
    "roic_estimated": "ROIC estimated (tax rate or cash not reported)",
    "peak_margin": "Net margin far above its own history (cyclical peak?)",
}


_YAML_CACHE: dict = {}


def _yaml_versioned(path: Path):
    """Parsed YAML, re-read whenever the file's modification time or size changes (PERF-15: an unversioned
    lru_cache kept serving the old rules in a running app after the YAML was edited, while the screener's outer
    cache key, a hash of the YAML, already said "new rules")."""
    st = path.stat()
    key = (st.st_mtime_ns, st.st_size)
    hit = _YAML_CACHE.get(path)
    if hit is None or hit[0] != key:
        hit = (key, yaml.safe_load(path.read_text(encoding="utf-8")))
        _YAML_CACHE[path] = hit
    return hit[1]


def load_rules() -> dict:
    return _yaml_versioned(_RULES_PATH)


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
                    higher_is_better: bool = True, reference: pd.Series = None) -> pd.Series:
    """0-100 mid-rank percentile within the smallest peer group (industry, sector, universe) with >= min_size
    REFERENCE members. Every stock is ranked against the reference members of its group only, so stocks that enter
    the universe through discovery screens do not move the scores of the others (audit F-11).
    reference: boolean Series (index = values.index); None = every stock is a reference member."""
    v = pd.to_numeric(values, errors="coerce").astype("float64")
    ref = (pd.Series(True, index=v.index) if reference is None
           else reference.reindex(v.index).fillna(False).astype(bool))
    out = pd.Series(np.nan, index=v.index, dtype=float)
    levels = (industry, sector, pd.Series("universe", index=v.index))
    for key in levels:
        k = key.reindex(v.index).fillna("").astype(str)
        valid = v.notna() & (k != "")
        todo = valid & out.isna()
        if not todo.any():
            continue
        pool_all = v[valid & ref]
        pool_key = k[valid & ref]
        for g, idx in k[todo].groupby(k[todo]).groups.items():
            pool = np.sort(pool_all[pool_key == g].to_numpy(float))
            if len(pool) < min_size:
                continue
            x = v.loc[idx].to_numpy(float)
            pct = (np.searchsorted(pool, x, "left") + np.searchsorted(pool, x, "right")) / 2 / len(pool) * 100
            out.loc[idx] = pct
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

    def _arr(df, col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy(float) if col in df.columns else np.full(len(df), np.nan)

    def agg(df):
        df = df.tail(5)
        rev, ni = _arr(df, "revenue"), _arr(df, "net_income")
        yrs = _arr(df, "year")
        out = {"rev_cagr": np.nan, "profit_years": np.nan, "margin_vol": np.nan, "ni_median": np.nan, "ni_cagr": np.nan,
               "ni_median_all": np.nan, "eps_cagr": np.nan, "nopat_median": np.nan, "invested_capital": np.nan,
               "last_revenue": np.nan, "margin_last": np.nan, "margin_median": np.nan, "n_years": float(len(df)),
               "roic_median": np.nan, "roic_estimated": False}
        if len(rev) and np.isfinite(rev[-1]):
            out["last_revenue"] = float(rev[-1])
        # PEG growth on DILUTED EPS (per share: buybacks and dilution count) — audit F-18
        eps = _arr(df, "eps_diluted")
        if np.isnan(eps).all():
            eps = _arr(df, "eps")
        growth_rev = rev
        if {"src_currency", "fx_to_eur"} <= set(df.columns):
            from core.valuation import verified_native_values
            factor = verified_native_values(df, np.ones(len(df)))
            factor = factor.to_numpy() if factor is not None else np.full(len(df), np.nan)
            growth_rev, eps = rev * factor, eps * factor
        e_mask = ~np.isnan(eps)
        e_ok, e_y = eps[e_mask], yrs[e_mask] if len(yrs) == len(eps) else np.full(e_mask.sum(), np.nan)
        if len(e_ok) >= 3 and e_ok[0] > 0 and e_ok[-1] > 0:
            span = e_y[-1] - e_y[0] if np.isfinite(e_y[[0, -1]]).all() else len(e_ok) - 1
            if span > 0:
                out["eps_cagr"] = ((e_ok[-1] / e_ok[0]) ** (1 / span) - 1) * 100
        # ROIC inputs (audit F-04): NOPAT = EBIT x (1 - tax), invested capital = equity + debt - cash, 2-year average
        ebit = _arr(df, "ebit")
        tax_raw = _arr(df, "tax_rate")
        tax = np.clip(np.nan_to_num(tax_raw, nan=0.21), 0.0, 0.35)
        nopat = ebit * (1 - tax)
        cash_raw = _arr(df, "cash")
        if (~np.isnan(nopat)).sum() >= 1:
            out["nopat_median"] = float(np.nanmedian(nopat[-3:])) if (~np.isnan(nopat[-3:])).any() else np.nan
        eq, dbt, cash = _arr(df, "total_equity"), _arr(df, "total_debt"), _arr(df, "cash")
        ic = eq + np.nan_to_num(dbt) - np.nan_to_num(cash)
        ic = np.where(np.isnan(eq) | np.isnan(dbt), np.nan, ic)       # need at least equity and debt
        ic_ok = ic[~np.isnan(ic)][-2:]
        if len(ic_ok):
            out["invested_capital"] = float(np.mean(ic_ok))
        # ROIC year by year on average opening/closing capital, then the median of the last 3 (A15).
        # E8: opening and closing capital must be ADJACENT fiscal years; a year that needed an assumed tax rate (21%)
        # or an assumed zero cash is flagged as estimated (it does not carry the evidence of an observed ROIC).
        roics, estimated = [], []
        for t in range(1, len(ic)):
            if np.isnan(nopat[t]) or np.isnan(ic[t]) or np.isnan(ic[t - 1]):
                continue
            if len(yrs) == len(ic) and np.isfinite(yrs[[t - 1, t]]).all() and yrs[t] - yrs[t - 1] != 1:
                continue
            avg_ic = (ic[t] + ic[t - 1]) / 2
            if avg_ic <= 0 or (np.isfinite(rev[t]) and avg_ic < 0.05 * rev[t]):
                continue
            roics.append(nopat[t] / avg_ic)
            estimated.append(bool(np.isnan(tax_raw[t]) or np.isnan(cash_raw[t]) or np.isnan(cash_raw[t - 1])))
        if roics:
            out["roic_median"] = float(np.median(roics[-3:])) * 100
            out["roic_estimated"] = any(estimated[-3:])
        # CAGR exponents use the YEARS between the endpoints, not the number of rows (a missing middle year used to
        # shorten the period and inflate growth — audit A15)
        yrs = _arr(df, "year")
        r_ok = ~np.isnan(growth_rev)
        if r_ok.sum() >= 3:
            rv, ry = growth_rev[r_ok], yrs[r_ok]
            span = ry[-1] - ry[0] if np.isfinite(ry[[0, -1]]).all() else r_ok.sum() - 1
            if rv[0] > 0 and rv[-1] > 0 and span > 0:
                out["rev_cagr"] = ((rv[-1] / rv[0]) ** (1 / span) - 1) * 100
        ok = ~np.isnan(ni)
        pos = ni[ok]
        py = yrs[ok]
        if len(pos) >= 3 and pos[0] > 0 and pos[-1] > 0:
            span = py[-1] - py[0] if np.isfinite(py[[0, -1]]).all() else len(pos) - 1
            if span > 0:
                out["ni_cagr"] = ((pos[-1] / pos[0]) ** (1 / span) - 1) * 100       # realised earnings growth
        if ok.sum() >= 2:
            out["ni_median"] = float(np.median(ni[ok][-3:]))      # through-the-cycle: one impairment year is not the story
            out["ni_median_all"] = float(np.median(ni[ok]))       # longest available window (cyclicals, audit F-22)
        if ok.sum() >= 3:
            out["profit_years"] = float((ni[ok] > 0).mean())
        m_ok = ok & (rev > 0)
        if m_ok.sum() >= 3:
            out["margin_vol"] = float(np.std(ni[m_ok] / rev[m_ok]) * 100)
        if m_ok.sum() >= 4 and m_ok[-1]:
            margins = ni[m_ok] / rev[m_ok]
            out["margin_last"], out["margin_median"] = float(margins[-1]), float(np.median(margins[:-1]))
        return pd.Series(out)

    cols = [c for c in ("year", "revenue", "net_income", "eps", "eps_diluted", "ebit", "tax_rate", "total_equity",
                        "total_debt", "cash", "src_currency", "fx_to_eur") if c in a.columns]
    hist = a.groupby("ticker")[cols].apply(agg)
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

    eur = p.groupby("ticker")["price_close"].apply(ret)
    # Momentum is a statement about the STOCK, not the currency: when the EUR conversion of each close is known
    # (fx_rate, price_scale) the return is measured in the stock's own currency.
    if {"fx_rate", "price_scale"} <= set(p.columns):
        local = p["price_close"] * p["price_scale"] / p["fx_rate"]
        local_ret = local.groupby(p["ticker"]).apply(ret)
        eur = local_ret.where(local_ret.notna(), eur)
    out = pd.DataFrame({"ret_12_1": eur * 100})
    if "volume" in p.columns:     # liquidity: median traded value of the last 60 sessions (EUR)
        traded = (p["price_close"] * p["volume"]).groupby(p["ticker"]).apply(lambda s: s.tail(60).median())
        out["adv_eur"] = traded
    last = p.groupby("ticker").tail(1).set_index("ticker")
    out["pct_from_ma200"] = _col(last, "pct_from_ma200")
    out["ma_signal"] = last["ma_signal"].astype(str) if "ma_signal" in last.columns else np.nan
    return out


def _sbc_last(hist_fcf) -> pd.Series:
    """Most recent annual stock-based compensation per ticker (EUR); empty when the warehouse has none yet."""
    if hist_fcf is None or hist_fcf.empty or "stock_based_comp" not in hist_fcf.columns:
        return pd.Series(dtype=float)
    h = hist_fcf.dropna(subset=["stock_based_comp"]).sort_values(["ticker", "year"])
    return h.groupby("ticker")["stock_based_comp"].last().astype(float)


def build_features(companies: pd.DataFrame, annual_fin: pd.DataFrame = None, prices: pd.DataFrame = None,
                   rules: dict = None, hist_fcf: pd.DataFrame = None) -> pd.DataFrame:
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
    f["is_cyclical"] = in_group("cyclical") if "cyclical" in grp else pd.Series(False, index=c.index)
    f["group"] = np.where(f["is_financial"], "financial", np.where(in_group("capex_heavy"), "capex_heavy", "standard"))

    mcap = _col(c, "market_cap").where(lambda s: s > 0)
    pe, eps = _col(c, "pe_ratio"), _col(c, "trailing_eps")
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

    # Free cash flow AFTER stock-based compensation: SBC is added back in operating cash flow but is paid by
    # owners through dilution. Without an SBC figure (older warehouse) the reported FCF is used and flagged.
    sbc = _sbc_last(hist_fcf).reindex(c.index)
    f["sbc_adjusted"] = sbc.notna()
    fcf_adj = fcf - sbc.fillna(0.0)
    f["fcf_yield"] = fcf_adj / mcap * 100
    rev_ttm = _col(c, "revenue_ttm").where(lambda s: s > 0)
    f["fcf_margin"] = (fcf_adj / rev_ttm * 100).where(sbc.notna() & rev_ttm.notna(), _col(c, "fcf_margin"))
    f["ev_ebitda"] = ev.where(ebitda > 0)
    f["ebitda_pos"] = ebitda.gt(0).where(ebitda.notna())          # NaN = unknown
    f["peg"] = np.nan                       # filled below from realised earnings growth (no analyst growth)
    f["ps"] = _col(c, "price_to_sales").where(lambda s: s > 0)
    f["pb"] = _col(c, "price_to_book").where(lambda s: s > 0)
    f["op_margin"], f["gross_margin"] = _col(c, "operating_margin"), _col(c, "gross_margin")
    f["roe"] = _col(c, "roe") * 100
    f["net_margin"] = (earn_ttm / _col(c, "revenue_ttm").where(lambda s: s > 0) * 100)
    f["current_ratio"] = _col(c, "current_ratio")
    f["fcf_conversion"] = (fcf_adj / earn_ttm).where(earn_ttm > 0)

    # Net debt = Yahoo's EV (which already nets cash) - market cap; EV = EV/EBITDA x EBITDA also works when
    # both are negative. Gross debt is the fallback (cash unknown -> conservative).
    ev_ratio = _col(c, "ev_to_ebitda")
    ev_abs = (ev_ratio * ebitda).where(lambda s: s > 0)
    net_debt = (ev_abs - mcap).where(ev_abs.notna() & mcap.notna(), debt)
    nd = (net_debt / ebitda).where(ebitda > 0)
    nd = nd.where(~(debt <= 0), 0.0)                                           # debt-free
    material = (net_debt > rules["quality"]["penalties"]["material_net_debt_pct_of_mcap"] * mcap).fillna(False)
    no_ebitda = (ebitda <= 0).fillna(False)
    net_cash = (net_debt <= 0).fillna(False)
    nd = nd.where(~(no_ebitda & material), 99.0)
    nd = nd.where(~(no_ebitda & ~material & net_cash), 0.0)           # loss-maker with net cash: no leverage
    # loss-maker with some net debt: leverage cannot be measured on negative EBITDA → not assessed (was 0 = best band)
    nd = nd.where(~(no_ebitda & ~material & ~net_cash & net_debt.notna()), np.nan)
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

    for col in ("rev_cagr", "profit_years", "margin_vol", "ni_median", "last_equity", "ni_median_all", "eps_cagr",
                "nopat_median", "invested_capital", "last_revenue", "margin_last", "margin_median", "roic_median"):
        f[col] = np.nan
    f["roic_estimated"] = False
    if annual_fin is not None and not annual_fin.empty:
        h = _history_features(annual_fin).reindex(f.index)
        for col in h.columns:
            f[col] = h[col]
    # cyclicals: through-the-cycle earnings over the longest window; flag margins far above their own history
    f["ni_median"] = f["ni_median_all"].where(f["is_cyclical"] & f["ni_median_all"].notna(), f["ni_median"])
    f["peak_margin"] = ((f["margin_median"] > 0) & (f["margin_last"] > 1.5 * f["margin_median"])).fillna(False)
    # ROIC = median NOPAT / average (equity + debt - cash). Invested capital that is negative or tiny vs revenue
    # (negative book equity from buybacks, cash-rich balance sheets) makes the ratio meaningless → NaN, not 100.
    ic = f["invested_capital"]
    ic_ok = ic.where((ic > 0) & ((ic >= 0.05 * f["last_revenue"]) | f["last_revenue"].isna()))
    roic = f["roic_median"].where(f["roic_median"].notna(), f["nopat_median"] / ic_ok * 100)   # per-year series first
    # legacy proxy only where EBIT is not in the warehouse yet, and only with positive book equity
    legacy = (f["ni_median"] / (f["last_equity"] + debt).where(lambda s: s > 0) * 100).where(f["last_equity"] > 0)
    f["roc"] = roic.where(f["nopat_median"].notna(), legacy)
    # Audit F05: only a year-by-year ROIC on ADJACENT capital pairs is observed ROIC. The pooled NOPAT / capital fallback
    # (non-adjacent or single-year history) still feeds the return-on-capital score, but as an ESTIMATED proxy.
    f["roc_is_roic"] = f["roic_median"].notna()
    pooled = f["roic_median"].isna() & f["nopat_median"].notna() & f["roc"].notna()
    f["roic_estimated"] = f["roic_estimated"].astype("boolean").fillna(False).astype(bool) | pooled
    if "ni_cagr" not in f:
        f["ni_cagr"] = np.nan

    # Earnings yield (%): trailing earnings and 3-year median earnings (through the cycle), averaged. No forward
    # P/E: it is an analyst forecast. Negative earnings = 0 (a fact, not a gap).
    trailing_ey = (100 / pe).where(pe > 0)
    normal_ey = (f["ni_median"] / mcap * 100).where(f["ni_median"] > 0)
    ey = pd.concat([trailing_ey, normal_ey], axis=1).mean(axis=1, skipna=True)
    ey = ey.where(~(f["is_cyclical"] & normal_ey.notna()), normal_ey)   # cyclicals: through-the-cycle only (F-22)
    ey = ey.where(~f["loss_making"], 0.0)                         # an economic loss earns nothing, whatever it earned before
    ey = ey.where(~f["gaap_loss_cash_positive"], ey * 0.5)        # impairment year: half credit for through-the-cycle earnings
    f["earn_yield"] = ey.where(ey.notna() | ~f["gaap_loss"], 0.0)
    # PEG on REALISED growth: trailing P/E over the diluted-EPS CAGR of the last (up to) 5 annual statements — per share,
    # like the P/E (F-18); net-income CAGR only when no EPS history exists. Yahoo's PEG uses analyst growth.
    g_raw = f["eps_cagr"].where(f["eps_cagr"].notna(), f["ni_cagr"])
    growth = g_raw.where(g_raw >= 2.0).clip(upper=60.0)
    f["peg"] = (pe / growth).where(pe > 0)
    f["negative_equity"] = (f["last_equity"] < 0).fillna(False)

    f["ret_12_1"], f["pct_from_ma200"], f["ma_signal"], f["adv_eur"] = np.nan, np.nan, np.nan, np.nan
    f["micro_cap"] = (mcap < rules["liquidity"]["micro_cap_eur"]).fillna(False)
    if prices is not None and not prices.empty:
        mo = _momentum_features(prices, rules).reindex(f.index)
        for col in mo.columns:
            f[col] = mo[col]
    return f


# ── scores ────────────────────────────────────────────────────────────────────────────────────
def _ref(f):
    return f["is_reference"] if "is_reference" in f.columns else None


def peer_group_labels(f: pd.DataFrame, min_size: int) -> pd.Series:
    """Which peer group each ticker's percentiles are ranked in, as text: 'industry: Telecom Services (14)', 'sector: ...'
    or 'whole universe (n)'. Same rule as peer_percentile (smallest of industry, sector, universe with >= min_size
    reference members); a metric a stock lacks can still fall to a larger group, so this is the group in the usual case."""
    ref = f["is_reference"].astype(bool) if "is_reference" in f.columns else pd.Series(True, index=f.index)
    out = pd.Series("", index=f.index, dtype=object)
    for level in ("industry", "sector"):
        key = f[level].fillna("").astype(str)
        n = key[ref].groupby(key[ref]).size()
        size = key.map(n).fillna(0).astype(int)
        todo = (out == "") & (key != "") & (size >= min_size)
        out[todo] = [f"{level}: {k} ({s})" for k, s in zip(key[todo], size[todo])]
    out[out == ""] = f"whole universe ({int(ref.sum())})"
    return out


def quality_components(f: pd.DataFrame, rules: dict) -> pd.DataFrame:
    q, mp = rules["quality"], rules["peers"]["min_size"]
    b = q["bands"]
    pct = lambda s, hib=True: peer_percentile(s, f["industry"], f["sector"], mp, hib, _ref(f))  # noqa: E731
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
    pct = lambda s, hib=True: peer_percentile(s, f["industry"], f["sector"], mp, hib, _ref(f))  # noqa: E731
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


def revisions_score(estimates: pd.DataFrame, index, rules: dict) -> pd.Series:
    """
    0-100 from the CHANGE in analysts' estimates over 30 days (marts.dim_forward_estimates): revision of the
    current- and next-year EPS estimate, and the balance of upgrades vs downgrades. 50 = no change.
    Estimate revisions are a flow with a documented drift; the level of the consensus (ratings, targets) is not used.
    """
    out = pd.Series(np.nan, index=index, dtype=float)
    if estimates is None or estimates.empty:
        return out
    r = rules["revisions"]
    e = estimates.drop_duplicates("ticker", keep="last").set_index("ticker").reindex(index)

    def pct(now, ago):
        now, ago = _col(e, now), _col(e, ago)
        return ((now - ago) / ago.abs().where(ago.abs() > 0) * 100)

    up, down = _col(e, "upgrades_30d"), _col(e, "downgrades_30d")
    moves = up + down
    balance = ((up - down) / moves).where(moves >= r["min_revisions"])
    comps = pd.DataFrame({
        "eps_cur_y": band(pct("eps_trend_cur_y", "eps_trend_cur_y_30d"), r["bands"]["eps_change_pct"]),
        "eps_next_y": band(pct("eps_trend_next_y", "eps_trend_next_y_30d"), r["bands"]["eps_change_pct"]),
        "net_revisions": band(balance, r["bands"]["net_revisions"]),
    })
    raw, cov = _weighted(comps, r["weights"])
    return _shrink(raw, cov, rules["coverage"]["full_at"]).where(cov >= 0.4)


def momentum_score(f: pd.DataFrame, rules: dict):
    m = rules["momentum"]["weights"]
    rank = peer_percentile(f["ret_12_1"], pd.Series("", index=f.index), pd.Series("", index=f.index),
                           rules["peers"]["min_size"], reference=_ref(f))
    trend = (50 * (f["pct_from_ma200"] > 0).astype(float) + 50 * f["ma_signal"].isin(_UPTREND).astype(float))
    trend = trend.where(f["pct_from_ma200"].notna() | f["ma_signal"].notna())
    comps = pd.DataFrame({"return_rank": rank, "trend": trend})
    raw, cov = _weighted(comps, m)
    return _shrink(raw, cov, rules["coverage"]["full_at"]), comps


_TICKERS_PATH = Path(__file__).resolve().parent.parent / "config" / "tickers.yaml"


def config_tickers() -> frozenset:
    """Tickers of the curated universe (config/tickers.yaml) — the reference set for peer percentiles."""
    try:
        return frozenset((_yaml_versioned(_TICKERS_PATH) or {}).get("tickers", {}) or {})
    except OSError:
        return frozenset()


def reference_mask(index, rules: dict, reference_tickers=None) -> pd.Series:
    """Reference members for percentiles: the curated universe (or the given tickers). Falls back to everyone when
    fewer than `peers.min_size` of the scored tickers are reference members (synthetic data, tests)."""
    ref_set = set(reference_tickers) if reference_tickers is not None else config_tickers()
    mask = pd.Series(index.isin(list(ref_set)), index=index)
    if mask.sum() < rules["peers"]["min_size"]:
        return pd.Series(True, index=index)
    return mask


# The measured values behind the Quality sub-scores (shown next to them, not only as 0-100 points):
# key → (feature column, multiplier to display units)
RAW_METRICS = {
    "roic_pct": ("roc", 1.0),                    # ROIC % (median of up to 3 years) — or the legacy return on capital
    "fcf_conversion": ("fcf_conversion", 1.0),   # FCF after SBC / trailing earnings (×)
    "net_debt_ebitda": ("net_debt_ebitda", 1.0), # net debt / EBITDA (×); 99 = debt without positive EBITDA
    "rev_cagr_pct": ("rev_cagr", 1.0),           # revenue CAGR over the stored years (%)
    "eps_cagr_pct": ("eps_cagr", 1.0),           # diluted EPS CAGR (%)
    "profit_years_pct": ("profit_years", 100.0), # share of stored years with a profit (%)
    "margin_vol_pp": ("margin_vol", 1.0),        # volatility of the net margin (percentage points)
    "fcf_margin_pct": ("fcf_margin", 1.0),       # FCF after SBC / revenue (%)
    "fcf_yield_pct": ("fcf_yield", 1.0),         # FCF after SBC / market cap (%) — the largest Value input
}


def score_universe(companies: pd.DataFrame, annual_fin: pd.DataFrame = None, prices: pd.DataFrame = None,
                   rules: dict = None, hist_fcf: pd.DataFrame = None, estimates: pd.DataFrame = None,
                   reference_tickers=None) -> pd.DataFrame:
    """
    Quality / Value / Momentum for every investable ticker (index = ticker).

    Columns: quality, value, momentum (0-100, NaN when nothing is observable), quality_coverage,
    value_coverage (% of weight observable), flags (red-flag text), missing (list of unknown inputs
    that matter), components (dict of sub-scores, None = unknown).
    """
    rules = rules or load_rules()
    f = build_features(companies, annual_fin, prices, rules, hist_fcf)
    if not f.empty:
        f["is_reference"] = reference_mask(f.index, rules, reference_tickers)
    if f.empty:
        return pd.DataFrame(columns=["quality", "value", "momentum", "revisions", "quality_coverage", "value_coverage",
                                     "flags", "missing", "components", "metrics", "adv_eur", "sbc_adjusted"])
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
    revisions = revisions_score(estimates, f.index, rules)
    liq = rules["liquidity"]
    info_flags = pd.DataFrame({
        "illiquid": (f["adv_eur"] < liq["min_adv_eur"]).fillna(False),
        "adv_unknown": f["adv_eur"].isna(),          # NaN < threshold is False: unknown used to pass silently (R3-D6)
        # E8: ROIC built on an assumed 21% tax rate or an assumed zero cash balance — estimated, not observed
        "roic_estimated": f["roic_estimated"].astype("boolean").fillna(False).astype(bool),
        "micro_cap": f["micro_cap"],
        "peak_margin": f["peak_margin"],
    })

    min_w = rules["missing_report_min_weight"]
    all_comps = pd.concat([qc.add_prefix("q_"), vc.add_prefix("v_"), mc.add_prefix("m_")], axis=1)
    rows_missing, rows_flags, rows_comps, rows_metrics = [], [], [], []
    for t in f.index:
        g = group[t]
        miss = [COMPONENT_LABELS[k] for k, w in qw[g].items() if w >= min_w and pd.isna(qc.at[t, k])]
        miss += [COMPONENT_LABELS[k] for k, w in vw[g].items() if w >= min_w and pd.isna(vc.at[t, k])]
        rows_missing.append(miss)
        rows_flags.append("; ".join([FLAG_LABELS[k] for k in flags.columns if flags.at[t, k]]
                                    + [FLAG_LABELS[k] for k in info_flags.columns if info_flags.at[t, k]]))
        rows_comps.append({k: (None if pd.isna(v) else round(float(v), 1)) for k, v in all_comps.loc[t].items()})
        rows_metrics.append({k: (None if pd.isna(f.at[t, col]) else float(f.at[t, col]) * mult)
                             for k, (col, mult) in RAW_METRICS.items()}
                            | {"roic_is_roic": bool(f.at[t, "roc_is_roic"]),
                               "roic_estimated": bool(info_flags.at[t, "roic_estimated"])})

    out = pd.DataFrame({
        "quality": quality.round(), "value": value.round(), "momentum": momentum.round(), "revisions": revisions.round(),
        "adv_eur": f["adv_eur"], "sbc_adjusted": f["sbc_adjusted"],
        "quality_coverage": (q_cov * 100).round(), "value_coverage": (v_cov * 100).round(),
        "peer_group": peer_group_labels(f, rules["peers"]["min_size"]),
        "flags": rows_flags, "missing": rows_missing, "components": rows_comps, "metrics": rows_metrics,
    }, index=f.index)
    return out
