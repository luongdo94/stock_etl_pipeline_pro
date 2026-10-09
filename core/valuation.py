"""
core/valuation.py — Intrinsic and relative valuation (pure functions).

DCF conventions
---------------
Base cash flow = latest annual cash-flow-statement FCF (operating cash flow − capex, after
interest → cash available to equity), stored in EUR by the ETL. Yahoo's `financialData.freeCashflow`
is only a fallback: on real data it understated MSFT/NVDA by 3–5x.
It is discounted at the COST OF EQUITY and gives equity value directly — debt must NOT be subtracted
again. Excess cash is not in the warehouse and is not added (conservative for cash-rich firms).

Growth fades linearly from the starting rate (year 1) to the terminal rate over a 10-year explicit
period. A 5-year fade is too short for reinvesting compounders and labelled most quality large caps
"overvalued" on real data.

When the price implies growth beyond what the model can represent, or the result looks implausible
(margin of safety > 150%), the DCF is reported as uninformative instead of driving a decision.
"""
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

RISK_FREE = 0.04             # fallback when no live 10Y yield is available


def _valuation_config() -> dict:
    """config/decision_rules.yaml → valuation section (DCF results are very sensitive to it)."""
    try:
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config" / "decision_rules.yaml",
                                  encoding="utf-8")) or {}
        return cfg.get("valuation", {}) or {}
    except OSError:
        return {}


def _configured(key: str, default: float) -> float:
    try:
        return float(_valuation_config().get(key, default))
    except (TypeError, ValueError):
        return default


_MINOR_UNITS = {"GBp": "GBP", "GBX": "GBP", "GBx": "GBP", "ZAc": "ZAR", "ILA": "ILS"}


def major_currency(ccy) -> Optional[str]:
    if not ccy or str(ccy).strip() in ("", "nan", "None"):
        return None
    c = str(ccy).strip()
    return _MINOR_UNITS.get(c, c.upper())


def currency_assumptions(ccy, macro: Optional[dict] = None) -> dict:
    """
    {risk_free, terminal_growth, currency, source} for cash flows earned in `ccy`.

    The risk-free rate and the long-run growth rate must come from the same monetary regime as the cash
    flows (an inflation-consistent pair). USD (and the pegged HKD) use the live US 10Y when the macro feed
    has it; every other currency uses its configured level, never the US yield.
    """
    cfg = _valuation_config().get("currencies", {}) or {}
    c = major_currency(ccy)
    entry = cfg.get(c) or cfg.get("default") or {"risk_free": RISK_FREE, "terminal_growth": 0.025}
    rf, tg = float(entry["risk_free"]), float(entry["terminal_growth"])
    source = f"{c or 'default'} assumptions (config)"
    if c in ("USD", "HKD"):
        live = risk_free_from_macro(macro)
        if macro and (macro.get("US10Y") or {}).get("val") and live != RISK_FREE:
            rf, source = live, "live US 10Y"
    return {"risk_free": rf, "terminal_growth": tg, "currency": c or "default", "source": source}


EQUITY_RISK_PREMIUM = _configured("equity_risk_premium", 0.05)   # mature-market ERP
TERMINAL_GROWTH = 0.025
EXPLICIT_YEARS = 10
GROWTH_FLOOR, GROWTH_CAP = -0.05, 0.20
GROWTH_BAND = 0.05   # earnings / FCF may move the anchor by at most ±5pp around revenue growth
REQUIRED_MARGIN_OF_SAFETY = _configured("required_margin_of_safety", 0.25)
FADE_KEEP = 0.7                   # share of a bank's normalised ROE assumed to persist (the rest converges to the cost of equity)
IMPLAUSIBLE_MARGIN = 1.5          # value > 2.5x price → suspect the inputs, not the market
# Sectors where free-cash-flow DCF is not meaningful (cash flow includes customer deposits / float)
DCF_NOT_APPLICABLE_SECTORS = {"banks", "capital markets", "financial services", "financials", "insurance"}
# Inside those sectors, fee businesses with no balance-sheet lending or float (exchanges, index / data providers):
# book value says nothing about them and their free cash flow is real, so they are valued with the FCFE DCF.
ASSET_LIGHT_INDUSTRIES = {"financial data & stock exchanges"}


def _finite(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def cost_of_equity(beta=None, risk_free: float = RISK_FREE, erp: float = EQUITY_RISK_PREMIUM) -> float:
    """CAPM with Blume-adjusted beta (⅔·β + ⅓, the standard pull toward 1), clipped to [0.6, 2.0]."""
    b = _finite(beta)
    b = 1.0 if b is None or b <= 0 else float(np.clip(2 / 3 * b + 1 / 3, 0.6, 2.0))
    return risk_free + b * erp


def fcf_cagr(values: Iterable) -> Optional[float]:
    """CAGR between the first and last POSITIVE free cash flow of an annual series (oldest → newest)."""
    v = [x for x in (_finite(x) for x in values) if x is not None]
    if len(v) < 2 or v[0] <= 0 or v[-1] <= 0:
        return None
    return (v[-1] / v[0]) ** (1 / (len(v) - 1)) - 1


def anchor_growth(revenue_growth=None, earnings_growth=None, hist_fcf_cagr=None,
                  default: float = 0.05, revenue_cagr=None) -> tuple:
    """
    Starting growth for the DCF, built the way an analyst would:
      1. Anchor on REVENUE: multi-year revenue CAGR from annual statements (preferred), else
         Yahoo's revenueGrowth (a single-quarter YoY — noisy: +44% for XOM on an oil spike).
      2. Earnings growth and FCF CAGR (each clipped to ±30%) may only pull the median of all
         inputs within ±5pp of that revenue anchor — margins cannot outgrow sales for 10 years.
      3. Clip to [-5%, 20%].
    Returns (growth, list of sources used).
    """
    rev_anchor = _finite(revenue_cagr)
    rev_label = "3Y revenue CAGR"
    if rev_anchor is None:
        rev_anchor, rev_label = _finite(revenue_growth), "revenue growth (latest quarter YoY)"
    eg, fc = _finite(earnings_growth), _finite(hist_fcf_cagr)
    named = {rev_label: rev_anchor,
             "earnings growth": float(np.clip(eg, -0.30, 0.30)) if eg is not None else None,
             "FCF CAGR": float(np.clip(fc, -0.30, 0.30)) if fc is not None else None}
    used = {k: v for k, v in named.items() if v is not None}
    if not used:
        return default, ["default"]
    g = float(np.median(list(used.values())))
    if rev_anchor is not None:
        g = float(np.clip(g, rev_anchor - GROWTH_BAND, rev_anchor + GROWTH_BAND))
    return float(np.clip(g, GROWTH_FLOOR, GROWTH_CAP)), list(used)


def revenue_cagr(annual_fin: Optional[pd.DataFrame], ticker: str, years: int = 3) -> Optional[float]:
    """Revenue CAGR over the last `years` annual statements (needs years+1 points)."""
    if annual_fin is None or annual_fin.empty or "revenue" not in annual_fin.columns:
        return None
    r = annual_fin[annual_fin["ticker"] == ticker].sort_values("year")["revenue"].dropna().tail(years + 1)
    return fcf_cagr(r) if len(r) >= 3 else None


def _after_sbc(rows: pd.DataFrame) -> pd.Series:
    """Free cash flow minus stock-based compensation (SBC is added back in operating cash flow although it is a
    real cost to owners — dilution). Years without an SBC figure are left unadjusted."""
    fcf = pd.to_numeric(rows["free_cash_flow"], errors="coerce")
    if "stock_based_comp" in rows.columns:
        fcf = fcf - pd.to_numeric(rows["stock_based_comp"], errors="coerce").fillna(0.0)
    return fcf


def normalized_statement_fcf(hist_fcf: pd.DataFrame, ticker: str, years: int = 3) -> Optional[float]:
    """Median of the last `years` annual statement FCFs after stock-based compensation (EUR) — damps one-off spikes and cycles."""
    if hist_fcf is None or hist_fcf.empty or "free_cash_flow" not in hist_fcf.columns:
        return None
    rows = hist_fcf[(hist_fcf["ticker"] == ticker) & hist_fcf["free_cash_flow"].notna()].sort_values("year")
    return _finite(_after_sbc(rows).tail(years).median()) if not rows.empty else None


def sbc_available(hist_fcf: pd.DataFrame, ticker: str, years: int = 3) -> bool:
    """True when the recent annual rows carry a stock-based-compensation figure (so the FCF above is adjusted)."""
    if hist_fcf is None or hist_fcf.empty or "stock_based_comp" not in hist_fcf.columns:
        return False
    rows = hist_fcf[(hist_fcf["ticker"] == ticker) & hist_fcf["free_cash_flow"].notna()].sort_values("year").tail(years)
    return bool(rows["stock_based_comp"].notna().any())


def dcf_equity_value(fcfe: float, growth: float, discount_rate: float,
                     terminal_growth: float = TERMINAL_GROWTH, years: int = EXPLICIT_YEARS) -> float:
    """Present value of FCFE: explicit years with growth fading to terminal, plus Gordon terminal value."""
    if discount_rate <= terminal_growth:
        raise ValueError("discount rate must exceed terminal growth")
    pv, cf = 0.0, float(fcfe)
    for t in range(1, years + 1):
        g_t = growth + (terminal_growth - growth) * (t - 1) / max(years - 1, 1)
        cf *= (1 + g_t)
        pv += cf / (1 + discount_rate) ** t
    terminal = cf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    return pv + terminal / (1 + discount_rate) ** years


def dcf_per_share(fcfe, shares, growth, discount_rate, terminal_growth=TERMINAL_GROWTH,
                  years=EXPLICIT_YEARS) -> Optional[float]:
    fcfe, shares = _finite(fcfe), _finite(shares)
    if not fcfe or fcfe <= 0 or not shares or shares <= 0:
        return None
    return dcf_equity_value(fcfe, growth, discount_rate, terminal_growth, years) / shares


def reverse_dcf_growth(price, fcfe, shares, discount_rate, terminal_growth=TERMINAL_GROWTH,
                       years=EXPLICIT_YEARS, lo=-0.5, hi=1.0) -> Optional[float]:
    """Starting growth rate the current price implies (bisection). None if outside [lo, hi]."""
    price = _finite(price)
    if not price or price <= 0 or dcf_per_share(fcfe, shares, 0.0, discount_rate, terminal_growth, years) is None:
        return None
    f = lambda g: dcf_per_share(fcfe, shares, g, discount_rate, terminal_growth, years) - price
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


@dataclass
class Scenario:
    name: str
    growth: float
    discount_rate: float
    value_per_share: Optional[float]


def dcf_scenarios(fcfe, shares, growth, discount_rate, terminal_growth=TERMINAL_GROWTH) -> dict:
    """Bear / base / bull: growth ∓5pp and discount rate ±1pp."""
    out = {}
    for name, dg, dr in (("bear", -0.05, +0.01), ("base", 0.0, 0.0), ("bull", +0.05, -0.01)):
        r = max(discount_rate + dr, terminal_growth + 0.01)
        g = float(np.clip(growth + dg, GROWTH_FLOOR - 0.05, GROWTH_CAP + 0.05))
        out[name] = Scenario(name, g, r, dcf_per_share(fcfe, shares, g, r, terminal_growth))
    return out


def sensitivity_table(fcfe, shares, growths, rates, terminal_growth=TERMINAL_GROWTH) -> pd.DataFrame:
    """Value per share for each (growth row × discount-rate column)."""
    data = {}
    for r in rates:
        col = []
        for g in growths:
            col.append(dcf_per_share(fcfe, shares, g, r, terminal_growth) if r > terminal_growth else None)
        data[f"{r:.1%}"] = col
    return pd.DataFrame(data, index=[f"{g:+.0%}" for g in growths])


def valuation_verdict(margin_of_safety: Optional[float], required: float = REQUIRED_MARGIN_OF_SAFETY) -> str:
    """Margin of safety = value / price − 1. Undervalued only with the REQUIRED cushion."""
    if margin_of_safety is None:
        return "NOT VALUED"
    if margin_of_safety >= required:
        return "UNDERVALUED"
    if margin_of_safety >= 0:
        return "BELOW VALUE — margin too thin"
    if margin_of_safety >= -0.15:
        return "FAIRLY VALUED"
    return "OVERVALUED"


def relative_valuation(companies: pd.DataFrame, ticker: str,
                       metrics=("pe_ratio", "forward_pe", "ev_to_ebitda", "price_to_sales", "price_to_book"),
                       min_peers: int = 4) -> dict:
    """
    Percentile of the ticker within its industry (fallback: sector) for each multiple —
    0 = cheapest, 100 = most expensive, positive values only — plus P/E vs its own 5Y average.
    """
    row = companies[companies["ticker"] == ticker]
    if row.empty:
        return {"group": None, "n_peers": 0, "percentiles": {}, "pe_vs_5y": None}
    row = row.iloc[0]
    group_col, peers = None, pd.DataFrame()
    for col in ("industry", "sector"):
        if col in companies.columns and pd.notna(row.get(col)):
            peers = companies[companies[col] == row[col]]
            if len(peers) >= min_peers:
                group_col = col
                break
    pct = {}
    for m in metrics:
        if m not in companies.columns or group_col is None:
            continue
        v = _finite(row.get(m))
        vals = pd.to_numeric(peers[m], errors="coerce")
        vals = vals[(vals > 0) & np.isfinite(vals)]
        if v is None or v <= 0 or len(vals) < min_peers:
            continue
        pct[m] = float((vals < v).mean() * 100)
    pe, pe5 = _finite(row.get("pe_ratio")), _finite(row.get("pe_5y_avg"))
    return {
        "group": f"{row.get(group_col)} ({group_col})" if group_col else None,
        "n_peers": int(len(peers)) if group_col else 0,
        "percentiles": pct,
        "pe_vs_5y": (pe / pe5) if pe and pe5 and pe > 0 and pe5 > 0 else None,
    }


def justified_pb_value(book_per_share, roe, discount_rate, growth) -> Optional[float]:
    """
    Value per share of a bank / insurer: book value x justified P/B, with P/B = (ROE - g) / (r - g).
    (Equivalent to a residual-income model with a constant ROE.) None when the franchise earns no more than
    it grows (ROE <= g) — then book value is not worth more than its parts and the formula is meaningless.
    """
    bv, roe, r, g = _finite(book_per_share), _finite(roe), _finite(discount_rate), _finite(growth)
    if bv is None or bv <= 0 or roe is None or r is None or g is None or r <= g or roe <= g:
        return None
    return bv * (roe - g) / (r - g)


def normalized_roe(annual_fin: Optional[pd.DataFrame], ticker: str, fallback=None, years: int = 4) -> Optional[float]:
    """Median of the last `years` annual net income / equity (through the cycle), clipped to [0, 25%]."""
    roes = []
    if annual_fin is not None and not annual_fin.empty and {"net_income", "total_equity"} <= set(annual_fin.columns):
        t = annual_fin[annual_fin["ticker"] == ticker].sort_values("year").tail(years)
        roes = [n / e for n, e in zip(t["net_income"], t["total_equity"])
                if _finite(n) is not None and _finite(e) is not None and e > 0]
    r = float(np.median(roes)) if len(roes) >= 2 else _finite(fallback)
    return None if r is None else float(np.clip(r, 0.0, 0.25))


def _financial_inputs(meta, price: float, annual_fin, ticker: str, coe: float, tg: float, ass: dict) -> dict:
    mcap = _finite(meta.get("market_cap"))
    shares = mcap / price if mcap and price else None
    pb = _finite(meta.get("price_to_book"))
    bvps = price / pb if pb and pb > 0 else None
    roe_norm = normalized_roe(annual_fin, ticker, fallback=meta.get("roe"))
    # Competitive fade: a return above (or below) the cost of equity does not last forever, so the sustainable ROE
    # is pulled 30% of the way toward it (only when above it). Without this, a reinsurer earning 20% is capitalised at 20% for ever.
    # Only EXCESS returns fade: a franchise below its cost of equity is not rescued by assuming it converges up.
    roe = None if roe_norm is None else (FADE_KEEP * roe_norm + (1 - FADE_KEEP) * coe if roe_norm > coe else roe_norm)
    scen = {}
    for name, d_roe, d_r in (("bear", -0.03, +0.01), ("base", 0.0, 0.0), ("bull", +0.03, -0.01)):
        r = max(coe + d_r, tg + 0.01)
        rr = None if roe is None else max(roe + d_roe, 0.0)
        scen[name] = Scenario(name, rr if rr is not None else 0.0, r, justified_pb_value(bvps, rr, r, tg))
    base, bear, bull = (scen[k].value_per_share for k in ("base", "bear", "bull"))
    implied = tg + (price / bvps) * (coe - tg) if bvps else None          # ROE the price already assumes
    if base is None:
        reliable, note = False, ("The sustainable ROE is not above long-run growth (or book value / ROE is missing) — "
                                 "a justified-P/B value cannot be computed. Judge on P/B, ROE trend and credit quality.")
    elif base / price - 1 > IMPLAUSIBLE_MARGIN:
        reliable, note = False, "Value is >2.5x the price — verify book value and ROE before acting."
    elif implied is not None and implied > 0.35:
        reliable, note = False, (f"The price implies a perpetual ROE of {implied:.0%} — beyond what a lender or insurer "
                                 f"can sustain; the model is not informative here.")
    else:
        reliable, note = True, None
    return {
        "model": "justified_pb", "fcfe": None, "fcf_source": "n/a (bank / insurer: cash flows include deposits and float)",
        "shares": shares, "growth": roe or 0.0,
        "growth_sources": [f"normalised ROE {roe_norm:.1%} (median of up to 4 annual ROEs), faded 30% toward the cost of equity"
                           if roe_norm is not None else "ROE unavailable"],
        "reliable": reliable, "note": note,
        "model_note": ("Justified P/B = (ROE − g) / (r − g) on current book value per share. ROE is the 4-year median "
                       "faded 30% toward the cost of equity (excess returns do not last); still sensitive to the credit "
                       "or underwriting cycle."),
        "risk_free": ass["risk_free"], "cost_of_equity": coe, "terminal_growth": tg, "currency": ass["currency"],
        "rate_source": ass["source"], "scenarios": scen, "base": base, "bear": bear, "bull": bull,
        "implied_growth": implied, "roe": roe, "roe_normalised": roe_norm, "book_per_share": bvps,
        "justified_pb": (roe - tg) / (coe - tg) if roe is not None and coe > tg else None,
        "sbc_adjusted": False,
    }


def latest_statement_fcf(hist_fcf: pd.DataFrame, ticker: str) -> Optional[float]:
    """Most recent annual cash-flow-statement FCF (EUR) for the ticker, if any."""
    if hist_fcf is None or hist_fcf.empty or "free_cash_flow" not in hist_fcf.columns:
        return None
    rows = hist_fcf[(hist_fcf["ticker"] == ticker) & hist_fcf["free_cash_flow"].notna()].sort_values("year")
    return _finite(rows["free_cash_flow"].iloc[-1]) if not rows.empty else None


def uses_book_model(meta, hist_fcf=None, ticker=None) -> bool:
    """True for banks, insurers, brokers and other balance-sheet financials (justified P/B); False for everything the
    free-cash-flow DCF can value, including asset-light fee businesses that report positive free cash flow."""
    if str(meta.get("sector") or "").strip().lower() not in DCF_NOT_APPLICABLE_SECTORS:
        return False
    if str(meta.get("industry") or "").strip().lower() in ASSET_LIGHT_INDUSTRIES:
        fcf = normalized_statement_fcf(hist_fcf, ticker) if hist_fcf is not None and ticker else None
        if fcf is None:
            fcf = _finite(meta.get("free_cashflow"))
        return not (fcf is not None and fcf > 0)
    return True


def dcf_reliability(sector, price, base_value, bull_value, implied_growth) -> tuple:
    """(reliable: bool, note: str|None). Unreliable DCFs must not drive BUY/AVOID."""
    if sector and str(sector).strip().lower() in DCF_NOT_APPLICABLE_SECTORS:
        return False, "Free-cash-flow DCF does not apply to banks/insurers — use P/B, ROE and relative valuation."
    if base_value is None:
        return False, "No positive free cash flow — DCF not possible."
    if implied_growth is None and bull_value is not None and price > bull_value:
        return False, ("The price implies more growth than the model can represent (above the bull case) — "
                       "DCF is not informative here; judge on relative valuation and growth durability.")
    if implied_growth is not None and implied_growth > GROWTH_CAP + 0.10:
        return False, (f"The price implies {implied_growth:+.0%} starting growth, beyond the model's "
                       f"{GROWTH_CAP:.0%} cap — DCF is not informative here.")
    if base_value / price - 1 > IMPLAUSIBLE_MARGIN:
        return False, ("Value is >2.5x the price — more likely a data problem (cash-flow definition, one-off, "
                       "currency) than a bargain. Verify the inputs before acting.")
    return True, None


def risk_free_from_macro(macro: dict) -> float:
    """^TNX is quoted in percent (4.3 = 4.3%)."""
    v = _finite((macro or {}).get("US10Y", {}).get("val"))
    return v / 100 if v and 0.1 < v < 20 else RISK_FREE


def valuation_inputs(meta, price: float, hist_fcf: pd.DataFrame, ticker: str, macro: dict,
                     annual_fin: Optional[pd.DataFrame] = None) -> dict:
    """
    Company-anchored intrinsic value inputs + scenarios + reverse model (EUR, per share).

    Industrials and tech: FCFE DCF. Banks, insurers and other financials: justified P/B (residual income).
    The risk-free rate and terminal growth come from the currency the cash flows are earned in.
    """
    ass = currency_assumptions(meta.get("currency"), macro)
    rf, tg = ass["risk_free"], ass["terminal_growth"]
    coe = cost_of_equity(meta.get("beta"), risk_free=rf)
    if uses_book_model(meta, hist_fcf, ticker):
        return _financial_inputs(meta, price, annual_fin, ticker, coe, tg, ass)

    # Cash-flow-statement FCF after stock-based compensation (EUR) first; Yahoo's levered FCF only as a fallback
    statement_fcf = normalized_statement_fcf(hist_fcf, ticker)
    sbc = sbc_available(hist_fcf, ticker)
    fcfe = statement_fcf if statement_fcf is not None else _finite(meta.get("free_cashflow"))
    fcf_source = (("cash-flow statement FCF (OCF − capex − stock-based comp), median of last 3 years" if sbc
                   else "cash-flow statement FCF (OCF − capex; stock comp not available), median of last 3 years")
                  if statement_fcf is not None else "Yahoo levered FCF")
    mcap = _finite(meta.get("market_cap"))
    shares = mcap / price if mcap and price else None
    hist = _after_sbc(hist_fcf[hist_fcf["ticker"] == ticker].sort_values("year")).tail(5) \
        if hist_fcf is not None and not hist_fcf.empty else []
    growth, sources = anchor_growth(meta.get("revenue_growth"), meta.get("earnings_growth"),
                                    fcf_cagr(hist), revenue_cagr=revenue_cagr(annual_fin, ticker))
    valuable = bool(fcfe and fcfe > 0 and shares)
    scen = dcf_scenarios(fcfe, shares, growth, coe, tg) if valuable else {}
    base = scen["base"].value_per_share if scen else None
    bull = scen["bull"].value_per_share if scen else None
    implied = reverse_dcf_growth(price, fcfe, shares, coe, tg) if valuable else None
    reliable, note = dcf_reliability(None, price, base, bull, implied)       # sector test already done by uses_book_model
    return {
        "model": "dcf", "fcfe": fcfe, "fcf_source": fcf_source, "shares": shares,
        "growth": growth, "growth_sources": sources, "reliable": reliable, "note": note,
        "risk_free": rf, "cost_of_equity": coe, "terminal_growth": tg, "currency": ass["currency"],
        "rate_source": ass["source"], "scenarios": scen,
        "base": base,
        "bear": scen["bear"].value_per_share if scen else None,
        "bull": bull,
        "implied_growth": implied, "sbc_adjusted": sbc,
    }
