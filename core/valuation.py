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
import contextvars
import copy
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd

RISK_FREE = 0.04             # fallback when no live 10Y yield is available


_CFG_CACHE: dict = {}
# Per-call memo of prepared issuer inputs, active only inside valuation_inputs (contextvar: thread/session-safe).
_MEMO: contextvars.ContextVar = contextvars.ContextVar("valuation_memo", default=None)


def _memo(key: tuple, build):
    """build() once per key within the current valuation_inputs call; every caller gets its own deep copy, so a
    caller that edits the result cannot change what the next one sees. Outside a call: plain build()."""
    memo = _MEMO.get()
    if memo is None:
        return build()
    if key not in memo:
        memo[key] = build()
    return copy.deepcopy(memo[key])


def _valuation_config() -> dict:
    """config/decision_rules.yaml → valuation section (DCF results are very sensitive to it). Parsed once per file
    version (audit F28: it was re-read from disk for every helper call of every issuer); an edit is picked up at once
    because the cache key is the file's modification time and size."""
    try:
        import yaml
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "config" / "decision_rules.yaml"
        st = p.stat()
        key = (st.st_mtime_ns, st.st_size)
        if _CFG_CACHE.get("key") != key:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            _CFG_CACHE.update(key=key, cfg=cfg.get("valuation", {}) or {})
        return dict(_CFG_CACHE["cfg"])
    except OSError:
        return {}


def _required(key: str, default: bool = True) -> bool:
    """Fail-closed switches in decision_rules.yaml → valuation (true unless explicitly set false)."""
    v = _valuation_config().get(key, default)
    return default if v is None else bool(v)


def _configured(key: str, default: float) -> float:
    try:
        return float(_valuation_config().get(key, default))
    except (TypeError, ValueError):
        return default


from core.currency import MINOR_UNITS as _MINOR_UNITS, major_currency  # noqa: E402,F401  (single definition, F-28)


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


def cost_of_equity(beta=None, risk_free: float = RISK_FREE, erp: Optional[float] = None) -> float:
    """CAPM with Blume-adjusted beta (⅔·β + ⅓, the standard pull toward 1), clipped to [0.6, 2.0]. The default ERP is read
    from the config at the call (it used to be fixed at import, so a long-running process kept the old premium)."""
    erp = _configured("equity_risk_premium", 0.05) if erp is None else erp
    b = _finite(beta)
    b = 1.0 if b is None or b <= 0 else float(np.clip(2 / 3 * b + 1 / 3, 0.6, 2.0))
    return risk_free + b * erp


def fcf_cagr(values: Iterable) -> Optional[float]:
    """CAGR between the first and last POSITIVE free cash flow of an annual series (oldest → newest)."""
    v = [x for x in (_finite(x) for x in values) if x is not None]
    if len(v) < 2 or v[0] <= 0 or v[-1] <= 0:
        return None
    return (v[-1] / v[0]) ** (1 / (len(v) - 1)) - 1


def cagr_by_year(years, values) -> Optional[float]:
    """CAGR between the first and last positive values, raised to 1 / (years between them) — a missing middle year
    no longer shortens the period (audit A15)."""
    pairs = [(float(y), float(v)) for y, v in zip(years, values)
             if _finite(y) is not None and _finite(v) is not None]
    if len(pairs) < 2 or pairs[0][1] <= 0 or pairs[-1][1] <= 0:
        return None
    span = pairs[-1][0] - pairs[0][0]
    return (pairs[-1][1] / pairs[0][1]) ** (1 / span) - 1 if span > 0 else None


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
    """Operating revenue CAGR in verified source currency; valuation amounts stay EUR."""
    columns = ["ticker", "year", "revenue", "src_currency", "fx_to_eur"]
    if annual_fin is None or annual_fin.empty or not set(columns).issubset(annual_fin.columns):
        return None
    rows = annual_fin[annual_fin["ticker"] == ticker].sort_values("year")[columns] \
        .dropna(subset=["year", "revenue"]).tail(years + 1)
    if len(rows) < 3:
        return None
    # Verify every selected year, including middle years; never drop unknown metadata.
    currencies, rates = rows["src_currency"], rows["fx_to_eur"].map(_finite)
    if currencies.isna().any() or currencies.nunique() != 1 or not str(currencies.iloc[0]).strip():
        return None
    if rates.isna().any() or (rates <= 0).any():
        return None
    return cagr_by_year(rows["year"], rows["revenue"] / rates)


# Cash paid to lenders / lessors that IFRS (and some US GAAP filers) report in FINANCING cash flow. OCF − capex of
# such a filer is cash BEFORE those payments; equity holders only get what is left (audit F-03).
FINANCING_DEDUCTIONS = ("interest_paid_cff", "lease_principal_paid")


def deduction_context(meta=None, ticker: Optional[str] = None) -> dict:
    """What the deduction resolver may use besides the reported fields.
      us_gaap  — a US-domiciled company with a US listing files a 10-K under US GAAP (foreign private issuers, the only
                 SEC registrants allowed IFRS, file 20-F / 40-F): ASC 230 puts interest paid in OPERATING cash flow and
                 ASC 842 puts operating-lease payments there too — both already inside FCF.
      sbc_material — the sector is one where share-based pay is routinely large (software, semis, internet, biotech…):
                 an unreported SBC there is never assumed away.
    No meta → no context: only reported evidence counts (the strict R4-02 behaviour)."""
    if meta is None:
        return {}
    get = meta.get if hasattr(meta, "get") else (lambda k, d=None: d)
    t = str(ticker or get("ticker") or "")
    country = str(get("country") or "").strip().lower()
    sector = f"{get('sector') or ''} {get('industry') or ''}".lower()
    keys = _valuation_config().get("sbc_material_sector_keywords") or SBC_MATERIAL_KEYWORDS
    return {"us_gaap": country in ("united states", "usa", "us") and "." not in t,
            "sbc_material": any(k in sector for k in keys), "sector": (get("sector") or "").strip() or "this sector",
            "allow_assumed": _required("allow_assumed_fcfe_deductions")}


SBC_MATERIAL_KEYWORDS = ("software", "saas", "semiconductor", "internet", "ecommerce", "e-commerce", "social", "biotech",
                         "platform", "cyber", "cloud", "fintech", "gaming", "interactive", "it services")
REIT_KEYWORDS = ("reit", "real estate")       # FCFE DCF not decision-grade for property owners (see valuation_inputs)
ASSUMED = "ASSUMED"                 # not reported anywhere, assumed with a stated reason — disclosed on the Decision


def resolve_deduction(r, item: str, ctx: dict, na: dict) -> tuple:
    """(status, amount_eur, source) of one deduction in one fiscal year. Evidence order: reported cash-flow field →
    SEC XBRL supplement (written into the field by the ETL, provenance in *_source) → accounting-standard rule →
    estimate → stated assumption → UNKNOWN. The amount is what the FCFE subtracts (0 for NOT_APPLICABLE / ASSUMED /
    UNKNOWN; UNKNOWN keeps the valuation from driving a trade)."""
    v = _finite(r.get(item))
    if v is not None:
        src = r.get(f"{item.split('_')[0]}_source") if item != "stock_based_comp" else r.get("sbc_source")
        src = str(src) if src and str(src) != "nan" else None
        if src and "year by column convention" in src:
            # the 10-K figure is real, but its year was inferred from the filing's column convention (review REV-F01)
            return ESTIMATED, abs(v), src
        return (CONFIRMED_ZERO if v == 0 else OBSERVED), abs(v), src
    if item == "stock_based_comp":
        if item in na:
            return NOT_APPLICABLE, 0.0, na[item]
        if ctx.get("allow_assumed") and ctx and not ctx.get("sbc_material"):
            return ASSUMED, 0.0, (f"stock-based compensation not reported (Yahoo, SEC) — assumed immaterial for "
                                  f"{ctx.get('sector')}; FCF may be overstated by the unreported amount")
        return UNKNOWN, 0.0, None
    if item == "interest_paid_cff":
        if _finite(r.get("interest_paid_cfo")) is not None:
            src = r.get("interest_source")
            return NOT_APPLICABLE, 0.0, (f"{src}: interest paid reported in operating cash flow"
                                         if src and str(src) != "nan" else
                                         "provider reports interest paid in operating cash flow (InterestPaidCFO)")
        if ctx.get("us_gaap"):
            return NOT_APPLICABLE, 0.0, ("US GAAP filer: ASC 230 classifies interest paid in operating cash flow, "
                                        "already inside FCF")
        deduct = _required("deduct_unclassified_interest")
        sup = _finite(r.get("interest_paid_supplemental"))
        if sup is not None and deduct:
            return ESTIMATED, abs(sup), ("interest paid known only as a supplemental amount — deducted as if in "
                                         "financing (conservative: double-counts it if it is already in operating CF)")
        iexp = _finite(r.get("interest_expense_is"))
        if iexp is not None and deduct:
            return ESTIMATED, abs(iexp), ("interest paid not reported — income-statement interest expense deducted "
                                          "(conservative: double-counts it if it is already in operating CF)")
        if item in na:
            return NOT_APPLICABLE, 0.0, na[item]
        return UNKNOWN, 0.0, None
    # lease principal
    if ctx.get("us_gaap"):
        # operating-lease cash is already in operating CF (ASC 842); only finance-lease principal is a financing claim.
        # The balance-sheet estimate would deduct operating leases twice, so it is not used for US GAAP filers.
        if item in na:
            return NOT_APPLICABLE, 0.0, na[item]
        if ctx.get("allow_assumed"):
            noted = r.get("lease_source")                 # the 10-K's own words, when it supports the assumption
            if noted and str(noted).startswith("immaterial per "):
                return ASSUMED, 0.0, ("US GAAP filer: operating leases already in operating cash flow (ASC 842); "
                                      f"finance leases stated as not material — {noted}")
            if noted and str(noted).startswith("no finance leases at period end per "):
                # absence at year-end is not a reported zero for the year (review REV-06) — still an assumption
                return ASSUMED, 0.0, ("US GAAP filer: operating leases already in operating cash flow (ASC 842); "
                                      f"no finance leases at the period end — {noted}")
            return ASSUMED, 0.0, ("US GAAP filer: operating-lease payments are already in operating cash flow "
                                  "(ASC 842); finance-lease principal not reported — assumed immaterial")
        return UNKNOWN, 0.0, None
    est = _finite(r.get("lease_principal_estimate"))
    if est is not None:
        return ESTIMATED, abs(est), str(r.get("lease_principal_estimate_source") or "balance-sheet lease liabilities")
    if item in na:
        return NOT_APPLICABLE, 0.0, na[item]
    return UNKNOWN, 0.0, None


def _after_sbc(rows: pd.DataFrame, ctx: Optional[dict] = None, ticker: Optional[str] = None) -> pd.Series:
    """Cash flow to equity: OCF − capex − stock-based compensation − interest paid in financing − lease principal.
    SBC is added back in operating cash flow although it is a real cost to owners (dilution). Interest classified in
    financing (IFRS option) and lease principal (IFRS 16) are real claims ahead of equity. A missing figure is left out
    of this REFERENCE number (fillna(0)); fcfe_deduction_status marks it UNKNOWN and the valuation is then not reliable
    for BUY/AVOID (R4-02) — the fill never reaches a decision as if it were a confirmed zero."""
    fcf = pd.to_numeric(rows["free_cash_flow"], errors="coerce")
    na = _not_applicable_registry().get(str(ticker), {}) if ticker else {}
    ctx = ctx or {}
    ded = [sum(resolve_deduction(r, item, ctx, na)[1] for item in DEDUCTION_ITEMS) for _, r in rows.iterrows()]
    return fcf - pd.Series(ded, index=rows.index, dtype=float)


BASIS_YEARS = 3
DEDUCTION_ITEMS = ("stock_based_comp",) + FINANCING_DEDUCTIONS
# Per-item status of an equity-cash-flow deduction (R4-02, sourced in R5). Known statuses:
#   OBSERVED / CONFIRMED_ZERO  a reported cash-flow figure (0 counts)
#   NOT_APPLICABLE             evidence the item is not a separate claim: the provider reports interest paid INSIDE
#                              operating cash flow (InterestPaidCFO), or an issuer entry with a source in
#                              decision_rules.yaml → valuation.fcfe_not_applicable
#   ESTIMATED                  lease principal estimated from the balance sheet (Yahoo has no cash-flow field for it):
#                              lease liabilities due within 12 months at the prior year-end — deducted, and disclosed
# A missing provider field without such evidence is UNKNOWN — never inferred to be zero, and never from the country.
OBSERVED, CONFIRMED_ZERO, NOT_APPLICABLE, UNKNOWN = "OBSERVED", "CONFIRMED_ZERO", "NOT_APPLICABLE", "UNKNOWN"
ESTIMATED = "ESTIMATED"
KNOWN_STATUSES = (OBSERVED, CONFIRMED_ZERO, NOT_APPLICABLE, ESTIMATED, "ASSUMED")


def _with_estimates(rows: pd.DataFrame) -> pd.DataFrame:
    """Adds lease_principal_estimate (+ its source) from the stored lease liabilities: the current portion at the
    PRIOR fiscal year-end (= principal falling due in this year), else this year-end's current portion; 0 when the
    total lease liability is reported as 0. Needs the ticker's full, year-sorted history (the prior year)."""
    if "lease_liability_current" not in rows.columns and "lease_liability_total" not in rows.columns:
        return rows
    rows = rows.copy()
    cur = pd.to_numeric(rows.get("lease_liability_current"), errors="coerce") \
        if "lease_liability_current" in rows.columns else pd.Series(np.nan, index=rows.index)
    tot = pd.to_numeric(rows.get("lease_liability_total"), errors="coerce") \
        if "lease_liability_total" in rows.columns else pd.Series(np.nan, index=rows.index)
    yrs = pd.to_numeric(rows["year"], errors="coerce")
    prior_ok = yrs.diff() == 1
    prior = cur.shift(1).where(prior_ok)
    est = prior.fillna(cur)
    src = pd.Series(np.where(prior.notna(), "current lease liabilities at the prior year-end",
                             np.where(cur.notna(), "current lease liabilities at this year-end", None)), index=rows.index)
    zero = (tot == 0) & est.isna()
    est = est.mask(zero, 0.0)
    src = src.mask(zero, "total lease liabilities reported as 0")
    rows["lease_principal_estimate"], rows["lease_principal_estimate_source"] = est, src
    return rows


def basis_rows(hist_fcf: pd.DataFrame, ticker: str, years: int = BASIS_YEARS) -> pd.DataFrame:
    """The annual rows the valuation uses: one row per fiscal year (the latest load when a year repeats), with a
    statement FCF, the most recent `years` of them. Selection is explicit so three rows can never mean two years.
    Lease-principal estimates are derived on the full history first (they need the prior year-end)."""
    if hist_fcf is None or hist_fcf.empty or "free_cash_flow" not in hist_fcf.columns:
        return pd.DataFrame(columns=["ticker", "year", "free_cash_flow"])

    def build():
        rows = hist_fcf[(hist_fcf["ticker"] == ticker) & hist_fcf["free_cash_flow"].notna()].copy()
        order = ["year", "_extracted_at"] if "_extracted_at" in rows.columns else ["year"]
        rows = rows.sort_values(order).drop_duplicates(subset=["year"], keep="last")
        return _with_estimates(rows).tail(years)
    return _memo(("basis", id(hist_fcf), ticker, years), build)


def basis_complete(rows: pd.DataFrame, years: int = BASIS_YEARS) -> bool:
    """`years` distinct, CONSECUTIVE fiscal years (a skipped year is a missing period, not a shorter median)."""
    ys = sorted(int(y) for y in pd.to_numeric(rows.get("year", pd.Series(dtype=float)), errors="coerce").dropna())
    return len(ys) == years and len(set(ys)) == years and ys[-1] - ys[0] == years - 1


def _not_applicable_registry() -> dict:
    """{TICKER: {item: source}} from decision_rules.yaml; an entry without a source note is ignored."""
    reg = _valuation_config().get("fcfe_not_applicable") or {}
    out = {}
    for t, items in (reg.items() if isinstance(reg, dict) else ()):
        if isinstance(items, dict):
            out[str(t)] = {str(k): str(v) for k, v in items.items() if v and str(v).strip()}
    return out


def fcfe_deduction_status(hist_fcf: pd.DataFrame, ticker: str, years: int = BASIS_YEARS,
                          ctx: Optional[dict] = None) -> dict:
    """{item: {"status": ..., "by_year": {year: status}, "source": str|None}} for SBC, interest paid in financing and
    lease principal, over the valuation basis years. An item is only as known as its worst year:
    UNKNOWN > ASSUMED > ESTIMATED > OBSERVED / NOT_APPLICABLE. `ctx` (deduction_context) enables the accounting-
    standard rules and the stated assumptions; without it only reported evidence counts."""
    return _memo(("status", id(hist_fcf), ticker, years, tuple(sorted((ctx or {}).items()))),
                 lambda: _deduction_status(hist_fcf, ticker, years, ctx))


def _deduction_status(hist_fcf, ticker, years, ctx) -> dict:
    rows = basis_rows(hist_fcf, ticker, years)
    na = _not_applicable_registry().get(str(ticker), {})
    ctx = ctx or {}
    out = {}
    for item in DEDUCTION_ITEMS:
        by_year, sources = {}, set()
        for _, r in rows.iterrows():
            status_y, _, src = resolve_deduction(r, item, ctx, na)
            by_year[int(r["year"])] = status_y
            if src and status_y != OBSERVED and status_y != CONFIRMED_ZERO or (src and str(src).startswith("SEC")):
                sources.add(src)
        vals = set(by_year.values())
        status = UNKNOWN if (not by_year or UNKNOWN in vals) else (
            ASSUMED if ASSUMED in vals else (
                NOT_APPLICABLE if vals == {NOT_APPLICABLE} else (ESTIMATED if ESTIMATED in vals else OBSERVED)))
        out[item] = {"status": status, "by_year": by_year,
                     "source": "; ".join(sorted(sources)) if sources else na.get(item)}
    return out


def _deduction_evidence_text(status: dict) -> str:
    """'; lease principal ESTIMATED (…); interest paid in financing NOT_APPLICABLE (…)' for the fcf_source caption."""
    names = {"stock_based_comp": "stock comp", "interest_paid_cff": "interest paid in financing",
             "lease_principal_paid": "lease principal"}
    parts = [f"{names[k]} {s['status'].replace('_', ' ').lower()}" + (f" ({s['source']})" if s.get("source") else "")
             for k, s in status.items() if s["status"] in (ESTIMATED, NOT_APPLICABLE, ASSUMED) or s.get("source")]
    return ("; " + "; ".join(parts)) if parts else ""


REFINANCING_STRESS_SHARE = 0.5


def refinancing_stress(hist_fcf: pd.DataFrame, ticker: str, fcfe) -> dict:
    """{net_repayment, share_of_fcfe, years}: median over the basis years of debt repaid − debt issued (EUR), and its
    share of the FCFE. None when the debt flows are not reported (unknown — disclosed, not assumed zero)."""
    rows = basis_rows(hist_fcf, ticker)
    if rows.empty or not {"debt_issued", "debt_repaid"} <= set(rows.columns):
        return {"net_repayment": None, "share_of_fcfe": None, "years": 0}
    rep = pd.to_numeric(rows["debt_repaid"], errors="coerce")
    iss = pd.to_numeric(rows["debt_issued"], errors="coerce")
    # Audit F32: a year with one leg unknown is UNKNOWN, not "that leg was zero" (issuance 0 + repayment unknown used
    # to read as no refinancing need at all)
    known = rep.notna() & iss.notna()
    incomplete = int((rep.notna() ^ iss.notna()).sum())
    if not known.any():
        return {"net_repayment": None, "share_of_fcfe": None, "years": 0, "incomplete_years": incomplete}
    net = (rep.abs() - iss.abs())[known]
    med = float(net.median())
    f = _finite(fcfe)
    share = (max(med, 0.0) / f) if f and f > 0 else None
    return {"net_repayment": med, "share_of_fcfe": share, "years": int(known.sum()), "incomplete_years": incomplete}


def fcfe_adjustments(hist_fcf: pd.DataFrame, ticker: str, years: int = BASIS_YEARS, ctx: Optional[dict] = None) -> list:
    """Financing deductions applied in every basis year — reported (incl. zero) or estimated (empty = none)."""
    st = fcfe_deduction_status(hist_fcf, ticker, years, ctx)
    return [c for c in FINANCING_DEDUCTIONS if st[c]["by_year"]
            and all(s in (OBSERVED, CONFIRMED_ZERO, ESTIMATED) for s in st[c]["by_year"].values())]


def normalized_statement_fcf(hist_fcf: pd.DataFrame, ticker: str, years: int = BASIS_YEARS,
                             ctx: Optional[dict] = None) -> Optional[float]:
    """Median of the basis years' statement FCF after the deductions that are reported (EUR) — damps spikes and cycles.
    A missing deduction is left out of the figure here; whether the figure may drive a trade is decided by
    fcfe_deduction_status (UNKNOWN → not reliable), never by this number."""
    rows = basis_rows(hist_fcf, ticker, years)
    return _finite(_after_sbc(rows, ctx, ticker).median()) if not rows.empty else None


def sbc_available(hist_fcf: pd.DataFrame, ticker: str, years: int = BASIS_YEARS) -> bool:
    """True only when the basis is `years` consecutive fiscal years AND each carries a reported SBC figure (a reported
    0 counts; a missing figure is unknown, not zero — R3-K3 / R4-02)."""
    rows = basis_rows(hist_fcf, ticker, years)
    if not basis_complete(rows, years):
        return False
    return all(s in (OBSERVED, CONFIRMED_ZERO)
               for s in fcfe_deduction_status(hist_fcf, ticker, years)["stock_based_comp"]["by_year"].values())


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


def terminal_value_share(fcfe, growth, discount_rate, terminal_growth=TERMINAL_GROWTH, years=EXPLICIT_YEARS):
    """Share of the DCF value that comes from the terminal value (0-1); None when not computable."""
    fcfe = _finite(fcfe)
    if not fcfe or fcfe <= 0 or discount_rate <= terminal_growth:
        return None
    total = dcf_equity_value(fcfe, growth, discount_rate, terminal_growth, years)
    pv, cf = 0.0, float(fcfe)
    for t in range(1, years + 1):
        cf *= 1 + growth + (terminal_growth - growth) * (t - 1) / max(years - 1, 1)
        pv += cf / (1 + discount_rate) ** t
    return 1 - pv / total if total > 0 else None


def _scenario_flows(fcfe, growth, r, tg, years=EXPLICIT_YEARS):
    """Per-period cash flows of the DCF (t = 1..years) and the terminal value at `years` (undiscounted)."""
    cf, flows = float(fcfe), []
    for t in range(1, years + 1):
        cf *= 1 + growth + (tg - growth) * (t - 1) / max(years - 1, 1)
        flows.append(cf)
    return flows, flows[-1] * (1 + tg) / (r - tg)


def scenario_irr(fcfe, shares, growth, discount_rate, terminal_growth, price, horizon: int = 4,
                 years: int = EXPLICIT_YEARS) -> Optional[float]:
    """
    Holding-period IRR of buying at `price` if the scenario materialises (audit A06): per-share cash flows to equity in
    years 1..H, plus the model's own value of everything after year H, received at H. Assumes cash flow to equity is
    distributed or reinvested at the cost of equity, a constant share count (SBC is already deducted from the cash
    flow) and no re-rating beyond the model. Property: price == scenario value  ->  IRR == discount rate.
    """
    fcfe, shares, price = _finite(fcfe), _finite(shares), _finite(price)
    if not fcfe or fcfe <= 0 or not shares or shares <= 0 or not price or price <= 0 or discount_rate <= terminal_growth:
        return None
    flows, tv = _scenario_flows(fcfe, growth, discount_rate, terminal_growth, years)
    H = int(min(max(horizon, 1), years))
    per_share = [f / shares for f in flows]
    exit_value = (sum(per_share[t - 1] / (1 + discount_rate) ** (t - H) for t in range(H + 1, years + 1))
                  + tv / shares / (1 + discount_rate) ** (years - H))
    cfs = per_share[:H]

    def npv(x):
        return sum(c / (1 + x) ** t for t, c in enumerate(cfs, 1)) + exit_value / (1 + x) ** H - price

    lo, hi = -0.95, 3.0
    if npv(lo) < 0 or npv(hi) > 0:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if npv(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


SCENARIO_PROBABILITIES = {"bear": 0.25, "base": 0.50, "bull": 0.25}
MAX_TERMINAL_SHARE = 0.80      # above this the value is mostly a perpetuity assumption (F-09)


def scenario_probabilities() -> dict:
    p = _valuation_config().get("scenario_probabilities") or SCENARIO_PROBABILITIES
    tot = sum(float(p.get(k, 0)) for k in ("bear", "base", "bull"))
    return {k: float(p.get(k, 0)) / tot for k in ("bear", "base", "bull")} if tot > 0 else dict(SCENARIO_PROBABILITIES)


def expected_value(scen: dict) -> Optional[float]:
    """Probability-weighted value of the bear / base / bull scenarios (None if any is missing)."""
    if not scen:
        return None
    vals = {k: (scen[k].value_per_share if k in scen else None) for k in ("bear", "base", "bull")}
    if any(v is None for v in vals.values()):
        return None
    p = scenario_probabilities()
    return sum(p[k] * vals[k] for k in vals)


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


def valuation_verdict(margin_of_safety: Optional[float], required: Optional[float] = None) -> str:
    """Margin of safety = value / price − 1. Undervalued only with the REQUIRED cushion (config, read at the call)."""
    required = _configured("required_margin_of_safety", 0.25) if required is None else required
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


RI_FADE_YEARS = 10                # residual-income horizon over which ROE converges to the cost of equity
DEFAULT_PAYOUT = 0.5              # share of earnings paid out (dividends + buybacks) when the provider does not say


def residual_income_value(book_per_share, roe, discount_rate, growth, payout=DEFAULT_PAYOUT,
                          years: int = RI_FADE_YEARS) -> Optional[float]:
    """
    Multi-stage residual-income value per share of a bank / insurer:
        V = B0 + Σ_{t=1..N} (ROE_t − r) · B_{t−1} / (1 + r)^t,
    with ROE fading LINEARLY from today's normalised ROE to the cost of equity over N years (competition erodes
    excess returns; nothing is capitalised for ever) and book value growing by retained earnings,
    B_t = B_{t−1} · (1 + ROE_t · (1 − payout)). After year N the franchise earns its cost of equity, so it adds nothing
    beyond book. A bank earning below its cost of equity is valued BELOW book (it destroys value while it converges).
    None when inputs are missing or r ≤ g.
    """
    bv, roe, r, g = _finite(book_per_share), _finite(roe), _finite(discount_rate), _finite(growth)
    if bv is None or bv <= 0 or roe is None or r is None or g is None or r <= g:
        return None
    keep = 1.0 - float(np.clip(_finite(payout) if _finite(payout) is not None else DEFAULT_PAYOUT, 0.0, 1.0))
    value, book = bv, bv
    for t in range(1, int(years) + 1):
        roe_t = roe + (r - roe) * t / years
        value += (roe_t - r) * book / (1 + r) ** t
        book *= 1 + max(roe_t, 0.0) * keep
    return float(value)


def payout_ratio(meta, price=None) -> Optional[float]:
    """(Dividends + net buybacks) / earnings = net payout yield / earnings yield = net payout yield × P/E.
    Both inputs are currency-free (audit F09: price is EUR while trailing EPS stays in the financial currency, so
    price / EPS mixed currencies). No positive P/E (losses) → unknown."""
    pe, npy = _finite(meta.get("pe_ratio")), _finite(meta.get("net_payout_yield_pct"))
    if pe is None or pe <= 0 or npy is None:
        return None
    return float(np.clip(npy / 100 * pe, 0.0, 1.0))


def pb_roe_peer_fit(companies: Optional[pd.DataFrame], ticker: str, min_peers: int = 8) -> Optional[dict]:
    """Relative value of a financial: P/B regressed on ROE across its sector peers (banks with banks, insurers with
    insurers). {fitted_pb, actual_pb, residual_pct, slope, r2, n} — a bank trading below the P/B its ROE earns in the
    peer group is cheap RELATIVE to peers (not necessarily cheap). None with too few usable peers."""
    if companies is None or companies.empty or not {"ticker", "sector", "price_to_book", "roe"} <= set(companies.columns):
        return None
    c = companies.drop_duplicates("ticker")
    me = c[c["ticker"] == ticker]
    if me.empty:
        return None
    sector = str(me["sector"].iloc[0])
    peers = c[c["sector"].astype(str) == sector][["ticker", "price_to_book", "roe"]].copy()
    peers["pb"], peers["roe_"] = pd.to_numeric(peers["price_to_book"], errors="coerce"), pd.to_numeric(peers["roe"], errors="coerce")
    peers = peers[(peers["pb"] > 0) & (peers["pb"] < 10) & peers["roe_"].between(-0.2, 0.5)]
    mine = peers[peers["ticker"] == ticker]
    if len(peers) < min_peers or mine.empty:
        return None
    slope, intercept = np.polyfit(peers["roe_"], peers["pb"], 1)
    fitted = intercept + slope * peers["roe_"]
    ss_res, ss_tot = float(((peers["pb"] - fitted) ** 2).sum()), float(((peers["pb"] - peers["pb"].mean()) ** 2).sum())
    my_fit = float(intercept + slope * mine["roe_"].iloc[0])
    actual = float(mine["pb"].iloc[0])
    return {"sector": sector, "fitted_pb": my_fit, "actual_pb": actual, "roe": float(mine["roe_"].iloc[0]),
            "residual_pct": (actual / my_fit - 1) * 100 if my_fit > 0 else None, "slope": float(slope),
            "intercept": float(intercept), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else None, "n": int(len(peers)),
            "peers": peers[["ticker", "pb", "roe_"]].rename(columns={"roe_": "roe"})}


def normalized_roe(annual_fin: Optional[pd.DataFrame], ticker: str, fallback=None, years: int = 4) -> Optional[float]:
    """Median of the last `years` annual net income / equity (through the cycle), clipped to [0, 25%]."""
    roes = []
    if annual_fin is not None and not annual_fin.empty and {"net_income", "total_equity"} <= set(annual_fin.columns):
        t = annual_fin[annual_fin["ticker"] == ticker].sort_values("year").tail(years)
        roes = [n / e for n, e in zip(t["net_income"], t["total_equity"])
                if _finite(n) is not None and _finite(e) is not None and e > 0]
    r = float(np.median(roes)) if len(roes) >= 2 else _finite(fallback)
    return None if r is None else float(np.clip(r, 0.0, 0.25))


CAPITAL_DEFAULTS = {"bank_min_cet1_pct": 11.0, "bank_max_npl_pct": 4.0, "insurer_min_solvency_pct": 150.0}


def capital_inputs_check(ticker: str, meta) -> tuple:
    """(reliable, note) for a bank / insurer from decision_rules.yaml → valuation.bank_capital_inputs[TICKER]:
    banks need cet1_pct ≥ bank_min_cet1_pct AND npl_pct ≤ bank_max_npl_pct; insurers need solvency_ratio_pct ≥
    insurer_min_solvency_pct. Every entry needs `source` and `as_of` (latest annual / Pillar 3 / SFCR report). Missing,
    unsourced or failing inputs keep the justified-P/B value as context only."""
    cfg = _valuation_config()
    th = {k: float(cfg.get(k, v)) for k, v in CAPITAL_DEFAULTS.items()}
    entry = ((cfg.get("bank_capital_inputs") or {}).get(str(ticker)) or {}) if ticker else {}
    if not entry or not str(entry.get("source") or "").strip() or not str(entry.get("as_of") or "").strip():
        return False, ("No sourced capital-adequacy / asset-quality inputs (CET1, NPLs; solvency ratio for insurers) — "
                       "the justified-P/B value is context only, not used for BUY/AVOID. Record them with source and "
                       "date in decision_rules.yaml → valuation.bank_capital_inputs.")
    # Audit F10: the date is evidence too — it must parse, not lie in the future, and not be older than the latest
    # annual / Pillar 3 cycle (bank_capital_max_age_days, default 400)
    as_of = pd.to_datetime(str(entry.get("as_of")), errors="coerce")
    max_age = int(cfg.get("bank_capital_max_age_days", 400))
    today = pd.Timestamp(date.today())
    if pd.isna(as_of):
        return False, f"Capital inputs dated {entry.get('as_of')!r} — not a date; not used for BUY/AVOID."
    if as_of > today:
        return False, f"Capital inputs dated {as_of:%Y-%m-%d}, in the future — not used for BUY/AVOID."
    if (today - as_of).days > max_age:
        return False, (f"Capital inputs from {as_of:%Y-%m-%d} are {(today - as_of).days} days old (limit {max_age}) — "
                       f"update them from the latest report; not used for BUY/AVOID.")
    insurer = "insur" in str(meta.get("sector") or "").lower() or "insur" in str(meta.get("industry") or "").lower()
    if insurer:
        sol = _finite(entry.get("solvency_ratio_pct"))
        if sol is None or sol < th["insurer_min_solvency_pct"]:
            return False, (f"Solvency ratio {sol if sol is not None else 'unknown'}% vs the "
                           f"{th['insurer_min_solvency_pct']:.0f}% floor ({entry['source']}, {entry['as_of']}) — "
                           f"not used for BUY/AVOID.")
        return True, None
    cet1, npl = _finite(entry.get("cet1_pct")), _finite(entry.get("npl_pct"))
    if cet1 is None or npl is None:
        return False, f"CET1 and NPL ratio both required ({entry['source']}) — not used for BUY/AVOID."
    if cet1 < th["bank_min_cet1_pct"] or npl > th["bank_max_npl_pct"]:
        return False, (f"CET1 {cet1:.1f}% (floor {th['bank_min_cet1_pct']:.0f}%) / NPL {npl:.1f}% (cap "
                       f"{th['bank_max_npl_pct']:.0f}%) per {entry['source']}, {entry['as_of']} — capital or asset "
                       f"quality fails; not used for BUY/AVOID.")
    return True, None


def _financial_inputs(meta, price: float, annual_fin, ticker: str, coe: float, tg: float, ass: dict) -> dict:
    mcap = _finite(meta.get("market_cap"))
    shares = mcap / price if mcap and price else None
    pb = _finite(meta.get("price_to_book"))
    bvps = price / pb if pb and pb > 0 else None
    roe_norm = normalized_roe(annual_fin, ticker, fallback=meta.get("roe"))
    # Residual income (P3/b): today's normalised ROE fades LINEARLY to the cost of equity over RI_FADE_YEARS, book grows
    # by retained earnings. (It replaced a constant-ROE justified P/B with a fixed 30% fade, which still capitalised an
    # above-cost ROE for ever and could not value a franchise earning below its cost of equity.)
    roe = roe_norm
    payout = payout_ratio(meta, price)
    scen = {}
    for name, d_roe, d_r in (("bear", -0.03, +0.01), ("base", 0.0, 0.0), ("bull", +0.03, -0.01)):
        r = max(coe + d_r, tg + 0.01)
        rr = None if roe is None else max(roe + d_roe, 0.0)
        scen[name] = Scenario(name, rr if rr is not None else 0.0, r,
                              residual_income_value(bvps, rr, r, tg, payout if payout is not None else DEFAULT_PAYOUT))
    base, bear, bull = (scen[k].value_per_share for k in ("base", "bear", "bull"))
    implied = tg + (price / bvps) * (coe - tg) if bvps else None          # perpetual ROE the price already assumes
    if base is None:
        reliable, note = False, ("Book value / ROE is missing — a residual-income value cannot be computed. Judge on "
                                 "P/B, ROE trend and credit quality.")
    elif roe is not None and roe <= tg:
        reliable, note = False, (f"Normalised ROE {roe:.1%} is not above long-run growth — the franchise destroys value "
                                 f"while it lasts (value below book). Judge on P/B, ROE trend and credit quality.")
    elif base / price - 1 > IMPLAUSIBLE_MARGIN:
        reliable, note = False, "Value is >2.5x the price — verify book value and ROE before acting."
    elif implied is not None and implied > 0.35:
        reliable, note = False, (f"The price implies a perpetual ROE of {implied:.0%} — beyond what a lender or insurer "
                                 f"can sustain; the model is not informative here.")
    elif _required("require_bank_capital_inputs"):
        # Four benign years of ROE capitalised on book value is not an underwriting of a lender or insurer (R3-K3).
        # Issuer-level capital / asset-quality inputs with a source unlock the model only when they pass (P1/3).
        reliable, note = capital_inputs_check(ticker, meta)
    else:
        reliable, note = True, None
    return {
        "model": "residual_income", "fcfe": None, "fcf_source": "n/a (bank / insurer: cash flows include deposits and float)",
        "shares": shares, "growth": roe or 0.0, "payout": payout,
        "growth_sources": [f"normalised ROE {roe_norm:.1%} (median of up to 4 annual ROEs), fading to the cost of equity "
                           f"over {RI_FADE_YEARS} years" if roe_norm is not None else "ROE unavailable"],
        "reliable": reliable, "note": note,
        "model_note": (f"Residual income: book value per share plus the discounted excess return (ROE − cost of equity) "
                       f"on book, with ROE fading linearly to the cost of equity over {RI_FADE_YEARS} years and book "
                       f"growing by retained earnings (payout "
                       f"{f'{payout:.0%}' if payout is not None else f'{DEFAULT_PAYOUT:.0%} assumed'}). Still sensitive "
                       f"to the credit or underwriting cycle."),
        "risk_free": ass["risk_free"], "cost_of_equity": coe, "terminal_growth": tg, "currency": ass["currency"],
        "rate_source": ass["source"], "scenarios": scen, "base": base, "bear": bear, "bull": bull,
        # audit F33: what the price implies here is a perpetual ROE (justified-P/B inverse), not a starting FCF growth
        "implied_growth": None, "implied_roe": implied, "roe": roe, "roe_normalised": roe_norm, "book_per_share": bvps,
        "justified_pb": (roe - tg) / (coe - tg) if roe is not None and coe > tg else None,
        "sbc_adjusted": False, "fcfe_adjustments": [], "terminal_share": None, "growth_note": None, "irr": {},
        "expected_value": expected_value(scen), "probabilities": scenario_probabilities(),
    }


def latest_statement_fcf(hist_fcf: pd.DataFrame, ticker: str) -> Optional[float]:
    """Most recent annual cash-flow-statement FCF (EUR) for the ticker, if any."""
    if hist_fcf is None or hist_fcf.empty or "free_cash_flow" not in hist_fcf.columns:
        return None
    rows = hist_fcf[(hist_fcf["ticker"] == ticker) & hist_fcf["free_cash_flow"].notna()].sort_values("year")
    return _finite(rows["free_cash_flow"].iloc[-1]) if not rows.empty else None


def uses_book_model(meta, hist_fcf=None, ticker=None) -> bool:
    """True for banks, insurers, brokers and other balance-sheet financials (residual income on book); False for everything the
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


def holding_horizon() -> int:
    """valuation.horizon_years — through the versioned config cache (PERF-05: it re-parsed the YAML once per issuer)."""
    try:
        return int(_valuation_config().get("horizon_years", 4))
    except (TypeError, ValueError):
        return 4


def _irrs(scen, fcfe, shares, price, tg) -> dict:
    """Bear / base / bull IRR (fractions) and their probability-weighted average."""
    if not scen:
        return {}
    H = holding_horizon()
    out = {k: scenario_irr(fcfe, shares, s.growth, s.discount_rate, tg, price, H) for k, s in scen.items()}
    p = scenario_probabilities()
    out["expected"] = (sum(p[k] * out[k] for k in ("bear", "base", "bull"))
                       if all(out.get(k) is not None for k in ("bear", "base", "bull")) else None)
    out["horizon_years"] = H
    return out


def valuation_inputs(meta, price: float, hist_fcf: pd.DataFrame, ticker: str, macro: dict,
                     annual_fin: Optional[pd.DataFrame] = None) -> dict:
    """Valuation of one issuer. Within this call the issuer's basis rows and deduction status are prepared once and
    reused (PERF-05: they were rebuilt ~7× per issuer); the memo lives only for the call, so nothing can go stale."""
    if _MEMO.get() is not None:
        return _valuation_inputs(meta, price, hist_fcf, ticker, macro, annual_fin)
    token = _MEMO.set({})
    try:
        return _valuation_inputs(meta, price, hist_fcf, ticker, macro, annual_fin)
    finally:
        _MEMO.reset(token)


def _valuation_inputs(meta, price: float, hist_fcf: pd.DataFrame, ticker: str, macro: dict,
                      annual_fin: Optional[pd.DataFrame] = None) -> dict:
    """
    Company-anchored intrinsic value inputs + scenarios + reverse model (EUR, per share).

    Industrials and tech: FCFE DCF. Banks, insurers and other financials: residual income on book value (ROE fading to the cost of equity).
    The risk-free rate and terminal growth come from the currency the cash flows are earned in.
    """
    ass = currency_assumptions(meta.get("currency"), macro)
    rf, tg = ass["risk_free"], ass["terminal_growth"]
    coe = cost_of_equity(meta.get("beta"), risk_free=rf)
    if uses_book_model(meta, hist_fcf, ticker):
        return _financial_inputs(meta, price, annual_fin, ticker, coe, tg, ass)

    # Cash-flow-statement FCF after stock-based compensation (EUR) first; Yahoo's levered FCF only as a fallback
    dctx = deduction_context(meta, ticker)               # accounting-standard rules + disclosed assumptions
    statement_fcf = normalized_statement_fcf(hist_fcf, ticker, ctx=dctx)
    sbc = sbc_available(hist_fcf, ticker)
    fcfe = statement_fcf if statement_fcf is not None else _finite(meta.get("free_cashflow"))
    adj = fcfe_adjustments(hist_fcf, ticker, ctx=dctx)
    fcf_source = (("cash-flow statement FCF (OCF − capex − stock-based comp" if sbc
                   else "cash-flow statement FCF (OCF − capex; stock comp not available")
                  + ("".join(f" − {a.replace('_', ' ')}" for a in adj))
                  + ("), median of last 3 years" if adj else
                     "; interest/lease payments in financing NOT deducted — overstated for IFRS filers), median of last 3 years")
                  + _deduction_evidence_text(fcfe_deduction_status(hist_fcf, ticker, ctx=dctx))
                  if statement_fcf is not None else "Yahoo levered FCF")
    mcap = _finite(meta.get("market_cap"))
    shares = mcap / price if mcap and price else None
    hist_rows = _with_estimates(hist_fcf[hist_fcf["ticker"] == ticker].sort_values("year")).tail(5) \
        if hist_fcf is not None and not hist_fcf.empty else pd.DataFrame(columns=["year", "free_cash_flow"])
    # Each fetch converts every year at one FX rate, which cancels in a CAGR; years kept from an earlier fetch carry
    # another rate (L606), so only the latest fetch counts. No fetch date -> FX unverified, source omitted.
    if "fetched_at" in hist_rows.columns and not hist_rows.empty:
        hist_rows = hist_rows[hist_rows["fetched_at"] == hist_rows["fetched_at"].iloc[-1]]
        hist_cagr = cagr_by_year(hist_rows["year"], _after_sbc(hist_rows, dctx, ticker))
    else:
        hist_cagr = None
    growth, sources = anchor_growth(meta.get("revenue_growth"), meta.get("earnings_growth"),
                                    hist_cagr, revenue_cagr=revenue_cagr(annual_fin, ticker))
    valuable = bool(fcfe and fcfe > 0 and shares)
    scen = dcf_scenarios(fcfe, shares, growth, coe, tg) if valuable else {}
    base = scen["base"].value_per_share if scen else None
    bull = scen["bull"].value_per_share if scen else None
    implied = reverse_dcf_growth(price, fcfe, shares, coe, tg) if valuable else None
    reliable, note = dcf_reliability(None, price, base, bull, implied)       # sector test already done by uses_book_model
    _sect = f"{meta.get('sector') or ''} {meta.get('industry') or ''}".lower()
    if reliable and any(k in _sect for k in REIT_KEYWORDS):
        # CFO − capex leaves out property ACQUISITIONS (investing, funded by new equity / debt), so a REIT's "FCFE" is
        # not what owners can take out — the DCF overstates value (Realty Income / VICI showed +130–140%). Context only
        # until an FFO / AFFO / NAV model exists.
        reliable, note = False, ("REIT: an FCFE DCF ignores property acquisitions funded by new equity and debt, so it "
                                 "overstates distributable cash — value shown as context only (an AFFO / NAV model is "
                                 "needed). Not used for BUY/AVOID.")
    rev_cagr = revenue_cagr(annual_fin, ticker)
    tv_share = terminal_value_share(fcfe, growth, coe, tg) if valuable else None
    growth_note = None
    if valuable and rev_cagr is None:
        growth_note = ("Growth anchor is not a multi-year revenue CAGR (fewer than 3 annual statements or unverified "
                       "currency/FX) — it rests on single-quarter YoY figures; treat the base value as low-evidence.")
    if reliable and tv_share is not None and tv_share > MAX_TERMINAL_SHARE:
        reliable, note = False, (f"{tv_share:.0%} of the value is terminal value — the result is a perpetuity "
                                 f"assumption, not a forecast. Not used for BUY/AVOID.")
    # Unverified claims ahead of equity (R3-K3): an unknown deduction is not a confirmed zero, so the value is shown
    # but cannot drive BUY / AVOID. Switch: valuation.require_verified_fcfe_deductions in decision_rules.yaml.
    # R4-02: EVERY applicable deduction must be known for EVERY one of 3 consecutive fiscal years — one verified item
    # no longer lets an unknown one through, and the filer's country is not evidence of its cash-flow classification.
    deduction_status = fcfe_deduction_status(hist_fcf, ticker, ctx=dctx)
    if reliable and valuable and _required("require_verified_fcfe_deductions"):
        _unknown = [k for k, s in deduction_status.items() if s["status"] == UNKNOWN]
        if statement_fcf is None:
            reliable, note = False, ("Only the provider's levered FCF is available — not reconciled to a cash-flow "
                                     "statement. Not used for BUY/AVOID.")
        elif not basis_complete(basis_rows(hist_fcf, ticker)):
            reliable, note = False, (f"Fewer than {BASIS_YEARS} consecutive annual cash-flow statements — the "
                                     f"normalised cash flow to equity is not established. Not used for BUY/AVOID.")
        elif _unknown:
            _names = {"stock_based_comp": "stock-based compensation", "interest_paid_cff": "interest paid in financing",
                      "lease_principal_paid": "lease principal"}
            reliable, note = False, (
                f"Unknown for at least one of the last {BASIS_YEARS} years: {', '.join(_names[k] for k in _unknown)} — "
                "unknown is not zero, so the cash flow to equity may be overstated. Not used for BUY/AVOID until each "
                "item is reported or recorded as not applicable with a source (decision_rules.yaml → "
                "valuation.fcfe_not_applicable).")
    # E6: the cash flow above assumes NET BORROWING = 0 (maturing debt is refinanced); it has no debt schedule.
    # Refinancing stress: if the company has been repaying more debt than it raised, by at least half of the FCFE,
    # the value depends on continued access to credit → not used for BUY/AVOID. Net borrowing is never ADDED.
    refi = refinancing_stress(hist_fcf, ticker, fcfe)
    fcf_source += " · assumes net borrowing = 0 (debt refinanced at maturity)"
    if reliable and valuable and refi["share_of_fcfe"] is not None and refi["share_of_fcfe"] >= REFINANCING_STRESS_SHARE:
        reliable, note = False, (f"Net debt repayment (median {refi['net_repayment']:,.0f} EUR a year) is "
                                 f"{refi['share_of_fcfe']:.0%} of the cash flow to equity — the value assumes the debt "
                                 f"can keep being refinanced. Not used for BUY/AVOID without a debt schedule.")
    return {
        "model": "dcf", "fcfe": fcfe, "fcf_source": fcf_source, "shares": shares, "refinancing": refi,
        "growth": growth, "growth_sources": sources, "reliable": reliable, "note": note,
        "risk_free": rf, "cost_of_equity": coe, "terminal_growth": tg, "currency": ass["currency"],
        "rate_source": ass["source"], "scenarios": scen,
        "base": base,
        "bear": scen["bear"].value_per_share if scen else None,
        "bull": bull,
        "implied_growth": implied, "sbc_adjusted": sbc,
        "fcfe_adjustments": adj, "fcfe_deduction_status": {k: s["status"] for k, s in deduction_status.items()},
        "fcfe_deduction_sources": {k: s.get("source") for k, s in deduction_status.items()},
        "terminal_share": tv_share, "growth_note": growth_note,
        "irr": _irrs(scen, fcfe, shares, price, tg),
        "expected_value": expected_value(scen) if scen else None, "probabilities": scenario_probabilities(),
    }
