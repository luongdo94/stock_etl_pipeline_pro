"""Highlights of a company's latest reported quarter (pure): what it reported, against what was expected and a year ago.

Built only from stored data: quarterly statements (marts.dim_quarterly_financials, amounts in EUR with the statement
currency and its EUR rate), quarterly free cash flow (EUR), the EPS pair of the report (raw.earnings_events, reporting
currency) and, when the ETL recorded it before the report, the revenue consensus (raw.estimate_snapshots, estimate
currency). Growth uses the local-currency rates the mart computes, so a moving exchange rate is not reported as growth.
Anything not stored is None — shown as N/A, never 0. Guidance, segments and management commentary are not in these
tables and are not invented here.
"""
import math
from typing import Optional

import pandas as pd

REPORT_WINDOW_DAYS = 100          # a quarter is reported within this many days of its end
IN_LINE = 0.005                   # |actual / estimate − 1| below this is "in line"


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _vs(actual, estimate) -> Optional[str]:
    a, e = _num(actual), _num(estimate)
    if a is None or e is None or e == 0:
        return None
    r = a / e - 1 if e > 0 else (a - e) / abs(e)
    return "In line" if abs(r) < IN_LINE else ("Beat" if r > 0 else "Miss")


def _margin(part, whole) -> Optional[float]:
    p, w = _num(part), _num(whole)
    return p / w * 100 if p is not None and w not in (None, 0) else None


def _report_event(events: pd.DataFrame, ticker: str, period_end: pd.Timestamp) -> Optional[pd.Series]:
    """The earnings event that reported the quarter ending `period_end` (first one within the reporting window)."""
    if events is None or events.empty:
        return None
    e = events[events["ticker"] == ticker].copy()
    if e.empty:
        return None
    ts = pd.to_datetime(e["earnings_ts"], utc=True).dt.tz_convert(None)
    start = pd.Timestamp(period_end)
    hit = e[(ts > start) & (ts <= start + pd.Timedelta(days=REPORT_WINDOW_DAYS))]
    return None if hit.empty else hit.loc[pd.to_datetime(hit["earnings_ts"], utc=True).idxmin()]


def _consensus_before(snaps: Optional[pd.DataFrame], ticker: str, report_ts) -> Optional[dict]:
    if snaps is None or snaps.empty or report_ts is None:
        return None
    ts = pd.Timestamp(report_ts)
    ts = (ts.tz_convert(None) if ts.tzinfo is not None else ts).normalize()
    s = snaps[(snaps["ticker"] == ticker) & (pd.to_datetime(snaps["as_of"]) < ts)]
    if s.empty:
        return None
    row = s.loc[pd.to_datetime(s["as_of"]).idxmax()]
    return {"as_of": pd.Timestamp(row["as_of"]).date(), "revenue": _num(row.get("rev_est_0q_avg")),
            "eps": _num(row.get("eps_est_0q_avg"))}


def latest_report(ticker: str, quarterly: pd.DataFrame, fcf_q: Optional[pd.DataFrame] = None,
                  events: Optional[pd.DataFrame] = None, snaps: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """{period, reported, currency, lines: [{label, value, unit, estimate, vs, yoy_pct, note}], sources} or None."""
    if quarterly is None or quarterly.empty or "ticker" not in quarterly.columns:
        return None
    q = quarterly[quarterly["ticker"] == ticker].sort_values("report_date")
    if q.empty:
        return None
    r = q.iloc[-1]
    end = pd.Timestamp(r["report_date"])
    year_ago = q[(pd.to_datetime(q["report_date"]) - (end - pd.DateOffset(years=1))).abs() <= pd.Timedelta(days=15)]
    ya = year_ago.iloc[-1] if not year_ago.empty else None
    ev = _report_event(events, ticker, end)
    reported = pd.Timestamp(ev["earnings_ts"]) if ev is not None else None
    cons = _consensus_before(snaps, ticker, reported)
    fx = _num(r.get("fx_to_eur"))
    ccy = r.get("src_currency") if isinstance(r.get("src_currency"), str) else None

    rev, ni, oi = _num(r.get("revenue")), _num(r.get("net_income")), _num(r.get("operating_income"))
    lines = []
    # revenue (EUR) and, when recorded before the report, the consensus — compared in the estimate's own currency
    rev_est_eur = cons["revenue"] * fx if cons and cons["revenue"] is not None and fx else None
    lines.append({"label": "Revenue", "value": rev, "unit": "EUR", "estimate": rev_est_eur,
                  "vs": _vs(rev, rev_est_eur), "yoy_pct": _num(r.get("revenue_growth_yoy_pct")),
                  "note": (f"consensus recorded {cons['as_of']}" if rev_est_eur is not None else
                           "no consensus recorded before this report")})
    eps_a = _num(ev["eps_actual"]) if ev is not None else None
    eps_e = _num(ev["eps_estimate"]) if ev is not None else None
    lines.append({"label": "EPS", "value": eps_a, "unit": ccy or "reporting currency", "estimate": eps_e,
                  "vs": _vs(eps_a, eps_e), "yoy_pct": _num(r.get("eps_growth_yoy_pct")),
                  "note": "" if ev is not None else "no report event stored for this quarter"})
    lines.append({"label": "Operating income", "value": oi, "unit": "EUR", "estimate": None, "vs": None,
                  "yoy_pct": _num(r.get("operating_income_growth_yoy_pct")), "note": ""})
    om, om_ya = _margin(oi, rev), (_margin(ya.get("operating_income"), ya.get("revenue")) if ya is not None else None)
    lines.append({"label": "Operating margin", "value": om, "unit": "%", "estimate": None, "vs": None,
                  "yoy_bps": (om - om_ya) * 100 if om is not None and om_ya is not None else None, "note": ""})
    lines.append({"label": "Net income", "value": ni, "unit": "EUR", "estimate": None, "vs": None,
                  "yoy_pct": _num(r.get("net_income_growth_yoy_pct")), "note": ""})
    fcf = fcf_ya = None
    if fcf_q is not None and not fcf_q.empty and {"ticker", "year", "quarter"} <= set(fcf_q.columns):
        f = fcf_q[fcf_q["ticker"] == ticker]
        cur = f[(f["year"] == end.year) & (f["quarter"] == (end.month - 1) // 3 + 1)]
        prev = f[(f["year"] == end.year - 1) & (f["quarter"] == (end.month - 1) // 3 + 1)]
        fcf = _num(cur["free_cash_flow"].iloc[0]) if not cur.empty else None
        fcf_ya = _num(prev["free_cash_flow"].iloc[0]) if not prev.empty else None
    lines.append({"label": "Free cash flow", "value": fcf, "unit": "EUR", "estimate": None, "vs": None,
                  "yoy_pct": (fcf / fcf_ya - 1) * 100 if fcf is not None and fcf_ya not in (None, 0) and fcf_ya > 0 else None,
                  "note": "EUR at each period's rate: the change includes currency moves"})
    return {"period": f"Q{(end.month - 1) // 3 + 1} {end.year} (quarter ended {end:%Y-%m-%d})",
            "reported": reported, "currency": ccy, "lines": lines,
            "sources": "Stored statements (Yahoo) converted to EUR, the report's EPS pair, and the consensus the ETL "
                       "recorded before the report when available."}


def facts_line(h: Optional[dict]) -> Optional[str]:
    """One line of the stored figures for the AI note's facts (so the note quotes the same numbers as the table)."""
    if not h:
        return None
    parts = []
    for line in h["lines"]:
        v = line["value"]
        if v is None:
            shown = "N/A"
        elif line["unit"] == "%":
            shown = f"{v:.1f}%"
        elif line["unit"] == "EUR":
            shown = fmt_amount(v)
        else:
            shown = f"{v:.2f} {line['unit']}"
        parts.append(f"{line['label']} {shown}")
    return f"- Stored figures for {h['period']}: " + "; ".join(parts) + "."


def fmt_amount(v: Optional[float]) -> str:
    """€12.87B / €410.2M / N/A."""
    if v is None:
        return "N/A"
    a = abs(v)
    s = f"{a / 1e9:,.2f}B" if a >= 1e9 else f"{a / 1e6:,.1f}M" if a >= 1e6 else f"{a:,.0f}"
    return ("-" if v < 0 else "") + "€" + s
