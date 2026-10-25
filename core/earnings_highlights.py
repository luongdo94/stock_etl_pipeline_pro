"""Highlights of a company's latest reported quarter (pure): what it reported, against what was expected and a year ago.

Built only from stored data: quarterly statements (marts.dim_quarterly_financials, amounts in EUR with the statement
currency and its EUR rate), quarterly free cash flow (EUR), the EPS pair of the report (raw.earnings_events, reporting
currency) and the revenue consensus (raw.estimate_snapshots, estimate currency) — only a snapshot recorded AFTER the
quarter ended and BEFORE it was reported: the provider's current-quarter (0q) estimate only rolls to the next quarter
once a report is out, so in that window it can only be the quarter just reported. Any other snapshot leaves the
comparison blank (its quarter is unverified). Growth uses the local-currency rates the mart computes, so a moving
exchange rate is not reported as growth.
The consensus and the EPS pair are in the company's FINANCIAL currency (Yahoo's financialCurrency); they are converted
with the statement's EUR rate only when that currency is the statement's (K8-07 — 8 real tickers report statements in
another currency), else they stay N/A for EUR. EPS is shown in EUR (K8-10); Beat / Miss uses the local pair, and the
EPS change is against the report a year earlier from the same EPS series as the growth history (K8-08).
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


def _consensus_before(snaps: Optional[pd.DataFrame], ticker: str, report_ts, period_end=None) -> Optional[dict]:
    """The last snapshot recorded after `period_end` and before the report day: the 0q consensus of that quarter."""
    if snaps is None or snaps.empty or report_ts is None or period_end is None:
        return None
    ts = pd.Timestamp(report_ts)
    ts = (ts.tz_convert(None) if ts.tzinfo is not None else ts).normalize()
    as_of = pd.to_datetime(snaps["as_of"])
    s = snaps[(snaps["ticker"] == ticker) & (as_of > pd.Timestamp(period_end)) & (as_of < ts)]
    if s.empty:
        return None
    row = s.loc[pd.to_datetime(s["as_of"]).idxmax()]
    return {"as_of": pd.Timestamp(row["as_of"]).date(), "revenue": _num(row.get("rev_est_0q_avg")),
            "eps": _num(row.get("eps_est_0q_avg"))}


def _eps_year_ago(events: pd.DataFrame, ticker: str, ts) -> Optional[float]:
    """Reported EPS of the report about a year before `ts` (330-400 days) — the growth history's basis."""
    if events is None or events.empty or ts is None:
        return None
    e = events[(events["ticker"] == ticker) & events["eps_actual"].notna()]
    t = pd.to_datetime(e["earnings_ts"], utc=True).dt.tz_convert(None)
    ts = pd.Timestamp(ts)
    ts = ts.tz_convert(None) if ts.tzinfo is not None else ts
    ago = e[(t >= ts - pd.Timedelta(days=400)) & (t <= ts - pd.Timedelta(days=330))]
    return _num(ago.loc[pd.to_datetime(ago["earnings_ts"], utc=True).idxmax(), "eps_actual"]) if not ago.empty else None


def _eps_yoy(eps, eps_ago) -> Optional[float]:
    return (eps / eps_ago - 1) * 100 if eps is not None and eps_ago is not None and eps_ago > 0 else None


def latest_report(ticker: str, quarterly: pd.DataFrame, fcf_q: Optional[pd.DataFrame] = None,
                  events: Optional[pd.DataFrame] = None, snaps: Optional[pd.DataFrame] = None,
                  estimate_currency: Optional[str] = None) -> Optional[dict]:
    """{period, reported, currency, lines: [{label, value, unit, estimate, vs, yoy_pct, note}], sources} or None.
    estimate_currency — the company's financial currency (that of the consensus and the EPS pair); unknown → neither
    is converted to EUR."""
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
    cons = _consensus_before(snaps, ticker, reported, end)
    fx = _num(r.get("fx_to_eur"))
    ccy = r.get("src_currency") if isinstance(r.get("src_currency"), str) else None

    rev, ni, oi = _num(r.get("revenue")), _num(r.get("net_income")), _num(r.get("operating_income"))
    lines = []
    # the consensus of THIS quarter (recorded between its end and its report), converted like the actuals
    est_ccy = str(estimate_currency).strip() if isinstance(estimate_currency, str) and estimate_currency.strip() else None
    same_ccy = est_ccy is not None and ccy is not None and est_ccy == ccy.strip()
    ccy_note = ("" if same_ccy else f"estimate currency {est_ccy} is not the statement's {ccy}" if est_ccy and ccy
                else "estimate currency unknown")
    has_cons = bool(cons and cons["revenue"] is not None)
    rev_est_eur = cons["revenue"] * fx if has_cons and fx and same_ccy else None
    lines.append({"label": "Revenue", "value": rev, "unit": "EUR", "estimate": rev_est_eur,
                  "vs": _vs(rev, rev_est_eur), "yoy_pct": _num(r.get("revenue_growth_yoy_pct")),
                  "note": (f"consensus recorded {cons['as_of']}, after the quarter ended" if rev_est_eur is not None
                           else f"consensus not compared: {ccy_note or 'no EUR rate'}" if has_cons
                           else "no consensus recorded between quarter end and report (period unverified)")})
    eps_a = _num(ev["eps_actual"]) if ev is not None else None
    eps_e = _num(ev["eps_estimate"]) if ev is not None else None
    to_eur = fx if fx and same_ccy else None
    eps_note = ("no report event stored for this quarter" if ev is None
                else "" if to_eur else f"not shown in EUR: {ccy_note or 'no EUR rate'} (vs estimate uses the local pair)")
    lines.append({"label": "EPS", "value": eps_a * to_eur if eps_a is not None and to_eur else None, "unit": "EUR/share",
                  "estimate": eps_e * to_eur if eps_e is not None and to_eur else None,
                  "vs": _vs(eps_a, eps_e), "yoy_pct": _eps_yoy(eps_a, _eps_year_ago(events, ticker, reported)),
                  "note": eps_note})
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
            "sources": "Stored statements (Yahoo) converted to EUR, the report's EPS pair, and the revenue consensus "
                       "recorded between quarter end and report when available."}


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
        elif line["unit"] == "EUR/share":
            shown = f"€{v:,.2f}/share"
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


def growth_history(ticker: str, events: Optional[pd.DataFrame], quarterly: Optional[pd.DataFrame], n: int = 8) -> pd.DataFrame:
    """The last `n` reports with EPS and revenue growth against the same quarter a year earlier.

    EPS comes from the report events (24 quarters stored, reporting currency, split-adjusted by the provider): the change
    is against the report about a year before (330-400 days), and a year-ago EPS that is zero or negative gives no
    percentage. Revenue growth comes from the stored statements (only the quarters the provider still serves, so older
    rows stay blank) and is matched to the report by its quarter end. Anything not stored stays NaN, never 0.
    Columns: Reported, Quarter end, EPS, EPS year ago, EPS YoY %, Revenue YoY %."""
    cols = ["Reported", "Quarter end", "EPS", "EPS year ago", "EPS YoY %", "Revenue YoY %"]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)
    e = events[(events["ticker"] == ticker) & events["eps_actual"].notna()].copy()
    if e.empty:
        return pd.DataFrame(columns=cols)
    e["_ts"] = pd.to_datetime(e["earnings_ts"], utc=True).dt.tz_convert(None)
    e = e.sort_values("_ts").reset_index(drop=True)
    q = pd.DataFrame()
    if quarterly is not None and not quarterly.empty and "ticker" in quarterly.columns:
        q = quarterly[quarterly["ticker"] == ticker].copy()
        q["report_date"] = pd.to_datetime(q["report_date"])
    rows = []
    for i in range(len(e)):
        ts, eps = e.loc[i, "_ts"], _num(e.loc[i, "eps_actual"])
        eps_ago = _eps_year_ago(e, ticker, ts)
        yoy = _eps_yoy(eps, eps_ago)
        end = rev = None
        if not q.empty:
            prior = q[(q["report_date"] < ts) & (q["report_date"] >= ts - pd.Timedelta(days=REPORT_WINDOW_DAYS))]
            if not prior.empty:
                last = prior.sort_values("report_date").iloc[-1]
                end, rev = last["report_date"], _num(last.get("revenue_growth_yoy_pct"))
        rows.append({"Reported": ts.normalize(), "Quarter end": end, "EPS": eps, "EPS year ago": eps_ago,
                     "EPS YoY %": yoy, "Revenue YoY %": rev})
    return pd.DataFrame(rows, columns=cols).tail(n).iloc[::-1].reset_index(drop=True)
