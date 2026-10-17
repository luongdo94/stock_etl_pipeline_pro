"""How fresh the data in use is (pure) — one answer for the sidebar, Data Health and the SIGNAL hub.

A successful ETL run is not fresh data (SIDEBAR_REFACTOR_REVIEW 3.3): this reports, separately, how old the newest
close is, what share of the universe has a recent close, how old the fundamentals are (and for what share of companies
that is known), when the last successful run started / finished and the data-quality state. Anything the inputs cannot
tell is UNKNOWN, never OK; the level is the worst of the parts, data quality included. This is a summary of the whole
universe — it does not confirm a single ticker or holding (the Decision checks its own price date).
"""
from datetime import date, datetime
from typing import Optional

import pandas as pd

from core.market_regime import INDICES, STALE_DAYS

PRICE_WARN_DAYS = 4          # newest close older than this (calendar days): prices are stale
FUNDAMENTALS_WARN_DAYS = 35  # = valuation.stale_fundamentals_days (decision_rules.yaml): older blocks a BUY
DATES_KNOWN_MIN = 0.50       # below this share of companies with a refresh date, fundamentals freshness is UNKNOWN
DATES_KNOWN_WARN = 0.90      # below this share it is WARN (the median describes only the dated part)
LEVELS = ("OK", "WARN", "STALE", "UNKNOWN")


def _age_days(d, today: date) -> Optional[int]:
    try:
        return (today - pd.Timestamp(d).date()).days
    except (TypeError, ValueError):
        return None


def data_freshness(prices: Optional[pd.DataFrame], companies: Optional[pd.DataFrame] = None,
                   etl_audit: Optional[pd.DataFrame] = None, dq_warnings: Optional[pd.DataFrame] = None,
                   today: Optional[date] = None) -> dict:
    """{"level", "prices": {...}, "fundamentals": {...}, "etl": {...}, "dq": {...}, "notes": [..]} — see module doc."""
    today = today or date.today()
    notes, levels = [], []
    out = {"prices": {"as_of": None, "age_days": None, "fresh_share": None, "stale": None, "tickers": 0},
           "fundamentals": {"median_age_days": None, "oldest_age_days": None, "stale_share": None,
                            "known_share": None, "undated": None, "companies": 0},
           "etl": {"status": None, "last_success_at": None, "basis": None, "age_hours": None},
           "dq": {"state": "UNKNOWN", "critical": 0, "warnings": 0}}

    # prices: the newest close, and the share of stocks within STALE_DAYS of it
    if prices is not None and not prices.empty and {"date", "ticker"} <= set(prices.columns):
        stocks = prices[~prices["ticker"].isin(INDICES)]
        last = pd.to_datetime(stocks.groupby("ticker")["date"].max())
        newest = last.max()
        behind = (newest - last).dt.days > STALE_DAYS
        p = out["prices"]
        p.update(as_of=newest.date(), age_days=_age_days(newest, today), tickers=int(len(last)),
                 stale=int(behind.sum()), fresh_share=float(1 - behind.mean()) if len(last) else None)
        if p["age_days"] is not None and p["age_days"] > PRICE_WARN_DAYS:
            levels.append("STALE")
            notes.append(f"newest close is {p['age_days']} days old ({p['as_of']})")
        elif p["stale"]:
            levels.append("WARN" if p["fresh_share"] < 0.95 else "OK")
            notes.append(f"{p['stale']} of {p['tickers']} stocks without a recent close")
        else:
            levels.append("OK")
    else:
        levels.append("UNKNOWN")
        notes.append("no prices loaded")

    # fundamentals: when each company row was last refreshed (info_updated_at)
    if companies is not None and not companies.empty and "info_updated_at" in companies.columns:
        comp = companies[~companies["ticker"].isin(INDICES)] if "ticker" in companies.columns else companies
        all_upd = pd.to_datetime(comp["info_updated_at"], errors="coerce")
        upd = all_upd.dropna()
        f = out["fundamentals"]
        f.update(companies=int(len(all_upd)), undated=int(all_upd.isna().sum()),
                 known_share=float(len(upd) / len(all_upd)) if len(all_upd) else None)
        if len(upd):
            ages = (pd.Timestamp(today) - upd).dt.days
            # the stale share counts undated companies as not known to be current (over all companies)
            f.update(median_age_days=int(ages.median()), oldest_age_days=int(ages.max()),
                     stale_share=float((ages > FUNDAMENTALS_WARN_DAYS).sum() / len(all_upd)))
            if f["undated"]:
                notes.append(f"{f['undated']} of {f['companies']} companies without a fundamentals refresh date")
            if f["known_share"] < DATES_KNOWN_MIN:
                levels.append("UNKNOWN")
            elif f["median_age_days"] > FUNDAMENTALS_WARN_DAYS:
                levels.append("STALE")
                notes.append(f"fundamentals are {f['median_age_days']} days old (median)")
            elif f["stale_share"] > 0.10 or f["known_share"] < DATES_KNOWN_WARN:
                levels.append("WARN")
                if f["stale_share"] > 0.10:
                    notes.append(f"{f['stale_share']:.0%} of companies' fundamentals older than "
                                 f"{FUNDAMENTALS_WARN_DAYS} days")
        else:
            levels.append("UNKNOWN")
            notes.append("fundamentals refresh dates unknown")
    else:
        levels.append("UNKNOWN")
        notes.append("fundamentals refresh dates unknown")

    # ETL: the last SUCCESSFUL run's end (a start time says nothing about completion)
    if etl_audit is not None and not etl_audit.empty and "status" in etl_audit.columns:
        e = out["etl"]
        latest = etl_audit.sort_values("start_time", ascending=False) if "start_time" in etl_audit.columns else etl_audit
        e["status"] = str(latest.iloc[0]["status"])
        ok = etl_audit[etl_audit["status"] == "SUCCESS"]
        if "start_time" in ok.columns:
            ok = ok.sort_values("start_time", ascending=False)
        if not ok.empty:
            row = ok.iloc[0]
            # the end time when recorded; otherwise the start, labelled as such (a start says nothing about completion)
            end = pd.to_datetime(row.get("end_time"), errors="coerce") if "end_time" in ok.columns else pd.NaT
            start = pd.to_datetime(row.get("start_time"), errors="coerce")
            at, basis = (end, "end") if pd.notna(end) else (start, "start")
            if pd.notna(at):
                e.update(last_success_at=at.to_pydatetime(), basis=basis)
                e["age_hours"] = round((datetime.now() - e["last_success_at"]).total_seconds() / 3600, 1)
        if e["status"] != "SUCCESS":
            levels.append("WARN")
            notes.append(f"last ETL run {e['status']}")
        if e["last_success_at"] is None:
            levels.append("UNKNOWN")
            notes.append("no successful ETL run recorded")
    else:
        levels.append("UNKNOWN")
        notes.append("no ETL run recorded")

    # data quality: one row per check per run; no rows = checks unavailable (UNKNOWN), not clean
    if dq_warnings is not None and not dq_warnings.empty and "violations" in dq_warnings.columns:
        v = dq_warnings[dq_warnings["violations"] > 0]
        crit = int(v["is_critical"].sum()) if "is_critical" in v.columns and not v.empty else 0
        warn = len(v) - crit
        out["dq"] = {"state": "CRIT" if crit else ("WARN" if warn else "CLEAN"), "critical": crit, "warnings": warn}
        if crit:
            levels.append("WARN")
            notes.append(f"{crit} critical data-quality check(s) violated")
        elif warn:
            levels.append("WARN")
            notes.append(f"{warn} data-quality check(s) with violations")
        else:
            levels.append("OK")
    else:
        levels.append("UNKNOWN")                 # not run / not loaded is not clean
        notes.append("data-quality checks unavailable")

    order = {"STALE": 0, "UNKNOWN": 1, "WARN": 2, "OK": 3}
    out["level"] = min(levels, key=order.get) if levels else "UNKNOWN"
    out["notes"] = notes
    return out
