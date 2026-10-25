"""
Independent-source reconciliation: every number in the warehouse comes from Yahoo; this checks a sample against two
sources that do not depend on it, and records where they disagree.

- PRICES (all markets) — TradingView's scanner API: the last close of each ticker, in local currency, against the
  stored close converted back to local currency (close_eur × price_scale / fx_rate). TradingView may already show an
  intraday price, so a ticker MATCHes when our close equals either its last price or its previous close (derived from
  the day's change) within `price_tolerance_pct`.
- ANNUAL STATEMENTS (US filers) — SEC EDGAR XBRL company facts: revenue and net income per fiscal year against
  raw.historical_financials (local currency = value / fx_to_eur). SEC asks for a User-Agent with a contact e-mail:
  set SEC_USER_AGENT (e.g. "Your Name you@example.com"); without it the SEC part is skipped. The filing dates are
  stored in raw.sec_filings — they are the point-in-time availability dates of the statements (etl/pit.py).

Results go to raw.source_reconciliation (latest result per ticker / check / period); MISMATCHes go to the exception
ledger and the dq_warnings table, and a systemic disagreement is a release-gate issue (etl/gates.py). Reconciliation
never fails the ETL by itself.
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

TV_SCAN = "https://scanner.tradingview.com/{market}/scan"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_CACHE_DIR = Path(__file__).resolve().parent.parent / "warehouse"

# Yahoo suffix → TradingView market
TV_MARKET = {
    "": "america", ".DE": "germany", ".F": "germany", ".L": "uk", ".PA": "france", ".AS": "netherlands",
    ".BR": "belgium", ".LS": "portugal", ".MI": "italy", ".MC": "spain", ".SW": "switzerland", ".ST": "sweden",
    ".CO": "denmark", ".OL": "norway", ".HE": "finland", ".VI": "austria", ".IR": "ireland", ".T": "japan",
    ".HK": "hongkong", ".TW": "taiwan", ".TWO": "taiwan", ".KS": "korea", ".KQ": "korea", ".SS": "china",
    ".SZ": "china", ".VN": "vietnam", ".AX": "australia", ".TO": "canada", ".V": "canada", ".SA": "brazil",
    ".NS": "india", ".BO": "india",
}
# the same unit under two spellings
_CCY_ALIASES = {"GBX": "GBp", "ZAC": "ZAc", "ILA": "ILA"}

REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet")
NET_INCOME_TAGS = ("NetIncomeLoss", "ProfitLoss")
ANNUAL_FORMS = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")
QUARTER_FORMS = ("10-Q", "10-Q/A")

RECON_DDL = """
    CREATE TABLE IF NOT EXISTS raw.sec_annual_history (
        ticker VARCHAR, period_end DATE, item VARCHAR, value DOUBLE, currency VARCHAR, tag VARCHAR,
        form VARCHAR, first_filed DATE, _loaded_at TIMESTAMP, value_filed DATE);
    ALTER TABLE raw.sec_annual_history ADD COLUMN IF NOT EXISTS value_filed DATE;
    CREATE TABLE IF NOT EXISTS raw.source_reconciliation (
        ticker VARCHAR, check_name VARCHAR, period VARCHAR, ours DOUBLE, theirs DOUBLE, diff_pct DOUBLE,
        status VARCHAR, source VARCHAR, detail VARCHAR, checked_at TIMESTAMP);
    CREATE TABLE IF NOT EXISTS raw.sec_filings (
        ticker VARCHAR, cik BIGINT, period_end DATE, form VARCHAR, fp VARCHAR, filed DATE, _loaded_at TIMESTAMP);
    CREATE TABLE IF NOT EXISTS raw.sec_cash_items (
        ticker VARCHAR, period_end DATE, item VARCHAR, value_usd DOUBLE, tag VARCHAR, form VARCHAR, filed DATE,
        _loaded_at TIMESTAMP);
"""
# Cash-flow facts Yahoo often lacks, from the 10-K XBRL (US filers): item → tags in order of preference
SEC_CASH_ITEMS = {
    "stock_based_comp": ("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"),
    "lease_principal_paid": ("FinanceLeasePrincipalPayments",),
    "interest_paid_cfo": ("InterestPaidNet", "InterestPaid"),         # US GAAP: interest paid sits in operating CF
}


def ensure_tables(conn) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(RECON_DDL)


def split_symbol(ticker: str) -> tuple:
    """'SAP.DE' → ('SAP', '.DE'); 'BRK-B' → ('BRK-B', '')."""
    t = str(ticker)
    if "." in t:
        base, suf = t.rsplit(".", 1)
        return base, "." + suf
    return t, ""


def tv_names(ticker: str) -> list:
    """Candidate TradingView names of a Yahoo symbol (BRK-B → BRK.B, BP.L → BP./BP, 0700.HK → 700)."""
    base, suf = split_symbol(ticker)
    names = {base, base.replace("-", ".")}
    if suf == ".L":
        names |= {base + ".", base.replace("-", ".") + "."}
    if suf == ".HK":
        names.add(base.lstrip("0") or "0")
    return sorted(names)


def _same_currency(ours, theirs) -> bool:
    if not ours or not theirs:
        return True                                     # unknown on one side: compare anyway (the price will tell)
    norm = lambda c: _CCY_ALIASES.get(str(c).upper(), str(c))     # noqa: E731
    return norm(ours) == norm(theirs)


def fetch_tv_quotes(tickers, post=None, timeout: int = 15) -> pd.DataFrame:
    """[ticker, tv_symbol, tv_close, tv_prev_close, tv_currency] for the tickers TradingView lists."""
    post = post or requests.post
    by_market = {}
    for t in tickers:
        market = TV_MARKET.get(split_symbol(t)[1])
        if market:
            by_market.setdefault(market, []).append(t)
    rows = []
    for market, ts in by_market.items():
        lookup = {}
        for t in ts:
            for n in tv_names(t):
                lookup.setdefault(n, t)
        names = sorted(lookup)
        for i in range(0, len(names), 400):
            chunk = names[i:i + 400]
            payload = {"filter": [{"left": "name", "operation": "in_range", "right": chunk},
                                  {"left": "is_primary", "operation": "equal", "right": True}],
                       "columns": ["name", "close", "change", "currency"], "range": [0, len(chunk) * 2],
                       "options": {"lang": "en"}}
            try:
                r = post(TV_SCAN.format(market=market), json=payload, timeout=timeout)
                data = r.json().get("data") or []
            except Exception as e:
                logger.warning(f"   ⚠️ TradingView quotes for {market} unavailable: {e}")
                continue
            for d in data:
                name, close, change, ccy = (d.get("d") or [None] * 4)[:4]
                t = lookup.get(name)
                if t is None or close is None:
                    continue
                prev = close / (1 + change / 100) if change not in (None, -100) else None
                rows.append({"ticker": t, "tv_symbol": d.get("s"), "tv_close": float(close),
                             "tv_prev_close": prev, "tv_currency": ccy})
    out = pd.DataFrame(rows, columns=["ticker", "tv_symbol", "tv_close", "tv_prev_close", "tv_currency"])
    return out.drop_duplicates("ticker")


def our_last_closes(conn, tickers) -> pd.DataFrame:
    """[ticker, date, close_local, currency] — the newest stored close back in quote currency."""
    return conn.execute("""
        SELECT ticker, CAST(MAX(date) AS DATE) AS date,
               arg_max(close * COALESCE(price_scale, 1) / NULLIF(COALESCE(fx_rate, 1), 0), date) AS close_local,
               arg_max(currency, date) AS currency
        FROM raw.stock_prices WHERE ticker = ANY(?) GROUP BY ticker""", [list(tickers)]).df()


def reconcile_prices(conn, tickers, tolerance_pct: float = 1.5, quotes: pd.DataFrame = None) -> pd.DataFrame:
    ours = our_last_closes(conn, tickers)
    quotes = fetch_tv_quotes(ours["ticker"].tolist()) if quotes is None else quotes
    m = ours.merge(quotes, on="ticker", how="left")
    newest = pd.to_datetime(m["date"]).max() if not m.empty else None
    now = pd.Timestamp.now()
    rows = []
    for r in m.itertuples():
        base = {"ticker": r.ticker, "check_name": "price_last_close", "period": str(r.date), "ours": r.close_local,
                "source": "TradingView", "checked_at": now}
        if newest is not None and pd.Timestamp(r.date) < newest - pd.Timedelta(days=4):
            rows.append({**base, "theirs": None, "diff_pct": None, "status": "STALE_HERE",
                         "detail": f"our last close is from {r.date} (newest {newest.date()}): not comparable"})
            continue
        if pd.isna(r.tv_close):
            rows.append({**base, "theirs": None, "diff_pct": None, "status": "NOT_COVERED",
                         "detail": f"not found on TradingView ({TV_MARKET.get(split_symbol(r.ticker)[1], 'no market')})"})
            continue
        if not _same_currency(r.currency, r.tv_currency):
            rows.append({**base, "theirs": r.tv_close, "diff_pct": None, "status": "CURRENCY_DIFF",
                         "detail": f"quoted in {r.currency} here, {r.tv_currency} on TradingView ({r.tv_symbol})"})
            continue
        cands = [c for c in (r.tv_close, r.tv_prev_close) if c is not None and pd.notna(c) and c > 0]
        diffs = [(r.close_local / c - 1) * 100 for c in cands] if pd.notna(r.close_local) else []
        best = min(diffs, key=abs) if diffs else None
        ok = best is not None and abs(best) <= tolerance_pct
        rows.append({**base, "theirs": cands[diffs.index(best)] if diffs else None,
                     "diff_pct": round(best, 3) if best is not None else None,
                     "status": "MATCH" if ok else "MISMATCH",
                     "detail": f"{r.tv_symbol}: last {r.tv_close:,.4g}, previous close "
                               f"{r.tv_prev_close if r.tv_prev_close is None else f'{r.tv_prev_close:,.4g}'} {r.tv_currency or ''}"})
    return pd.DataFrame(rows)


# ── SEC EDGAR ─────────────────────────────────────────────────────────────────────────────────────────────────────
def sec_user_agent(env=None):
    ua = (env or os.environ).get("SEC_USER_AGENT", "").strip()
    return ua if "@" in ua else None


def sec_cik_map(user_agent: str, get=None, cache_days: int = 7) -> dict:
    """{TICKER: cik} from SEC's company_tickers.json (cached in warehouse/)."""
    cache = _CACHE_DIR / "sec_company_tickers.json"
    try:
        if cache.exists() and datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime) < timedelta(days=cache_days):
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            r = (get or requests.get)(SEC_TICKERS, headers={"User-Agent": user_agent}, timeout=30)
            r.raise_for_status()
            data = r.json()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data), encoding="utf-8")
    except Exception as e:
        logger.warning(f"   ⚠️ SEC ticker map unavailable: {e}")
        return {}
    return {str(v["ticker"]).upper().replace(".", "-"): int(v["cik_str"]) for v in data.values()}


def _duration_days(e) -> int:
    try:
        return (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days
    except (KeyError, ValueError, TypeError):
        return -1


def facts_annual(facts: dict, tags, currency: str = "USD") -> pd.DataFrame:
    """[period_end, value, first_filed, form] per fiscal year from the first tag that reports annual values.
    Value = the LATEST filing for that period (restatements included, as Yahoo shows); first_filed = when the
    figure was first public (point in time)."""
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    for tag in tags:
        units = (gaap.get(tag) or {}).get("units") or {}
        entries = [e for e in units.get(currency, [])
                   if e.get("form") in ANNUAL_FORMS and 350 <= _duration_days(e) <= 380]
        if not entries:
            continue
        df = pd.DataFrame(entries)
        df["end"] = pd.to_datetime(df["end"])
        df["filed"] = pd.to_datetime(df["filed"])
        df = df.sort_values("filed")
        g = df.groupby("end")
        return pd.DataFrame({"period_end": g["end"].first(), "value": g["val"].last(), "first_filed": g["filed"].first(),
                             "form": g["form"].first(), "tag": tag}).reset_index(drop=True)
    return pd.DataFrame(columns=["period_end", "value", "first_filed", "form", "tag"])


HISTORY_COLUMNS = ["period_end", "value", "first_filed", "form", "tag", "item", "currency", "value_filed"]


def facts_annual_history(facts: dict, currency: str, additional_facts=()) -> pd.DataFrame:
    """Latest value with its value_filed date, retaining first_filed; at most 15 annual periods."""
    tags = {"revenue": REVENUE_TAGS,
            "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",
                                    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
            "capex": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
            "stock_based_comp": SEC_CASH_ITEMS["stock_based_comp"]}
    rows = []
    for source in (facts, *additional_facts):
        for item, names in tags.items():
            seen = set()
            for tag in names:
                frame = facts_annual(source, (tag,), currency)
                if frame.empty:
                    continue
                frame = frame[~frame.period_end.isin(seen)]
                seen.update(frame.period_end)
                entries = source["facts"]["us-gaap"][tag]["units"][currency]
                dates = pd.DataFrame([e for e in entries if e.get("form") in ANNUAL_FORMS
                                      and 350 <= _duration_days(e) <= 380]).groupby("end")["filed"].max()
                frame["_filed"] = frame.period_end.dt.strftime("%Y-%m-%d").map(dates)
                rows.extend(frame.assign(item=item, currency=currency).to_dict("records"))
    if not rows:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.DataFrame(rows)
    df["first_filed"] = df.groupby(["item", "period_end"])["first_filed"].transform("min")
    df = df.sort_values("_filed", kind="stable").drop_duplicates(["item", "period_end"], keep="last").rename(columns={"_filed": "value_filed"})
    periods = df["period_end"].drop_duplicates().sort_values().tail(15)
    return (df[df["period_end"].isin(periods)].sort_values("period_end", kind="stable")[HISTORY_COLUMNS]
            .reset_index(drop=True))


def facts_filings(facts: dict, tags=REVENUE_TAGS + NET_INCOME_TAGS) -> pd.DataFrame:
    """[period_end, form, fp, filed] — first filing date of every reported period (quarters and years)."""
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    rows = []
    for tag in tags:
        for unit in ((gaap.get(tag) or {}).get("units") or {}).values():
            for e in unit:
                if e.get("form") in ANNUAL_FORMS + QUARTER_FORMS and e.get("end") and e.get("filed"):
                    rows.append({"period_end": e["end"], "form": e["form"], "fp": e.get("fp"), "filed": e["filed"]})
    if not rows:
        return pd.DataFrame(columns=["period_end", "form", "fp", "filed"])
    df = pd.DataFrame(rows)
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
    df["filed"] = pd.to_datetime(df["filed"]).dt.date
    return df.sort_values("filed").drop_duplicates(["period_end", "form"]).reset_index(drop=True)


def facts_cash_items(facts: dict) -> pd.DataFrame:
    """[period_end, item, value_usd, tag, form, filed] — annual (10-K, ~1-year duration) SBC, finance-lease principal and
    interest paid, latest filing per period (restatements included)."""
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    rows = []
    for item, tags in SEC_CASH_ITEMS.items():
        for tag in tags:
            entries = [e for e in ((gaap.get(tag) or {}).get("units") or {}).get("USD", [])
                       if e.get("form") in ANNUAL_FORMS and 350 <= _duration_days(e) <= 380]
            if not entries:
                continue
            df = pd.DataFrame(entries).sort_values("filed").drop_duplicates("end", keep="last")
            rows += [{"period_end": pd.to_datetime(r.end).date(), "item": item, "value_usd": abs(float(r.val)),
                      "tag": tag, "form": r.form, "filed": pd.to_datetime(r.filed).date()} for r in df.itertuples()]
            break                                            # first tag with data wins for this item
    return pd.DataFrame(rows, columns=["period_end", "item", "value_usd", "tag", "form", "filed"])


def apply_sec_supplements(conn) -> dict:
    """Fill raw.hist_fcf deductions the provider left empty with the SEC XBRL figure of the same fiscal year (USD → EUR
    at that year's statement rate), recording the source in *_source. Re-applied every run (a Yahoo reload rewrites the
    row). Never overwrites a reported value. {item: rows filled}."""
    out = {}
    try:
        from etl.load import ensure_hist_fcf
        ensure_hist_fcf(conn)
        ensure_tables(conn)
        src_col = {"stock_based_comp": "sbc_source", "lease_principal_paid": "lease_source",
                   "interest_paid_cfo": "interest_source"}
        for item, col in src_col.items():
            # SEC-derived values are re-derived from raw.sec_cash_items on every run, so a figure written under an
            # older rule (the USD = EUR fallback, review REV-08) does not survive; provider values are untouched
            conn.execute(f"UPDATE raw.hist_fcf SET {item} = NULL, {col} = NULL WHERE {col} LIKE 'SEC XBRL %'")
            n0 = conn.execute(f"SELECT COUNT(*) FROM raw.hist_fcf WHERE {item} IS NOT NULL").fetchone()[0]
            conn.execute(f"""
                UPDATE raw.hist_fcf f SET {item} = s.value_usd * fx.rate,
                       {col} = 'SEC XBRL ' || s.tag || ' (' || s.form || ', filed ' || CAST(s.filed AS VARCHAR) || ')'
                FROM raw.sec_cash_items s
                JOIN (SELECT ticker, EXTRACT(YEAR FROM date) AS y, arg_max(fx_to_eur, date) AS rate
                      FROM raw.historical_financials WHERE src_currency = 'USD' GROUP BY 1, 2) fx
                  ON fx.ticker = s.ticker AND fx.y = EXTRACT(YEAR FROM s.period_end)
                WHERE f.ticker = s.ticker AND f.year = EXTRACT(YEAR FROM s.period_end) AND s.item = '{item}'
                  AND isfinite(fx.rate) AND fx.rate > 0     -- a missing USD→EUR rate is not 1 (review REV-08)
                  AND f.{item} IS NULL""")
            filled = conn.execute(f"SELECT COUNT(*) FROM raw.hist_fcf WHERE {item} IS NOT NULL").fetchone()[0] - n0
            if filled:
                out[item] = int(filled)
        if out:
            logger.info("   🧩 SEC XBRL filled missing cash-flow deductions: " + ", ".join(f"{k} {v}" for k, v in out.items()))
    except duckdb.Error as e:
        logger.warning(f"   ⚠️ SEC supplements not applied: {e}")
    return out


def compare_statements(ticker: str, ours: pd.DataFrame, sec: dict, tolerance_pct: float) -> list:
    """ours: [date, revenue, net_income, src_currency, fx_to_eur]; sec: {'revenue': df, 'net_income': df}."""
    now = pd.Timestamp.now()
    rows = []
    for field, sdf in sec.items():
        if sdf.empty:
            continue
        for o in ours.itertuples():
            if pd.isna(getattr(o, field)) or not o.fx_to_eur:
                continue
            gap = (sdf["period_end"] - pd.Timestamp(o.date)).abs()
            if gap.min() > pd.Timedelta(days=10):
                continue
            s = sdf.loc[gap.idxmin()]
            local = getattr(o, field) / o.fx_to_eur
            diff = (local / s["value"] - 1) * 100 if s["value"] else None
            ok = diff is not None and abs(diff) <= tolerance_pct
            rows.append({"ticker": ticker, "check_name": f"annual_{field}", "period": str(pd.Timestamp(o.date).date()),
                         "ours": local, "theirs": float(s["value"]), "diff_pct": None if diff is None else round(diff, 3),
                         "status": "MATCH" if ok else "MISMATCH", "source": "SEC EDGAR",
                         "detail": f"{s['tag']} ({s['form']}, first filed {pd.Timestamp(s['first_filed']).date()})",
                         "checked_at": now})
    return rows


def sec_targets(conn, tickers, cik_map: dict, limit: int) -> list:
    """US filers with annual statements, least recently checked first."""
    cands = [t for t in tickers if split_symbol(t)[1] == "" and t.upper() in cik_map]
    if not cands:
        return []
    last = dict(conn.execute("""SELECT ticker, MAX(checked_at) FROM raw.source_reconciliation
                                WHERE source = 'SEC EDGAR' GROUP BY 1""").fetchall())
    have = {r[0] for r in conn.execute("SELECT DISTINCT ticker FROM raw.historical_financials").fetchall()}
    cands = [t for t in cands if t in have]
    return sorted(cands, key=lambda t: (last.get(t) is not None, last.get(t) or datetime.min, t))[:limit]


def reconcile_sec(conn, tickers, user_agent: str, tolerance_pct: float = 3.0, limit: int = 60, get=None,
                  pause: float = 0.12, cik_map=None, targets=None) -> pd.DataFrame:
    get = get or requests.get
    cik_map = sec_cik_map(user_agent, get) if cik_map is None else cik_map
    targets = sec_targets(conn, tickers, cik_map, limit) if targets is None else targets
    overrides = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "sec_cik_overrides.yaml")
                               .read_text(encoding="utf-8")) or {}
    rows, filings, cash_items, history = [], [], [], []
    history_tickers = []
    for t in targets:
        cik = cik_map[t.upper()]
        try:
            r = get(SEC_FACTS.format(cik=cik), headers={"User-Agent": user_agent}, timeout=30)
            if r.status_code == 404:                     # no XBRL facts for this filer: not covered, not an error
                continue
            r.raise_for_status()
            facts = r.json()
        except Exception as e:
            logger.warning(f"SEC facts {t}: {e}")
            rows.append({"ticker": t, "check_name": "annual_statements", "status": "ERROR",
                         "source": "SEC EDGAR", "detail": str(e), "checked_at": pd.Timestamp.now()})
            continue
        finally:
            time.sleep(pause)                            # SEC fair-access limit: ≤ 10 requests / second
        ours = conn.execute("""SELECT date, revenue, net_income, src_currency, fx_to_eur FROM raw.historical_financials
                               WHERE ticker = ? ORDER BY date DESC LIMIT 4""", [t]).df()
        ccy = (ours["src_currency"].dropna().iloc[0] if ours["src_currency"].notna().any() else "USD") or "USD"
        sec = {"revenue": facts_annual(facts, REVENUE_TAGS, ccy), "net_income": facts_annual(facts, NET_INCOME_TAGS, ccy)}
        found = compare_statements(t, ours, sec, tolerance_pct)
        rows += found or [{"ticker": t, "check_name": "annual_statements", "period": None, "ours": None, "theirs": None,
                           "diff_pct": None, "status": "NOT_COVERED", "source": "SEC EDGAR",
                           "detail": f"no comparable {ccy} annual revenue / net income in the XBRL facts",
                           "checked_at": pd.Timestamp.now()}]
        extra_facts, history_failed = [], False
        for extra_cik in overrides.get(t.upper(), []):
            try:
                r = get(SEC_FACTS.format(cik=extra_cik), headers={"User-Agent": user_agent}, timeout=30)
                r.raise_for_status()
                extra_facts.append(r.json())
            except Exception as e:
                history_failed = True
                logger.warning("SEC history %s CIK %s: %s; retaining previous history", t, extra_cik, e,
                               extra={"event": "sec_history_retained", "ticker": t, "cik": extra_cik,
                                      "reason": str(e)})
            finally:
                time.sleep(pause)
        if not history_failed:
            history.append(facts_annual_history(facts, ccy, extra_facts).assign(ticker=t))
            history_tickers.append(t)
        items = facts_cash_items(facts)
        if not items.empty:
            items.insert(0, "ticker", t)
            cash_items.append(items)
        f = facts_filings(facts)
        if not f.empty:
            f.insert(0, "cik", cik)
            f.insert(0, "ticker", t)
            filings.append(f)
    if history:
        hdf = pd.DataFrame([r for h in history for r in h.to_dict("records")], columns=history[0].columns)
        hdf["_loaded_at"] = pd.Timestamp.now()
        conn.register("history_tmp", hdf)
        conn.execute("DELETE FROM raw.sec_annual_history WHERE ticker = ANY(?)", [history_tickers])
        conn.execute("""INSERT INTO raw.sec_annual_history
                        (ticker, period_end, item, value, currency, tag, form, first_filed, _loaded_at, value_filed)
                        SELECT ticker, period_end, item, value, currency, tag, form, first_filed, _loaded_at,
                               value_filed FROM history_tmp""")
        conn.unregister("history_tmp")
    if cash_items:
        cdf = pd.concat(cash_items, ignore_index=True)
        cdf["_loaded_at"] = pd.Timestamp.now()
        conn.register("ci_tmp", cdf)
        conn.execute("DELETE FROM raw.sec_cash_items WHERE ticker IN (SELECT DISTINCT ticker FROM ci_tmp)")
        conn.execute("""INSERT INTO raw.sec_cash_items SELECT ticker, period_end, item, value_usd, tag, form, filed,
                        _loaded_at FROM ci_tmp""")
        conn.unregister("ci_tmp")
    if filings:
        fdf = pd.concat(filings, ignore_index=True)
        fdf["_loaded_at"] = pd.Timestamp.now()
        conn.register("fil_tmp", fdf)
        conn.execute("DELETE FROM raw.sec_filings WHERE ticker IN (SELECT DISTINCT ticker FROM fil_tmp)")
        conn.execute("INSERT INTO raw.sec_filings SELECT ticker, cik, period_end, form, fp, filed, _loaded_at FROM fil_tmp")
        conn.unregister("fil_tmp")
    return pd.DataFrame(rows)


def store(conn, results: pd.DataFrame) -> int:
    """Keep the latest result per (ticker, check_name, period)."""
    if results is None or results.empty:
        return 0
    cols = ["ticker", "check_name", "period", "ours", "theirs", "diff_pct", "status", "source", "detail", "checked_at"]
    df = results.reindex(columns=cols)
    conn.register("rec_tmp", df)
    conn.execute("""DELETE FROM raw.source_reconciliation r USING rec_tmp n
                    WHERE r.ticker = n.ticker AND r.check_name = n.check_name
                      AND COALESCE(r.period, '') = COALESCE(n.period, '')""")
    # a price check replaces the previous price check of that ticker whatever its date
    conn.execute("""DELETE FROM raw.source_reconciliation r USING rec_tmp n
                    WHERE r.ticker = n.ticker AND r.check_name = 'price_last_close' AND n.check_name = 'price_last_close'""")
    conn.execute(f"INSERT INTO raw.source_reconciliation SELECT {', '.join(cols)} FROM rec_tmp")
    conn.unregister("rec_tmp")
    return len(df)


def summary(results: pd.DataFrame) -> dict:
    """{source: (compared, mismatched)} — compared excludes NOT_COVERED / CURRENCY_DIFF."""
    out = {}
    if results is None or results.empty:
        return out
    for src, g in results.groupby("source"):
        cmp = g[g["status"].isin(["MATCH", "MISMATCH"])]
        out[src] = (len(cmp), int((cmp["status"] == "MISMATCH").sum()))
    return out


def run_reconciliation(conn, tickers, cfg: dict) -> pd.DataFrame:
    """Pipeline step on the shadow warehouse → this run's results (also stored)."""
    from etl import exceptions as _exc
    ensure_tables(conn)
    parts = []
    equities = [t for t in tickers if not t.startswith("^") and not t.endswith("=X")]
    try:
        parts.append(reconcile_prices(conn, equities, cfg["price_tolerance_pct"]))
    except Exception as e:
        logger.warning(f"   ⚠️ price reconciliation skipped: {e}")
    ua = sec_user_agent()
    if ua:
        try:
            parts.append(reconcile_sec(conn, equities, ua, cfg["fundamental_tolerance_pct"], cfg["sec_max_tickers_per_run"]))
        except Exception as e:
            logger.warning(f"   ⚠️ SEC reconciliation skipped: {e}")
    else:
        logger.info("   ℹ️ SEC reconciliation off: set SEC_USER_AGENT='Your Name you@example.com' to enable it")
    parts = [p for p in parts if p is not None and not p.empty]
    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    store(conn, res)
    for r in res[res["status"] == "MISMATCH"].itertuples() if not res.empty else []:
        _exc.record(r.ticker, "reconciliation", f"{r.source} disagrees on {r.check_name} ({r.period})",
                    f"ours {r.ours:,.6g} vs {r.theirs:,.6g} ({r.diff_pct:+.2f}%) · {r.detail}")
    for src, (n, bad) in summary(res).items():
        logger.info(f"   🔎 {src}: {n} compared, {bad} mismatched"
                    + (f" ({bad / n:.0%})" if n else ""))
    return res
