"""
Earnings events: announcement timestamps with the EPS estimate, the reported EPS and the surprise, for ~6 years per ticker.

raw.earnings_surprise only has the last 4 quarters keyed by fiscal-quarter END, which cannot tell when the market learnt
the number. The price reaction needs the announcement itself, so this table stores Yahoo's earnings dates (with time of
day) and the reported figures. EPS stays in the reporting currency (`surprise_pct` is a ratio and needs no conversion).

Refreshed weekly (etl_config refresh_intervals.earnings_hours). A failure never fails the ETL run.
"""
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterable

from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

TABLE = "raw.earnings_events"
COLUMNS = ("ticker", "earnings_ts", "eps_estimate", "eps_actual", "surprise_pct", "_extracted_at")
DDL = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        ticker VARCHAR, earnings_ts TIMESTAMPTZ, eps_estimate DOUBLE, eps_actual DOUBLE, surprise_pct DOUBLE,
        _extracted_at TIMESTAMP
    )"""


def ensure_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(DDL)


def needs_refresh(conn: duckdb.DuckDBPyConnection, threshold_hours: int = 168) -> bool:
    try:
        last = conn.execute(f"SELECT MAX(_extracted_at) FROM {TABLE}").fetchone()[0]
    except duckdb.Error:
        return True
    return last is None or (pd.Timestamp.now() - pd.Timestamp(last)).total_seconds() / 3600 > threshold_hours


def parse_earnings_dates(ticker: str, raw: pd.DataFrame, now=None) -> pd.DataFrame:
    """Yahoo `get_earnings_dates` frame → rows of COLUMNS. Surprise is stored as a fraction (6.74% → 0.0674), as
    reported: a surprise above 500 % is a near-zero estimate (−0.05 → −0.37), not a units error — the analysis flags it
    (core.earnings.small_base) instead of the loader dropping a real miss."""
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    d = raw.copy()
    idx = pd.to_datetime(d.index, utc=True, errors="coerce")
    est = pd.to_numeric(d.get("EPS Estimate"), errors="coerce")
    act = pd.to_numeric(d.get("Reported EPS"), errors="coerce")
    sur = pd.to_numeric(d.get("Surprise(%)"), errors="coerce") / 100.0
    out = pd.DataFrame({"ticker": ticker, "earnings_ts": idx, "eps_estimate": est.to_numpy(), "eps_actual": act.to_numpy(),
                        "surprise_pct": sur.to_numpy(), "_extracted_at": now or datetime.now()})
    out = out[out["earnings_ts"].notna()]
    return out.drop_duplicates(subset=["ticker", "earnings_ts"]).reset_index(drop=True)[list(COLUMNS)]


def _fetch_one(ticker: str, limit: int) -> pd.DataFrame:
    import yfinance as yf
    for attempt in range(3):
        try:
            return parse_earnings_dates(ticker, yf.Ticker(ticker).get_earnings_dates(limit=limit))
        except Exception as e:                      # rate limits are transient
            if attempt == 2:
                logger.debug(f"   earnings dates {ticker}: {e}")
            time.sleep(1.5 * (attempt + 1) + random.random())
    return pd.DataFrame(columns=COLUMNS)


def extract_earnings_events(tickers: Iterable[str], limit: int = 24, max_workers: int = 6) -> pd.DataFrame:
    """Announcement history for the given equities (indices, FX and futures are skipped)."""
    keys = [t for t in tickers if not str(t).startswith("^") and not str(t).endswith(("=X", "=F"))]
    logger.info(f"📅 EARNINGS EVENTS: fetching announcement history for {len(keys)} equities...")
    frames, ok = [], 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, t, limit): t for t in keys}
        for f in as_completed(futures):
            df = f.result()
            if not df.empty:
                frames.append(df)
                ok += 1
    logger.info(f"   ✅ Earnings events for {ok}/{len(keys)} tickers")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


SAME_EVENT_DAYS = 20          # a fetched stamp this close to a stored one is the same report (quarters are ~90 d apart)
SAME_FIGURES_DAYS = 60        # ... or this close when both carry the same reported EPS pair (a reported date moved)
SAME_STAMP_DAYS = 3           # rows of ONE side this close are re-timed stamps of one report (anchor-based, never chained)
_FIGURES = ("eps_estimate", "eps_actual", "surprise_pct")


class SeedError(RuntimeError):
    """The stored earnings history exists but could not be carried into the new warehouse."""


def _reported(r) -> bool:
    return pd.notna(r["eps_actual"]) or pd.notna(r["surprise_pct"])


def _full(r) -> bool:
    return pd.notna(r["eps_actual"]) and pd.notna(r["eps_estimate"])


def _dedupe(d: pd.DataFrame) -> pd.DataFrame:
    """One row per report within one side: rows within SAME_STAMP_DAYS of the group's first stamp are re-timed stamps of
    it; the most complete (then the most recently fetched) is kept — a partial 20:00 stamp never hides the reported 21:00
    one, whatever the input order."""
    if d.empty:
        return d
    d = d.sort_values("earnings_ts").reset_index(drop=True)
    groups, cur = [], [0]
    for i in range(1, len(d)):
        if d.loc[i, "earnings_ts"] - d.loc[cur[0], "earnings_ts"] <= pd.Timedelta(days=SAME_STAMP_DAYS):
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    keep = []
    for g in groups:
        rows = d.loc[g]
        # a complete EPS pair (estimate and actual) outranks "more fields present": a (1.0, NULL, 0.1) row ties a
        # (1.0, 1.1, NULL) row on count and must not win it, or the published actual is lost
        full = rows["eps_actual"].notna() & rows["eps_estimate"].notna()
        score = rows[list(_FIGURES)].notna().sum(axis=1)
        order = pd.DataFrame({"full": full, "score": score, "ext": pd.to_datetime(rows["_extracted_at"], errors="coerce"),
                              "ts": rows["earnings_ts"]}).sort_values(["full", "score", "ext", "ts"], ascending=False)
        keep.append(order.index[0])
    return d.loc[keep].reset_index(drop=True)


def _pair_up(old: pd.DataFrame, new: pd.DataFrame, now) -> list:
    """One-to-one (new index, old index) pairs of the same report. Nearest first within SAME_EVENT_DAYS; then reported
    rows with identical EPS figures within SAME_FIGURES_DAYS (a moved reported date); then the earliest unmatched stored
    UPCOMING date with the earliest unmatched fetched one (a rescheduled date) — an old date nothing replaces is kept."""
    pairs, used_o, used_n = [], set(), set()

    def take(cands):
        for _, i, j in sorted(cands):
            if i not in used_n and j not in used_o:
                pairs.append((i, j))
                used_n.add(i)
                used_o.add(j)
    def stale_vs_future(n, o):          # an unreported row from the past does not stand for a future date
        return not _reported(n) and n.earnings_ts < now and not _reported(o) and o.earnings_ts >= now
    near = [(abs(n.earnings_ts - o.earnings_ts), i, j) for i, n in new.iterrows() for j, o in old.iterrows()
            if abs(n.earnings_ts - o.earnings_ts) <= pd.Timedelta(days=SAME_EVENT_DAYS) and not stale_vs_future(n, o)]
    take(near)
    same = [(abs(n.earnings_ts - o.earnings_ts), i, j) for i, n in new.iterrows() if i not in used_n and _full(n)
            for j, o in old.iterrows() if j not in used_o and _full(o)
            and n.eps_estimate == o.eps_estimate and n.eps_actual == o.eps_actual
            and abs(n.earnings_ts - o.earnings_ts) <= pd.Timedelta(days=SAME_FIGURES_DAYS)]
    take(same)
    # a rescheduled date: both still ahead (a stale unreported row from the past is not a replacement for a future date)
    up_n = [i for i in new.index if i not in used_n and not _reported(new.loc[i]) and new.loc[i, "earnings_ts"] >= now]
    up_o = [j for j in old.index if j not in used_o and not _reported(old.loc[j]) and old.loc[j, "earnings_ts"] >= now]
    if up_n and up_o:
        i, j = min(up_n, key=lambda k: new.loc[k, "earnings_ts"]), min(up_o, key=lambda k: old.loc[k, "earnings_ts"])
        if abs(new.loc[i, "earnings_ts"] - old.loc[j, "earnings_ts"]) <= pd.Timedelta(days=SAME_FIGURES_DAYS):
            pairs.append((i, j))
    return pairs


def _combine(o: pd.Series, n: pd.Series) -> pd.Series:
    """The merged row of one report: the fetched stamp, and ONE observation of the figures —
      * fetched row complete (estimate and actual) → the fetched figures (the provider's surprise only when the stored row
        has the same EPS pair; otherwise it is derived from the pair downstream);
      * fetched row partial, stored row complete → the stored figures whole (a partial or estimate-only response never
        erases or rewrites a published EPS pair); likewise a fetched row that reports nothing (no actual, no surprise)
        never replaces a stored row that does, even one with only an actual or only a surprise;
      * neither complete → the fetched row, its estimate filled from the stored one when it has none."""
    row = n.copy()
    if _full(n):
        if pd.isna(n["surprise_pct"]) and _full(o) and n["eps_actual"] == o["eps_actual"] \
                and n["eps_estimate"] == o["eps_estimate"]:
            row["surprise_pct"] = o["surprise_pct"]
    elif _full(o) or (_reported(o) and not _reported(n)) or (pd.notna(o["eps_actual"]) and pd.isna(n["eps_actual"])):
        for c in _FIGURES:                      # the stored observation whole — also an actual-only / surprise-only one
            row[c] = o[c]
    elif pd.isna(n["eps_estimate"]) and not _reported(n):
        row["eps_estimate"] = o["eps_estimate"]
    return row


def merge_events(old: pd.DataFrame, new: pd.DataFrame, now=None) -> pd.DataFrame:
    """History-preserving merge of one ticker's stored and fetched events.

    Yahoo returns the latest `limit` events, so a refresh is a window: stored events it does not return are kept.
    Fetched and stored rows are matched ONE-TO-ONE as the same report (_pair_up); a matched pair becomes one row
    (_combine); a fetched row nothing matches is added; a stored row nothing matches is kept — in particular the
    absence of a date from a response is no evidence that it was cancelled, and other upcoming dates of the ticker
    survive a replacement of one of them. Re-timed stamps within one side are one report (_dedupe)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    old = old.copy() if old is not None else pd.DataFrame(columns=COLUMNS)
    new = new.copy()
    for d in (old, new):
        d["earnings_ts"] = pd.to_datetime(d["earnings_ts"], utc=True)
    old, new = _dedupe(old), _dedupe(new)
    pairs = _pair_up(old, new, now)
    paired_n, paired_o = {i for i, _ in pairs}, {j for _, j in pairs}
    rows = [_combine(old.loc[j], new.loc[i]) for i, j in pairs]
    rows += [new.loc[i] for i in new.index if i not in paired_n]
    rows += [old.loc[j] for j in old.index if j not in paired_o]
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows)[list(COLUMNS)].sort_values("earnings_ts").reset_index(drop=True)


def seed_from_production(conn: duckdb.DuckDBPyConnection, prod_path: str, changes: dict = None) -> int:
    """FULL refresh: the shadow warehouse starts empty, so the merge would not see the stored history. Copy production's
    events into the shadow before the fetched ones are merged (and instead of the key-based carry-forward afterwards,
    which re-added the old stamp of a moved event and could not refill a reported EPS). Renamed symbols
    (`changes` {old: new}) are renamed in the copy — the history continues under the new symbol, one row per report.

    Returns the rows copied. No production file, or one without the table (a first run) → 0. A production history that
    exists but cannot be read or copied completely raises SeedError: publishing the new warehouse would lose it."""
    ensure_table(conn)
    if not prod_path or not Path(prod_path).exists():
        return 0
    if conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]:
        return 0                                    # incremental shadow: already a copy of production
    try:
        conn.execute(f"ATTACH '{prod_path}' AS prod_ee (READ_ONLY)")
    except duckdb.Error as e:
        raise SeedError(f"production warehouse cannot be attached: {e}") from e
    try:
        try:
            df = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM prod_ee.{TABLE}").df()
        except duckdb.CatalogException:
            return 0                                # production has no earnings events yet
        if df.empty:
            return 0
        df["earnings_ts"] = pd.to_datetime(df["earnings_ts"], utc=True)
        renamed = {o: n for o, n in (changes or {}).items() if o in set(df["ticker"])}
        expected = len(df)
        if renamed:
            # the history of OLD continues under NEW: reconcile the two with the matcher a refresh uses (the same report
            # dated 01/07 under OLD and 11/07 under NEW is one report), NEW's own rows winning
            parts = []
            for new_sym in sorted(set(renamed.values())):
                olds = [o for o, n in renamed.items() if n == new_sym]
                carried = df[df["ticker"].isin(olds)].assign(ticker=new_sym)
                own = df[df["ticker"] == new_sym]
                parts.append(merge_events(carried, own, now=None) if len(own) else _dedupe(carried))
            keep = df[~df["ticker"].isin(set(renamed) | set(renamed.values()))]
            df = pd.concat([keep] + parts, ignore_index=True)
            expected = len(df)
        conn.register("ee_seed", df[list(COLUMNS)])
        try:
            conn.execute(f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) SELECT {', '.join(COLUMNS)} FROM ee_seed")
        finally:
            conn.unregister("ee_seed")
        copied = int(conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])
        if copied != expected:
            raise SeedError(f"{copied:,} of {expected:,} production earnings events copied")
        return copied
    except duckdb.Error as e:
        raise SeedError(f"production earnings events cannot be copied: {e}") from e
    finally:
        conn.execute("DETACH prod_ee")


def load_earnings_events(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, now=None) -> int:
    """Merge the fetched events into the stored history (merge_events), per ticker present in df; a re-run never
    duplicates. `now` (default: the clock) decides which dates are still ahead. Returns the number of fetched rows."""
    ensure_table(conn)
    if df is None or df.empty:
        return 0
    tickers = df["ticker"].unique().tolist()
    stored = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM {TABLE} WHERE ticker = ANY(?)", [tickers]).df()
    merged = pd.concat([merge_events(stored[stored["ticker"] == t], g, now=now) for t, g in df.groupby("ticker")],
                       ignore_index=True)
    conn.register("ee_tmp", merged)
    try:
        conn.execute("BEGIN TRANSACTION")
        conn.execute(f"DELETE FROM {TABLE} WHERE ticker = ANY(?)", [tickers])
        conn.execute(f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) SELECT {', '.join(COLUMNS)} FROM ee_tmp")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.unregister("ee_tmp")
    logger.info(f"✅ Merged {len(df)} fetched rows → {TABLE} ({len(merged)} stored for {len(tickers)} tickers)")
    return len(df)
