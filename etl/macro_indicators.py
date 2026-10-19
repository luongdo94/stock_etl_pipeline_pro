"""Economic indicators (latest release + previous) from free official sources into raw.macro_indicators.

config/macro_indicators.yaml lists them. One failing source never stops the others and never erases what was stored:
a failed fetch keeps the previous row (its `fetched_at` shows how old it is). Pure transforms and parsers are separate
from the network calls so they can be tested without a network.
"""
import io
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import duckdb
import pandas as pd
import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "macro_indicators.yaml"
TABLE = "raw.macro_indicators"
COLUMNS = ["id", "region", "name", "unit", "period", "value", "previous_period", "previous_value", "source", "series",
           "frequency", "max_age_days", "fetched_at"]
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}?format=JSON&lang=EN&{query}"
_UA = {"User-Agent": "stock-etl-pipeline/1.0 (personal research)"}


def load_config(path: Path = CONFIG_PATH) -> list:
    with open(path, encoding="utf-8") as f:
        return list((yaml.safe_load(f) or {}).get("indicators") or [])


def _get(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
        return r.read()


def parse_fred(csv_text: str) -> pd.Series:
    """observation_date,<series> CSV → float Series indexed by period start (missing '.' values dropped)."""
    df = pd.read_csv(io.StringIO(csv_text))
    if df.shape[1] < 2:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    return s[s.index.notna()].dropna().sort_index()


def parse_eurostat(doc: dict) -> pd.Series:
    """Eurostat JSON-stat (one series: every other dimension has size 1) → float Series indexed by period start."""
    times = doc["dimension"]["time"]["category"]["index"]               # {'2026-08': 7, ...}
    values = doc.get("value") or {}
    pos = {int(k): v for k, v in values.items()} if isinstance(values, dict) else dict(enumerate(values))
    out = {pd.Period(t).to_timestamp(): pos[i] for t, i in times.items() if i in pos and pos[i] is not None}
    return pd.Series(out, dtype=float).sort_index()


def transform(s: pd.Series, how: str) -> pd.Series:
    """level | yoy (vs 12 observations earlier — monthly series) | mom (vs the previous observation) | diff."""
    s = s.dropna()
    if how == "level":
        return s
    if how == "yoy":
        return (s / s.shift(12) - 1).mul(100).dropna()
    if how == "mom":
        return (s / s.shift(1) - 1).mul(100).dropna()
    if how == "diff":
        return s.diff().dropna()
    raise ValueError(f"unknown transform {how!r}")


def latest_two(s: pd.Series) -> Optional[dict]:
    """{period, value, previous_period, previous_value} of the two newest observations (previous None if only one)."""
    s = s.dropna().sort_index()
    if s.empty:
        return None
    cur = s.index[-1]
    prev = s.index[-2] if len(s) > 1 else None
    return {"period": cur.date(), "value": float(s.iloc[-1]), "previous_period": prev.date() if prev is not None else None,
            "previous_value": float(s.iloc[-2]) if prev is not None else None}


def fetch_series(ind: dict, fetch: Callable[[str], bytes] = _get, nbs_get: Optional[Callable] = None,
                 cache: Optional[dict] = None) -> pd.Series:
    if ind["source"] == "nbs":                                  # China: the statistics bureau's English releases (etl/nbs_china.py)
        from etl import nbs_china
        return nbs_china.fetch(ind, nbs_get or nbs_china.http_get, cache)
    if ind["source"] == "fred":
        return parse_fred(fetch(FRED.format(series=ind["series"])).decode("utf-8"))
    if ind["source"] == "eurostat":
        q = urllib.parse.urlencode({**ind.get("params", {}), "sinceTimePeriod": f"{datetime.now().year - 3}-01"})
        return parse_eurostat(json.loads(fetch(EUROSTAT.format(dataset=ind["dataset"], query=q)).decode("utf-8")))
    raise ValueError(f"unknown source {ind['source']!r}")


def ensure_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        id VARCHAR PRIMARY KEY, region VARCHAR, name VARCHAR, unit VARCHAR, period DATE, value DOUBLE,
        previous_period DATE, previous_value DOUBLE, source VARCHAR, series VARCHAR, frequency VARCHAR,
        max_age_days INTEGER, fetched_at TIMESTAMP)""")


def refresh(conn: duckdb.DuckDBPyConnection, indicators: Optional[list] = None, fetch: Callable[[str], bytes] = _get,
            now: Optional[datetime] = None, nbs_get: Optional[Callable] = None) -> dict:
    """Fetch every indicator; upsert the ones that came back. Returns {'ok': [...ids], 'failed': {id: error}}."""
    indicators = load_config() if indicators is None else indicators
    now = now or datetime.now()
    ensure_table(conn)
    ok, failed, cache = [], {}, {}                                # cache: the NBS release list / pages, shared by its indicators
    for ind in indicators:
        try:
            got = latest_two(transform(fetch_series(ind, fetch, nbs_get, cache), ind.get("transform", "level")))
            if got is None:
                raise ValueError("no observations")
        except Exception as e:                                   # one source down must not stop the others
            failed[ind["id"]] = f"{type(e).__name__}: {e}"[:160]
            continue
        row = {"id": ind["id"], "region": ind["region"], "name": ind["name"], "unit": ind.get("unit", ""), **got,
               "source": ind["source"], "series": ind.get("series") or ind.get("dataset") or ind.get("key"),
               "frequency": ind.get("frequency", ""), "max_age_days": int(ind.get("max_age_days", 90)), "fetched_at": now}
        conn.execute(f"DELETE FROM {TABLE} WHERE id = ?", [row["id"]])
        conn.execute(f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                     [row[c] for c in COLUMNS])
        ok.append(ind["id"])
    if failed:
        logger.warning(f"   ⚠️ Economic indicators not refreshed ({len(failed)}): "
                       + "; ".join(f"{k} ({v})" for k, v in list(failed.items())[:5]))
    logger.info(f"   🌐 Economic indicators: {len(ok)} refreshed, {len(failed)} kept as stored")
    return {"ok": ok, "failed": failed}
