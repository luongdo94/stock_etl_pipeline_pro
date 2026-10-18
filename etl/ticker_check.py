"""
Ticker configuration checks.

- validate_config: offline structure check of config/tickers.yaml (run by the test suite, so a typo, a duplicated key
  or an inconsistent symbol change fails CI instead of a production run);
- main (`python run.py --check-tickers`): asks Yahoo whether every configured symbol still quotes and, for those that
  do not, searches for the company name so a renamed listing can be moved to `symbol_changes`.
"""
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "tickers.yaml"
REGIONS = {"US", "EU", "JP", "CN", "TW", "KR", "VN", "OTHER"}   # also the regional benchmark of the Earnings tab
REQUIRED = ("name", "sector", "region")


class _UniqueKeyLoader(yaml.SafeLoader):
    """yaml.safe_load silently keeps the LAST of two equal keys; a duplicated ticker must be an error."""


def _no_duplicates(loader, node, deep=False):
    keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
    dup = sorted({str(k) for k in keys if keys.count(k) > 1})
    if dup:
        raise ValueError(f"duplicated key(s) in tickers.yaml: {', '.join(dup)}")
    return loader.construct_mapping(node, deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def read_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_UniqueKeyLoader) or {}


def validate_config(cfg: dict) -> list:
    """Problems found in a parsed tickers.yaml ([] = valid)."""
    from etl.extract import _YAHOO_SYMBOL
    problems = []
    tickers = cfg.get("tickers") or {}
    changes = cfg.get("symbol_changes") or {}
    delisted = cfg.get("delisted") or {}
    for t, meta in tickers.items():
        # indices (^…), currency pairs (…=X) and futures (CL=F, GC=F) are Yahoo symbols too
        if not (_YAHOO_SYMBOL.match(str(t)) or str(t).startswith("^") or str(t).endswith(("=X", "=F"))):
            problems.append(f"{t}: not a Yahoo symbol")
        missing = [k for k in REQUIRED if not (meta or {}).get(k)]
        if missing:
            problems.append(f"{t}: missing {', '.join(missing)}")
        elif meta["region"] not in REGIONS:
            problems.append(f"{t}: unknown region {meta['region']!r} (expected one of {sorted(REGIONS)})")
    for old, new in changes.items():
        if old in tickers:
            problems.append(f"symbol_changes {old} → {new}: the OLD symbol is still configured")
        if new not in tickers:
            problems.append(f"symbol_changes {old} → {new}: the NEW symbol is not configured")
        if new in changes:
            problems.append(f"symbol_changes {old} → {new}: chained change ({new} → {changes[new]}); point {old} at the final symbol")
    for t in delisted:
        if t in tickers:
            problems.append(f"delisted {t}: still configured as an active ticker")
        if t in changes:
            problems.append(f"delisted {t}: also listed as a symbol change")
    return problems


def quoting(symbols: list, period: str = "1mo") -> set:
    """Symbols with at least one price in `period` (one batched Yahoo request)."""
    import yfinance as yf
    data = yf.download(symbols, period=period, progress=False, auto_adjust=True, group_by="ticker", threads=True)
    ok = set()
    for s in symbols:
        try:
            col = data[s]["Close"] if len(symbols) > 1 else data["Close"]
            if col.dropna().size:
                ok.add(s)
        except KeyError:
            continue
    return ok


def suggest(name: str, max_results: int = 5) -> list:
    """[(symbol, exchange)] Yahoo search hits for a company name."""
    import yfinance as yf
    try:
        return [(q.get("symbol"), q.get("exchange")) for q in yf.Search(name, max_results=max_results).quotes
                if q.get("quoteType") in (None, "EQUITY")]
    except Exception as e:
        logger.debug(f"search {name!r} failed: {e}")
        return []


def main() -> int:
    cfg = read_config()
    problems = validate_config(cfg)
    for p in problems:
        print(f"❌ config: {p}")
    tickers = cfg.get("tickers") or {}
    symbols = sorted(tickers)
    print(f"Checking {len(symbols)} symbols on Yahoo…")
    alive = set()
    for i in range(0, len(symbols), 100):
        alive |= quoting(symbols[i:i + 100])
    dead = [s for s in symbols if s not in alive]
    for s in dead:
        hits = suggest(tickers[s].get("name") or s)
        print(f"⚠️ {s} ({tickers[s].get('name')}): no price in the last month — candidates: "
              f"{', '.join(f'{h} [{x}]' for h, x in hits) or 'none (delisted? move it to `delisted:`)'}")
    print(f"{len(alive)} quoting, {len(dead)} not quoting, {len(problems)} config problem(s).")
    if dead:
        print("A renamed listing goes to `symbol_changes: {OLD: NEW}` (rename the key under `tickers:` too); a delisted "
              "one moves from `tickers:` to `delisted:`.")
    return 1 if (dead or problems) else 0
