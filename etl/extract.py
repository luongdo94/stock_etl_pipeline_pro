# etl/extract.py
import yfinance as yf
import pandas as pd
import inspect
import logging
import yaml
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import numpy as np
try:
    from yahooquery import Ticker as YQTicker
except ImportError:
    YQTicker = None

from etl import exceptions as _exc   # per-ticker exception ledger (marts.etl_exceptions)
from etl import fetch_guard as _fg   # provider guard: classify / cool down / circuit breaker / no-data cache

logger = logging.getLogger(__name__)

def _batch_statuses(batch, succeeded, frame=None, error: Optional[BaseException] = None) -> dict:
    """Per-ticker status of one batch request for the provider guard. A ticker that produced records is OK; otherwise
    the provider's own per-symbol answer when there is one; a failed request is classified from its exception; an
    unexplained miss is None (= one retry, not counted as a refusal)."""
    succeeded = set(succeeded)
    by_symbol = dict(getattr(frame, "attrs", {}).get("by_symbol") or {}) if frame is not None else {}
    overall = getattr(frame, "attrs", {}).get("status") if frame is not None else None
    out = {}
    for t in batch:
        if t in succeeded:
            out[t] = _fg.OK
        elif error is not None:
            out[t] = _fg.classify_exception(error)
        elif t in by_symbol:
            out[t] = by_symbol[t]
        elif overall == _fg.BLOCKED:
            out[t] = _fg.BLOCKED
        else:
            out[t] = None
    return out


def _guarded_retry(guard, failed, fetch_one, label: str) -> None:
    """ONE retry for refused / errored tickers (never for 'no data'), throttled by the guard; replaces the old
    pass 2 / pass 3 (the latter rotated browser User-Agents and is gone). fetch_one(ticker) -> fetch_guard status."""
    cands = guard.retry_candidates(failed)
    if not cands or guard.tripped:
        return
    logger.info(f"🔄 {label}: one retry for {len(cands)} refused / unexplained ticker(s)...")
    for t in cands:
        if guard.tripped:
            break
        try:
            s = fetch_one(t)
        except Exception as e:
            logger.debug(f"   ↳ retry {t}: {type(e).__name__}: {e}")
            s = _fg.classify_exception(e)
        guard.after_one(t, s)


def _merged_status(*frames) -> pd.DataFrame:
    """Status attrs for a frame built from several provider calls (e.g. income statement + balance sheet)."""
    st = [getattr(f, "attrs", {}).get("status") for f in frames if f is not None]
    by = {}
    for f in frames:
        for k, v in (getattr(f, "attrs", {}).get("by_symbol") or {}).items():
            by[k] = v if by.get(k) in (None, _fg.NO_DATA) else by[k]
    status = _fg.OK if _fg.OK in st else (_fg.BLOCKED if _fg.BLOCKED in st else _fg.NO_DATA)
    return status, by


def _run_guarded(name: str, ticker_keys, fetch, process, succeeded: set, label: str, batch_size: int = 40) -> dict:
    """Batch pass + ONE guarded retry for a statement-style extractor.
    fetch(symbols, asynchronous) -> DataFrame carrying attrs status / by_symbol (see _fetch_statement);
    process(frame) adds the tickers it could use to `succeeded`."""
    import time, random
    guard = _fg.Guard(name)
    asked = guard.active(ticker_keys)
    i = 0
    while i < len(asked) and not guard.tripped:
        bs = max(5, int(batch_size * guard.batch_scale))
        batch = asked[i:i + bs]
        i += bs
        logger.info(f"   📦 {label} batch ({min(i, len(asked))}/{len(asked)})...")
        frame, err = None, None
        try:
            frame = fetch(batch, True)
            process(frame)
        except Exception as e:
            logger.warning(f"   ⚠️ {label} batch failed: {e}")
            err = e
        guard.after_batch(_batch_statuses(batch, succeeded & set(batch), frame, err))
        if i < len(asked):
            time.sleep(1.0 + random.random())

    def _one(t):
        frame = fetch(t, False)
        process(frame)
        if t in succeeded:
            return _fg.OK
        attrs = getattr(frame, "attrs", {})
        s = (attrs.get("by_symbol") or {}).get(t) or attrs.get("status")
        return s if s in (_fg.NO_DATA, _fg.BLOCKED) else (_fg.NO_DATA if s == _fg.OK else _fg.ERROR)

    _guarded_retry(guard, [t for t in asked if t not in succeeded], _one, label)
    return guard.finish(ticker_keys, succeeded)


def _run_guarded_modules(name: str, ticker_keys, fetch, parse, succeeded: set, label: str, batch_size: int = 40,
                         pause=(1.0, 2.0)) -> dict:
    """Batch pass + ONE guarded retry for module-style endpoints answering {symbol: data}.
    fetch(symbols, asynchronous) -> dict (or an error payload); parse(ticker, data) -> True when records were added."""
    import time, random
    guard = _fg.Guard(name)
    asked = guard.active(ticker_keys)

    def _status(t, data, ok):
        if ok:
            succeeded.add(t)
            return _fg.OK
        s = _fg.classify(data)
        return s if s != _fg.OK else _fg.NO_DATA          # a well-formed answer without what we need = no data

    i = 0
    while i < len(asked) and not guard.tripped:
        bs = max(5, int(batch_size * guard.batch_scale))
        batch = asked[i:i + bs]
        i += bs
        logger.info(f"   📦 {label} batch ({min(i, len(asked))}/{len(asked)})...")
        statuses = {}
        try:
            raw = fetch(batch, True)
            if not isinstance(raw, dict):
                statuses = {t: _fg.classify(raw) for t in batch}
            else:
                for t in batch:
                    statuses[t] = _status(t, raw.get(t), parse(t, raw.get(t)))
        except Exception as e:
            logger.warning(f"   ⚠️ {label} batch failed: {e}")
            statuses = _batch_statuses(batch, set(), error=e)
        guard.after_batch(statuses)
        if i < len(asked):
            time.sleep(random.uniform(*pause))

    def _one(t):
        raw = fetch(t, False)
        data = raw.get(t) if isinstance(raw, dict) else raw
        return _status(t, data, parse(t, data))

    _guarded_retry(guard, [t for t in asked if t not in succeeded], _one, label)
    return guard.finish(ticker_keys, succeeded)


def load_tickers_config():
    """Load tickers from config file."""
    config_path = Path(__file__).parent.parent / "config" / "tickers.yaml"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config.get("tickers", {})
    except Exception as e:
        logger.warning(f"⚠️ Failed to load tickers config: {e}. Using empty config.")
        return {}

def load_sector_overrides() -> dict:
    """{ticker: sector} corrections from tickers.yaml `sector_overrides:` - they win over the configured sector and over
    the sector a discovery screen assigned (the scores treat banks / insurers / cyclicals by sector name)."""
    config_path = Path(__file__).parent.parent / "config" / "tickers.yaml"
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = (yaml.safe_load(f) or {}).get("sector_overrides") or {}
        return {str(k): str(v) for k, v in raw.items() if v}
    except Exception as e:
        logger.warning(f"⚠️ sector_overrides not read: {e}")
        return {}


def load_sector_aliases() -> dict:
    """{old sector name: canonical name} from tickers.yaml `sector_aliases:` - one name per kind of business, applied to
    configured AND discovered tickers (a screen invents its own labels)."""
    config_path = Path(__file__).parent.parent / "config" / "tickers.yaml"
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = (yaml.safe_load(f) or {}).get("sector_aliases") or {}
        return {str(k): str(v) for k, v in raw.items() if v and str(k) != str(v)}
    except Exception as e:
        logger.warning(f"⚠️ sector_aliases not read: {e}")
        return {}


def load_sector_by_industry() -> tuple:
    """(catch-all sector names, {Yahoo industry: sector}) from tickers.yaml `generic_sectors:` / `sector_by_industry:`."""
    config_path = Path(__file__).parent.parent / "config" / "tickers.yaml"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return ({str(s) for s in cfg.get("generic_sectors") or []},
                {str(k): str(v) for k, v in (cfg.get("sector_by_industry") or {}).items() if v})
    except Exception as e:
        logger.warning(f"⚠️ sector_by_industry not read: {e}")
        return set(), {}


# --- TRADINGVIEW AUTO-DISCOVERY ---
_YAHOO_SYMBOL = __import__("re").compile(r"^[A-Z0-9][A-Z0-9\-&]*(\.[A-Z]{1,3})?$")


_EU_COUNTRIES = {"Germany", "France", "United Kingdom", "Netherlands", "Spain", "Italy", "Switzerland", "Sweden",
                 "Denmark", "Norway", "Finland", "Belgium", "Austria", "Portugal", "Ireland", "Poland", "Luxembourg"}
_REGION_OF_COUNTRY = {"United States": "US", "Japan": "JP", "China": "CN", "Hong Kong": "CN", "Taiwan": "TW",
                      "South Korea": "KR", "Korea": "KR", "Vietnam": "VN", **{c: "EU" for c in _EU_COUNTRIES}}


def region_code(country) -> str:
    """TradingView country → the region codes of config/tickers.yaml (scanner filter, regional earnings benchmark)."""
    return _REGION_OF_COUNTRY.get(str(country or "").strip(), "OTHER")


def tv_to_yahoo_symbol(raw_ticker: str, suffix: str):
    """TradingView 'EXCHANGE:SYMBOL' symbol part → Yahoo ticker, or None when it cannot be mapped safely.
    LSE symbols carry a trailing dot ('BP.') and use a dot for the share class ('BT.A'); Yahoo writes BP.L / BT-A.L
    (the review found 'BP..L' in the universe, 2026-10-14)."""
    raw = str(raw_ticker or "").strip().upper().rstrip(".")
    if not raw or " " in raw or "/" in raw:
        return None
    if "." in raw:                                   # share class: BT.A → BT-A (Yahoo convention)
        raw = raw.replace(".", "-")
    sym = f"{raw}{suffix}"
    return sym if _YAHOO_SYMBOL.match(sym) else None


class Discovery(dict):
    """{yahoo ticker: meta} of a TradingView scan, plus `failed`: the presets that did not answer (a partial scan)."""

    def __init__(self, *args, failed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed = list(failed or [])


def fetch_dynamic_tv_tickers(base_tickers=None):
    """TradingView screens → Discovery. Raises when EVERY preset failed (a failed scan is not "nothing found" — review
    SIG-R04); a partial scan lists its failed presets in `.failed`. ETL callers keep the previous universe on error."""
    if base_tickers is None:
        base_tickers = {}
    
    import requests
    tv_presets = {
        "value_stocks": {
            "filters": [
                {"left": "market_cap_basic", "operation": "greater", "right": 2000000000},
                {"left": "close", "operation": "greater", "right": 5},
                {"left": "volume", "operation": "greater", "right": 500000},
                {"left": "price_earnings_ttm", "operation": "less", "right": 15},
                {"left": "price_earnings_ttm", "operation": "greater", "right": 0},
                {"left": "price_book_ratio", "operation": "less", "right": 1.5},
                {"left": "price_book_ratio", "operation": "greater", "right": 0},
                {"left": "dividend_yield_recent", "operation": "greater", "right": 2}
            ],
            "sort": {"sortBy": "price_earnings_ttm", "sortOrder": "asc"}
        },
        "growth_at_reasonable_price": {
            "filters": [
                {"left": "market_cap_basic", "operation": "greater", "right": 2000000000},
                {"left": "close", "operation": "greater", "right": 5},
                {"left": "volume", "operation": "greater", "right": 500000},
                {"left": "earnings_per_share_yoy_growth_ttm", "operation": "greater", "right": 15},
                {"left": "total_revenue_yoy_growth_ttm", "operation": "greater", "right": 10},
                {"left": "price_earnings_ttm", "operation": "less", "right": 25},
                {"left": "price_earnings_ttm", "operation": "greater", "right": 0}
            ],
            "sort": {"sortBy": "earnings_per_share_yoy_growth_ttm", "sortOrder": "desc"}
        },
        "breakout_momentum": {
            "filters": [
                {"left": "market_cap_basic", "operation": "greater", "right": 2000000000},
                {"left": "close", "operation": "greater", "right": 5},
                {"left": "volume", "operation": "greater", "right": 2000000},
                {"left": "SMA50", "operation": "greater", "right": "SMA200"},
                {"left": "close", "operation": "greater", "right": "SMA50"},
                {"left": "RSI", "operation": "greater", "right": 60},
                {"left": "RSI", "operation": "less", "right": 75}
            ],
            "sort": {"sortBy": "RSI", "sortOrder": "desc"}
        },
        "quality_compounders": {
            "filters": [
                {"left": "market_cap_basic", "operation": "greater", "right": 5000000000},
                {"left": "close", "operation": "greater", "right": 5},
                {"left": "volume", "operation": "greater", "right": 500000},
                {"left": "return_on_invested_capital", "operation": "greater", "right": 15},
                {"left": "return_on_equity", "operation": "greater", "right": 20},
                {"left": "operating_margin", "operation": "greater", "right": 15},
                {"left": "debt_to_equity", "operation": "less", "right": 0.5}
            ],
            "sort": {"sortBy": "return_on_invested_capital", "sortOrder": "desc"}
        },
        "high_yield_dividend": {
            "filters": [
                {"left": "market_cap_basic", "operation": "greater", "right": 5000000000},
                {"left": "close", "operation": "greater", "right": 5},
                {"left": "volume", "operation": "greater", "right": 500000},
                {"left": "dividend_yield_recent", "operation": "greater", "right": 4},
                {"left": "payout_ratio", "operation": "less", "right": 60},
                {"left": "payout_ratio", "operation": "greater", "right": 0},
                {"left": "total_revenue_yoy_growth_ttm", "operation": "greater", "right": 0},
                {"left": "debt_to_equity", "operation": "less", "right": 1.0}
            ],
            "sort": {"sortBy": "dividend_yield_recent", "sortOrder": "desc"}
        }
    }
    
    global_markets = ["america", "vietnam", "uk", "germany", "france", "japan", "hongkong", "china", "australia", "canada", "india", "brazil", "taiwan", "korea"]
    
    prefix_map = {
        "NASDAQ": "", "NYSE": "", "AMEX": "", "OTC": "",
        "XETR": ".DE", "FWB": ".DE", "DUS": ".DE", "MUN": ".DE", "TRADEGATE": ".DE", "BER": ".DE", "GETTEX": ".DE",
        "LSE": ".L", "HOSE": ".VN", "HNX": ".VN", "UPCOM": ".VN",
        "TSE": ".T", "FSE": ".F", "EURONEXT": ".PA", "PAR": ".PA",
        "ASX": ".AX", "TSX": ".TO", "TSXV": ".V", "CSE": ".CN",
        "HKEX": ".HK", "SSE": ".SS", "SZSE": ".SZ", "NSE": ".NS", "BSE": ".BO",
        "BMFBOVESPA": ".SA", "TWSE": ".TW", "TPEX": ".TWO", "KRX": ".KS", "KOSDAQ": ".KQ"
    }

    dynamic_tickers = {}
    
    import re
    def normalize_name(n):
        if not n: return ""
        n = re.sub(r'[^\w\s]', '', n.lower())
        for suffix in ["inc", "corp", "corporation", "ltd", "limited", "company", "co", "plc", "nv", "sa", "ag"]:
            n = re.sub(fr'\b{suffix}\b', '', n)
        return re.sub(r'\s+', ' ', n).strip()

    # Pre-populate seen_names to avoid fetching aliases of existing companies
    seen_names = set()
    for meta in base_tickers.values():
        clean_name = normalize_name(meta.get("name", ""))
        if len(clean_name) > 3:
            seen_names.add(clean_name)
        
    url = "https://scanner.tradingview.com/global/scan"
    failed = []
    
    for preset_name, preset_data in tv_presets.items():
        payload = {
            "filter": preset_data["filters"],
            "options": {"lang": "en"},
            "markets": global_markets,
            "symbols": {"query": {"types": ["stock"]}, "tickers": []},
            "columns": ["name", "description", "sector", "country", "exchange"],
            "sort": preset_data["sort"],
            "range": [0, 20]
        }
        
        try:
            r = requests.post(url, json=payload, timeout=10)
            r.raise_for_status()                              # 429 / 5xx is a failure, not an empty screen
            data = r.json()
            if not isinstance(data, dict) or "data" not in data:
                raise ValueError("response without a 'data' field")
            for d in (data.get('data') or []):
                tv_symbol = d['s']
                parts = tv_symbol.split(':')
                if len(parts) < 2: continue
                exchange = parts[0]
                raw_ticker = parts[1]
                
                f = d['d']
                name = f[1]
                
                # Exclude duplicate companies (e.g. cross-listed)
                clean_name = normalize_name(name)
                
                # Exclude preferred stocks / depositary shares explicitly by name
                name_lower = name.lower()
                if any(x in name_lower for x in ["preferred", "pfd", "depositary share", "warrant", "right"]):
                    continue
                    
                # Check if this clean_name starts with any seen_name or vice versa
                if any(clean_name.startswith(sn) or sn.startswith(clean_name) for sn in seen_names):
                    continue
                    
                seen_names.add(clean_name)
                
                # Determine Yahoo Finance ticker
                suffix = prefix_map.get(exchange, None)
                if suffix is None:
                    # Skip unknown/obscure exchanges
                    continue
                    
                yf_ticker = tv_to_yahoo_symbol(raw_ticker, suffix)

                # Clean up weird tickers with spaces or invalid characters
                if yf_ticker is None or "-" in raw_ticker:
                    continue
                    
                dynamic_tickers[yf_ticker] = {
                    "name": name,
                    "sector": f[2] if f[2] else "N/A",
                    "region": region_code(f[3] if len(f) > 3 else None),
                    "discovery_source": f"TV_{preset_name.upper()}"
                }
        except Exception as e:
            failed.append(f"{preset_name} ({type(e).__name__})")
            logger.warning(f"⚠️ Failed to fetch TV tickers for {preset_name}: {e}")

    if len(failed) == len(tv_presets):
        raise RuntimeError("TradingView discovery failed for every preset: " + ", ".join(failed))
    logger.info(f"🔮 TV Auto-Discovery found {len(dynamic_tickers)} unique dynamic tickers globally"
                + (f" (failed: {', '.join(failed)})" if failed else "") + ".")
    return Discovery(dynamic_tickers, failed=failed)

def get_combined_tickers():
    base_tickers = load_tickers_config()
    try:
        dynamic = fetch_dynamic_tv_tickers(base_tickers)
        # Merge them (base takes precedence to prevent overwriting known config)
        return {**dynamic, **base_tickers}
    except Exception as e:
        logger.error(f"⚠️ Error during TV auto-discovery: {e}")
        return base_tickers

# Static universe from config/tickers.yaml. The TradingView-discovered tickers are resolved once per
# pipeline run (etl.universe.resolve_universe) and passed explicitly — this module no longer touches the
# network when it is imported (the dashboard and every test import it).
TICKERS = load_tickers_config()

def get_equity_tickers(tickers_pool: dict = TICKERS) -> dict:
    """
    Filter a pool of tickers to include only corporate equities.
    Excludes Indices (starting with ^) and ETFs/Benchmarks (sector: Benchmark).
    """
    return {
        t: meta for t, meta in tickers_pool.items() 
        if not t.startswith('^') and meta.get('sector') != 'Benchmark'
    }

# quote currency always verified: London lines trade in GBp / GBP / USD / EUR; Johannesburg quotes in ZAc (cents)
VERIFY_QUOTE_CURRENCY_SUFFIXES = (".L", ".IL", ".JO")


def _guess_currency(ticker: str) -> str:
    """Heuristic to guess currency for fast FX pre-fetching."""
    if ticker.endswith(".T"): return "JPY"
    if any(ticker.endswith(s) for s in [".DE", ".PA", ".AS", ".MI", ".MC", ".LS", ".BR"]): return "EUR"
    if ".CO" in ticker: return "DKK"
    if ".HK" in ticker: return "HKD"
    if any(ticker.endswith(s) for s in [".SS", ".SZ"]): return "CNY"
    if any(ticker.endswith(s) for s in [".L", ".IL"]): return "GBp"  # UK stocks trade in pence
    if any(ticker.endswith(s) for s in [".TO", ".V"]): return "CAD"
    if ".AX" in ticker: return "AUD"
    if ".SW" in ticker: return "CHF"
    if ".ST" in ticker: return "SEK"
    if ".HE" in ticker: return "EUR" # Finland
    if ".OL" in ticker: return "NOK"
    if ".TW" in ticker: return "TWD"
    if any(ticker.endswith(s) for s in [".KS", ".KQ"]): return "KRW"
    if any(ticker.endswith(s) for s in [".BO", ".NS"]): return "INR"
    if ticker.endswith(".SA"): return "BRL"
    if ticker.endswith(".MX"): return "MXN"
    if ticker.endswith(".SI"): return "SGD"
    if ticker.endswith(".JO"): return "ZAR"
    return "USD"

# ── FX to EUR (single implementation for every extractor) ─────────────────────
# Minor currency units Yahoo uses for QUOTES (prices, targets). Company-level amounts
# (revenue, cash flow, market cap) are always reported in the MAJOR unit.
from core.currency import MINOR_UNITS as _MINOR_UNITS, major_currency  # noqa: E402  (single definition, F-28)
_FX_CACHE: dict = {}   # major currency → EUR rate (None = unavailable this run)


def prefetch_fx_to_eur(currencies) -> None:
    """Download the latest {CCY}EUR=X close for every currency not cached yet."""
    majors = sorted({major_currency(c) for c in currencies if c} - {"EUR", None} - set(_FX_CACHE))
    if not majors:
        return
    tickers = [f"{m}EUR=X" for m in majors]
    close = pd.DataFrame()
    try:
        dl = yf.download(tickers, period="7d", progress=False, auto_adjust=False)
        if not dl.empty and "Close" in dl.columns.get_level_values(0):
            close = dl["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
    except Exception as e:
        logger.error(f"   ❌ FX download failed for {tickers}: {e}")
    for m, t in zip(majors, tickers):
        series = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
        if series.empty:
            logger.error(f"   ❌ No FX rate for {m}→EUR — {m} amounts will be left EMPTY, not stored unconverted.")
            _FX_CACHE[m] = None
        else:
            _FX_CACHE[m] = float(series.iloc[-1])


def fx_to_eur(ccy: Optional[str]) -> Optional[float]:
    """
    Multiplier converting an amount in `ccy` to EUR, or None when unknown.
    Minor units are handled: GBp (pence) → GBP rate / 100. 'GBP' means pounds.
    Callers must treat None as "cannot convert" — never fall back to 1.0.
    """
    if not ccy:
        return None
    major = major_currency(ccy)
    if major == "EUR":
        rate = 1.0
    else:
        if major not in _FX_CACHE:
            prefetch_fx_to_eur([major])
        rate = _FX_CACHE.get(major)
        if rate is None:
            return None
    return rate / 100.0 if str(ccy) in _MINOR_UNITS else rate


def _statement_currency(t_data: pd.DataFrame, ticker: str) -> str:
    """Currency of a yahooquery financial statement (currencyCode column), else the market's major unit."""
    if "currencyCode" in t_data.columns:
        codes = t_data["currencyCode"].dropna()
        if not codes.empty:
            return str(codes.iloc[-1])
    return major_currency(_guess_currency(ticker))


def _one_period_per_key(df: pd.DataFrame, keys: list, stage: str) -> pd.DataFrame:
    """Cash-flow rows are keyed by fiscal YEAR (or calendar year/quarter). Audit F14: a fiscal-year change gives two
    period ends in one key, and the second silently replaced the first downstream. The latest period end is kept, and
    every dropped period is recorded in the exception ledger with both dates."""
    if df is None or df.empty or "period_end" not in df.columns or not set(keys) <= set(df.columns):
        return df
    from etl import exceptions as _exc
    df = df.sort_values("period_end")
    dup = df.duplicated(keys, keep="last")
    for r in df[dup].itertuples():
        kept = df[~dup & (df[keys] == pd.Series({k: getattr(r, k) for k in keys})).all(axis=1)]["period_end"]
        _exc.record(r.ticker, stage, "two fiscal periods in one key (fiscal-year change?) — the latest is kept",
                    f"{'/'.join(str(getattr(r, k)) for k in keys[1:])}: dropped period ending {r.period_end}, kept "
                    f"{kept.iloc[0] if len(kept) else '?'}")
    return df[~dup].reset_index(drop=True)


def _row_currencies(t_data: pd.DataFrame, ticker: str) -> tuple:
    """(currency per row, EUR rate per row) for a statement frame. Audit F13: one currency used to be applied to every
    period, so a filer that switched from USD to EUR reporting had its old USD accounts converted as EUR. Each period
    now keeps its own currencyCode (rows without one fall back to the statement's currency); a row whose currency has
    no EUR rate gets NaN and is dropped by the caller, instead of the whole ticker."""
    fallback = _statement_currency(t_data, ticker)
    ccy = (t_data["currencyCode"].where(t_data["currencyCode"].notna(), fallback).astype(str)
           if "currencyCode" in t_data.columns else pd.Series(fallback, index=t_data.index))
    rates = {c: fx_to_eur(c) for c in ccy.unique()}
    return ccy, ccy.map(lambda c: rates.get(c) if rates.get(c) is not None else np.nan).astype(float)


FX_MIN_COVERAGE = 0.9     # a direct {CUR}EUR=X series observed on fewer of the dates than this is replaced


def _with_usd_cross_rates(fx_raw: pd.DataFrame, start, end) -> pd.DataFrame:
    """Replace sparse direct {CUR}EUR=X columns by the same-day cross rate EUR=X / {CUR}=X (both quoted per USD).
    A date where the cross is missing keeps the direct value; nothing is back-filled. Non-USD-cross failures leave
    the direct column as it was (and the ticker is skipped later with a logged exception)."""
    if fx_raw is None or fx_raw.empty:
        return fx_raw
    sparse = [c for c in fx_raw.columns
              if str(c).endswith("EUR=X") and fx_raw[c].notna().mean() < FX_MIN_COVERAGE and str(c) != "USDEUR=X"]
    if not sparse:
        return fx_raw
    legs = {c: f"{str(c)[:-5].upper()}=X" for c in sparse}
    try:
        dl = yf.download(["EUR=X"] + sorted(set(legs.values())), start=start, end=end, progress=False)
        close = dl["Close"] if not dl.empty else pd.DataFrame()
    except Exception as e:
        logger.warning(f"    ⚠️ USD cross-rate download failed ({e}) — sparse FX pairs kept as they are")
        return fx_raw
    if "EUR=X" not in getattr(close, "columns", []):
        return fx_raw
    out = fx_raw.copy()
    idx = out.index.union(close.index)
    out = out.reindex(idx)
    for c, leg in legs.items():
        if leg not in close.columns:
            continue
        cross = (close["EUR=X"] / close[leg]).reindex(idx)
        before = out[c].notna().mean()
        out[c] = cross.combine_first(out[c])
        logger.info(f"    💱 {c}: direct series covered {before:.0%} of dates → USD cross rate used "
                    f"({out[c].notna().mean():.0%})")
    return out


def _fetch_statement(yq, method: str, frequency: str) -> pd.DataFrame:
    """
    Call a yahooquery financial-statement method (income_statement / balance_sheet / cash_flow).

    - `trailing=False` is passed only if the installed yahooquery accepts it: balance_sheet() has no
      such parameter (TypeError otherwise), income_statement()/cash_flow() do.
    - yahooquery returns a str/dict error payload instead of a DataFrame when Yahoo has no data
      for a symbol; that is "no data" (empty frame), not a crash.
    """
    fn = getattr(yq, method)
    try:
        accepts_trailing = "trailing" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        accepts_trailing = False
    kwargs = {"frequency": frequency}
    if accepts_trailing:
        kwargs["trailing"] = False
    result = fn(**kwargs)
    if not isinstance(result, pd.DataFrame):
        logger.debug(f"   ↳ {method}({frequency}) returned {type(result).__name__} (no data): {str(result)[:120]}")
        out = pd.DataFrame()
        # what the provider actually said, per symbol (a dict {symbol: message}) or overall — read by the guard
        out.attrs["status"] = _fg.classify(result)
        out.attrs["by_symbol"] = ({str(k): _fg.classify(v) for k, v in result.items()}
                                  if isinstance(result, dict) else {})
        return out
    result.attrs["status"] = _fg.OK if not result.empty else _fg.NO_DATA
    return result


def _safe_float(val):
    """None-safe float cast — no FX conversion. Use for per-share metrics
    (EPS, target price) that Yahoo already reports in local currency."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def _overlap_days() -> int:
    """Days re-downloaded before the watermark (config/etl_config.yaml: incremental_load.overlap_buffer_days)."""
    from etl.config_manager import get_etl_config
    return int(get_etl_config()["incremental_load"]["overlap_buffer_days"])


WATERMARK_BUCKET_DAYS = 7   # tickers whose watermarks lie within this many days share one batch download


def watermark_buckets(watermarks: dict, overlap_days: int, bucket_days: int = WATERMARK_BUCKET_DAYS) -> list:
    """[(start datetime, [tickers])] — tickers grouped by watermark, oldest first. A bucket starts at its OLDEST
    watermark minus the overlap, so every ticker re-downloads at least `overlap_days` before its own watermark; a
    ticker joins the open bucket while its watermark is within `bucket_days` of that bucket's oldest one."""
    out = []
    for t, d in sorted(watermarks.items(), key=lambda kv: (kv[1], kv[0])):
        if out and (d - out[-1][0]).days <= bucket_days:
            out[-1][1].append(t)
        else:
            out.append((d, [t]))
    return [(datetime.combine(d, datetime.min.time()) - timedelta(days=overlap_days), ts) for d, ts in out]


def price_backfill_targets(conn, watermarks, lookback_days, end=None):
    """Retry truncated histories at most weekly; attempts live in the shadow alongside price rows."""
    conn.execute("""CREATE TABLE IF NOT EXISTS raw.price_backfill_attempts (
        ticker VARCHAR PRIMARY KEY, attempted_at TIMESTAMP, rows_found INTEGER)""")
    if not watermarks:
        return []
    end = pd.Timestamp(end if end is not None else datetime.now())
    first = dict(conn.execute("SELECT ticker, MIN(date) FROM raw.stock_prices GROUP BY ticker").fetchall())
    attempts = dict(conn.execute("SELECT ticker, attempted_at FROM raw.price_backfill_attempts").fetchall())
    cutoff = end - timedelta(days=lookback_days) + timedelta(days=30)
    return [t for t in watermarks if first.get(t) is not None and pd.Timestamp(first[t]) > cutoff
            and (attempts.get(t) is None or end - pd.Timestamp(attempts[t]) >= timedelta(days=7))]


def extract_stock_prices(
    tickers: dict = TICKERS,
    lookback_days: int = 365,
    watermarks: dict = None
) -> pd.DataFrame:
    """
    EXTRACT TURBO: Vectorized extraction of daily OHLCV data.

    Incremental Load Support:
        If `watermarks` is provided (dict of {ticker: last_date}), each ticker
        will only download data from (last_date - 2 days) onward. The 2-day
        overlap buffer handles timezone mismatches and market holiday gaps.

        New tickers (not in watermarks) always get the full `lookback_days`
        so historical data is bootstrapped correctly on first run.
        The pipeline also removes short histories from the watermarks; an empty dict bootstraps all of them.
    """
    end_date   = datetime.now()
    all_ticker_list = list(tickers.keys())

    # ── INCREMENTAL: Calculate per-ticker start dates ─────────────────────────
    buckets = []
    if watermarks is not None:
        # Only the requested tickers' watermarks count, and tickers are downloaded in buckets of similar watermarks
        # (PERF-02): one stale or halted name (EA stopped at 2026-08-04) used to widen the download of every other
        # ticker to its date. Each ticker still starts at or before its own watermark minus the overlap buffer.
        buckets = watermark_buckets({t: d for t, d in watermarks.items() if t in tickers}, _overlap_days())
        start_date = min((b[0] for b in buckets), default=end_date - timedelta(days=_overlap_days()))
        new_tickers = [t for t in all_ticker_list if t not in watermarks]
        if new_tickers:
            # For brand-new tickers, we need to use the full lookback
            # We handle this by doing two separate downloads
            full_start = end_date - timedelta(days=lookback_days)
            logger.info(f"  📦 {len(new_tickers)} new tickers detected → Full bootstrap ({lookback_days}d)")
        else:
            full_start = None
        for b_start, b_tickers in buckets:
            names = f" ({', '.join(b_tickers)})" if len(b_tickers) <= 5 else ""
            logger.info(f"  ⚡ {len(b_tickers)} existing tickers → Incremental from {b_start.date()}{names}")
    else:
        start_date = end_date - timedelta(days=lookback_days)
        full_start = None
        new_tickers = []
        logger.info(f"🚀 FULL EXTRACT: Downloading {len(all_ticker_list)} tickers ({lookback_days}d history)...")

    all_frames = []

    def _batch_download(batch, start):
        # A batch-level API error (rate limit, network) must not abort the whole run:
        # treat it as "nothing received" and let PASS 2 retry ticker by ticker.
        try:
            return yf.download(batch, start=start, end=end_date, auto_adjust=True,
                               progress=False, group_by='column')
        except Exception as e:
            logger.warning(f"⚠️ Batch price download failed for {len(batch)} tickers: {e}")
            return pd.DataFrame()

    # ── BATCH DOWNLOAD: Incremental (or Full if no watermarks) ───────────────
    existing_tickers = [t for t in all_ticker_list if t not in new_tickers]
    ticker_start = {t: b_start for b_start, b_tickers in buckets for t in b_tickers}
    for b_start, b_tickers in (buckets if watermarks is not None else [(start_date, existing_tickers)] if existing_tickers else []):
        raw_prices = _batch_download(b_tickers, b_start)

        if raw_prices.empty:
            logger.warning(f"⚠️ No price data returned for {len(b_tickers)} existing tickers in the incremental window.")
        else:
            all_frames.append(("existing", b_tickers, raw_prices))

    # ── BATCH DOWNLOAD: Full history for brand-new tickers ───────────────────
    if new_tickers and full_start:
        raw_new = _batch_download(new_tickers, full_start)
        if not raw_new.empty:
            all_frames.append(("new", new_tickers, raw_new))

    # ── PASS 2: SURGICAL RECOVERY FOR FAILED TICKERS ─────────────────────────
    # Identify tickers that were requested but are missing from the batch results
    received_tickers = set()
    for _, _tl, _df in all_frames:
        if isinstance(_df.columns, pd.MultiIndex):
            received_tickers.update(_df.columns.get_level_values(1).unique())
        elif len(_tl) == 1:
            # Single ticker case (a one-ticker batch or bucket)
            received_tickers.add(_tl[0])

    failed_tickers = [t for t in all_ticker_list if t not in received_tickers]
    
    if failed_tickers:
        logger.info(f"🔄 PASS 2: Surgical Recovery for {len(failed_tickers)} failed price downloads: {failed_tickers}")
        for ticker in failed_tickers:
            try:
                # Use single-ticker history call which is often more resilient than batch download
                t_obj = yf.Ticker(ticker)
                # Determine correct start date for this ticker
                t_start = (ticker_start.get(ticker, start_date) if ticker in existing_tickers
                           else (full_start if full_start else start_date))
                
                s_df = t_obj.history(start=t_start, end=end_date, auto_adjust=True)
                if not s_df.empty:
                    # Format to match batch download structure for downstream processing
                    all_frames.append(("recovered", [ticker], s_df))
                    logger.info(f"   ✅ Recovered price data for: {ticker}")
                else:
                    logger.warning(f"   ❌ Recovery failed for {ticker}: No data returned.")
            except Exception as e:
                logger.warning(f"   ❌ Recovery error for {ticker}: {e}")

    if not all_frames:
        if watermarks is None:
            raise ValueError("❌ No price data returned from Yahoo Finance.")
        # Incremental run with nothing new (weekend/holiday): skip currency + FX API calls
        logger.info("ℹ️ No new price rows in the incremental window.")
        return pd.DataFrame()

    # 2. RESOLVE CURRENCIES: Heuristic-first, API only for ambiguous tickers
    # _guess_currency() correctly identifies ~95% of non-US tickers by suffix.
    # Only "USD" results (tickers where the heuristic couldn't determine) go to the live API.
    currencies = {}
    ticker_keys = list(all_ticker_list)

    # Pass 1: Heuristic resolution (free, instant)
    # E4: London listings (.L / .IL) quote in GBp, GBP, USD or EUR depending on the line — the suffix does not tell
    # which, and a wrong guess mis-scales the whole history by 100x. They are always verified against the quote.
    ambiguous = []
    for t in ticker_keys:
        guessed = _guess_currency(t)
        if guessed != "USD" and not t.endswith(VERIFY_QUOTE_CURRENCY_SUFFIXES):
            currencies[t] = guessed  # High-confidence non-USD: HK, JP, EU, etc.
        else:
            # Could be US stock OR a non-suffix ticker we can't determine — check via API
            ambiguous.append(t)

    def fetch_currency(t):
        guess = _guess_currency(t)
        try:
            cur = yf.Ticker(t).fast_info.get("currency")
            if cur:
                if guess != "USD" and major_currency(cur) != major_currency(guess):
                    logger.info(f"   💱 {t}: quote currency {cur} (suffix suggested {guess})")
                return t, cur
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch currency for {t}: {e}")
        # unverified: keep the heuristic, but say so (a suffix guess is not evidence)
        _exc.record(t, "prices", f"quote currency UNVERIFIED — suffix guess {guess} used")
        return t, guess

    # Pass 2: API for ambiguous tickers only (most are genuine USD stocks)
    if ambiguous:
        logger.info(f"   💱 Resolving currency for {len(ambiguous)} ambiguous tickers via API...")
        from etl.config_manager import get_etl_config
        max_workers = int(get_etl_config()["extraction"]["max_workers"])
        batch_size  = 80  # Larger batch since we expect fewer failures
        for i in range(0, len(ambiguous), batch_size):
            batch = ambiguous[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_tick = {executor.submit(fetch_currency, t): t for t in batch}
                for future in as_completed(future_to_tick):
                    t, cur = future.result()
                    currencies[t] = cur
            if i + batch_size < len(ambiguous):
                import time
                time.sleep(1.0)

    solved = sum(1 for t in ticker_keys if t not in [a for a in ambiguous])
    logger.info(f"   💱 Currency resolved: {solved} via heuristic, {len(ambiguous)} via API")

    unique_currencies = {major_currency(c) for c in currencies.values() if major_currency(c) not in (None, "EUR")}
    fx_data = pd.DataFrame()

    if unique_currencies:
        # Standardize on {CUR}EUR=X format for direct conversion to Euro
        fx_tickers = [f"{c}EUR=X" for c in unique_currencies]
        logger.info(f"    💱 Downloading FX rates to EUR for: {unique_currencies}")
        # FX history must cover the earliest price row (new tickers bootstrap from full_start)
        fx_start = min(d for d in (start_date, full_start) if d is not None)
        _fx_dl = yf.download(fx_tickers, start=fx_start, end=end_date, progress=False)
        if _fx_dl.empty or "Close" not in _fx_dl.columns.get_level_values(0):
            # Storing local-currency prices as EUR would silently corrupt the warehouse
            raise RuntimeError(f"❌ FX download returned no data for {fx_tickers} — aborting price extract.")
        _fx_raw = _fx_dl["Close"]
        if isinstance(_fx_raw, pd.Series):
            # Must use the "{CUR}EUR=X" name: the lookup below searches for exactly that column.
            # Naming it after the bare currency made the lookup miss → prices left unconverted.
            _fx_raw = _fx_raw.to_frame(name=fx_tickers[0])
        # Some direct pairs are nearly empty on Yahoo (TWDEUR=X / VNDEUR=X carried ONE value in 10 days, 2026-10):
        # the old back-fill spread today's rate over five years of prices; without it the tickers were dropped.
        # Same-day cross rates via USD (EUR=X = EUR per USD, {CUR}=X = CUR per USD) replace a sparse direct series.
        _fx_raw = _with_usd_cross_rates(_fx_raw, fx_start, end_date)
        # Forward fill only: a back-fill wrote a LATER rate into earlier dates (look-ahead, R3-D5). Dates before the
        # first available rate get no rate and are dropped below.
        fx_data = _fx_raw.ffill()

    # 3. VECTORIZED NORMALIZATION & FORMATTING
    # Process each download batch (may have 1 or 2: existing tickers + new tickers)
    frames = []
    for _label, _ticker_list, _raw_prices in all_frames:
        for ticker in _ticker_list:
            try:
                # Handle MultiIndex correctly (yfinance returns MultiIndex if tickers provided as list)
                if isinstance(_raw_prices.columns, pd.MultiIndex):
                    if ticker in _raw_prices.columns.get_level_values(1):
                        df = _raw_prices.xs(ticker, axis=1, level=1).copy()
                    else:
                        logger.warning(f"  ⚠️ Ticker {ticker} not found in download results")
                        continue
                else:
                    df = _raw_prices.copy()

                # 🏆 DEFENSIVE SHIELD: Ensure we have full OHLC data. 
                # Dropping rows with NaN in any price column to prevent "garbage-in" corruption.
                df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
                if df.empty: continue

                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]

                # Apply FX normalization (EUR is the baseline for ETL). The conversion is recorded per row
                # (currency, fx_rate, price_scale): close_eur = close_local * fx_rate / price_scale, so it can be
                # audited or reversed without downloading again.
                currency = currencies.get(ticker, "EUR")
                scale_factor = 1.0
                if currency != "EUR":
                    # E4: quotes in a MINOR unit (GBp, GBX, ZAc, ILA — core.currency.MINOR_UNITS) use the MAJOR
                    # currency's rate and are divided by 100; the old code special-cased GBp only.
                    _major = major_currency(currency)
                    fx_col = f"{_major}EUR=X" if f"{_major}EUR=X" in fx_data.columns else (
                        f"{currency}EUR=X" if f"{currency}EUR=X" in fx_data.columns else None)
                    if fx_data.empty or fx_col is None:
                        logger.error(f"  ❌ {ticker}: no EUR rate for {currency} — ticker skipped (a price is never stored unconverted)")
                        _exc.record(ticker, 'prices', 'no EUR rate for the currency')
                        continue
                    rates = fx_data[[fx_col]].reset_index()
                    rates.columns = ["date", "fx_rate"]
                    rates = rates.dropna(subset=["fx_rate"])
                    # Each price date takes the latest rate known ON or BEFORE it — never a later one (R3-D5)
                    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
                    rates["date"] = pd.to_datetime(rates["date"]).astype("datetime64[ns]")
                    df = pd.merge_asof(df.sort_values("date"), rates.sort_values("date"), on="date",
                                       direction="backward")
                    _no_rate = df["fx_rate"].isna()
                    if _no_rate.all():
                        logger.error(f"  ❌ {ticker}: EUR rate for {currency} has no value for its dates — ticker skipped")
                        _exc.record(ticker, 'prices', 'EUR rate has no value for the price dates')
                        continue
                    if _no_rate.any():
                        logger.warning(f"  ⚠️ {ticker}: {int(_no_rate.sum())} early row(s) before the first {currency} "
                                       f"rate dropped (no future rate is back-filled)")
                        df = df[~_no_rate].copy()
                    if currency in _MINOR_UNITS:
                        scale_factor = 100.0  # minor unit (pence, cents, agorot) → major unit
                    for col in ["open", "high", "low", "close"]:
                        df[col] = (df[col] * df["fx_rate"]) / scale_factor
                else:
                    df["fx_rate"] = 1.0
                df["currency"] = currency
                df["price_scale"] = scale_factor

                # Metadata
                meta = tickers[ticker]
                df["ticker"]       = ticker
                df["company"]      = meta["name"]
                df["sector"]       = meta["sector"]
                df["region"]       = meta["region"]
                df["_extracted_at"] = datetime.now()

                frames.append(df)
            except Exception as e:
                logger.warning(f"  ⚠️ Error processing {ticker}: {e}")
                continue

    if not frames:
        logger.warning("⚠️ No frames to process — returning empty DataFrame")
        return pd.DataFrame()

    final_df = pd.concat(frames, ignore_index=True)
    final_df = final_df.dropna(subset=["close"])
    final_df["date"] = pd.to_datetime(final_df["date"])

    mode = "INCREMENTAL" if watermarks is not None else "FULL"
    logger.info(f"✅ {mode} EXTRACT COMPLETE: {len(final_df):,} rows across {final_df['ticker'].nunique()} tickers")
    return final_df


def extract_company_info(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Parallelized extraction of company fundamentals using yahooquery for batching.
    """
    logger.info(f"🚀 TURBO METADATA: Fetching info for {len(tickers)} companies using yahooquery...")
    records = []
    
    if not YQTicker:
        logger.warning("⚠️ yahooquery not installed. Falling back to yfinance (unstable for metadata).")
        return pd.DataFrame()

    ticker_keys = list(tickers.keys())
    
    # ── 1. PRE-FETCH FX RATES (one shared, fail-loud implementation) ──────────
    prefetch_fx_to_eur(_guess_currency(t) for t in ticker_keys)

    # ── 2. BATCH EXTRACTION VIA YAHOOQUERY (Pass 1) ───────────────────────────
    batch_size = 40
    failed_tickers = []
    
    def process_data_modules(ticker, data, fx_rate, meta):
        """Helper to parse raw yahooquery modules into a record."""
        if not isinstance(data, dict):
            return None

        # Extract modules (Yahoo returns an error STRING instead of a dict for a module it refuses)
        _mod = lambda name: data.get(name) if isinstance(data.get(name), dict) else {}   # noqa: E731
        summary    = _mod('summaryDetail')
        profile    = _mod('assetProfile')
        stats      = _mod('defaultKeyStatistics')
        financials = _mod('financialData')
        price_mod  = _mod('price')
        calendar   = _mod('calendarEvents')  # Contains ex-dividend & pay dates

        # A rate-limited / refused response arrives as a dict with empty modules. It used to be stored as a
        # "successful" all-NULL record — never retried, and the UPSERT overwrote the previous good row (on
        # 2026-10-14, 173 of 553 US companies lost market cap, P/E, revenue, beta). No core data = failed fetch:
        # retried by the next pass, and if it still fails the previous row is kept.
        quote_type = str(price_mod.get('quoteType') or 'EQUITY').upper()
        if not (summary or price_mod):
            return None
        # An equity without a market cap AND without any statement data is an empty (refused) response. A market
        # cap of 0 next to real data (Michelin, 2026-10-15) is kept: the field is simply unknown for that record.
        if quote_type == 'EQUITY' and not (summary.get('marketCap') or price_mod.get('marketCap')) \
                and not (financials or stats or profile):
            return None

        # Determine currency and FX rates (split by financial vs trading to fix ADR anomalies)
        # Statement amounts are in the financial currency's MAJOR unit (GBP pounds, never pence);
        # quotes (target price) are in the trading currency, which can be a minor unit (GBp);
        # market cap is a company-level amount → trading currency's major unit.
        financial_currency = major_currency(
            financials.get('financialCurrency') or summary.get('currency') or _guess_currency(ticker))
        trading_currency = summary.get('currency') or price_mod.get('currency') or financial_currency

        fin_fx_rate   = fx_to_eur(financial_currency)
        quote_fx_rate = fx_to_eur(trading_currency)
        size_fx_rate  = fx_to_eur(major_currency(trading_currency))
        if fin_fx_rate is None or quote_fx_rate is None:
            logger.error(f"   ❌ {ticker}: no EUR rate for {financial_currency}/{trading_currency} — money fields left empty")

        def _conv(val, rate):
            if rate is None or val is None or (isinstance(val, (float, int)) and pd.isna(val)): return None
            try: return float(val) * rate
            except (TypeError, ValueError): return None

        norm_fin_val   = lambda v: _conv(v, fin_fx_rate)
        norm_trad_val  = lambda v: _conv(v, quote_fx_rate)
        norm_size_val  = lambda v: _conv(v, size_fx_rate)

        record = {
            "ticker":          ticker,
            "quote_type":      price_mod.get('quoteType', 'EQUITY'),
            "company":         meta.get("name") or price_mod.get("shortName") or ticker,
            "sector":          meta.get("sector") or profile.get("sector", "N/A"),
            "industry":        profile.get("industry") or None,  # Granular sub-category from Yahoo
            "region":          meta.get("region") or "N/A",
            "market_cap":      norm_size_val(summary.get('marketCap') or price_mod.get('marketCap') or None),
            "pe_ratio":        summary.get('trailingPE'),
            "forward_pe":      summary.get('forwardPE'),
            "revenue_ttm":     norm_fin_val(financials.get('totalRevenue')),
            "employees":       profile.get('fullTimeEmployees'),
            "country":         profile.get('country'),
            "currency":        financial_currency,  # financial currency (major unit) of EPS / estimates
            "total_debt":      norm_fin_val(financials.get('totalDebt')),
            "ebitda":          norm_fin_val(financials.get('ebitda')),
            "gross_margin":    financials.get('grossMargins'),
            "operating_margin":financials.get('operatingMargins'),
            # ── Per-share EPS stays in the financial currency on purpose: it is only used as a
            # ratio (forward vs trailing) and views convert estimates via `currency` when displayed.
            "trailing_eps":    _safe_float(stats.get('trailingEps')),
            "forward_eps":     _safe_float(stats.get('forwardEps')),
            "roe":             financials.get('returnOnEquity'),
            "free_cashflow":   norm_fin_val(financials.get('freeCashflow')),
            "price_to_book":   stats.get('priceToBook'),
            "beta":            stats.get('beta'),
            # target_mean_price is analyst consensus in trading currency — must convert to EUR using trad_fx_rate
            # to be consistent with price_close (which is also stored in EUR after FX normalization).
            # Keeping it as-is would cause massive upside distortion for JPY/TWD/CNY stocks.
            "target_mean_price": norm_trad_val(financials.get('targetMeanPrice')),
            "recommendation_key": financials.get('recommendationKey'),
            "peg_ratio":       stats.get('trailingPegRatio') or stats.get('pegRatio'),
            "price_to_sales":  summary.get('priceToSalesTrailing12Months'),
            "ev_to_ebitda":    stats.get('enterpriseToEbitda'),
            "revenue_growth":  financials.get('revenueGrowth'),
            "earnings_growth": financials.get('earningsGrowth'),
            "current_ratio":   financials.get('currentRatio'),
            "quick_ratio":     financials.get('quickRatio'),
            "debt_to_equity":  financials.get('debtToEquity'),
            "short_ratio":     stats.get('shortRatio'),
            "short_percent_of_float": stats.get('shortPercentOfFloat'),
            "inst_ownership":  stats.get('heldPercentInstitutions'),
            "insider_ownership":stats.get('heldPercentInsiders'),
            "fx_fin_to_eur":   fin_fx_rate,    # rate applied to the financial-currency amounts above
            "fx_quote_to_eur": quote_fx_rate,  # rate applied to quotes (target price)
            "_extracted_at":   datetime.now(),
        }

        # ── RATIO CONSISTENCY (audit F-15) ───────────────────────────────────────
        # When the listing currency differs from the reporting currency (ADRs), Yahoo's P/E, P/B and EV/EBITDA can
        # combine a USD price with local-currency fundamentals. Rebuild them from EUR-converted COMPANY-level amounts
        # (no per-share / ADS-ratio ambiguity); book value per share cannot be made consistent → left unknown.
        if major_currency(trading_currency) != financial_currency and fin_fx_rate and size_fx_rate:
            mcap_eur = record.get("market_cap")
            ni_eur = norm_fin_val(stats.get('netIncomeToCommon'))
            cash_eur = norm_fin_val(financials.get('totalCash'))
            record["pe_ratio"] = (mcap_eur / ni_eur) if (mcap_eur and ni_eur and ni_eur > 0) else (
                None if (ni_eur is not None and ni_eur <= 0) else record["pe_ratio"])
            if mcap_eur and record.get("ebitda") and record["ebitda"] > 0 and record.get("total_debt") is not None:
                record["ev_to_ebitda"] = (mcap_eur + record["total_debt"] - (cash_eur or 0.0)) / record["ebitda"]
            record["price_to_book"] = None
            logger.debug(f"   ↳ {ticker}: {trading_currency} listing / {financial_currency} reporting — ratios rebuilt in EUR")

        # ── DIVIDEND DATES (from calendarEvents — avoids live API calls on Cloud) ────
        def _parse_div_date(val):
            """Safely parse dividend date from various formats returned by yahooquery."""
            if val is None: return None
            try:
                if hasattr(val, 'strftime'):
                    return val.strftime('%Y-%m-%d')
                s = str(val).strip()
                if s in ('', 'None', 'NaT', 'nan', '0'): return None
                # Handle epoch timestamps (Yahoo sometimes returns raw unix timestamps)
                if s.isdigit() and len(s) >= 9:
                    import datetime as _dt
                    return _dt.datetime.utcfromtimestamp(int(s)).strftime('%Y-%m-%d')
                return pd.to_datetime(s).strftime('%Y-%m-%d')
            except Exception:
                return None

        # calendarEvents.dividends is a list of {exDividendDate, date} dicts in yahooquery
        cal_dividends = calendar.get('dividends', [])
        ex_div_raw, pay_raw = None, None
        if cal_dividends:
            # Take the most recent entry
            last_entry = cal_dividends[-1] if isinstance(cal_dividends, list) else {}
            ex_div_raw = last_entry.get('exDividendDate')
            pay_raw    = last_entry.get('date')
        else:
            # Fallback: some yahooquery versions place dates directly in calendarEvents
            ex_div_raw = calendar.get('exDividendDate')
            pay_raw    = calendar.get('dividendDate')

        record['ex_dividend_date'] = _parse_div_date(ex_div_raw)
        record['pay_date']         = _parse_div_date(pay_raw)

        # Sanitize Yield
        dy  = summary.get('dividendYield')
        tdy = summary.get('trailingAnnualDividendYield')
        
        def _sanitize_yield(val):
            if val is None or (isinstance(val, float) and pd.isna(val)): return None
            v = float(val)
            if v > 1.0: v = v / 100.0
            return v if 0.0 < v <= 0.25 else None
        
        # Yahoo divides a dividend in the MAJOR unit (GBP 0.11) by a quote in the MINOR unit
        # (1404 GBp) for trailingAnnualDividendYield → 100x too small for LSE (Rolls-Royce showed 0.01%).
        if tdy is not None and str(trading_currency) in _MINOR_UNITS:
            try: tdy = float(tdy) * 100
            except (TypeError, ValueError): tdy = None
        _tdy = _sanitize_yield(tdy)
        _dy  = _sanitize_yield(dy)
        # Store 0.0 for confirmed non-payers (yfinance returns None for stocks with no dividend).
        # This distinguishes "confirmed zero" from "data missing due to fetch error".
        # trailingAnnualDividendYield is present (even as 0.0) when yfinance has coverage.
        _has_coverage = (tdy is not None) or (dy is not None)
        if _tdy is not None:
            record["dividend_yield"] = _tdy
        elif _dy is not None:
            record["dividend_yield"] = _dy
        elif _has_coverage:
            record["dividend_yield"] = 0.0   # Confirmed non-payer: yfinance responded with 0
        else:
            record["dividend_yield"] = None  # True data gap: yfinance had no coverage at all
        
        return record

    guard = _fg.Guard("company_info")
    asked = guard.active(ticker_keys)

    def _status_of(ticker, data, record):
        if record:
            return _fg.OK
        s = _fg.classify(data)
        return s if s != _fg.OK else None          # data but no usable record: unexplained → one retry

    i = 0
    while i < len(asked) and not guard.tripped:
        bs = max(5, int(batch_size * guard.batch_scale))
        batch = asked[i:i + bs]
        i += bs
        logger.info(f"   🔍 Fetching metadata batch ({min(i, len(asked))}/{len(asked)})...")
        statuses = {}
        try:
            all_data = YQTicker(batch, asynchronous=True).all_modules
            if not isinstance(all_data, dict):
                statuses = {t: _fg.classify(all_data) for t in batch}
            else:
                for ticker in batch:
                    data = all_data.get(ticker)
                    meta = tickers.get(ticker, {"name": ticker, "sector": "N/A", "region": "N/A"})
                    record = process_data_modules(ticker, data, None, meta)
                    if record:
                        records.append(record)
                    statuses[ticker] = _status_of(ticker, data, record)
        except Exception as e:
            logger.warning(f"  ⚠️ Metadata batch failed: {e}")
            statuses = _batch_statuses(batch, set(), error=e)
        guard.after_batch(statuses)
        if i < len(asked):
            import time, random
            time.sleep(1.0 + random.random())

    def _one(ticker):
        data = YQTicker(ticker, asynchronous=False).all_modules
        data = data.get(ticker) if isinstance(data, dict) else data
        record = process_data_modules(ticker, data, None, tickers.get(ticker, {"name": ticker, "sector": "N/A",
                                                                               "region": "N/A"}))
        if record:
            records.append(record)
        return _status_of(ticker, data, record) or _fg.ERROR

    _guarded_retry(guard, [t for t in asked if t not in {r["ticker"] for r in records}], _one, "Metadata")
    guard.finish(ticker_keys, {r["ticker"] for r in records})

    missing = [t for t in ticker_keys if t not in {r["ticker"] for r in records}]
    if missing:
        logger.warning(f"   ⚠️ {len(missing)} companies without usable metadata this run (kept their previous row, "
                       f"if any): {', '.join(missing[:15])}{' …' if len(missing) > 15 else ''}")
    logger.info(f"✅ Metadata Extraction: {len(records)}/{len(ticker_keys)} successful.")
    return pd.DataFrame(records)



def extract_historical_financials(tickers: dict = None) -> pd.DataFrame:
    """
    Parallelized extraction of annual financials (Revenue, EPS).
    Defaults to equity-only tickers to skip ETFs and Indices.
    """
    if tickers is None:
        tickers = get_equity_tickers()
    
    logger.info(f"🚀 TURBO FINANCIALS: Fetching reports for {len(tickers)} equities...")
    all_data = []
    ticker_keys = list(tickers.keys())
    # AG PRO: Filter out Indices and ETFs from fundamental extraction
    ticker_keys = sorted([t for t in ticker_keys if not t.startswith('^') and t not in ['SPY']])

    
    # 1. FX rates come from the shared fx_to_eur() (per statement currency)

    def process_yq_fin(df, successful_set):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        
        # yahooquery returns a df with 'symbol' in index or as a column
        if 'symbol' in df.index.names:
            df = df.reset_index()
            
        row_map = {
            "TotalRevenue": "revenue", 
            "BasicEPS": "eps", 
            "DilutedEPS": "eps_diluted",
            "NetIncome": "net_income",
            "StockholdersEquity": "total_equity",
            # ROIC inputs (audit F-04)
            "EBIT": "ebit",
            "TaxRateForCalcs": "tax_rate",
            "TotalDebt": "total_debt",
            "CashCashEquivalentsAndShortTermInvestments": "cash",
            "CashAndCashEquivalents": "cash_only",
            # income-statement interest: the stand-in when the cash-flow statement does not report interest PAID
            "InterestExpense": "interest_expense",
            "InterestExpenseNonOperating": "interest_expense_nonop",
        }
        
        for ticker in df['symbol'].unique():
            t_data = df[df['symbol'] == ticker].copy()
            currency, fx_rate = _row_currencies(t_data, ticker)
            if fx_rate.isna().all():
                logger.error(f"   ❌ {ticker}: no EUR rate for {sorted(set(currency))} — statements skipped")
                continue
            
            # Map columns and normalize
            found_cols = [c for c in row_map.keys() if c in t_data.columns]
            if not found_cols: continue
            
            t_filtered = t_data[["asOfDate"] + found_cols].rename(columns={"asOfDate": "date"}).rename(columns=row_map)
            
            # Ensure all target columns exist even if missing from source
            if "cash_only" in t_filtered.columns:
                t_filtered["cash"] = (t_filtered["cash"].fillna(t_filtered["cash_only"]) if "cash" in t_filtered.columns
                                      else t_filtered["cash_only"])
                t_filtered = t_filtered.drop(columns=["cash_only"])
            if "interest_expense_nonop" in t_filtered.columns:
                t_filtered["interest_expense"] = (t_filtered["interest_expense"].fillna(t_filtered["interest_expense_nonop"])
                                                  if "interest_expense" in t_filtered.columns
                                                  else t_filtered["interest_expense_nonop"])
                t_filtered = t_filtered.drop(columns=["interest_expense_nonop"])
            for col in ["revenue", "net_income", "total_equity", "eps", "eps_diluted", "ebit", "total_debt", "cash",
                        "interest_expense"]:
                if col not in t_filtered.columns:
                    t_filtered[col] = pd.NA
                else:
                    t_filtered[col] = pd.to_numeric(t_filtered[col], errors="coerce") * fx_rate
            if "tax_rate" not in t_filtered.columns:
                t_filtered["tax_rate"] = pd.NA                       # a ratio: no FX
            
            t_filtered["ticker"] = ticker
            t_filtered["src_currency"] = currency.to_numpy()   # provenance per period: amounts = reported * fx_to_eur
            t_filtered["fx_to_eur"] = fx_rate.to_numpy()
            t_filtered = t_filtered[t_filtered["fx_to_eur"].notna()]
            all_data.append(t_filtered)
            successful_set.add(ticker)

    # â”€â”€ Batch pass + ONE guarded retry (etl/fetch_guard.py: refusals cool down, "no data" is not retried) â”€â”€
    successful_tickers = set()

    def _fetch(symbols, asynchronous):
        yq = YQTicker(symbols, asynchronous=asynchronous)
        inc_df = _fetch_statement(yq, 'income_statement', 'a')
        bal_df = _fetch_statement(yq, 'balance_sheet', 'a')
        status, by = _merged_status(inc_df, bal_df)
        if not inc_df.empty and not bal_df.empty:
            if 'symbol' in inc_df.index.names: inc_df = inc_df.reset_index()
            if 'symbol' in bal_df.index.names: bal_df = bal_df.reset_index()
            fin_df = pd.merge(inc_df, bal_df, on=['symbol', 'asOfDate'], how='outer', suffixes=('', '_bal'))
        else:
            fin_df = (inc_df if not inc_df.empty else bal_df).copy()
        fin_df.attrs.update(status=status, by_symbol=by)
        return fin_df

    _run_guarded("annual_financials", ticker_keys, _fetch, lambda f: process_yq_fin(f, successful_tickers),
                 successful_tickers, "Annual financials")

    if not all_data: return pd.DataFrame()
    final_df = pd.concat(all_data, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])
    # De-duplicate to prevent DuckDB Constraint Errors (e.g. NBIS 2021-12-31)
    final_df = final_df.drop_duplicates(subset=["ticker", "date"], keep="first")
    return final_df


def extract_quarterly_financials(tickers: dict = None) -> pd.DataFrame:
    """
    Parallelized extraction of quarterly financials.
    Defaults to equity-only tickers to skip ETFs and Indices.
    """
    if tickers is None:
        tickers = get_equity_tickers()
    
    logger.info(f"🚀 TURBO QUARTERLY: Fetching reports for {len(tickers)} equities...")
    all_data = []
    ticker_keys = list(tickers.keys())
    # AG PRO: Filter out Indices and ETFs from fundamental extraction
    ticker_keys = sorted([t for t in ticker_keys if not t.startswith('^') and t not in ['SPY']])

    
    # 1. FX rates come from the shared fx_to_eur() (per statement currency)

    def process_yq_q_fin(df, successful_set):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        
        if 'symbol' in df.index.names:
            df = df.reset_index()
            
        row_map = {
            "TotalRevenue": "revenue", 
            "BasicEPS": "eps", 
            "DilutedEPS": "eps_diluted",
            "NetIncome": "net_income",
            "OperatingIncome": "operating_income",
            "StockholdersEquity": "total_equity"
        }
        
        for ticker in df['symbol'].unique():
            t_data = df[df['symbol'] == ticker].copy()
            currency, fx_rate = _row_currencies(t_data, ticker)
            if fx_rate.isna().all():
                logger.error(f"   ❌ {ticker}: no EUR rate for {sorted(set(currency))} — statements skipped")
                continue
            
            found_cols = [c for c in row_map.keys() if c in t_data.columns]
            if not found_cols: continue
            
            t_filtered = t_data[["asOfDate"] + found_cols].rename(columns={"asOfDate": "date"}).rename(columns=row_map)
            
            # Ensure all target columns exist even if missing from source
            for col in ["revenue", "net_income", "total_equity", "eps", "eps_diluted", "operating_income"]:
                if col not in t_filtered.columns:
                    t_filtered[col] = pd.NA
                else:
                    t_filtered[col] = pd.to_numeric(t_filtered[col], errors="coerce") * fx_rate.to_numpy()
            
            t_filtered["ticker"] = ticker
            t_filtered["src_currency"] = currency.to_numpy()   # provenance per period: amounts = reported * fx_to_eur
            t_filtered["fx_to_eur"] = fx_rate.to_numpy()
            t_filtered = t_filtered[t_filtered["fx_to_eur"].notna()]
            all_data.append(t_filtered)
            successful_set.add(ticker)

    # â”€â”€ Batch pass + ONE guarded retry (etl/fetch_guard.py: refusals cool down, "no data" is not retried) â”€â”€
    successful_tickers = set()

    def _fetch(symbols, asynchronous):
        yq = YQTicker(symbols, asynchronous=asynchronous)
        inc_df = _fetch_statement(yq, 'income_statement', 'q')
        bal_df = _fetch_statement(yq, 'balance_sheet', 'q')
        status, by = _merged_status(inc_df, bal_df)
        if not inc_df.empty and not bal_df.empty:
            if 'symbol' in inc_df.index.names: inc_df = inc_df.reset_index()
            if 'symbol' in bal_df.index.names: bal_df = bal_df.reset_index()
            fin_df = pd.merge(inc_df, bal_df, on=['symbol', 'asOfDate'], how='outer', suffixes=('', '_bal'))
        else:
            fin_df = (inc_df if not inc_df.empty else bal_df).copy()
        fin_df.attrs.update(status=status, by_symbol=by)
        return fin_df

    _run_guarded("quarterly_financials", ticker_keys, _fetch, lambda f: process_yq_q_fin(f, successful_tickers),
                 successful_tickers, "Quarterly financials")

    if not all_data: return pd.DataFrame()
    final_df = pd.concat(all_data, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])
    # De-duplicate to prevent DuckDB Constraint Errors
    final_df = final_df.drop_duplicates(subset=["ticker", "date"], keep="first")
    
    logger.info(f"✅ Quarterly Financials Extraction: {len(successful_tickers)} companies successful.")
    return final_df

PAYOUT_YEARS, PAYOUT_MIN_YEARS, PAYOUT_MAX_YIELD = 3, 2, 0.20


def payout_evidence(cf_df: pd.DataFrame, fx_rate: float, mcap_eur: float = 0.0,
                    dividend_confirmed_zero: bool = False) -> dict:
    """NET shareholder payout, NORMALISED ANNUAL (mean of the last ≤3 annual cash-flow statements — not a trailing
    twelve months), with evidence (E5). Per item: OBSERVED / CONFIRMED_ZERO (every year reported, all 0) / PARTIAL
    (a year missing) / UNKNOWN (never reported). A missing field is never a zero:
      buyback_ttm (net) = repurchases − issuance, only when BOTH are fully reported;
      dividends_paid_ttm only when reported every year, or confirmed 0 by the provider's dividend rate;
      fewer than PAYOUT_MIN_YEARS statements → INSUFFICIENT; payout yield > 20% of market cap → INVALID.
    Column names keep their historical '_ttm' suffix for compatibility; the values are normalised annual."""
    cf = cf_df.reset_index() if "symbol" in getattr(cf_df.index, "names", []) else cf_df
    if "asOfDate" in cf.columns:
        cf = cf.sort_values("asOfDate")
    recent = cf.tail(PAYOUT_YEARS)
    years = len(recent)

    def _item(col, outflow):
        if col not in recent.columns:
            return None, "UNKNOWN"
        s = pd.to_numeric(recent[col], errors="coerce")
        if s.isna().all():
            return None, "UNKNOWN"
        if s.isna().any():
            return None, "PARTIAL"
        v = (-s if outflow else s).clip(lower=0) * fx_rate      # cash OUT is negative in Yahoo statements
        return float(v.mean()), ("CONFIRMED_ZERO" if (v == 0).all() else "OBSERVED")

    rep, rep_s = _item("RepurchaseOfCapitalStock", True)
    iss, iss_s = _item("IssuanceOfCapitalStock", False)
    div, div_s = _item("CashDividendsPaid", True)
    if div is None and div_s == "UNKNOWN" and dividend_confirmed_zero:
        div, div_s = 0.0, "CONFIRMED_ZERO"
    net = rep - iss if (rep is not None and iss is not None) else None
    notes = [f"repurchase {rep_s}", f"issuance {iss_s}", f"dividends {div_s}"]
    if years < PAYOUT_MIN_YEARS:
        status, net, div = "INSUFFICIENT", None, None
        notes.append(f"only {years} annual statement(s)")
    elif mcap_eur and mcap_eur > 1_000_000 and (abs(net or 0) + (div or 0)) / mcap_eur > PAYOUT_MAX_YIELD:
        status, net, div = "INVALID", None, None
        notes.append("payout yield > 20% of market cap")
    elif net is not None and div is not None:
        status = "OBSERVED"
    elif net is not None or div is not None:
        status = "PARTIAL"
    else:
        status = "UNKNOWN"
    return {"buyback_ttm": net, "dividends_paid_ttm": div, "repurchase_avg": rep, "issuance_avg": iss,
            "payout_years": years, "payout_status": status, "payout_note": "; ".join(notes)}


def extract_cashflows(tickers: dict = TICKERS) -> pd.DataFrame:
    """
    Parallelized extraction of annual cashflows via yahooquery with surgical retry.
    Focuses on share buyback and dividends paid.
    """
    logger.info(f"🚀 TURBO CASHFLOWS: Fetching buybacks/dividends for {len(tickers)} companies via yahooquery...")
    records = []
    ticker_keys = list(tickers.keys())
    # AG PRO: Filter out Indices and ETFs from fundamental extraction 
    ticker_keys = sorted([t for t in ticker_keys if not t.startswith('^') and t not in ['SPY']])


    # 1. FX rates come from the shared fx_to_eur() (statement currencyCode)

    def fetch_single(ticker, session=None):
        try:
            if session is not None:
                yq = YQTicker(ticker, session=session)
            else:
                yq = YQTicker(ticker)
            cf_df = _fetch_statement(yq, 'cash_flow', 'a')
            if cf_df is None or (isinstance(cf_df, pd.DataFrame) and cf_df.empty):
                return None
            
            if 'symbol' in cf_df.index.names:
                cf_df = cf_df.reset_index()
            
            # Statement currency (e.g. DKK for a USD-listed ADR) — no more DKK guessing
            currency = _statement_currency(cf_df, ticker)
            fx_rate  = fx_to_eur(currency)
            if fx_rate is None:
                logger.error(f"   ❌ {ticker}: no EUR rate for {currency} — cashflow skipped")
                return None
            try:
                summary = yq.summary_detail.get(ticker, {})
                summary = summary if isinstance(summary, dict) else {}
                mcap_eur = (summary.get("marketCap") or 0) * (fx_to_eur(major_currency(summary.get("currency"))) or 0)
            except Exception:
                summary, mcap_eur = {}, 0
            # trailingAnnualDividendRate == 0 is Yahoo confirming a non-payer; absent is unknown
            div_zero = summary.get("trailingAnnualDividendRate") == 0 and not summary.get("dividendRate")
            ev = payout_evidence(cf_df, fx_rate, mcap_eur, dividend_confirmed_zero=div_zero)
            if ev["payout_status"] == "INVALID":
                logger.warning(f"   ⚠️ {ticker}: implausible payout yield — stored as INVALID (unknown), not as zero")
            return {"ticker": ticker, **ev}
        except Exception as e:
            logger.warning(f"  ⚠️ Cashflow fetch failed for {ticker}: {e}")
            return None

    # â”€â”€ Concurrent pass + ONE guarded retry (etl/fetch_guard.py). fetch_single swallows the provider's answer, so a
    # miss is an ERROR (retried once, never cached as "no data"); a batch that fails entirely cools the guard down.
    import time
    successful_tickers = set()
    guard = _fg.Guard("cashflows")
    asked = guard.active(ticker_keys)
    i = 0
    while i < len(asked) and not guard.tripped:
        bs = max(5, int(40 * guard.batch_scale))
        batch = asked[i:i + bs]
        i += bs
        logger.info(f"   ðŸ’¸ Cash-flow batch ({min(i, len(asked))}/{len(asked)})...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_single, t): t for t in batch}
            for future in as_completed(futures):
                res = future.result()
                if res:
                    records.append(res)
                    successful_tickers.add(futures[future])
        statuses = {t: (_fg.OK if t in successful_tickers else _fg.ERROR) for t in batch}
        if batch and not (successful_tickers & set(batch)):
            statuses = {t: _fg.BLOCKED for t in batch}           # a whole batch with nothing usable: treat as refused
        guard.after_batch(statuses)
        if i < len(asked):
            time.sleep(1.5)

    def _one(t):
        res = fetch_single(t)
        if res:
            records.append(res)
            successful_tickers.add(t)
            return _fg.OK
        return _fg.ERROR

    _guarded_retry(guard, [t for t in asked if t not in successful_tickers], _one, "Cash flow")
    guard.finish(ticker_keys, successful_tickers)

    logger.info(f"✅ Cashflow extracted for {len(records)}/{len(tickers)} tickers")
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "buyback_ttm", "dividends_paid_ttm"])


def extract_historical_fcf(tickers: dict = None) -> pd.DataFrame:
    """
    Extract historical annual Free Cash Flow (FCF) with surgical retry.
    Defaults to equity-only tickers to skip ETFs and Indices.
    """
    if tickers is None:
        tickers = get_equity_tickers()
        
    if not tickers:
        return pd.DataFrame()
        
    logger.info(f"🚀 HISTORICAL FCF: Fetching {len(tickers)} tickers via yahooquery...")
    records = []
    ticker_keys = list(tickers.keys())
    # AG PRO: Filter out Indices and ETFs from fundamental extraction
    ticker_keys = sorted([t for t in ticker_keys if not t.startswith('^') and t not in ['SPY']])


    if not YQTicker:
        logger.warning("⚠️ yahooquery not installed — skipping historical FCF extraction.")
        return pd.DataFrame()

    def lease_lookup(bs) -> dict:
        """{(symbol, fiscal year): (current lease liability EUR, total lease liability EUR)} from the balance sheet.
        Yahoo's cash flow has NO lease-principal field (verified on live data, 10 issuers, 2026-10): the principal
        repaid in a year is estimated by the lease liability falling due within 12 months at the prior year-end."""
        out = {}
        if not isinstance(bs, pd.DataFrame) or bs.empty:
            return out
        bs = bs.reset_index() if 'symbol' in bs.index.names else bs
        for _, r in bs.iterrows():
            try:
                key = (r.get('symbol'), pd.to_datetime(r.get('asOfDate')).year)
            except (TypeError, ValueError):
                continue
            ccy = r.get('currencyCode') if isinstance(r.get('currencyCode'), str) else \
                major_currency(_guess_currency(r.get('symbol')))
            rate = fx_to_eur(ccy)
            if rate is None:
                continue
            cur, tot = r.get('CurrentCapitalLeaseObligation'), r.get('CapitalLeaseObligations')
            out[key] = (abs(float(cur)) * rate if cur is not None and not pd.isna(cur) else None,
                        abs(float(tot)) * rate if tot is not None and not pd.isna(tot) else None)
        return out

    def fetch_bs(yq):
        try:
            return _fetch_statement(yq, 'balance_sheet', 'a')
        except Exception as e:                       # lease inputs are optional: their absence stays UNKNOWN
            logger.debug(f"   ↳ balance sheet for lease liabilities unavailable: {e}")
            return pd.DataFrame()

    def process_fcf_df(df, records_list, successful_set, bs=None):
        leases = lease_lookup(bs)
        if isinstance(df, pd.DataFrame) and not df.empty:
            if 'symbol' in df.index.names:
                df = df.reset_index()
            for _, row in df.iterrows():
                ticker = row.get('symbol')
                as_of  = row.get('asOfDate')
                if not ticker or not as_of: continue
                try: year = pd.to_datetime(as_of).year
                except: continue
                
                fcf, ocf, capex = row.get('FreeCashFlow'), row.get('OperatingCashFlow'), row.get('CapitalExpenditure')
                if pd.isna(fcf) and pd.isna(ocf): continue
                if pd.isna(fcf) and not pd.isna(ocf) and not pd.isna(capex):
                    fcf = ocf + capex
                # Stored in EUR (statement currencyCode → EUR) so FCF is comparable with revenue,
                # market cap and the DCF. Views must NOT convert again.
                ccy = row.get('currencyCode') if isinstance(row.get('currencyCode'), str) else \
                    major_currency(_guess_currency(ticker))
                rate = fx_to_eur(ccy)
                if rate is None:
                    continue
                fcf   = fcf * rate if not pd.isna(fcf) else fcf
                ocf   = ocf * rate if not pd.isna(ocf) else ocf
                capex = capex * rate if not pd.isna(capex) else capex
                # Stock-based compensation is added back in operating cash flow although it costs owners
                # (dilution); kept separately so valuation can use FCF after SBC.
                sbc = row.get('StockBasedCompensation')
                sbc = float(sbc) * rate if sbc is not None and not pd.isna(sbc) else None
                # Payments to lenders / lessors reported in FINANCING (IFRS interest option, IFRS 16 lease principal):
                # OCF − capex of such filers is cash before them (audit F-03). Stored as positive EUR amounts.
                # Field names verified on live Yahoo data (2026-10): InterestPaidCFF / InterestPaidCFO say WHERE
                # interest is classified (CFO = already inside OCF − capex, so no financing deduction applies);
                # InterestPaidSupplementalData gives the amount but not the classification. There is no cash-flow
                # lease-principal field; the balance sheet's lease liabilities are stored for an estimate instead.
                def _first(*keys):
                    for k in keys:
                        v = row.get(k)
                        if v is not None and not pd.isna(v):
                            return abs(float(v)) * rate
                    return None
                interest_cff = _first('InterestPaidCFF')
                interest_cfo = _first('InterestPaidCFO')
                interest_sup = _first('InterestPaidSupplementalData')
                lease_principal = _first('RepaymentOfLeaseLiabilities', 'PrincipalPaymentsOfLeases',
                                         'PaymentsOfLeaseLiabilities', 'LeasePayments')
                lease_cur, lease_tot = leases.get((ticker, int(year)), (None, None))
                # E6: ordinary debt flows, for the refinancing stress of the FCFE (net borrowing is NOT added to it)
                debt_issued = _first('IssuanceOfDebt', 'LongTermDebtIssuance')
                debt_repaid = _first('RepaymentOfDebt', 'LongTermDebtPayments')

                records_list.append({
                    "ticker": ticker, "year": int(year), "period_end": pd.to_datetime(as_of).date(),
                    "free_cash_flow": float(fcf) if not pd.isna(fcf) else None,
                    "operating_cash_flow": float(ocf) if not pd.isna(ocf) else None,
                    "capex": float(capex) if not pd.isna(capex) else None,
                    "stock_based_comp": sbc,
                    "interest_paid_cff": interest_cff,
                    "interest_paid_cfo": interest_cfo,
                    "interest_paid_supplemental": interest_sup,
                    "lease_principal_paid": lease_principal,
                    "lease_liability_current": lease_cur,
                    "lease_liability_total": lease_tot,
                    "debt_issued": debt_issued,
                    "debt_repaid": debt_repaid,
                    "_extracted_at": datetime.now(),
                })
                successful_set.add(ticker)

    # â”€â”€ Batch pass + ONE guarded retry (etl/fetch_guard.py: refusals cool down, "no data" is not retried) â”€â”€
    successful_tickers = set()

    def _fetch(symbols, asynchronous):
        yq = YQTicker(symbols, asynchronous=asynchronous)
        cf_df = _fetch_statement(yq, 'cash_flow', 'a')
        cf_df.attrs["_bs"] = fetch_bs(yq) if not cf_df.empty else None
        return cf_df

    _run_guarded("hist_fcf", ticker_keys, _fetch,
                 lambda f: process_fcf_df(f, records, successful_tickers, f.attrs.get("_bs")),
                 successful_tickers, "Annual cash flow")

    logger.info(f"✅ Historical FCF: {len(successful_tickers)} tickers successful.")
    return _one_period_per_key(pd.DataFrame(records), ["ticker", "year"], "annual_fcf")


def extract_quarterly_fcf(tickers: dict = None) -> pd.DataFrame:
    """
    Extract historical quarterly Free Cash Flow (FCF) with surgical retry.
    Defaults to equity-only tickers to skip ETFs and Indices.
    """
    if tickers is None:
        tickers = get_equity_tickers()
        
    if not tickers:
        return pd.DataFrame()
        
    logger.info(f"🚀 QUARTERLY FCF: Fetching {len(tickers)} tickers via yahooquery...")
    records = []
    ticker_keys = list(tickers.keys())
    # AG PRO: Filter out Indices and ETFs from fundamental extraction
    ticker_keys = sorted([t for t in ticker_keys if not t.startswith('^') and t not in ['SPY']])


    if not YQTicker:
        logger.warning("⚠️ yahooquery not installed — skipping quarterly FCF extraction.")
        return pd.DataFrame()

    def process_q_fcf_df(df, records_list, successful_set):
        if isinstance(df, pd.DataFrame) and not df.empty:
            if 'symbol' in df.index.names:
                df = df.reset_index()
            for _, row in df.iterrows():
                ticker = row.get('symbol')
                as_of  = row.get('asOfDate')
                if not ticker or not as_of: continue
                try:
                    dt = pd.to_datetime(as_of)
                    year, quarter = dt.year, (dt.month - 1) // 3 + 1
                except: continue
                
                fcf, ocf, capex = row.get('FreeCashFlow'), row.get('OperatingCashFlow'), row.get('CapitalExpenditure')
                if pd.isna(fcf) and pd.isna(ocf): continue
                if pd.isna(fcf) and not pd.isna(ocf) and not pd.isna(capex):
                    fcf = ocf + capex
                # Stored in EUR (statement currencyCode → EUR) so FCF is comparable with revenue,
                # market cap and the DCF. Views must NOT convert again.
                ccy = row.get('currencyCode') if isinstance(row.get('currencyCode'), str) else \
                    major_currency(_guess_currency(ticker))
                rate = fx_to_eur(ccy)
                if rate is None:
                    continue
                fcf   = fcf * rate if not pd.isna(fcf) else fcf
                ocf   = ocf * rate if not pd.isna(ocf) else ocf
                capex = capex * rate if not pd.isna(capex) else capex
                
                records_list.append({
                    "ticker": ticker, "year": int(year), "quarter": int(quarter), "period_end": dt.date(),
                    "free_cash_flow": float(fcf) if not pd.isna(fcf) else None,
                    "operating_cash_flow": float(ocf) if not pd.isna(ocf) else None,
                    "capex": float(capex) if not pd.isna(capex) else None,
                    "_extracted_at": datetime.now(),
                })
                successful_set.add(ticker)

    # â”€â”€ Batch pass + ONE guarded retry (etl/fetch_guard.py: refusals cool down, "no data" is not retried) â”€â”€
    successful_tickers = set()

    def _fetch(symbols, asynchronous):
        return _fetch_statement(YQTicker(symbols, asynchronous=asynchronous), 'cash_flow', 'q')

    _run_guarded("quarterly_fcf", ticker_keys, _fetch, lambda f: process_q_fcf_df(f, records, successful_tickers),
                 successful_tickers, "Quarterly cash flow")

    logger.info(f"✅ Quarterly FCF: {len(successful_tickers)} tickers successful.")
    return _one_period_per_key(pd.DataFrame(records), ["ticker", "year", "quarter"], "quarterly_fcf")

    
def extract_earnings_calendar(tickers: dict = None) -> pd.DataFrame:
    """
    Fetch upcoming earnings dates and estimates.
    Defaults to equity-only tickers to skip ETFs and Indices.
    """
    if tickers is None:
        tickers = get_equity_tickers()
        
    if not tickers:
        return pd.DataFrame()
        
    logger.info(f"📅 Fetching earnings calendar for {len(tickers)} equities...")
    records = []
    ticker_keys = [t for t in tickers.keys() if not t.startswith("^")]
    successful_tickers = set()
    
    # â”€â”€ Batch pass + ONE guarded retry (etl/fetch_guard.py: refusals cool down, "no data" is not retried) â”€â”€
    def _parse(ticker, data):
        if not (isinstance(data, dict) and 'earnings' in data):
            return False
        earn = data['earnings']
        e_date = earn.get('earningsDate')
        if not (isinstance(e_date, list) and e_date):
            return False
        d_obj = e_date[0]
        if isinstance(d_obj, str):
            try:
                d_obj = datetime.strptime(d_obj.split(' ')[0], '%Y-%m-%d').date()
            except ValueError:
                d_obj = None
        if not d_obj:
            return False
        records.append({"ticker": ticker, "earnings_date": d_obj, "eps_avg": earn.get('earningsAverage'),
                        "rev_avg": earn.get('revenueAverage')})
        return True

    if YQTicker:
        _run_guarded_modules("earnings_calendar", ticker_keys,
                             lambda s, asynchronous: YQTicker(s, asynchronous=False).calendar_events,
                             _parse, successful_tickers, "Earnings calendar", pause=(2.0, 4.0))

    logger.info(f"✅ Earnings Calendar: {len(successful_tickers)} tickers successful.")
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "earnings_date", "eps_avg", "rev_avg"])


def extract_forward_estimates(tickers: dict = None) -> pd.DataFrame:
    """
    Extract analyst forward estimates (EPS & Revenue) for current/next quarter and year.
    Uses yahooquery earningsTrend module — batch pass + one guarded retry (etl/fetch_guard.py).

    Returns one row per ticker with flattened columns:
        - eps_est_{period}_avg/low/high/growth/n_analysts
        - rev_est_{period}_avg/low/high/growth
        - eps_trend_{period}_current/7d/30d/60d/90d
        - eps_rev_{period}_up7d/up30d/down7d/down30d
    Period suffixes: 0q (current quarter), 1q (next quarter), 0y (this year), 1y (next year).
    """
    if tickers is None:
        tickers = get_equity_tickers()

    if not tickers or not YQTicker:
        return pd.DataFrame()

    logger.info(f"🔭 FORWARD ESTIMATES: Fetching earningsTrend for {len(tickers)} equities...")
    records = []
    ticker_keys = sorted([
        t for t in tickers.keys()
        if not t.startswith('^') and t not in ['SPY']
    ])
    successful_tickers: set = set()

    # Period tags as returned by Yahoo: current Q, next Q, this year, next year
    _PERIODS = ["0q", "+1q", "0y", "+1y"]
    _PERIOD_ALIAS = {"0q": "0q", "+1q": "1q", "0y": "0y", "+1y": "1y"}

    def _parse_trend(ticker: str, raw: dict):
        """Flatten earningsTrend payload for one ticker into a single record.

        yahooquery get_modules("earningsTrend") returns raw[ticker] as the
        earningsTrend object directly (keys: "trend", "defaultMethodology", "maxAge").
        """
        if not isinstance(raw, dict):
            return None
        # raw IS the earningsTrend object — "trend" is a direct key
        trend_list = raw.get("trend", [])
        if not trend_list:
            return None

        record: dict = {"ticker": ticker, "_extracted_at": datetime.now()}

        for item in trend_list:
            period = item.get("period", "")
            if period not in _PERIODS:
                continue
            alias = _PERIOD_ALIAS[period]

            ee = item.get("earningsEstimate", {}) or {}
            re = item.get("revenueEstimate", {}) or {}
            et = item.get("epsTrend", {}) or {}
            er = item.get("epsRevisions", {}) or {}

            # EPS Estimates
            record[f"eps_est_{alias}_avg"]         = _safe_float(ee.get("avg"))
            record[f"eps_est_{alias}_low"]         = _safe_float(ee.get("low"))
            record[f"eps_est_{alias}_high"]        = _safe_float(ee.get("high"))
            record[f"eps_est_{alias}_growth"]      = _safe_float(ee.get("growth"))
            record[f"eps_est_{alias}_n_analysts"]  = ee.get("numberOfAnalysts")

            # Revenue Estimates
            record[f"rev_est_{alias}_avg"]         = _safe_float(re.get("avg"))
            record[f"rev_est_{alias}_low"]         = _safe_float(re.get("low"))
            record[f"rev_est_{alias}_high"]        = _safe_float(re.get("high"))
            record[f"rev_est_{alias}_growth"]      = _safe_float(re.get("growth"))

            # EPS Trend (revision history)
            record[f"eps_trend_{alias}_current"]   = _safe_float(et.get("current"))
            record[f"eps_trend_{alias}_7d_ago"]    = _safe_float(et.get("7daysAgo"))
            record[f"eps_trend_{alias}_30d_ago"]   = _safe_float(et.get("30daysAgo"))
            record[f"eps_trend_{alias}_60d_ago"]   = _safe_float(et.get("60daysAgo"))
            record[f"eps_trend_{alias}_90d_ago"]   = _safe_float(et.get("90daysAgo"))

            # EPS Revisions (upgrade/downgrade counts)
            record[f"eps_rev_{alias}_up7d"]        = er.get("upLast7days")
            record[f"eps_rev_{alias}_up30d"]       = er.get("upLast30days")
            record[f"eps_rev_{alias}_down7d"]      = er.get("downLast7Days")
            record[f"eps_rev_{alias}_down30d"]     = er.get("downLast30days")

        # Only return if at least one period was parsed
        if len(record) <= 2:
            return None
        return record

    # â”€â”€ Batch pass + ONE guarded retry (etl/fetch_guard.py: refusals cool down, "no data" is not retried) â”€â”€
    def _parse(ticker, data):
        rec = _parse_trend(ticker, data if isinstance(data, dict) else {})
        if rec:
            records.append(rec)
            return True
        return False

    _run_guarded_modules("forward_estimates", ticker_keys,
                         lambda s, asynchronous: YQTicker(s, asynchronous=asynchronous).get_modules("earningsTrend"),
                         _parse, successful_tickers, "Forward estimates")

    logger.info(f"✅ Forward Estimates: {len(successful_tickers)}/{len(ticker_keys)} tickers successful, {len(records)} records.")
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def extract_earnings_history(tickers: dict = None) -> pd.DataFrame:
    """
    Fetch historical EPS Actual vs Estimate (Earnings Surprise) for the last 4 quarters.
    Uses yahooquery's earning_history attribute.

    All monetary amounts (eps_actual, eps_estimate, eps_difference) are normalised
    to EUR using the same FX pre-fetch pattern as the rest of the ETL pipeline.
    `surprise_pct` is a dimensionless ratio and is stored as-is.

    Returns DataFrame with columns:
        ticker, quarter_date, eps_actual, eps_estimate, eps_difference, surprise_pct,
        currency, period, _extracted_at
    """
    if tickers is None:
        tickers = get_equity_tickers()

    if not tickers or not YQTicker:
        return pd.DataFrame()

    logger.info(f"📊 EARNINGS SURPRISE: Fetching history for {len(tickers)} equities...")
    records = []
    ticker_keys = [t for t in tickers.keys() if not t.startswith("^")]
    successful_tickers = set()

    def _eur_rate(ticker_sym: str, reported_currency: str):
        """Multiplier reported_currency → EUR (None = unknown). EPS is in the major unit (GBP, not GBp)."""
        ccy = reported_currency.strip() if reported_currency else major_currency(_guess_currency(ticker_sym))
        return fx_to_eur(ccy)

    def _to_eur(val, rate):
        """None-safe multiply."""
        if val is None or rate is None:
            return None
        try:
            return float(val) * rate
        except (TypeError, ValueError):
            return None

    def process_history(df_raw, ticker_symbol):
        """Parse earning_history DataFrame for a single ticker."""
        if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
            return
        df = df_raw.reset_index() if df_raw.index.name else df_raw.copy()
        # filter to this ticker if multi-ticker df
        if "symbol" in df.columns:
            df = df[df["symbol"] == ticker_symbol]
        for _, row in df.iterrows():
            try:
                quarter_date = pd.to_datetime(row.get("quarter")).date()
            except Exception:
                continue
            reported_ccy = str(row.get("currency") or "")
            rate = _eur_rate(ticker_symbol, reported_ccy)
            records.append({
                "ticker":         ticker_symbol,
                "quarter_date":   quarter_date,
                "eps_actual":     _to_eur(row.get("epsActual"),     rate),  # EUR
                "eps_estimate":   _to_eur(row.get("epsEstimate"),   rate),  # EUR
                "eps_difference": _to_eur(row.get("epsDifference"), rate),  # EUR
                "surprise_pct":   row.get("surprisePercent"),               # dimensionless
                "currency":       reported_ccy,   # keep original for audit
                "period":         row.get("period", ""),
                "_extracted_at":  datetime.now(),
            })
        successful_tickers.add(ticker_symbol)

    # ── Batch pass + ONE guarded retry (etl/fetch_guard.py: refusals cool down, "no data" is not retried) ──
    def _fetch(symbols, asynchronous):
        raw = YQTicker(symbols, asynchronous=asynchronous).earning_history
        if isinstance(raw, pd.DataFrame):
            raw = raw.copy()
            raw.attrs["status"] = _fg.OK if not raw.empty else _fg.NO_DATA
            return raw
        out = pd.DataFrame()
        out.attrs["status"] = _fg.classify(raw)
        out.attrs["by_symbol"] = {str(k): _fg.classify(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
        return out

    def _process(raw):
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return
        df = raw.reset_index() if "symbol" in raw.index.names else raw.copy()
        if "symbol" not in df.columns:
            return
        for ticker in df["symbol"].unique():
            process_history(df[df["symbol"] == ticker], ticker)

    _run_guarded("earnings_history", ticker_keys, _fetch, _process, successful_tickers, "Earnings history")

    logger.info(f"✅ Earnings Surprise: {len(successful_tickers)} tickers successful, {len(records)} quarter records.")
    if not records:
        return pd.DataFrame(columns=["ticker", "quarter_date", "eps_actual", "eps_estimate", "eps_difference", "surprise_pct", "currency", "period", "_extracted_at"])
    df_out = pd.DataFrame(records)
    df_out["quarter_date"] = pd.to_datetime(df_out["quarter_date"])
    df_out = df_out.drop_duplicates(subset=["ticker", "quarter_date"], keep="last")
    return df_out
