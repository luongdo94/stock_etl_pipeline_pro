"""
Provider guard for the Yahoo extractors (ETL review 2026-10-15).

The extractors used to treat every failure the same way: three passes, the last with a randomised browser
User-Agent. On the 2026-10-14 full run pass 2 recovered 0 tickers and pass 3 recovered 12 of 158 failures — most
failures were permanent "no data" answers, retried for ~30 minutes — while REAL blocking (rate-limited responses that
arrive as dicts of empty modules, no HTTP error) was not recognised at all and the next batch went out 1–2 s later.

This module gives every extractor:
  classify()      OK / BLOCKED / NO_DATA / ERROR for any yahooquery payload or exception
  Guard           per-extractor circuit breaker: a batch that comes back mostly BLOCKED triggers an exponential
                  cool-down (and smaller batches); past the cool-down budget the extractor stops calling the provider
                  for the rest of the run (previous data is kept: incremental UPSERTs, full-refresh carry-forward)
  NoDataCache     tickers the provider has no data for are not asked again for `no_data_ttl_days` (per extractor),
                  persisted in warehouse/provider_no_data.json so it survives full rebuilds
  ledger          BLOCKED / NO_DATA tickers and the guard's counters go to the per-ticker exception ledger
No identity rotation: requests go out with the library's normal session, throttled, never disguised.
"""
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

OK, BLOCKED, NO_DATA, ERROR = "OK", "BLOCKED", "NO_DATA", "ERROR"
_NO_DATA_HINTS = ("not found", "no fundamentals", "no data", "delisted", "does not exist", "no earnings",
                  "unavailable")      # e.g. 'Cash Flow data unavailable for ROG.SW' (live answer, 2026-10-15)
_BLOCK_HINTS = ("too many", "429", "unauthorized", "401", "403", "forbidden", "crumb", "rate limit", "blocked",
                "invalid response", "temporarily", "try again")
CACHE_PATH = Path(__file__).resolve().parent.parent / "warehouse" / "provider_no_data.json"
DEFAULTS = {"block_share": 0.3, "cooldown_base_s": 30.0, "cooldown_cap_s": 300.0, "cooldown_budget_s": 900.0,
            "retry_pause_s": 1.5, "no_data_ttl_days": 14}


def _text_status(text: str) -> str:
    t = str(text).lower()
    if any(h in t for h in _BLOCK_HINTS):
        return BLOCKED
    if any(h in t for h in _NO_DATA_HINTS):
        return NO_DATA
    return ERROR


def classify(payload) -> str:
    """Status of one ticker's yahooquery payload: a frame / dict with data = OK, an error string = NO_DATA or
    BLOCKED by its text, a dict whose modules are all empty = BLOCKED (how a refused request actually looks)."""
    if payload is None:
        return BLOCKED
    if isinstance(payload, pd.DataFrame):
        return OK if not payload.empty else NO_DATA
    if isinstance(payload, str):
        return _text_status(payload)
    if isinstance(payload, dict):
        if not payload:
            return BLOCKED
        vals = list(payload.values())
        if all(isinstance(v, str) for v in vals):
            return _text_status(" ".join(vals))
        if all((isinstance(v, (dict, list)) and not v) or isinstance(v, str) for v in vals):
            return BLOCKED
        return OK
    if isinstance(payload, (list, tuple)):
        return OK if payload else NO_DATA
    return OK


def classify_exception(e: BaseException) -> str:
    return BLOCKED if _text_status(f"{type(e).__name__} {e}") == BLOCKED else ERROR


class NoDataCache:
    """{extractor: {ticker: expiry ISO date}} in a small JSON file next to the warehouse."""

    def __init__(self, path: Path = CACHE_PATH, today: Optional[date] = None):
        self.path, self.today = Path(path), today or date.today()
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def hit(self, extractor: str, ticker: str) -> bool:
        exp = (self.data.get(extractor) or {}).get(ticker)
        return bool(exp) and date.fromisoformat(exp) >= self.today

    def add(self, extractor: str, ticker: str, ttl_days: int) -> None:
        self.data.setdefault(extractor, {})[ticker] = (self.today + timedelta(days=int(ttl_days))).isoformat()

    def drop(self, extractor: str, ticker: str) -> None:
        (self.data.get(extractor) or {}).pop(ticker, None)

    def save(self) -> None:
        live = {x: {t: e for t, e in m.items() if date.fromisoformat(e) >= self.today} for x, m in self.data.items()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(live, indent=1, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as e:
            logger.warning(f"   ⚠️ provider no-data cache not saved: {e}")


_CACHE: Optional[NoDataCache] = None


def cache() -> NoDataCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = NoDataCache()
    return _CACHE


def reset_cache(c: Optional[NoDataCache] = None) -> None:
    global _CACHE
    _CACHE = c


class Guard:
    """One extractor's throttle + circuit breaker + bookkeeping for one run."""

    def __init__(self, extractor: str, settings: Optional[dict] = None, sleep=None, no_data_cache=None):
        self.name = extractor
        self.cfg = {**DEFAULTS, **(settings or _settings())}
        self.sleep = sleep or (lambda s: time.sleep(s))         # looked up at call time (tests patch time.sleep)
        self.cache = no_data_cache or cache()
        self.level = 0
        self.cooled = 0.0
        self.tripped = False
        self.status: dict = {}

    # ── before asking ──────────────────────────────────────────────────────────────────────────────────────────
    def active(self, tickers) -> list:
        """Tickers worth asking: not answered 'no data' within the TTL (cached ones are counted as NO_DATA)."""
        out = []
        for t in tickers:
            if self.cache.hit(self.name, t):
                self.status[t] = NO_DATA
            else:
                out.append(t)
        skipped = len(tickers) - len(out)
        if skipped:
            logger.info(f"   ⏭️  {self.name}: {skipped} ticker(s) skipped — provider had no data for them recently")
        return out

    # ── after asking ───────────────────────────────────────────────────────────────────────────────────────────
    def mark(self, ticker: str, status: str) -> None:
        self.status[ticker] = status
        if status == NO_DATA:
            self.cache.add(self.name, ticker, self.cfg["no_data_ttl_days"])
        elif status == OK:
            self.cache.drop(self.name, ticker)

    def after_batch(self, statuses: dict) -> None:
        """Record a batch; cool down when it was mostly refused, relax again after a clean batch."""
        for t, s in statuses.items():
            self.mark(t, s)
        if not statuses:
            return
        blocked = sum(1 for s in statuses.values() if s == BLOCKED)
        if blocked / len(statuses) >= self.cfg["block_share"]:
            self._cool_down(f"{blocked}/{len(statuses)} refused")
        elif self.level:
            self.level -= 1

    def after_one(self, ticker: str, status: str) -> None:
        """Single-ticker retry: a refusal cools down at once; otherwise a short polite pause."""
        self.mark(ticker, status)
        if status == BLOCKED:
            self._cool_down(f"{ticker} refused")
        else:
            self.sleep(self.cfg["retry_pause_s"])

    def _cool_down(self, why: str) -> None:
        if self.tripped:
            return
        wait = min(self.cfg["cooldown_base_s"] * (2 ** self.level), self.cfg["cooldown_cap_s"])
        if self.cooled + wait > self.cfg["cooldown_budget_s"]:
            self.tripped = True
            logger.warning(f"   🧯 {self.name}: provider keeps refusing ({why}) — circuit open, no more requests this "
                           f"run (previous data is kept)")
            return
        logger.warning(f"   🧊 {self.name}: provider is refusing requests ({why}) — cooling down {wait:.0f}s")
        self.sleep(wait)
        self.cooled += wait
        self.level += 1

    @property
    def batch_scale(self) -> float:
        """Shrink batches while cooling down (1, 1/2, 1/4 … ≥ 1/8)."""
        return max(1.0 / (2 ** self.level), 0.125)

    def retry_candidates(self, tickers) -> list:
        """Failed tickers worth ONE more try: refused or errored, never 'no data'."""
        return [t for t in tickers if self.status.get(t) in (BLOCKED, ERROR, None)]

    # ── reporting ──────────────────────────────────────────────────────────────────────────────────────────────
    def finish(self, all_tickers, succeeded) -> dict:
        """Close the run: unresolved tickers keep their last status (BLOCKED when the circuit opened before them),
        ledger entries, cache saved, counters logged and returned."""
        from etl import exceptions as _exc
        ok = set(succeeded)
        counts = {OK: 0, NO_DATA: 0, BLOCKED: 0, ERROR: 0}
        for t in all_tickers:
            s = OK if t in ok else self.status.get(t, BLOCKED if self.tripped else ERROR)
            counts[s] = counts.get(s, 0) + 1
            if s == BLOCKED:
                _exc.record(t, self.name, "BLOCKED by the provider (rate limit) — previous data kept")
            elif s == NO_DATA:
                _exc.record(t, self.name, f"provider has NO DATA — not asked again for {self.cfg['no_data_ttl_days']} days")
            elif s == ERROR:
                _exc.record(t, self.name, "request error after one retry — previous data kept")
        self.cache.save()
        logger.info(f"   📊 {self.name}: ok {counts[OK]} · no data {counts[NO_DATA]} · blocked {counts[BLOCKED]} · "
                    f"errors {counts[ERROR]} · cooled {self.cooled:.0f}s{' · CIRCUIT OPEN' if self.tripped else ''}")
        return {**counts, "cooled_s": self.cooled, "tripped": self.tripped}


def _settings() -> dict:
    try:
        from etl.config_manager import get_etl_config
        return dict(get_etl_config().get("provider") or {})
    except Exception:
        return {}
