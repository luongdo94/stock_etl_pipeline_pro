"""
Per-user data for LOCAL_DEV_MODE (no Supabase): watchlist, portfolio and alert rules as JSON files under
warehouse/local_user/. One local user, one machine — never used when a real login is configured.
"""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = Path(os.environ.get("LOCAL_STORE_DIR") or ROOT / "warehouse" / "local_user")
_STORE_LOCK = Lock()


def _path(name: str) -> Path:
    return STORE_DIR / f"{name}.json"


class LocalStoreError(OSError):
    """A local file exists but cannot be read or parsed. Callers must not overwrite it (audit F23)."""


def read(name: str, default):
    """`default` only when the file does not exist yet. A corrupt or unreadable file raises LocalStoreError and is left
    untouched (it used to read as empty, and the next save replaced it)."""
    p = _path(name)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LocalStoreError(f"{p} could not be read ({e}); it was left untouched — fix or move it, then reload") from e


def write(name: str, data) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=STORE_DIR,
                            prefix=f".{name}-", suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
        try:
            json.dump(data, handle, ensure_ascii=False, indent=1, default=str)
        except Exception:
            handle.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        tmp.replace(_path(name))                  # never leave a half-written file
    finally:
        tmp.unlink(missing_ok=True)


def _versioned(name, default):
    raw = read(name, default)
    if isinstance(raw, dict) and "rev" in raw and "data" in raw:
        return raw["data"], int(raw["rev"])
    return raw, 0                                # legacy JSON


def load_versioned(name, default):
    """Read rows and their revision together, under the same process-wide lock as writes."""
    with _STORE_LOCK:
        return _versioned(name, default)


def replace_versioned(name, data, expected_rev=None):
    with _STORE_LOCK:
        _, rev = _versioned(name, [] if name == "watchlist" else {})
        if expected_rev is not None and rev != expected_rev:
            raise LocalStoreError("Not saved: changed in another session — reload")
        write(name, {"rev": rev + 1, "data": data})
        return rev + 1


# watchlist: list of records keyed like the dashboard's columns
def load_watchlist_records() -> list:
    return load_versioned("watchlist", [])[0]


def save_watchlist_records(records: list, expected_rev=None) -> int:
    return replace_versioned("watchlist", records, expected_rev)


def load_portfolio() -> dict:
    return load_versioned("portfolio", {})[0]


def save_portfolio(shares: dict, costs: dict, expected_rev=None) -> int:
    from math import isfinite

    data = {t: {"shares": float(qty), "cost": float(costs[t])
                if costs.get(t) is not None and isfinite(float(costs[t])) else None} for t, qty in shares.items()}
    return replace_versioned("portfolio", data, expected_rev)


def load_trades() -> list:
    return read("trades", [])


def append_trades(rows: list) -> None:
    write("trades", load_trades() + list(rows))


def load_alerts() -> list:
    return read("alerts", [])


def add_alert(ticker, metric, condition, threshold) -> None:
    rules = load_alerts()
    rules.append({"id": max([r["id"] for r in rules], default=0) + 1, "ticker": ticker, "metric": metric,
                  "condition": condition, "threshold": float(threshold)})
    write("alerts", rules)


def delete_alert(rule_id) -> None:
    write("alerts", [r for r in load_alerts() if r["id"] != rule_id])
