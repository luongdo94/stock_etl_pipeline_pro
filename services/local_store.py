"""
Per-user data for LOCAL_DEV_MODE (no Supabase): watchlist, portfolio and alert rules as JSON files under
warehouse/local_user/. One local user, one machine — never used when a real login is configured.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = Path(os.environ.get("LOCAL_STORE_DIR") or ROOT / "warehouse" / "local_user")


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
    tmp = _path(name).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(_path(name))                      # never leave a half-written file


# watchlist: list of records keyed like the dashboard's columns
def load_watchlist_records() -> list:
    return read("watchlist", [])


def save_watchlist_records(records: list) -> None:
    write("watchlist", records)


def load_portfolio() -> dict:
    return read("portfolio", {})


def save_portfolio(shares: dict, costs: dict) -> None:
    write("portfolio", {t: {"shares": float(shares.get(t, 0)), "cost": float(costs.get(t, 0))} for t in shares})


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
