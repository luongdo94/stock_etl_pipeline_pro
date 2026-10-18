"""Strategy Lab presentation logic (pure): N/A formatting, neutral comparison, run identity, sample metadata.

The Lab compares timing rules on ONE stock's own history, in-sample. Nothing here crowns a winner."""
import math
from typing import Optional

import pandas as pd

BASIS = "adjusted EUR closes, signal filled at the next close; buy & hold is fee-free"
EXPLORATORY = "Exploratory, in-sample: the rules are scored on the same history they were chosen from."


def finite(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fmt(v, spec: str = "{:.2f}", suffix: str = "") -> str:
    f = finite(v)
    return "N/A" if f is None else spec.format(f) + suffix


def win_rate(r: dict) -> Optional[float]:
    """None (N/A) when no trade was completed — a rule that never traded did not win 0% of its trades."""
    return finite(r.get("win_rate")) if (r.get("n_trades") or 0) > 0 else None


def run_spec(ticker, sl, tp, capital, fee, version) -> dict:
    return {"ticker": ticker, "sl": int(sl), "tp": int(tp), "capital": float(capital), "fee": float(fee),
            "snapshot": version}


def stale_reason(run: Optional[dict], spec: dict) -> Optional[str]:
    """Why the stored result no longer matches the controls / the data (None = it is current)."""
    if not run:
        return None
    old = run["spec"]
    changed = [label for k, label in (("ticker", "ticker"), ("sl", "stop loss"), ("tp", "take profit"),
                                      ("capital", "capital"), ("fee", "transaction cost")) if old.get(k) != spec.get(k)]
    if changed:
        return "controls changed (" + ", ".join(changed) + ")"
    if old.get("snapshot") != spec.get("snapshot"):
        return "the data snapshot was refreshed"
    return None


def validate(prices: pd.DataFrame, min_bars: int = 60) -> Optional[str]:
    if prices is None or prices.empty:
        return "No price history for this ticker."
    if len(prices) < min_bars:
        return f"Only {len(prices)} price bars — at least {min_bars} are needed."
    if "daily_return_pct" not in prices.columns or prices["daily_return_pct"].notna().sum() < min_bars - 1:
        return "Too few daily returns on file for this ticker."
    return None


def comparison_rows(results: list) -> pd.DataFrame:
    """One row per rule, in the order they were run (the experimental demo last). NaN/None stay None → shown as N/A."""
    rows = [{"Rule": r["strategy"], "Return %": finite(r["total_return"]), "vs buy & hold (pts)":
             finite(r["total_return"] - r["bnh_return"]), "Sharpe": finite(r["sharpe"]),
             "Max DD %": finite(r["max_dd"]), "Win rate %": win_rate(r), "Trades": int(r["n_trades"]),
             "Days in market %": (100.0 * r["bars_held"] / r["n_bars"]) if r.get("n_bars") else None,
             "Lookahead": bool(r.get("lookahead")), "Missing returns": int(r.get("missing_returns", 0))}
            for r in results]
    return pd.DataFrame(rows)


def neutral_summary(results: list) -> str:
    """Descriptive, not a verdict: how many fair (no-lookahead) rules ended above buy & hold in this window."""
    fair = [r for r in results if not r.get("lookahead")]
    if not fair:
        return ""
    bnh = fair[0]["bnh_return"]
    above = sum(1 for r in fair if r["total_return"] > r["bnh_return"])
    return (f"In this window {above} of {len(fair)} timing rules ended above buy & hold ({bnh:+.1f}%, fee-free). "
            f"That describes the past of this one stock; it does not say which rule to use.")


def sample_meta(r: dict) -> str:
    d = pd.to_datetime(pd.Series(r["dates_arr"]))
    held = f"{r['bars_held']} of {r['n_bars']} bars in the market" if r.get("n_bars") else ""
    return (f"{d.min():%d %b %Y} → {d.max():%d %b %Y} · {r['n_bars']} bars · {held} · "
            f"{r['n_trades']} completed trade(s) · {BASIS}.")
