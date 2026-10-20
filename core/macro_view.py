"""Display logic of the economic-indicator table (pure)."""
from datetime import datetime

import pandas as pd

ORDER = {"US": 0, "EU": 1, "JP": 2, "CN": 3}
REGION_NAMES = {"US": "United States", "EU": "Euro area", "JP": "Japan", "CN": "China"}


def period_label(period, frequency: str) -> str:
    """'Sep 2026' for monthly, '2026 Q2' for quarterly series (the stored period is the first day of the period)."""
    p = pd.Timestamp(period)
    return f"{p.year} Q{(p.month - 1) // 3 + 1}" if str(frequency).lower().startswith("q") else f"{p:%b %Y}"


def table_for_display(df: pd.DataFrame, today=None) -> pd.DataFrame:
    """Rows ordered by region, with `stale` (period older than max_age_days) and `change` (value − previous)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["id", "region", "name", "unit", "period", "value", "previous_period", "previous_value", "source", "series", "frequency", "max_age_days", "fetched_at", "stale", "change"])
    today = pd.Timestamp(today or datetime.now()).normalize()
    d = df.copy()
    d["period"] = pd.to_datetime(d["period"])
    d["age_days"] = (today - d["period"]).dt.days
    d["stale"] = d["age_days"] > d["max_age_days"]
    d["change"] = d["value"] - d["previous_value"]
    d["previous_label"] = [
        f"{v:.2f} ({period_label(p, f) if pd.notna(p) else 'period unknown'})" if pd.notna(v) else "N/A"
        for v, p, f in zip(d["previous_value"], d["previous_period"], d["frequency"])]
    d["collected_label"] = pd.to_datetime(d["fetched_at"], errors="coerce").map(
        lambda p: f"Collected {p:%d %b %Y}" if pd.notna(p) else "Collection date unknown")
    d["status_label"] = [f"STALE — observation period {age:.0f} d old" if stale else ""
                         for age, stale in zip(d["age_days"], d["stale"])]
    d["_o"] = d["region"].map(ORDER).fillna(9)
    return d.sort_values(["_o", "id"], kind="mergesort").drop(columns="_o").reset_index(drop=True)
