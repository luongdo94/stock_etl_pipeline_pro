"""Reusable in-section chart time-range control.

Replaces the global sidebar range control ("Option B").
Renders a compact segmented control directly in the section that uses it, returning
the window (horizon, t_start, t_end) and the filtered price dataframes (prices, spy_prices).
"""
from datetime import date, timedelta
from typing import NamedTuple, Optional

import pandas as pd
import streamlit as st

from core import market_regime as mr
from core.market_regime import horizon_start
from ui.sidebar import HORIZONS, _SAVED_CUSTOM, _SAVED_HORIZON, _clamp, _range
from ui.sticky import kept_key


class ChartRangeResult(NamedTuple):
    window: tuple  # (horizon, t_start, t_end)
    prices: pd.DataFrame
    spy_prices: pd.DataFrame

    @property
    def horizon(self) -> str:
        return self.window[0]

    @property
    def t_start(self) -> pd.Timestamp:
        return self.window[1]

    @property
    def t_end(self) -> pd.Timestamp:
        return self.window[2]


def allowed_horizons(
    min_date: date,
    max_date: date,
    prices_full: Optional[pd.DataFrame] = None,
    min_sessions: Optional[int] = None,
) -> list[str]:
    """Return horizons that can provide at least `min_sessions` trading sessions."""
    if min_sessions is None or min_sessions <= 1:
        return list(HORIZONS)

    allowed = []
    for h in HORIZONS:
        if h == "Custom":
            if prices_full is not None and not prices_full.empty:
                total_sessions = prices_full["date"].nunique()
            else:
                total_sessions = (max_date - min_date).days * 5 // 7
            if total_sessions >= min_sessions:
                allowed.append("Custom")
            continue

        st_date = horizon_start(h, min_date, max_date)
        if prices_full is not None and not prices_full.empty:
            sessions = prices_full.loc[
                (prices_full["date"] >= pd.Timestamp(st_date)) & (prices_full["date"] <= pd.Timestamp(max_date)),
                "date",
            ].nunique()
        else:
            sessions = (max_date - st_date).days * 5 // 7

        if sessions >= min_sessions:
            allowed.append(h)
    return allowed


def resolve_range(
    prices_full: pd.DataFrame,
    horizon: Optional[str] = None,
    custom_range: Optional[tuple] = None,
    min_sessions: Optional[int] = None,
) -> ChartRangeResult:
    """Compute the clamped date window and filtered price frames without rendering UI."""
    if prices_full is None or prices_full.empty:
        today = date.today()
        t_start = t_end = pd.Timestamp(today)
        empty = pd.DataFrame(columns=["date", "ticker", "price_close"])
        return ChartRangeResult((horizon or "1Y", t_start, t_end), empty, empty)

    min_date = pd.to_datetime(prices_full["date"].min()).date()
    max_date = pd.to_datetime(prices_full["date"].max()).date()

    allowed = allowed_horizons(min_date, max_date, prices_full, min_sessions=min_sessions)
    h = horizon or "1Y"
    if h not in allowed and allowed:
        h = allowed[0]

    start, end = _range(h, min_date, max_date, custom_range)
    t_start, t_end = pd.Timestamp(start), pd.Timestamp(end)

    in_window = (prices_full["date"] >= t_start) & (prices_full["date"] <= t_end)
    indices_list = mr.INDICES
    spy_prices = prices_full[in_window & (prices_full["ticker"] == "SPY")]
    prices = prices_full[in_window & ~prices_full["ticker"].isin(indices_list)]

    return ChartRangeResult((h, t_start, t_end), prices, spy_prices)


def chart_range(
    ctx: dict,
    *,
    min_sessions: Optional[int] = None,
    label: str = "Chart range — applies to the charts below",
    key: str = "time_horizon_ctrl",
) -> ChartRangeResult:
    """Render a compact segmented control for chart time-horizon and return window + filtered frames.

    Persists choice in the shared session keys (`_range_horizon`, `_range_custom`) so choices survive tab switching.
    """
    prices_full = ctx.get("prices_full")
    if prices_full is None or prices_full.empty:
        return resolve_range(prices_full, horizon="1Y", min_sessions=min_sessions)

    min_date = pd.to_datetime(prices_full["date"].min()).date()
    max_date = pd.to_datetime(prices_full["date"].max()).date()

    allowed = allowed_horizons(min_date, max_date, prices_full, min_sessions=min_sessions)
    ss = st.session_state

    # One shared choice (_SAVED_HORIZON) for every section; each control is a widget of its own. A widget value that
    # did not change since its last run (no click this run) follows the shared choice, so a range picked on another tab
    # shows here; a click this run wins. A section that needs min_sessions shows its shortest allowed range instead of a
    # too-short shared choice WITHOUT overwriting it (Market Pulse keeps the user's 1M after a Portfolio visit).
    saved_h = ss.get(_SAVED_HORIZON) or "1Y"
    saved_rng = ss.get(_SAVED_CUSTOM) or ss.get("custom_range")
    seen_key, auto_key = f"_{key}__seen", f"_{key}__auto"
    clicked = key in ss and ss.get(key) != ss.get(seen_key)
    if not clicked or ss.get(key) not in allowed:
        start_h = saved_h if saved_h in allowed else (allowed[0] if allowed else "1Y")
        ss[key] = start_h
        ss[auto_key] = start_h if start_h != saved_h else None
    horizon = st.segmented_control(label, allowed, key=key) or ss.get(key) or (allowed[0] if allowed else "1Y")
    if horizon != ss.get(auto_key):                           # a real choice, not the min_sessions fallback
        ss[_SAVED_HORIZON] = horizon
        ss[auto_key] = None
    elif saved_h != horizon:
        st.caption(f"Showing {horizon}: {saved_h} gives fewer than {min_sessions} sessions for these figures.")
    ss[seen_key] = horizon

    rng = saved_rng
    if horizon == "Custom":
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            rng = (max(max_date - timedelta(days=365), min_date), max_date)
        restored = tuple(_clamp(d, min_date, max_date) for d in _range("Custom", min_date, max_date, rng))
        cur_custom = ss.get("custom_range")
        if not (isinstance(cur_custom, (list, tuple)) and cur_custom and all(min_date <= d <= max_date for d in cur_custom)):
            ss["custom_range"] = restored

        with st.form("chart_custom_range_form", border=False):
            picked = st.date_input("Custom dates", key="custom_range", min_value=min_date, max_value=max_date)
            st.form_submit_button("Apply", width="stretch")

        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            ss[_SAVED_CUSTOM] = rng = tuple(picked)
            ss[kept_key("custom_range")] = rng
        else:
            st.caption("Pick a start and an end date — the last full range stays applied.")
            rng = restored

    res = resolve_range(prices_full, horizon=horizon, custom_range=rng if horizon == "Custom" else None, min_sessions=min_sessions)
    return res
