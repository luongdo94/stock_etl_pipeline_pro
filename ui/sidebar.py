"""Sidebar (SIDEBAR_REFACTOR_REVIEW): data status first, then the controls of the current page, macro (collapsed),
account last.

- DATA — how fresh the data in use is (core.freshness: newest close, share of fresh closes, fundamentals age, last
  successful ETL run, data quality; UNKNOWN when not known) and "Check for new data" (a newer snapshot is loaded and
  only the snapshot caches are cleared; it never runs the ETL). Per-ticker details are in 🩺 Data Health.
- PAGE CONTROLS — the chart range, shown only on the tabs that use it; it never changes a Decision.
- MACRO — rates and economy with their source; an unknown value is N/A, never 0.
"""
from datetime import timedelta

import streamlit as st

from core.market_regime import horizon_start

HORIZONS = ["1D", "1W", "1M", "3M", "6M", "1Y", "YTD", "3Y", "5Y", "ALL", "Custom"]
# the tabs whose content reads the range-filtered prices (an inventory of views/: ctx["prices"], spy_prices, t_start…).
# Stock Analysis → Risks & data → Price range fits and calibrates on the full history, independent of the chart range.
RANGE_TABS = {"🌐 Market Pulse", "🔬 Stock Analysis", "💼 Portfolio"}
# the applied choice, kept outside the widget keys: Streamlit drops a widget's state on a run that does not render it
_SAVED_HORIZON, _SAVED_CUSTOM = "_range_horizon", "_range_custom"

_CSS = """
<style>
.sb-section-label { font-size: 0.72rem; letter-spacing: 0.08em; color:inherit; text-transform: uppercase;
    margin: 12px 0 6px 0; border-bottom: 1px solid rgba(128,128,128,0.25); padding-bottom: 3px; font-weight: 700; }
.sb-card { padding: 8px 10px; background:var(--secondary-background-color, transparent); border: 1px solid rgba(128,128,128,0.25);
    border-radius: 6px; margin-bottom: 6px; font-size: 0.8rem; color:inherit; line-height: 1.45; }
.sb-muted { color:inherit; font-size: 0.74rem; }
.sb-macro-row { display: flex; justify-content: space-between; align-items: center; padding: 4px 8px;
    border-radius: 4px; margin-bottom: 3px; background:var(--secondary-background-color, transparent);
    border: 1px solid rgba(128,128,128,0.25); font-size: 0.8rem; color:inherit; }
.sb-macro-label { color:inherit; font-size: 0.74rem; }
</style>
"""
_LEVEL = {"OK": ("#2ecc71", "Up to date"), "WARN": ("#f1c40f", "Check"), "STALE": ("#e74c3c", "Stale"),
          "UNKNOWN": ("#8899aa", "Unknown")}


def _label(text):
    st.sidebar.markdown(f"<div class='sb-section-label'>{text}</div>", unsafe_allow_html=True)


# ── DATA ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def render_data_status(fresh: dict, n_exceptions: int = 0):
    """The freshness summary and the 'Check for new data' button."""
    st.sidebar.markdown(_CSS, unsafe_allow_html=True)
    _label("Data")
    colour, word = _LEVEL.get(fresh.get("level"), _LEVEL["UNKNOWN"])
    p, f, e, dq = fresh["prices"], fresh["fundamentals"], fresh["etl"], fresh["dq"]
    px = (f"Prices to <b>{p['as_of']:%d %b %Y}</b> ({p['age_days']} d)" if p["as_of"] else "Prices: <b>unknown</b>")
    fresh_share = f" · {p['fresh_share']:.0%} current" if p["fresh_share"] is not None else ""
    fund = (f"Fundamentals: median {f['median_age_days']} d old" if f["median_age_days"] is not None
            else "Fundamentals: refresh date unknown")
    if f.get("undated") and f["median_age_days"] is not None:
        fund += f" ({f['undated']} of {f['companies']} undated)"
    if e["last_success_at"]:
        verb = "finished" if e["basis"] == "end" else "started"
        etl = f"Last good ETL {verb} {e['last_success_at']:%d %b %H:%M} ({e['age_hours']:.0f} h ago)"
    else:
        etl = "Last good ETL: unknown"
    dq_txt = {"CLEAN": "checks clean", "WARN": f"{dq['warnings']} check warning(s)",
              "CRIT": f"{dq['critical']} critical check(s)", "UNKNOWN": "checks unavailable"}[dq["state"]]
    dq_col = {"CLEAN": "#2ecc71", "WARN": "#f1c40f", "CRIT": "#e74c3c", "UNKNOWN": "#8899aa"}[dq["state"]]
    exc = f"<br><span class='sb-muted'>{n_exceptions} ticker exception(s) — Data Health</span>" if n_exceptions else ""
    st.sidebar.markdown(
        f"<div class='sb-card'><span style='color:{colour};font-weight:700;'>● {word}</span><br>{px}{fresh_share}<br>"
        f"<span class='sb-muted'>{fund}<br>{etl}</span><br>"
        f"<span class='sb-muted'>Data quality: <span style='color:{dq_col};font-weight:700;'>● {dq_txt}</span></span>{exc}</div>", unsafe_allow_html=True)
    if st.sidebar.button("Check for new data", width="stretch", key="sb_check_data",
                         help="Looks for a newer warehouse snapshot and reloads only what depends on it. It does not "
                              "run the ETL or fetch new data from the providers."):
        from services.db import check_for_new_data
        res = check_for_new_data(st.session_state.get("_data_version"))
        st.session_state["_data_check"] = res
        if res["status"] == "updated":
            st.rerun()      # the app records the version it actually loads (not the one found here)
    res = st.session_state.get("_data_check")
    if res:
        (st.sidebar.success if res["status"] == "updated" else
         st.sidebar.info if res["status"] == "latest" else st.sidebar.warning)(res["message"])


# ── PAGE CONTROLS ─────────────────────────────────────────────────────────────────────────────────────────────────
def _clamp(d, min_date, max_date):
    return min(max(d, min_date), max_date)


def _range(horizon, min_date, max_date, rng=None):
    """(start, end) with min_date <= start <= end <= max_date — a stored range outside the warehouse (older than its
    first date, after an ETL changed the bounds) is clamped at both ends, an inverted one is put in order."""
    start, end = horizon_start(horizon, min_date, max_date), max_date
    if horizon == "Custom":
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            start, end = sorted(rng)
        elif isinstance(rng, (list, tuple)) and len(rng) == 1:
            start = rng[0]
    start, end = _clamp(start, min_date, max_date), _clamp(end, min_date, max_date)
    return min(start, end), end


def render_horizon(min_date, max_date, active_tab=None):
    """Deprecated: chart range control moved directly into sections (ui.range_control).

    Returns (horizon, start, end) clamped to warehouse bounds without rendering sidebar UI.
    """
    ss = st.session_state
    saved_h = ss.get("time_horizon_ctrl") or ss.get(_SAVED_HORIZON) or "1Y"
    saved_rng = ss.get("custom_range") or ss.get(_SAVED_CUSTOM)
    start, end = _range(saved_h, min_date, max_date, saved_rng)
    return saved_h, start, end


# ── MACRO ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def _num(d, k):
    try:
        v = float((d or {}).get(k, {}).get("val"))
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def render_macro_panel(macro, fred=None, live=None):
    """Rates and economy, collapsed; N/A when a value is unknown; values are context, not signals.

    `macro` is what the snapshot stores (dated closes). "Fetch live" asks the providers (Yahoo, FRED) once and shows the
    result here, labelled, for display only: the Decision's risk-free rate and the regime keep reading the stored closes.
    `live` = {"version", "macro", "fred", "at"} as kept by the button; ignored once the snapshot changed."""
    ss = st.session_state
    with st.sidebar.expander("Macro — US rates & economy", expanded=False):
        if st.button("Fetch live rates & economy", key="sb_live_macro", width="stretch",
                     help="Asks Yahoo (rates, VIX, dollar) and FRED (CPI, unemployment, Fed funds) now. Shown here only — "
                          "the Decision and the regime keep using the closes stored in the snapshot."):
            from datetime import datetime
            from services.market_data import fetch_fred_macro, fetch_macro_data
            with st.spinner("Asking the providers..."):
                ss["_live_markets"] = live = {"version": ss.get("_data_version"), "macro": fetch_macro_data(ss.get("_data_version")),
                                              "fred": fetch_fred_macro(), "at": datetime.now()}
        if live and live.get("version") != ss.get("_data_version"):
            live = None                                   # fetched for another snapshot
        shown = (live or {}).get("macro") or macro
        fred = (live or {}).get("fred") or fred or {}
        tnx, irx = _num(shown, "US10Y"), _num(shown, "US2Y")
        spread = tnx - irx if tnx is not None and irx is not None else None
        cpi, un, ff = _num(fred, "CPI"), _num(fred, "UNRATE"), _num(fred, "FEDFUNDS")
        fmt = lambda v, f: "N/A" if v is None else format(v, f) + "%"  # noqa: E731
        month = str(fred.get("CPI", {}).get("date") or "")[:7] or "n/a"
        rows = [("US 10Y yield", fmt(tnx, ".2f")), ("10Y – 3M spread", fmt(spread, ".2f")),
                ("CPI YoY", fmt(cpi, ".2f")), ("Unemployment", fmt(un, ".1f")), ("Fed Funds", fmt(ff, ".2f"))]
        st.markdown("".join(f"<div class='sb-macro-row'><span class='sb-macro-label'>{k}</span><b>{v}</b></div>"
                            for k, v in rows), unsafe_allow_html=True)
        if live:
            st.caption(f"Live from Yahoo / FRED, fetched {live['at']:%d %b %H:%M} — display only. The Decision and the "
                       "regime read the stored closes.")
        else:
            src = (macro or {}).get("US10Y", {}).get("source") or "unavailable"
            st.caption(f"Rates: {src} (the 10Y also prices the DCF). The 3M bill and the economy (CPI, unemployment, Fed "
                       "funds) are not stored: press Fetch live. Context only — not a signal.")
        if live:
            st.caption(f"Economy: FRED, latest {month}. 3M = 13-week T-bill (^IRX).")


# ── PHONE LAYOUT SWITCH ───────────────────────────────────────────────────────────────────────────────────────────
def is_phone_layout() -> bool:
    """Read the choice persisted by the Phone layout toggle."""
    from ui.sticky import kept_key
    return bool(st.session_state.get(kept_key("ui_phone_layout"), False))


def render_phone_toggle() -> bool:
    """Persist the toggle; the URL only seeds a session without a prior choice."""
    from ui.sticky import kept_key, sticky as _sticky
    if "_ui_phone_layout_initialized" not in st.session_state:
        if kept_key("ui_phone_layout") not in st.session_state:
            st.session_state[kept_key("ui_phone_layout")] = st.session_state.get(
                "ui_phone_layout", st.query_params.get("phone") == "1")
        st.session_state["_ui_phone_layout_initialized"] = True
    _sticky(
        st.sidebar.toggle,
        "Phone layout",
        False,
        "ui_phone_layout",
        help="Optimises tables, charts and views for phone screens (360-430 px)."
    )
    return is_phone_layout()
