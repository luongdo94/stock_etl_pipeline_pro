"""View: 🔭 Stock Scanner — the research queue.

One Decision per stock comes from the shared table (ctx["m_df"], core.screener — the same Decision the Stock Analysis page and
the assistant show). This page only selects and orders those rows: a summary of the universe, six views (Candidates, Held,
Watchlist, Needs data, Changed, All), filters and a selectable table that opens the stock in Stock Analysis. No view adds a
score or a recommendation of its own (core/scanner_views.py is pure and tested without Streamlit).
"""
import hashlib

import pandas as pd
import streamlit as st

from core import scan_presets as presets
from core import scanner_views as sv
from core.scan_presets import ELITE, FAIR, SOLID
from core.scoring import SCORE_VERSION
from ui.icons import render_header
from ui.watchlist_status import status_label
from ui.sticky import forget, kept_key, sticky, sticky_choice, sticky_multi, sticky_pills

# Default columns answer the four questions of a long-horizon screen, each with a number every stock has:
#   is the business good (Quality), is the price cheap (Value, FCF yield), what does the price already assume
#   (Growth Gap = delivered revenue growth − growth implied by the price), and is there an event / red flag now
#   (Next ER, Flags, Conflicts) — plus Why: the Decision's own first reason, so a row explains itself.
DEFAULT_COLS = ["Ticker", "Company", "Verdict", "Why", "Quality", "Value", "Momentum", "Revisions", "FCF Yield (%)",
                "Growth Gap (pp)", "Next ER (days)", "Flags", "Conflicts"]
# Detail one click away; near-duplicates removed (dividend yield ≈ net payout, ρ 0.99; gross ≈ net debt/EBITDA, ρ 0.80)
EXTRA_COLS = ["Confidence", "Sector", "MCap (B)", "MoS (%)", "Exp. Return (%/yr)", "Implied Growth (%)", "P/E (Fwd)",
              "EV/EBITDA", "PEG", "ROIC (%)", "ROE (%)", "FCF Margin (%)", "FCF Conversion (x)", "Net Debt/EBITDA",
              "Net Payout (%)", "Insider 6M (%)", "ADV (EUR M)", "Action", "Smart Money", "vs MA200 (%)", "Z-Score",
              "RSI (14)", "Coverage (%)", "Data issue"]
SORTS = {"Decision, then Quality, Value": None, "Quality ↓": ("Quality", False), "Value ↓": ("Value", False),
         "FCF Yield ↓": ("FCF Yield (%)", False), "Next earnings ↑": ("Next ER (days)", True), "Ticker A–Z": ("Ticker", True)}
STATE_KEYS = ("scan_view", "scan_search", "p_region_filter", "p_sector_filter", "scan_mode", "scan_sort", "scan_skip_er",
              "custom_sm_filter", "scan_extra_cols", "scan_min_q", "scan_min_v", "scan_min_m", "scan_rsi", "scan_pe",
              "scan_z", "scan_peg", "scan_fcf", "scan_archived", "radar_limit")


def _clean_conflicts(text) -> str:
    """The earnings blackout has its own column (Next ER); keep the other open conflicts."""
    parts = [x for x in str(text or "").split("; ") if x and not x.startswith("Earnings in ")]
    return "; ".join(parts)


def style_opportunity_df(df):
    def highlight_action(val):
        val_str = str(val).upper()
        if "RESEARCH" in val_str: return 'color: #3498db; font-weight: bold'
        if "STRONG SETUP" in val_str: return 'color: #2ecc71; font-weight: 800'
        elif "FAVOURABLE" in val_str and "UN" not in val_str: return 'color: #27ae60; font-weight: bold'
        elif "BUY" in val_str: return 'color: #27ae60; font-weight: bold'
        elif "UNFAVOURABLE" in val_str or "SELL" in val_str or "AVOID" in val_str: return 'color: #e74c3c; font-weight: bold'
        elif "NOT ENOUGH" in val_str: return 'color:inherit'
        elif "REDUCE" in val_str or "WEAKENING" in val_str: return 'color: #e67e22; font-weight: bold'
        return 'color: #f1c40f'  # HOLD

    def highlight_smart_money(val):
        val_str = str(val).upper()
        if "ACCUMULATION" in val_str: return 'color: #2ecc71'
        elif "DISTRIBUTION" in val_str: return 'color: #e74c3c'
        return ''

    def color_pos_neg(val):
        try:
            v = float(val)
            if v > 0: return 'color: #2ecc71'
            elif v < 0: return 'color: #e74c3c'
        except: pass
        return ''

    def color_zscore(val):
        try:
            v = float(val)
            if v <= -2.0: return 'color: #2ecc71; font-weight: bold'
            elif v >= 2.0: return 'color: #e74c3c; font-weight: bold'
        except: pass
        return ''

    def color_rsi(val):
        try:
            v = float(val)
            if v < 35: return 'color: #2ecc71; font-weight: bold'
            elif v > 65: return 'color: #e74c3c; font-weight: bold'
        except: pass
        return ''

    def color_peg(val):
        try:
            v = float(val)
            if v > 0 and v <= 1.0: return 'color: #2ecc71'
            elif v > 2.5: return 'color: #e74c3c'
        except: pass
        return ''

    def color_debt(val):
        try:
            v = float(val)
            if v < 1.5: return 'color: #2ecc71'
            elif v > 3.0: return 'color: #e74c3c'
        except: pass
        return ''

    def color_high_good(val, threshold):
        try:
            if float(val) >= threshold: return 'color: #2ecc71'
        except: pass
        return ''

    def color_fcf_yield(val):
        try:
            v = float(val)
            if v >= 5: return 'color: #2ecc71'
            if v < 0: return 'color: #e74c3c'
        except: pass
        return ''

    def color_gap(val):
        try:
            v = float(val)
            if v >= 0: return 'color: #2ecc71'
            if v <= -20: return 'color: #e74c3c'
        except: pass
        return ''

    def color_er(val):
        try:
            if float(val) <= 7: return 'color: #f39c12; font-weight: bold'
        except: pass
        return ''

    def on(*cols):
        return [c for c in cols if c in df.columns]

    styler = df.style.map(highlight_action, subset=on("Verdict", "Action")) \
                     .map(highlight_smart_money, subset=on("Smart Money")) \
                     .map(color_pos_neg, subset=on("vs MA200 (%)")) \
                     .map(color_zscore, subset=on("Z-Score")) \
                     .map(color_rsi, subset=on("RSI (14)")) \
                     .map(color_peg, subset=on("PEG")) \
                     .map(color_debt, subset=on("Debt/EBITDA")) \
                     .map(color_debt, subset=on("Net Debt/EBITDA")) \
                     .map(lambda x: color_high_good(x, 15.0), subset=on("ROE (%)", "ROIC (%)")) \
                     .map(color_fcf_yield, subset=on("FCF Yield (%)")) \
                     .map(color_gap, subset=on("Growth Gap (pp)")) \
                     .map(color_er, subset=on("Next ER (days)"))
    return styler


_COL_CFG = {
        "Ticker":          st.column_config.TextColumn("Ticker", width="small"),
        "Company":         st.column_config.TextColumn("Company", width="medium"),
        "Sector":          st.column_config.TextColumn("Sector", width="small"),
        "Verdict":         st.column_config.TextColumn("Decision", width="medium",
                                                   help="The recommendation (same as the Decision Summary in Stock Analysis). "
                                                        "▲ / · / ▼ = timing context (trend, quality, value, volume flow) supportive / neutral / against. "
                                                        "The arrow never changes the recommendation."),
        "MoS (%)":         st.column_config.NumberColumn("MoS", format="%+d%%",
                                                     help="Margin of safety vs base-case DCF value (blank = DCF not informative)"),
        "Action":          st.column_config.TextColumn("Signal", width="small",
                                                 help="Technical + quality composite — an input to the Decision, not a recommendation"),
        "Quality":         st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%d",
                                                     help=f"How good is the business? Returns on capital, margins, stability, balance sheet, cash conversion (sector-relative). {SOLID}+ = sound, {ELITE}+ = excellent."),
        "Value":           st.column_config.ProgressColumn("Value", min_value=0, max_value=100, format="%d",
                                                     help="How cheap is the price? FCF yield, EV/EBITDA, earnings yield, PEG, shareholder yield — vs sector peers and absolute bands. Analyst ratings are not used."),
        "Momentum":        st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%d",
                                                     help="12-1 month return rank + trend. Timing only: never part of Quality or Value."),
        "Revisions":       st.column_config.ProgressColumn("Revisions", min_value=0, max_value=100, format="%d",
                                                     help="30-day change in analysts' EPS estimates and upgrade/downgrade balance. Context only — not part of Quality or Value."),
        "ADV (EUR M)":     st.column_config.NumberColumn("ADV €M", format="%.1f",
                                                   help="Median daily traded value, last 60 sessions. Under 1 is illiquid."),
        "Flags":           st.column_config.TextColumn("Red flags", width="medium",
                                                 help="Loss-making, debt without EBITDA, high net debt/EBITDA, uncovered dividend, negative equity"),
        "Smart Money":     st.column_config.TextColumn("Smart Money", width="small"),
        "RSI (14)":        st.column_config.NumberColumn("RSI", format="%d"),
        "Z-Score":         st.column_config.NumberColumn("Z-Score", format="%+.2f"),
        "vs MA200 (%)":    st.column_config.NumberColumn("vs MA200", format="%+.1f%%"),
        "P/E (Fwd)":       st.column_config.NumberColumn("Fwd P/E", format="%.1fx"),
        "EV/EBITDA":       st.column_config.NumberColumn("EV/EBITDA", format="%.1fx"),
        "PEG":             st.column_config.NumberColumn("PEG", format="%.2f"),
        "FCF Margin (%)":  st.column_config.NumberColumn("FCF Margin", format="%.1f%%"),
        "ROE (%)":         st.column_config.NumberColumn("ROE", format="%.1f%%"),
        "Yield (%)":       st.column_config.NumberColumn("Yield", format="%.2f%%"),
        "Net Payout (%)":  st.column_config.NumberColumn("Net Payout", format="%.2f%%"),
        "Debt/EBITDA":     st.column_config.NumberColumn("Debt/EBITDA", format="%.2fx",
                                                         help="Gross debt / EBITDA. The Quality score uses NET debt."),
        "ROIC (%)":        st.column_config.NumberColumn("ROIC", format="%.1f%%",
                                                         help="NOPAT / invested capital, median of up to 3 years "
                                                              "(blank for banks / insurers and when not measurable)."),
        "Net Debt/EBITDA": st.column_config.NumberColumn("Net Debt/EBITDA", format="%.2fx",
                                                         help="What the Quality score uses; negative = net cash."),
        "FCF Conversion (x)": st.column_config.NumberColumn("FCF Conv.", format="%.2fx",
                                                            help="Free cash flow after stock comp / net income. "
                                                                 "Below ~0.8x: earnings not backed by cash."),
        "FCF Yield (%)":   st.column_config.NumberColumn("FCF Yield", format="%.1f%%",
                                                         help="Free cash flow after stock comp / market cap — what the price "
                                                              "earns in cash today (blank for banks / insurers)."),
        "Growth Gap (pp)": st.column_config.NumberColumn("Growth vs priced-in", format="%+.0f pp",
                                                         help="Starting FCF growth the model anchors on the company's own "
                                                              "history (revenue / earnings / FCF CAGR) minus the starting FCF "
                                                              "growth the price IMPLIES (reverse DCF, same model). ≥ 0: the "
                                                              "price asks for no more than history; ≤ −20: it needs much "
                                                              "faster growth than the company has shown."),
        "Implied Growth (%)": st.column_config.NumberColumn("Priced-in growth", format="%+.0f%%",
                                                            help="Starting FCF growth (fading over 10 years) that makes the "
                                                                 "DCF equal to today's price."),
        "Exp. Return (%/yr)": st.column_config.NumberColumn("Exp. return", format="%+.1f%%",
                                                            help="Probability-weighted IRR of the bear / base / bull DCF over "
                                                                 "the holding horizon (only where the valuation is reliable)."),
        "Next ER (days)":  st.column_config.NumberColumn("Next ER", format="%d d",
                                                         help="Days to the next earnings report (Yahoo; may be an estimate). "
                                                              "≤ 7: no new position into the release."),
        "MCap (B)":        st.column_config.NumberColumn("MCap €B", format="%.1f"),
        "Insider 6M (%)":  st.column_config.NumberColumn("Insider 6M", format="%+.2f%%",
                                                         help="Net insider buying over 6 months, % of insider "
                                                              "holdings (US filers; blank = no data)."),
}


def _reset():
    for k in STATE_KEYS:
        st.session_state.pop(k, None)
        st.session_state.pop(kept_key(k), None)
    forget("scan_table_")


def _open_analysis(ticker):
    """Callback (runs before the next script run): Stock Analysis builds its selector from these two keys, and only when the
    selector's own key does not exist yet — so both are set, and the tab."""
    ss = st.session_state
    ss["active_ticker"] = ticker
    ss["deep_ticker_selector"] = ticker
    ss["active_tab"] = "🔬 Stock Analysis"


def _show_all():
    st.session_state.pop("scan_view", None)                       # the pills are rebuilt from the remembered choice
    st.session_state[kept_key("scan_view")] = "All"


def _summary(scope, changes, ctx):
    counts = sv.decision_counts(scope)
    er = sv.earnings_soon(scope)
    issue = int(sv.needs_data_mask(scope).sum()) if len(scope) else 0
    cols = st.columns(8)
    for col, (label, n) in zip(cols, [("BUY", counts[sv.STANCES[0]]), ("RESEARCH", counts[sv.STANCES[1]]),
                                       ("HOLD / WATCH", counts[sv.STANCES[2]]), ("AVOID / TRIM", counts[sv.STANCES[3]]),
                                       ("NOT ENOUGH DATA", counts[sv.STANCES[4]]), ("Needs data", int(issue)),
                                       (f"Earnings ≤{sv.ER_SOON_DAYS}d", er["soon"])]):
        col.metric(label, n, help="Stocks, in the region / sector / search scope below.")
    n_ch = len((changes or {}).get("changes") or [])
    cols[7].metric("Changed", n_ch if (changes or {}).get("previous") else "N/A",
                   help="Stance changes between the two newest recorded daily snapshots. N/A until two exist.")
    notes = []
    if (changes or {}).get("rules_changed") and n_ch:
        notes.append(f"the {n_ch} change(s) include a rules / score version change between the two snapshots "
                     f"({changes.get('rule_changes', 0)} are that alone)")
    if counts["none"]:
        notes.append(f"{counts['none']} without a valid stance")
    if er["unknown"]:
        notes.append(f"{er['unknown']} with no known report date (not 'no report')")
    if notes:
        st.caption("In scope: " + " · ".join(notes) + ".")


def _user():
    """The signed-in user's lists with the outcome of THEIR read (the SIGNAL hub's cached read — no second request)."""
    from ui.signal_hub import _user_data
    return _user_data()


def _changes(ctx):
    from core import signal_feed as feed
    from services.db import load_decision_snapshots
    try:
        return feed.decision_changes(load_decision_snapshots(ctx.get("data_version")))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def _selected_panel(shown, ctx):
    """Click a row to read its Decision in full and open it in Stock Analysis."""
    sel = st.session_state.get("_scan_selected")
    if not sel:
        st.caption("Select a row (the box at its left) to read the full Decision and open it in Stock Analysis.")
        return
    row = shown[shown["Ticker"] == sel]
    if row.empty:
        return
    r = row.iloc[0].to_dict()
    st.markdown(f"##### {sel} · {r.get('Company', '')} — **{r.get('Verdict', 'N/A')}**" +
                (f" · confidence {r['Confidence']}" if r.get("Confidence") else ""))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Why**")
        for x in (sv.as_list(r.get("_reasons")) or ([r["Why"]] if isinstance(r.get("Why"), str) and r["Why"] else [])):
            st.markdown(f"- {x}")
    with c2:
        st.markdown("**What could change it**")
        items = sv.as_list(r.get("_warnings")) + sv.as_list(r.get("_invalidation"))
        if r.get("Conflicts"):
            items = [f"Conflict: {x}" for x in str(r["Conflicts"]).split("; ") if x] + items
        for x in items[:5] or ["—"]:
            st.markdown(f"- {x}")
    with c3:
        st.markdown("**Data**")
        issue = r.get("Data issue")
        st.markdown(f"- {issue}" if issue else "- no data issue flagged")
        if isinstance(r.get("Peer group"), str) and r["Peer group"]:
            st.markdown(f"- Scores ranked against **{r['Peer group']}**")
        for x in sv.as_list(r.get("_confidence_notes"))[:3]:
            st.markdown(f"- {x}")
    ok = sel in set(ctx["current_universe"])
    st.button("Open in Stock Analysis", key="scan_open", disabled=not ok, on_click=_open_analysis, args=(sel,),
              help=None if ok else "This ticker is not in the loaded universe.")
    if not ok:
        st.caption("Not in the loaded universe, so it cannot be opened in Stock Analysis.")


def render(ctx):
    """Render the 🔭 Stock Scanner tab. ctx is the context dict built in app.py."""
    # The shell's screener table — computed once with the statement FCF and the stored risk-free rate, so the Decision
    # column here matches the Stock Analysis Decision Summary.
    m_df = ctx['m_df']
    render_header("search", "Stock Scanner — your research queue", level="###")
    if m_df.empty:
        st.info("The screener table is empty — there is no price / score data in this snapshot yet.")
        return
    latest = ctx["prices_full"]["date"].max() if "prices_full" in ctx and not ctx["prices_full"].empty else None
    changes = _changes(ctx)

    # ── scope: region · sector · search ────────────────────────────────────────────────────────────────────────────
    s1, s2, s3 = st.columns([1, 1, 1.4])
    with s1:
        selected_region = sticky_choice("Region", ["🌎 All Regions"] + sorted(m_df["Region"].dropna().unique().tolist()),
                                        "🌎 All Regions", "p_region_filter", format_func=lambda value: value.removeprefix("🌎 "))
    with s2:
        selected_sector = sticky_choice("Sector", ["🌍 All Sectors"] + sorted(m_df["Sector"].dropna().unique().tolist()),
                                        "🌍 All Sectors", "p_sector_filter", format_func=lambda value: value.removeprefix("🌍 "))
    with s3:
        query = sticky(st.text_input, "Search ticker or company", "", "scan_search", placeholder="e.g. AAPL or Apple")
    scope = m_df
    if selected_region != "🌎 All Regions":
        scope = scope[scope["Region"] == selected_region]
    if selected_sector != "🌍 All Sectors":
        scope = scope[scope["Sector"] == selected_sector]
    scope = sv.search(scope, query)
    st.caption(f"Prices to {latest:%d %b %Y}" if latest is not None else "Price date unknown")
    _summary(scope, changes, ctx)

    # ── view ────────────────────────────────────────────────────────────────────────────────────────────────────────
    view = sticky_pills("View", sv.VIEWS, "Candidates", "scan_view", label_visibility="collapsed")
    user = _user() if view in ("Held", "Watchlist") else None
    include_archived = False
    if view == "Watchlist":
        include_archived = sticky(st.checkbox, "Include closed / invalidated ideas", False, "scan_archived")
    if view in ("Held", "Watchlist") and user is not None:
        if not st.session_state.get("user_id"):
            st.info("Sign in to see your holdings and watchlist.")
            return
        if not (user["holdings_ok"] if view == "Held" else user["watchlist_ok"]):
            st.warning(f"Your {'portfolio' if view == 'Held' else 'watchlist'} could not be read, so this view cannot be "
                       "shown (that is not an empty list).")
            return
    _members = view in sv.MEMBERSHIP_VIEWS
    base = sv.select_view(m_df if _members else scope, view, holdings=(user or {}).get("holdings"),
                          watchlist=(user or {}).get("watchlist"), changes=changes, include_archived=include_archived)
    if _members:       # a name that has a screener row but is outside the scope is out of the view, not "missing"
        base = sv.within_scope(base, scope, m_df,
                               narrowed=(selected_region != "🌎 All Regions" or selected_sector != "🌍 All Sectors"),
                               query=query)
    if view == "Changed" and (changes or {}).get("error"):
        st.warning("Decision snapshots could not be read. Try again using Retry decision snapshots.")
        if st.button("Retry decision snapshots", key="scan_retry_changes"):
            from services.db import load_decision_snapshots
            load_decision_snapshots.clear()
            st.rerun()
        return
    if view == "Changed" and not (changes or {}).get("previous"):
        st.info("Decision changes need two daily snapshots (written by the ETL) — none to compare yet. That is not '0 changes'.")
        return
    f_df = sv.with_data_issue(base) if "_valuation" in base.columns else base

    # ── filters (a preset is a screen of observations, not a Decision) ──────────────────────────────────────────────
    can_filter = {"Quality", "RSI (14)", "Smart Money"} <= set(f_df.columns)
    scan_presets = presets.options()
    preset_labels = {presets.ALL: "All Stock Universe",
                     **{p.label: p.label.split(" ", 1)[1] for p in presets.PRESETS},
                     presets.SEPARATORS["opportunity"]: "OPPORTUNITY",
                     presets.SEPARATORS["risk"]: "RISK / WARNING"}
    # A preset removed in an update must not break a session that still has it selected.
    # (the widget key is absent on a run after a tab in which the Scanner was not drawn: judge the remembered value then)
    _cur_mode = st.session_state.get("scan_mode", st.session_state.get(kept_key("scan_mode")))
    if _cur_mode is not None and _cur_mode not in scan_presets:
        st.session_state.pop("scan_mode", None)
        st.session_state.pop(kept_key("scan_mode"), None)
    p1, p2 = st.columns([1.6, 1])
    with p1:
        scan_mode = sticky_choice("Intelligence Strategy Preset", scan_presets, presets.ALL, "scan_mode",
                                  format_func=preset_labels.__getitem__)
    with p2:
        sort_label = sticky_choice("Sort", list(SORTS), "Decision, then Quality, Value", "scan_sort")
    if view == "Changed":
        st.caption("Preset, refinements and sort do not apply to this view: it lists recorded stance changes in a fixed order.")
    if can_filter:
        preset = presets.get(scan_mode)
        if preset is not None:
            f_df = presets.apply(f_df, scan_mode)
            getattr(st, preset.level)(f"{preset_labels[preset.label].split(' (')[0]}: {preset.note}")
        elif scan_mode in presets.SEPARATORS.values():
            st.warning("Please select a valid screening preset.")

    with st.expander("Custom Refinement Sliders"):
        if not can_filter:
            st.caption("These filters do not apply to this view.")
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            min_score = sticky(st.slider, "Min Quality", 0, "scan_min_q", min_value=0, max_value=100)
            min_value = sticky(st.slider, "Min Value", 0, "scan_min_v", min_value=0, max_value=100)
            min_mom = sticky(st.slider, "Min Momentum (timing)", 0, "scan_min_m", min_value=0, max_value=100)
        with rcol2:
            rsi_range = sticky(st.slider, "RSI Range", (0, 100), "scan_rsi", min_value=0, max_value=100)
            max_pe = sticky(st.slider, "Max Forward P/E", 200, "scan_pe", min_value=0, max_value=200)
            z_score_range = sticky(st.slider, "Z-Score Range", (-5.0, 5.0), "scan_z", min_value=-5.0, max_value=5.0, step=0.1)
        with rcol3:
            peg_range = sticky(st.slider, "PEG Range", (-5.0, 10.0), "scan_peg", min_value=-5.0, max_value=10.0, step=0.1)
            min_fcf_yield = sticky(st.slider, "Min FCF yield (%)", -10.0, "scan_fcf", min_value=-10.0, max_value=15.0, step=0.5,
                                   help="Free cash flow after stock comp / market cap. Leave at -10 to keep everything.")
            skip_er = sticky(st.checkbox, "Hide stocks reporting within 7 days", False, "scan_skip_er",
                             help="No new position into a release: the Decision blocks BUYs in that window anyway.")
            filter_sm = sticky_choice("Smart Money", ["🌎 All", "🟢 Accumulation only", "🔴 Distribution only"],
                                      "🌎 All", "custom_sm_filter", format_func=lambda value: value.split(" ", 1)[1])
    # A refinement applies only once moved off its default — otherwise stocks with an unknown value
    # (no forward P/E, no PEG) were silently removed from every preset, including "All".
    n_before = len(f_df)
    if can_filter:
        if min_score > 0:
            f_df = f_df[f_df["Quality"] >= min_score]
        if min_value > 0:
            f_df = f_df[f_df["Value"] >= min_value]
        if min_mom > 0:
            f_df = f_df[f_df["Momentum"] >= min_mom]
        if tuple(rsi_range) != (0, 100):
            f_df = f_df[f_df["RSI (14)"].between(*rsi_range)]
        if max_pe < 200:
            f_df = f_df[f_df["P/E (Fwd)"].between(0, max_pe, inclusive="right")]   # loss-makers excluded
        if tuple(z_score_range) != (-5.0, 5.0):
            f_df = f_df[f_df["Z-Score"].between(*z_score_range)]
        if tuple(peg_range) != (-5.0, 10.0):
            f_df = f_df[f_df["PEG"].between(*peg_range)]
        if min_fcf_yield > -10.0:
            f_df = f_df[f_df["FCF Yield (%)"] >= min_fcf_yield]
        if skip_er:
            f_df = f_df[~(f_df["Next ER (days)"] <= 7)]
        if "Accumulation only" in filter_sm:
            f_df = f_df[f_df["Smart Money"].str.contains("ACCUMULATION", na=False)]
        elif "Distribution only" in filter_sm:
            f_df = f_df[f_df["Smart Money"].str.contains("DISTRIBUTION", na=False)]

    # ── result ──────────────────────────────────────────────────────────────────────────────────────────────────────
    extras = sticky_multi("More columns", EXTRA_COLS, [], "scan_extra_cols",
                          help="Detail behind the scores. Hidden by default: Quality, Value and Momentum already summarise them.")
    display_cols = sv.columns_for(view, DEFAULT_COLS, extras, f_df.columns)
    if "Conflicts" in f_df.columns:
        f_df = f_df.assign(Conflicts=f_df["Conflicts"].map(_clean_conflicts))
    spec = SORTS[sort_label]
    if view == "Changed":
        ordered = f_df                                              # data / mixed causes first, as the view builds it
    elif spec is None:
        ordered = sv.sort_default(f_df)
    else:
        ordered = sv.sort_by(f_df, spec[0], spec[1])
    st.markdown(f"**{len(ordered)} of {len(base)}** in “{view}”"
                + (f" ({len(base)} before the refinements)" if n_before != len(ordered) and can_filter else "")
                + (" — changes with a data or mixed cause first, methodology-only changes last." if view == "Changed"
                   else f" — sorted by {sort_label.lower()}."))
    if view == "Candidates" and ordered["Decision"].eq("BUY CANDIDATE").any():
        from core.track_record import evidence_headline
        from services.db import evidence
        try:
            status = evidence_headline(evidence(ctx["prices_full"]))
        except Exception:
            status = "unavailable"
        st.caption("BUY CANDIDATE means worth researching, not an order. "
                   f"Track Record evidence that these signals earn excess returns: {status}.")
    if ordered.empty:
        if view == "Candidates":
            st.info("No stock is a candidate under the current rules in this scope — the rules were not loosened.")
            st.button("Show all stocks", key="scan_show_all", on_click=_show_all)
        else:
            st.info("Nothing in this view with the current scope and filters.")
    st.caption("▲ timing context supportive · neutral ▼ against (trend, quality, value, volume flow). "
               "The label is the recommendation; the arrow is context only.")

    if 'radar_limit' not in st.session_state:
        st.session_state.radar_limit = 50
    display_df = ordered[display_cols].reset_index(drop=True)
    if "Plan status" in display_df:
        display_df["Plan status"] = display_df["Plan status"].map(status_label)
    paged_df = display_df.iloc[:st.session_state.radar_limit]
    paged_full = ordered.reset_index(drop=True).iloc[:st.session_state.radar_limit]
    if not paged_df.empty:
        # a new membership / order / page is a new table: a selection must not point at another row
        sig = hashlib.sha1(("|".join(paged_df["Ticker"].astype(str)) + view).encode()).hexdigest()[:10]
        event = st.dataframe(style_opportunity_df(paged_df), use_container_width=True, height=550, hide_index=True,
                             column_config={c: v for c, v in _COL_CFG.items() if c in display_cols},
                             on_select="rerun", selection_mode="single-row", key=f"scan_table_{sig}")
        rows = event.selection.rows if event is not None else []
        st.session_state["_scan_selected"] = (paged_df.iloc[rows[0]]["Ticker"] if rows and rows[0] < len(paged_df) else None)
        _selected_panel(paged_full, ctx)

    if len(display_df) > st.session_state.radar_limit:
        if st.button(f"Load All (Showing {st.session_state.radar_limit} of {len(display_df)})", width="stretch"):
            st.session_state.radar_limit = len(display_df)
            st.rerun()
    elif len(display_df) > 50:
        if st.button("Reset to Top 50", width="stretch"):
            st.session_state.radar_limit = 50
            st.rerun()

    b1, b2, _ = st.columns([1.3, 1, 3])
    _applied = can_filter
    _refine = []
    if _applied:
        for label, on in (("min_quality", min_score > 0), ("min_value", min_value > 0), ("min_momentum", min_mom > 0),
                          ("rsi", tuple(rsi_range) != (0, 100)), ("max_fwd_pe", max_pe < 200),
                          ("z_score", tuple(z_score_range) != (-5.0, 5.0)), ("peg", tuple(peg_range) != (-5.0, 10.0)),
                          ("min_fcf_yield", min_fcf_yield > -10.0), ("hide_earnings_7d", bool(skip_er)),
                          ("smart_money", "All" not in filter_sm)):
            if on:
                _refine.append(label)
    meta = {"view": view, "scope": f"{selected_region} / {selected_sector} / search '{query}'",
            "preset": scan_mode if _applied else "not applied to this view",
            "refinements": ", ".join(_refine) if _applied else "not applied to this view",
            "sort": sort_label if view != "Changed" else "fixed order (not applied)",
            "include_archived": include_archived, "data_version": ctx.get("data_version") or "unknown",
            "rows": len(ordered), "prices_to": f"{latest:%Y-%m-%d}" if latest is not None else "unknown",
            "note": "Decision = the app's one recommendation; scores are rules, not forecasts"}
    b1.download_button("Export filtered results (CSV)", data=sv.export_csv(ordered, display_cols, meta=meta),
                       file_name=f"scanner_{view.lower().replace(' ', '_')}.csv", mime="text/csv", key="scan_export",
                       disabled=ordered.empty, help="All filtered rows (not only the page shown), the columns shown.")
    b2.button("Reset view", key="scan_reset", on_click=_reset, help="Back to the default view and filters.")

    # ── Score methodology (core/scoring.py, thresholds in config/scoring_rules.yaml) ─────────
    def _card(colour, title, body, extra):
        return (f"<div style='background:var(--secondary-background-color, transparent); border-left:3px solid {colour}; padding:10px 12px; border-radius:6px;'>"
                f"<div style='font-size:0.78rem; color:{colour}; font-weight:800;'>{title}</div>"
                f"<div style='font-size:0.72rem; color:inherit; margin-top:4px;'>{body}</div>"
                f"<div style='font-size:0.68rem; color:inherit; margin-top:6px;'>{extra}</div></div>")
    with st.expander("How the scores work"):
        st.markdown(
            "<div style='margin-top:16px; padding:14px 18px; background:var(--secondary-background-color, transparent); border:1px solid rgba(128,128,128,0.25); border-radius:10px;'>"
            "<div style='font-size:0.78rem; font-weight:700; color:inherit; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>"
            f"Scores {SCORE_VERSION} — independent 0-100 numbers per stock, ranked against sector peers</div>"
            "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:10px;'>"
            + _card("#2ecc71", "QUALITY — the business", "Return on capital 20 · margins 25 · growth &amp; stability 20 · balance sheet (net debt/EBITDA) 20 · FCF conversion 15. Banks/insurers: ROE, net margin, stability.",
                    f"Red flags subtract points: loss-making, debt without EBITDA, net debt/EBITDA &gt; 5x, dividend not covered by FCF, negative equity. Tiers: ELITE ≥ {ELITE} · SOLID ≥ {SOLID} · FAIR ≥ {FAIR}.")
            + _card("#3498db", "VALUE — the price", "FCF yield 25 · EV/EBITDA 20 · earnings yield 20 · PEG 15 · shareholder yield 10 · P/S 10, each half peer percentile, half absolute band.",
                    "No analyst forecasts at all (not even forward P/E): trailing and 3-year-median earnings, PEG on realised growth, FCF after stock compensation. Shareholder yield is halved when the dividend is not covered.")
            + _card("#f1c40f", "MOMENTUM — timing only", "12-1 month return rank across the universe (70%) + price above MA200 / golden cross (30%). RSI and Z-score are shown but not scored.",
                    "Measured in the stock's own currency. Never enters Quality or Value. A separate Revisions score tracks changes in analysts' estimates.")
            + "</div>"
            "<div style='margin-top:8px; font-size:0.68rem; color:inherit;'>Unknown inputs are excluded and the rest re-weighted — never scored as zero. "
            "With less than 80% of a score's inputs observable it is pulled toward 50 (see Coverage in the Decision confidence). "
            "The Decision uses Quality as a floor (a BUY needs ≥ 50) and Value as a cross-check on the DCF. "
            "Whether any score predicts returns is tested on the Track Record tab.</div></div>", unsafe_allow_html=True)

    with st.expander("How to read the table"):
        st.write("""
        - **Decision** is the recommendation; **MoS** (margin of safety vs the DCF base value) is its evidence. Blank = DCF not informative.
        - The arrow after the Decision is the **timing context** (the Signal): ▲ supportive, · neutral, ▼ against — trend, quality, value, reward/risk, volume flow. Add the "Action" column for the full Signal label.
        - **Quality / Value / Momentum** are independent 0-100 ranks vs sector peers; **Revisions** is the 30-day change in analysts' EPS estimates.
        - A cheap stock with Quality below 45 is a value-trap candidate; a high Quality stock with RSI < 30 is a pullback to research, not a signal.
        """)

    # External: TradingView's own screener — fetched only on request, not scored here, no Decision
    from views import scanner_discovery
    scanner_discovery.render()
