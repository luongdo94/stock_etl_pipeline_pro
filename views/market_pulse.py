"""View: 🌐 Market Pulse — the market context for your research.

One regime result (core.market_regime, the same one the header and SIGNAL show) is stated first with its reasons and the
inputs that were unknown; the SPY trend and the breadth follow the sidebar range for display only. The external
TradingView widgets load only when switched on. The optional Codex macro brief offers portfolio context on request;
it never changes a stock's Decision.
"""
import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import market_regime as mr
from services import auth
from ui.components import phone_safe_chart
from ui.icons import render_header
from ui.range_control import chart_range
from ui.sticky import sticky, sticky_segmented

_HEATMAP = """
        <!-- TradingView Market Heatmap Widget -->
        <div class="tradingview-widget-container" style="height:500px; width:100%">
          <div class="tradingview-widget-container__widget" style="height:500px; width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
          {
            "exchanges": [],
            "dataSource": "SPX500",
            "grouping": "sector",
            "blockSize": "market_cap_basic",
            "blockColor": "change",
            "locale": "en",
            "symbolUrl": "",
            "colorTheme": "dark",
            "hasTopBar": true,
            "isDataSetEnabled": true,
            "isZoomEnabled": true,
            "hasSymbolTooltip": true,
            "isMonoSize": false,
            "width": "100%",
            "height": "500"
          }
          </script>
        </div>
        """
_CALENDAR = """
        <!-- TradingView Economic Calendar Widget -->
        <div class="tradingview-widget-container" style="height:500px; width:100%">
          <div class="tradingview-widget-container__widget" style="height:500px; width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
          {
            "colorTheme": "dark",
            "isTransparent": true,
            "width": "100%",
            "height": "500",
            "locale": "en",
            "importanceFilter": "0,1",
            "countryFilter": "us,eu,gb,jp,cn,de,fr"
          }
          </script>
        </div>
        """
_CROSS = """
        <!-- TradingView Market Overview (Full) Widget -->
        <div class="tradingview-widget-container" style="height:400px; width:100%">
          <div class="tradingview-widget-container__widget" style="height:400px; width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
          {
            "colorTheme": "dark",
            "dateRange": "YTD",
            "showChart": true,
            "locale": "en",
            "isTransparent": true,
            "width": "100%",
            "height": "400",
            "tabs": [
              {
                "title": "US Indices",
                "symbols": [
                  {"s": "FOREXCOM:SPXUSD", "d": "S&P 500"},
                  {"s": "FOREXCOM:NSXUSD", "d": "Nasdaq 100"},
                  {"s": "CBOE:VIX",        "d": "VIX"},
                  {"s": "TVC:DJI",         "d": "Dow Jones"}
                ],
                "originalTitle": "US Indices"
              },
              {
                "title": "Europe",
                "symbols": [
                  {"s": "XETR:DAX",          "d": "DAX"},
                  {"s": "EURONEXT:CAC40",    "d": "CAC 40"},
                  {"s": "INDEX:SX5E",        "d": "STOXX 50"},
                  {"s": "TVC:FTSE100",       "d": "FTSE 100"}
                ],
                "originalTitle": "Europe"
              },
              {
                "title": "Macro",
                "symbols": [
                  {"s": "TVC:GOLD",   "d": "Gold"},
                  {"s": "TVC:USOIL", "d": "Oil WTI"},
                  {"s": "TVC:DXY",   "d": "Dollar Index"},
                  {"s": "FX:EURUSD", "d": "EUR/USD"}
                ],
                "originalTitle": "Macro"
              }
            ]
          }
          </script>
        </div>
        """

_NEXT_STEPS = (("Look for ideas", "🔭 Stock Scanner", "Scanner — Decisions of the whole universe"),
               ("Check what you hold", "💼 Portfolio", "Portfolio — your holdings and their Decisions"),
               ("One stock's range — Risks & data → Price range", "🔬 Stock Analysis",
                "Stock Analysis — Risks & data → Price range"))


def _go(tab):
    st.session_state["active_tab"] = tab
    if tab == "🔬 Stock Analysis":
        st.session_state["sa_group"] = "Risks & data"
        st.session_state["_sa_group"] = "Risks & data"


def _answer(ctx):
    """The first thing on the page: the regime, why, what is unknown, and where to look next."""
    summ = mr.regime_summary(
        {"regime": ctx["regime"], "scoring_regime": ctx["_macro_regime"]}, ctx["conf_score_global"],
        [r for r in ctx["conf_reason_str"].split(", ") if r and not r.startswith("No warning")],
        breadth=ctx["latest_breadth_global"], vix=ctx["_vix_val"], dxy_pct=ctx["_dxy_pct"],
        tnx_chg=(ctx.get("macro") or {}).get("US10Y", {}).get("chg"))
    colour = ctx["regime_ui_color"]
    latest = ctx["df_spy_global"]["date"].max() if not ctx["df_spy_global"].empty else None
    why = "".join(f"<li>{r}</li>" for r in summ["reasons"]) or "<li>No warning among the inputs.</li>"
    gaps = (f"<li><b>Incomplete inputs:</b> {', '.join(summ['incomplete'])} — scored at the middle of their range, "
            "so the regime is less certain than it looks.</li>" if summ["incomplete"] else "")
    flag = ("<li>Rates &amp; USD pressure flag (10Y yield and the dollar rising together; a heuristic, not a forecast).</li>"
            if summ["rates_usd_flag"] else "")
    st.markdown(f"""
    <div style='border:1px solid {colour}55; border-left:5px solid {colour}; border-radius:10px; padding:12px 18px;
                margin:6px 0 10px 0; background:var(--secondary-background-color, transparent);'>
      <div style='display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;'>
        <span style='color:{colour}; font-size:1.5rem; font-weight:900;'>{summ['label']}</span>
        <span style='color:inherit; font-size:0.78rem;'>raw heuristic score {summ['score']}/{summ['max_score']}
          · {"prices to " + f"{latest:%d %b %Y}" if latest is not None else "price date unknown"}</span>
      </div>
      <div style='font-size:0.82rem; margin-top:4px;'>{ctx['regime_advice']}</div>
      <ul style='margin:6px 0 0 16px; padding:0; font-size:0.82rem;'>{why}{gaps}{flag}</ul>
    </div>""", unsafe_allow_html=True)
    cols = st.columns(len(_NEXT_STEPS))
    for col, (label, tab, tip) in zip(cols, _NEXT_STEPS):
        col.button(label, key=f"mp_go_{tab}", help=tip, width="stretch", on_click=_go, args=(tab,))
    _your_review()
    st.caption("Context for your research, not an instruction: the regime does not change any stock's Decision. The charts "
               "below follow the chart range; the regime above always uses the latest data.")


def _your_review():
    """The first lines of what SIGNAL has for you (same items, same badge) — or that it has not been loaded for you."""
    from services.db import _expected
    prev = st.session_state.get("_sig_preview")
    if not st.session_state.get("user_id"):
        st.caption("Sign in to see what needs your review (holdings, watchlist, your rules).")
        return
    if not prev or prev.get("uid") != st.session_state.get("user_id") or prev.get("version") != _expected():
        st.caption("Your review is not loaded yet — open SIGNAL (top right).")
        return
    st.caption(f"Review preview evaluated at {prev.get('evaluated_at', 'unknown time')}. "
               "Open SIGNAL for the latest review.")
    if not prev["readable"]:
        st.warning("Your portfolio or watchlist could not be read, so SIGNAL cannot say what needs your review.")
        return
    if not prev["top"]:
        st.success("Nothing in your holdings or watchlist needs your review at the preview time (SIGNAL).")
        return
    tag = {"action": "ACT", "review": "REVIEW", "info": "INFO"}
    lines = "".join(f"<li><b>{tag.get(i['severity'], i['severity'].upper())}</b> · "
                    f"{'' if i['ticker'] == '—' or str(i['message']).startswith(str(i['ticker'])) else html.escape(str(i['ticker'])) + ' · '}"
                    f"{html.escape(str(i['message']))}</li>" for i in prev["top"])
    more = f" — {prev['total'] - len(prev['top'])} more in SIGNAL" if prev["total"] > len(prev["top"]) else ""
    st.markdown(f"<div style='font-size:0.82rem; margin:2px 0 6px 0;'><b>For you:</b> {prev['n']} item(s) need review"
                f"{more}<ul style='margin:2px 0 0 16px; padding:0;'>{lines}</ul></div>", unsafe_allow_html=True)


def _economic_data(ctx):
    """Stored observation periods and their collection dates; no inferred release dates."""
    from core.macro_view import REGION_NAMES, period_label, table_for_display
    from services.db import load_macro_indicators
    st.markdown("#### Economic data — latest observation vs the previous one")
    stored = load_macro_indicators(ctx.get("data_version"))
    if stored.attrs.get("unavailable") == "read_error":
        st.warning("Economic data could not be read. Reload to retry; this does not mean no observations were stored.")
        return
    t = table_for_display(stored)
    if t.empty:
        st.info("No economic observations are stored in this snapshot (the table may be absent or empty).")
        return
    show = pd.DataFrame({
        "Region": t["region"].map(REGION_NAMES).fillna(t["region"]), "Indicator": t["name"],
        "Observation period": [period_label(p, f) for p, f in zip(t["period"], t["frequency"])],
        "Latest": t["value"], "Previous": t["previous_label"], "Change": t["change"], "Unit": t["unit"],
        "Status": t["status_label"], "Source": t["source"].str.upper(), "Collection": t["collected_label"]})
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "Latest": st.column_config.NumberColumn(format="%.2f"),
        "Previous": st.column_config.TextColumn(help="Previous stored value and its observation period, when known."),
        "Change": st.column_config.NumberColumn(format="%+.2f")})
    st.caption("Observation period is the period measured; Collected is when this app retrieved each row, not its release date. "
               "Release dates and consensus forecasts are not stored. STALE means the observation period is older than "
               "the series' age limit, not that its source stopped updating. Context only; it never feeds a Decision. "
               "Sources: FRED (US, Japan), Eurostat/FRED (euro area), NBS English releases (China).")


def _macro_facts(ctx):
    import json
    from core.macro_view import table_for_display
    from services.db import load_macro_indicators
    stored = load_macro_indicators(ctx.get("data_version"))
    if stored.attrs.get("unavailable") == "read_error":
        raise RuntimeError("Economic data could not be read; this is not an empty dataset.")
    macro = ctx.get("macro") or {}
    prices = ctx["prices_full"]
    facts = dict(regime=mr.regime_summary(
        {"regime": ctx["regime"], "scoring_regime": ctx["_macro_regime"]}, ctx["conf_score_global"],
        [r for r in ctx["conf_reason_str"].split(", ") if r and not r.startswith("No warning")],
        breadth=ctx["latest_breadth_global"], vix=ctx["_vix_val"], dxy_pct=ctx["_dxy_pct"],
        tnx_chg=macro.get("US10Y", {}).get("chg")), regime_advice=ctx.get("regime_advice"),
        VIX=ctx["_vix_val"], DXY_5d_pct_or_daily_fallback=mr.dxy_5d_move(prices, ctx["_dxy_pct"]),
        DXY_daily_pct=ctx["_dxy_pct"], US10Y=macro.get("US10Y"), macro_sources_dates=macro,
        breadth_pct=ctx["latest_breadth_global"], data_version=ctx.get("data_version"))
    facts["DXY_stored_dates"] = prices.loc[prices["ticker"] == "DX-Y.NYB", "date"].sort_values().tail(6).astype(str).tolist()
    facts["DXY_window"] = "Up to 5 stored sessions in native units; daily fallback if unavailable."
    for name, frame in (("economic_data", table_for_display(stored)),
                        ("sector_rotation", ctx.get("tv_sector_rotation", pd.DataFrame()))):
        facts[name] = json.loads(frame.to_json(orient="records", date_format="iso"))
    for name in ("df_spy_global", "breadth_ts_global"):
        frame = ctx.get(name, pd.DataFrame())
        facts[name + "_as_of"] = str(frame["date"].max()) if not frame.empty else None
    return facts


def _macro_brief(ctx):
    import json
    from datetime import datetime
    from time import perf_counter
    from core.macro_brief import SCHEMA, build_prompt, validate
    from services import codex_local
    from ui.codex_options import render_codex_options, codex_label
    from ui.sticky import sticky_radio
    if not codex_local.codex_path():
        st.caption("Codex runs on the local app only")
        return
    with st.expander("Macro brief (Codex)"):
        model, effort = render_codex_options("macro")
        language = sticky_radio("Language", ["English", "Vietnamese"], "English", "mp_macro_language")
        signed_in = bool(st.session_state.get("user_id"))
        spec = (ctx.get("data_version"), model or codex_local.DEFAULT_MODEL, effort or codex_local.DEFAULT_EFFORT,
                language, signed_in, st.session_state.get("user_id"))
        if auth.can_use("write the macro brief (it runs Codex on this machine)", "mp_macro") and                 st.button("Write macro brief — Codex with web search, ~2–4 min", key="mp_macro_write"):
            started = perf_counter()
            portfolio = None
            if signed_in:                  # read on the click only: the landing tab must not value the book every rerun
                try:
                    from ui.assistant import _toolbox
                    portfolio = _toolbox(ctx).run("my_portfolio", {})[0]
                    if "error" in portfolio:
                        raise RuntimeError(portfolio["error"])
                except Exception as exc:
                    st.error(f"Portfolio could not be read: {exc}")
                    return
            try:
                with st.spinner("Codex is writing the macro brief with web search..."):
                    prompt = build_prompt(_macro_facts(ctx), portfolio, datetime.now().date(), language)
                    brief = validate(json.loads(codex_local.run_codex(
                        prompt, timeout=600, model=model, effort=effort, web=True, schema=SCHEMA)))
                    if brief is None or portfolio is None and brief["portfolio_notes"]:
                        raise ValueError("Invalid macro brief JSON: required fields, types or allowed values do not match.")
                st.session_state["mp_macro_brief"] = dict(spec=spec, brief=brief, model=codex_label(model, effort),
                    elapsed=perf_counter() - started, written_at=datetime.now().isoformat(timespec="seconds"))
            except Exception as exc:
                st.error(f"Macro brief failed (Codex / JSON / app data): {exc}")
                return
        saved = st.session_state.get("mp_macro_brief")
        if not saved:
            return
        if saved["spec"] != spec:
            st.warning("Stale — inputs or options changed. Write the macro brief again.")
            return
        brief = saved["brief"]
        st.markdown(brief["headline"])
        st.markdown("**Regime read** — " + brief["regime_read"])
        for region in brief["regions"]:
            st.markdown(f"**{region['region']}** — {region['summary']}")
            for point in region["data_points"]:
                st.markdown("- " + point)
        st.markdown("**Risks**")
        for point in brief["risks"]:
            st.markdown("- " + point)
        st.markdown("**Watch next**")
        st.dataframe(pd.DataFrame(brief["watch_next"]), hide_index=True, width="stretch")
        posture = brief["posture"]
        st.info(f"Portfolio posture: **{posture['risk_level']}**")
        st.markdown(posture["cash"] + "\n\n" + posture["rationale"])
        for tilt in posture["tilts"]:
            st.markdown("- " + tilt)
        if brief["portfolio_notes"]:
            st.markdown("**Portfolio notes**")
            for note in brief["portfolio_notes"]:
                st.markdown("- " + note)
        st.markdown("**Web facts — sources and dates**")
        st.dataframe(pd.DataFrame(brief["web_facts"]), hide_index=True, width="stretch")
        st.caption(f"Written by {saved['model']} · {saved['written_at']} · {saved['elapsed']:.1f}s · "
                   f"facts_as_of: {brief['facts_as_of']} · confidence: {brief['confidence']}")
        st.caption("Context only, not advice; it never changes a stock's Decision.")


def render(ctx):
    """Render the 🌐 Market Pulse tab. ctx is the context dict built in app.py."""
    breadth_ts_global = ctx['breadth_ts_global']
    df_spy_global = ctx['df_spy_global']
    t_end = ctx['t_end']
    t_start = ctx['t_start']
    tv_sector_rotation = ctx['tv_sector_rotation']
    render_header("pulse", "Market Pulse — the market context for your research", level="###")
    _answer(ctx)
    _macro_brief(ctx)
    st.markdown("---")

    # ROW 2: PRIMARY CHARTS
    range_res = chart_range(ctx)
    t_start, t_end = range_res.t_start, range_res.t_end
    # Filter SPY and breadth to the selected time horizon
    df_spy_filtered = df_spy_global[(df_spy_global["date"] >= t_start) & (df_spy_global["date"] <= t_end)]
    breadth_filtered = breadth_ts_global[(breadth_ts_global["date"] >= t_start) & (breadth_ts_global["date"] <= t_end)]

    c1, c2 = st.columns(2)
    with c1:
        render_header("activity", "Index Trend (SPY + MA 50/200)")
        if df_spy_filtered["price_close"].notna().sum() >= 2:
            fig_spy = go.Figure()
            fig_spy.add_trace(go.Scatter(
                x=df_spy_filtered["date"], y=df_spy_filtered["price_close"],
                name="SPY Price", mode='lines',
                line=dict(color="#0284C7", width=2.5)
            ))
            fig_spy.add_trace(go.Scatter(
                x=df_spy_filtered["date"], y=df_spy_filtered["ma_50"],
                name="MA 50", mode='lines',
                line=dict(color="#3498db", width=1.5, dash='dot')
            ))
            fig_spy.add_trace(go.Scatter(
                x=df_spy_filtered["date"], y=df_spy_filtered["ma_200"],
                name="MA 200", mode='lines',
                line=dict(color="#e67e22", width=1.5)
            ))
            fig_spy.update_layout(
                height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="SPY in € (includes EUR/USD moves)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(phone_safe_chart(fig_spy), use_container_width=True)
        else:
            _closes = df_spy_filtered.dropna(subset=["price_close"])
            if not _closes.empty:
                _last = _closes.iloc[-1]
                st.caption(f"SPY: {pd.Timestamp(_last['date']):%Y-%m-%d} · €{_last['price_close']:,.2f}")
            st.info("SPY: need at least 2 closes for the selected period.")

    with c2:
        render_header("dna", "Market Breadth (% Stocks > MA 50)")
        if not breadth_filtered.empty:
            fig_br = go.Figure()
            fig_br.add_trace(go.Scatter(
                x=breadth_filtered["date"], y=breadth_filtered["breadth_pct"],
                name="Breadth %", mode='lines',
                line=dict(color="#2ecc71", width=2),
                fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.08)'
            ))
            # Reference levels
            fig_br.add_hline(y=70, line_dash="dot", line_color="rgba(46, 204, 113, 0.5)",
                             annotation_text="70% Bullish Zone", annotation_position="top left",
                             annotation_font=dict(size=10, color="#2ecc71"))
            fig_br.add_hline(y=50, line_dash="dash", line_color="rgba(128,128,128,0.6)",
                             annotation_text="50% Neutral", annotation_position="top left",
                             annotation_font=dict(size=10))
            fig_br.add_hline(y=30, line_dash="dot", line_color="rgba(231, 76, 60, 0.5)",
                             annotation_text="30% Oversold", annotation_position="top left",
                             annotation_font=dict(size=10, color="#e74c3c"))
            fig_br.update_layout(
                height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(title="% Above MA50", range=[0, 100], ticksuffix="%"),
                xaxis_title="", showlegend=False
            )
            st.plotly_chart(phone_safe_chart(fig_br), use_container_width=True)
        else:
            st.info("No breadth history in this snapshot.")





    st.markdown("---")

    st.markdown("#### ETF Sector Rotation")
    _asof = ""
    if "_extracted_at" in tv_sector_rotation.columns and tv_sector_rotation["_extracted_at"].notna().any():
        _asof = f" Stored {tv_sector_rotation['_extracted_at'].max():%d %b %Y %H:%M}."
    st.caption("Price change of the 11 US SPDR sector ETFs (Yahoo, USD, dividend-adjusted closes; 1 week = 5 sessions, "
               "1 / 3 / 6 months = 21 / 63 / 126, YTD from the last close of last year). Price performance, not fund flows "
               "— estimated creations/redemptions are shown separately below." + _asof)

    from etl.sector_rotation import BENCHMARK
    _periods = {"1 week": "perf_1w", "1 month": "perf_1m", "3 months": "perf_3m", "6 months": "perf_6m",
                "YTD": "perf_ytd"}
    _periods = {k: c for k, c in _periods.items() if c in tv_sector_rotation.columns}     # older snapshots: 1m / 3m only
    if not tv_sector_rotation.empty and _periods:
        c_period, c_rel = st.columns([3, 1])
        with c_period:
            _period = sticky_segmented("Period", list(_periods), "1 month", "mp_rotation_period",
                                       label_visibility="collapsed")
        _col = _periods[_period]
        _bench = tv_sector_rotation.loc[tv_sector_rotation["sector"] == BENCHMARK, _col]
        _bench = float(_bench.iloc[0]) if not _bench.empty and pd.notna(_bench.iloc[0]) else None
        with c_rel:
            _rel = sticky(st.toggle, "vs S&P 500", False, "mp_rotation_rel", disabled=_bench is None,
                          help="Each sector's change minus SPY's over the same period (percentage points): above 0 = "
                               "it beat the market. Rotation is about relative strength, not up or down.")
        # a sector without enough history for this period is left out (unknown, not 0%)
        rot_df = tv_sector_rotation[tv_sector_rotation["sector"] != BENCHMARK].dropna(subset=[_col]).copy()
        _x = _col
        if _rel and _bench is not None:
            rot_df["vs_spy"] = rot_df[_col] - _bench
            _x = "vs_spy"
        rot_df = rot_df.sort_values(_x, ascending=True)
        _unit = " pts" if _x == "vs_spy" else "%"

        # Plotly Horizontal Bar Chart
        fig = px.bar(
            rot_df,
            x=_x,
            y='sector',
            orientation='h',
            text=rot_df[_x].apply(lambda x: f"{x:+.1f}{_unit}"),
            color=_x,
            color_continuous_scale=['#e74c3c', '#2c3e50', '#2ecc71'],
            color_continuous_midpoint=0
        )

        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title=(f"{_period}: change minus S&P 500 (percentage points)" if _x == "vs_spy"
                         else f"{_period} performance (%)"),
            yaxis_title="",
            coloraxis_showscale=False,
        )

        fig.update_traces(
            textposition='outside',
            textfont_size=11,
            cliponaxis=False
        )

        st.plotly_chart(phone_safe_chart(fig), use_container_width=True)
        st.caption(f"S&P 500 (SPY) over {_period.lower() if _period != 'YTD' else 'the year to date'}: "
                   + (f"{_bench + 0:+.2f}% in USD (the € index chart above includes EUR/USD moves)."
                      if _bench is not None else "not available in this snapshot."))
    else:
        st.info("Sector ETF performance is not in this snapshot (it is collected by the ETL).")

    _rotation_context(tv_sector_rotation, ctx.get("data_version"))
    st.markdown("---")


    st.markdown("---")
    _economic_data(ctx)
    st.markdown("---")
    st.markdown("#### External market tools (TradingView)")
    import streamlit.components.v1 as _mp_comp
    if not sticky(st.toggle, "Load the external widgets", False, "mp_external",
                  help="Heatmap, economic calendar and cross-asset performance are TradingView widgets: they load scripts from "
                       "tradingview.com in your browser, are not part of the warehouse snapshot and never feed the Decision."):
        st.caption("Not loaded. Switch on to see the S&P 500 sector heatmap, the economic calendar and cross-asset performance.")
        return
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("##### S&P 500 Sector Heatmap")
        _mp_comp.html(_HEATMAP, height=520)
    with c2:
        st.markdown("##### Economic Calendar")
        _mp_comp.html(_CALENDAR, height=520)
    st.markdown("##### Global Cross-Asset Performance")
    _mp_comp.html(_CROSS, height=420)


def _rotation_context(rotation, generation=None):
    from core.sector_rotation import estimate_flows, rotation_warnings
    from etl.config_manager import load_config
    from etl.sector_rotation import BENCHMARK
    from services.db import load_etf_shares_snapshots
    try:
        snapshots = load_etf_shares_snapshots(generation)
    except Exception:
        snapshots = pd.DataFrame()
        st.info("ETF flow history could not be read.")
    flows = estimate_flows(snapshots)
    if rotation.empty:                      # no sector data is unknown, not "no warnings"
        st.caption("Rotation warnings need the sector ETF performance above.")
    else:
        warnings = rotation_warnings(rotation, load_config("etl_config")["sector_rotation"], flows)
        for item in warnings[:5]:
            (st.warning if item["severity"] == "warning" else st.info)(item["message"])
        if not warnings:
            st.info("No rotation warnings for this snapshot.")
    st.caption("Context only — not used in the Decision.")
    if not flows.empty:
        show = flows[flows.sector != BENCHMARK].copy()
        display = {"Sector": show.sector, "Snapshot": show.as_of.dt.strftime("%Y-%m-%d")}
        from services.market_data import stored_fx_rate
        _fx = stored_fx_rate("USD", with_date=True)      # stored USD -> EUR rate; unknown -> N/A, never $ shown as EUR
        _usd_eur, _fx_date = _fx if _fx else (None, None)
        for period in ("1w", "1m"):
            label = period.upper()
            display[f"{label} EUR (€M)"] = show[f"flow_{period}_usd"].map(
                lambda v: f"{v * _usd_eur / 1e6:+.2f}" if pd.notna(v) and _usd_eur else "N/A")
            display[f"{label} % AUM"] = show[f"flow_{period}_pct"].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "N/A")
            display[f"{label} note"] = show[f"flow_{period}_note"]
        st.dataframe(pd.DataFrame(display), hide_index=True, width="stretch")
    history_note = (f"history collected by this app since {min(flows.history_start)}." if not flows.empty
                    else "No ETF share snapshots collected yet; flows are N/A.")
    _fx_note = ""
    if not flows.empty:
        _fx_note = (f" Amounts in EUR at the stored USD rate {_usd_eur:.4f} ({_fx_date or 'date unknown'}); % AUM is on "
                    f"the USD figures." if _usd_eur else " No stored USD rate: EUR amounts are N/A.")
    st.caption("Estimated from changes in shares outstanding reported by Yahoo (creations / redemptions × price); "
               + history_note + _fx_note)
