"""View: 🧪 Strategy Lab — evidence summary + exploratory timing comparison on one stock."""
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import lab_view as lv
from core.backtest import STATIC_QUALITY, run_backtest_simulation
from core.track_record import evidence_headline
from services import db
from ui.components import phone_safe_chart, render_metric_tile
from ui.icons import render_header
from ui.sticky import follow_active_ticker, my_tickers, remember_ticker, sticky, sticky_choice, ticker_options

RULES = [
    "Trend Following (MA20/50 Cross)",
    "RSI Mean Reversion (30/70)",
    "Z-Score Mean Reversion (60-bar price deviation)",
    "Buy on Dip (Uptrend + Oversold)",
    "RSI 50 Cross above MA50 (Price>MA50, RSI crosses 50)",
]
DEMO = f"{STATIC_QUALITY} (today's Quality ≥ 75 applied to past bars — lookahead, not evidence)"
_SKIP = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]


def _evidence_block(prices_full):
    try:
        ev = db.evidence(prices_full)
    except Exception as e:           # a read error is not "no history"
        ev = {"ok": False, "label": f"Evidence not computable: {e}"}
    head = evidence_headline(ev)
    gate = True
    try:
        from core.decision import load_rules
        gate = bool((load_rules().get("evidence") or {}).get("require_validated_evidence_for_buy", True))
    except Exception:
        pass
    st.markdown("**What is supported, what is not**")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.metric("System evidence (Quality, 12 months)", head)
    with c2:
        st.caption(f"{ev.get('label', '')}  \nEvidence required for BUY: **{'On' if gate else 'Off'}** (owner setting). "
                   f"The timing experiments below are a different question and do not change this status. "
                   f"Details: the Track Record tab.")


def render(ctx):
    all_tickers = ctx['all_tickers']
    format_ticker = ctx['format_ticker']
    prices_full = ctx['prices_full']
    reco_df = ctx['reco_df']

    render_header("activity", "Strategy Lab — Evidence & Experiments")
    _evidence_block(prices_full)
    st.markdown("---")

    st.markdown("#### Compare timing rules on one stock")
    st.caption(f"{lv.EXPLORATORY} A single stock, closes only, today's universe: this is not evidence for a 3-5-year "
               f"fundamental decision. The simulation uses the full price history loaded in the dashboard (up to ~3 years).")

    col1, col2 = st.columns([1, 2])
    options = [t for t in all_tickers if t not in _SKIP]
    with col1:
        follow_active_ticker("slab_ticker", options)
        cur = st.session_state.get("_kept::slab_ticker")
        options = ticker_options(options, cur if cur in options else None, my_tickers())
        ticker = st.selectbox("Ticker", options, index=0, format_func=format_ticker, key="slab_ticker") if options else None
        st.session_state["_kept::slab_ticker"] = ticker
        remember_ticker(ticker)
        sl_c, tp_c = st.columns(2)
        with sl_c:
            sl = sticky(st.slider, "Stop loss (%)", 8, "slab_sl", min_value=0, max_value=30,
                        help="Closing-price stop, filled at the next close. 0 does NOT disable it: any loss triggers an exit.")
        with tp_c:
            tp = sticky(st.slider, "Take profit (%)", 25, "slab_tp", min_value=0, max_value=100,
                        help="0 disables the take-profit.")
        capital = sticky(st.number_input, "Initial capital (€)", 10000, "slab_cap", min_value=1000, max_value=1_000_000,
                         step=1000)
        fee = sticky(st.slider, "Transaction cost per fill (%)", 0.1, "slab_fee", min_value=0.0, max_value=1.0, step=0.05)
        demo = sticky(st.toggle, "Include the Static Quality demo (lookahead — methodology only)", False, "slab_demo")
        run = st.button("Compare all rules", type="primary", width="stretch")

    spec = lv.run_spec(ticker, sl, tp, capital, fee, db.opened_version(), demo)

    with col2:
        if run and ticker:
            bars = prices_full[prices_full["ticker"] == ticker].sort_values("date").copy()
            problem = lv.validate(bars)
            if problem:
                st.session_state.pop("slab_run", None)          # never leave an old result under a failed run
                st.warning(problem)
            else:
                results, names = [], RULES + ([DEMO] if demo else [])
                bar = st.progress(0)
                for i, name in enumerate(names):
                    bar.progress((i + 1) / len(names), text=f"Simulating: {name}")
                    r = run_backtest_simulation(ticker, bars, name, sl / 100, tp / 100, fee / 100, capital, reco_df)
                    if r:
                        results.append(r)
                bar.empty()
                if results:
                    st.session_state["slab_run"] = {"spec": spec, "results": results, "at": time.strftime("%H:%M")}
                else:
                    st.session_state.pop("slab_run", None)
                    st.warning("The simulation returned no result for this ticker.")

        saved = st.session_state.get("slab_run")
        if not saved:
            st.info("Choose a ticker and press **Compare all rules**.")
            return
        results, rspec = saved["results"], saved["spec"]
        why = lv.stale_reason(saved, spec)
        if why:
            st.warning(f"Showing the run of {saved['at']} for {rspec['ticker']}: {why}. Press Compare to refresh.")
        comp = lv.comparison_rows(results)
        render_header("trophy", f"Rule comparison — {rspec['ticker']}")
        gaps = int(comp["Missing returns"].max() or 0)
        if gaps:
            st.warning(f"{gaps} trading day(s) without a return inside {rspec['ticker']}'s history were counted as 0% — "
                       f"read these results as indicative.")
        st.markdown(f"<div style='border-left:4px solid #8899aa; padding:10px 14px; margin-bottom:12px; color:inherit;'>"
                    f"{lv.neutral_summary(results)}</div>", unsafe_allow_html=True)
        st.dataframe(comp, width="stretch", hide_index=True, column_config={
            "Return %": st.column_config.NumberColumn("Return", format="%.1f%%"),
            "vs buy & hold (pts)": st.column_config.NumberColumn("vs buy & hold", format="%+.1f"),
            "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f", help="N/A when the rule never traded or had no variance."),
            "Max DD %": st.column_config.NumberColumn("Max DD", format="%.1f%%"),
            "Win rate %": st.column_config.NumberColumn("Win rate", format="%.0f%%", help="N/A when no trade was completed."),
            "Days in market %": st.column_config.NumberColumn("In market", format="%.0f%%"),
            "Lookahead": st.column_config.CheckboxColumn("Lookahead", help="Uses today's Quality on past bars: optimistic by construction."),
        })
        st.caption(lv.sample_meta(results[0]))

        names = [r["strategy"] for r in results]
        pick = sticky_choice("Show details for", names, names[0], "slab_pick")
        r = next(x for x in results if x["strategy"] == pick)
        m = st.columns(5)
        with m[0]: render_metric_tile("Total return", lv.fmt(r["total_return"], "{:+.1f}", "%"), delta=r["total_return"])
        with m[1]: render_metric_tile("vs buy & hold", lv.fmt(r["total_return"] - r["bnh_return"], "{:+.1f}", " pts"),
                                      delta=r["total_return"] - r["bnh_return"])
        with m[2]: render_metric_tile("Sharpe", lv.fmt(r["sharpe"]))
        with m[3]: render_metric_tile("Max drawdown", lv.fmt(r["max_dd"], "{:.1f}", "%"))
        with m[4]:
            wr = lv.win_rate(r)
            render_metric_tile("Win rate", "N/A (no completed trade)" if wr is None else f"{wr:.0f}% ({r['n_trades']} trades)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r["dates_arr"], y=r["bnh_curve"], name="Buy & hold (fee-free)",
                                 line=dict(color="#64748b", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=r["dates_arr"], y=r["equity_curve"], name=r["strategy"], line=dict(color="#0284C7", width=3)))
        fig.update_layout(height=420, yaxis_title="Equity (€)", hovermode="x unified",
                          legend=dict(orientation="h", y=1.05))
        st.plotly_chart(phone_safe_chart(fig), use_container_width=True)
        with st.expander("Trade log"):
            trades = pd.DataFrame(r["trade_log"]).drop(columns=["_pnl"], errors="ignore")
            if "Action" in trades:
                trades["Action"] = trades["Action"].replace({"🟢 BUY": "BUY", "🔴 SELL": "SELL"})
            st.dataframe(trades, width="stretch", hide_index=True)
