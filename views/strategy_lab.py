"""View: 🧪 Strategy Lab"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.backtest import run_backtest_simulation
from ui.components import render_metric_tile
from ui.icons import render_header


def render(ctx):
    """Render the 🧪 Strategy Lab tab. ctx is the context dict built in app.py."""
    all_tickers = ctx['all_tickers']
    format_ticker = ctx['format_ticker']
    prices_full = ctx['prices_full']
    reco_df = ctx['reco_df']

    render_header("activity", "Strategy Backtesting Engine — Signal Simulator")

    st.markdown("""
    <div style='background:rgba(0,255,204,0.05); border:1px solid rgba(0,255,204,0.2);
                border-radius:8px; padding:12px 16px; margin-bottom:16px; font-size:0.85rem; color:#aaa;'>
    <span style='color:#00ffcc; font-weight:900;'>[INFO]</span> <b>How it works:</b> Select a trading rule or run a tournament to find the best logic for a specific ticker.
    The engine simulates every signal on <b>the full price history loaded in the dashboard</b> (up to ~3 years, independent of the sidebar horizon).
    </div>
    """, unsafe_allow_html=True)


    bt_col1, bt_col2 = st.columns([1, 2])

    with bt_col1:
        st.markdown("#### Trading Rule Configuration")
        _bt_options = [t for t in all_tickers if t not in ["^VIX","SPY","^GSPC","^DJI","^IXIC"]]
        
        with st.form("backtest_form"):
            bt_ticker = st.selectbox("Select Ticker to Backtest", options=_bt_options, format_func=format_ticker, key="bt_ticker_form")
            
            st.markdown("###### Risk Management")
            sl_col, tp_col = st.columns(2)
            with sl_col: stop_loss = st.slider("Stop Loss (%)", 0, 30, 8)
            with tp_col: take_profit = st.slider("Take Profit (%)", 0, 100, 25)
                
            st.markdown("###### Capital Constraints")
            initial_capital = st.number_input("Initial Capital (€)", 1000, 1_000_000, 10000, step=1000)
            tx_cost_v = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, step=0.05)
            
            run_backtest = st.form_submit_button("▶ Run All Strategies & Find Best", width="stretch", type="primary")

    with bt_col2:
        if run_backtest and bt_ticker:
            tx_cost_pct = tx_cost_v / 100.0
            sl_pct = stop_loss / 100.0
            tp_pct = take_profit / 100.0
            
            # Full loaded history — the sidebar horizon (default 1Y) left only a handful of trades
            bt_prices = prices_full[prices_full["ticker"] == bt_ticker].sort_values("date").copy()
            
            all_strats = [
                "Institutional Quality Pulse (AI Score > 75)",
                "Trend Following (MA20/50 Cross)", 
                "RSI Mean Reversion (30/70)",
                "Z-Score Mean Reversion (Deep Value)",
                "Buy on Dip (Uptrend + Oversold)",
                "Multi-Indicator Breakout (Price>MA50 + RSI>50)"
            ]
            results = []
            progress_bar = st.progress(0)
            for idx, s in enumerate(all_strats):
                progress_bar.progress((idx + 1) / len(all_strats), text=f"Simulating: {s}")
                r = run_backtest_simulation(bt_ticker, bt_prices, s, sl_pct, tp_pct, tx_cost_pct, initial_capital, reco_df)
                if r: results.append(r)
            progress_bar.empty()
            
            if results:
                st.session_state["bt_leaderboard"] = results
                best_res = max(results, key=lambda x: (not x["lookahead"],
                                                       x["sharpe"] if x["sharpe"] == x["sharpe"] else float("-inf")))
                st.session_state["bt_results"] = best_res

        # ── RENDER RESULTS ────────────────────────────────────────────────────
        if "bt_results" in st.session_state and st.session_state["bt_results"]:
            r = st.session_state["bt_results"]
            l_board = st.session_state.get("bt_leaderboard")
            
            if l_board:
                render_header("trophy", f"Strategy Tournament Leaderboard — {r['ticker']}")
                
                # Build Comparison Table
                comp_data = []
                for s_res in l_board:
                    comp_data.append({
                        "Strategy": s_res["strategy"],
                        "Return %": s_res["total_return"],
                        "Sharpe": s_res["sharpe"],
                        "Max DD %": s_res["max_dd"],
                        "Win Rate %": s_res["win_rate"],
                        "Trades": s_res["n_trades"],
                        "Lookahead": s_res.get("lookahead", False),
                    })

                # Lookahead-biased rows sink to the bottom so iloc[0] is always a fair winner
                comp_df = pd.DataFrame(comp_data).sort_values(["Lookahead", "Sharpe"], ascending=[True, False],
                                                              na_position="last")

                # Only call a "winner" if it actually beats doing nothing (buy & hold) after costs
                best = comp_df.iloc[0]
                best_strat_name = best["Strategy"]
                _bnh = r["bnh_return"]
                if best["Return %"] > _bnh and best["Sharpe"] == best["Sharpe"] and best["Sharpe"] > 0:
                    st.markdown(f"""
                    <div style='background:rgba(46, 204, 113, 0.1); border-left:4px solid #2ecc71; padding:15px; border-radius:4px; margin-bottom:20px;'>
                        <span style='color:#2ecc71; font-weight:800; font-size:1.1rem;'>BEST RULE: {best_strat_name}</span><br>
                        <span style='color:#bbb; font-size:0.9rem;'>For {r['ticker']}: {best['Return %']:+.1f}% vs buy &amp; hold {_bnh:+.1f}% (Sharpe {best['Sharpe']:.2f}).</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background:rgba(241, 196, 15, 0.08); border-left:4px solid #f1c40f; padding:15px; border-radius:4px; margin-bottom:20px;'>
                        <span style='color:#f1c40f; font-weight:800; font-size:1.1rem;'>NO RULE BEAT BUY &amp; HOLD</span><br>
                        <span style='color:#bbb; font-size:0.9rem;'>For {r['ticker']}, buy &amp; hold returned {_bnh:+.1f}%; the best rule ({best_strat_name}) returned {best['Return %']:+.1f}%. Trading these signals would have cost money here.</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.dataframe(comp_df, width="stretch", hide_index=True,
                             column_config={
                                 "Return %": st.column_config.NumberColumn("Return", format="%.1f%%"),
                                 "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                                 "Max DD %": st.column_config.NumberColumn("Max DD", format="%.1f%%"),
                                 "Win Rate %": st.column_config.NumberColumn("Win Rate", format="%.0f%%"),
                                 "Lookahead": st.column_config.CheckboxColumn("⚠️ Lookahead", help="Uses today's Quality Score on historical bars — results are optimistic and excluded from the ranking."),
                             })
                st.caption("⚠️ The winner is picked in-sample on the same history it is scored on — "
                           "treat it as a hypothesis to validate on newer data, not as an expected return.")
                st.markdown("---")

            # Main Metrics (of best/selected)
            st.caption(f"Showing detailed analytics for: **{r['strategy']}**")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: render_metric_tile("Total Return",  f"{r['total_return']:+.1f}%", delta=r["total_return"])
            with m2: render_metric_tile("vs Buy&Hold",   f"{r['total_return']-r['bnh_return']:+.1f}%", delta=r["total_return"]-r["bnh_return"])
            with m3: render_metric_tile("Sharpe Ratio",  f"{r['sharpe']:.2f}")
            with m4: render_metric_tile("Max Drawdown",  f"{r['max_dd']:.1f}%")
            with m5: render_metric_tile("Win Rate",      f"{r['win_rate']:.0f}% ({r['n_trades']} trades)")

            # Chart
            fig_bt = go.Figure()
            # Overlay B&H
            fig_bt.add_trace(go.Scatter(x=r["dates_arr"], y=r["bnh_curve"], name="Buy & Hold", line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dot")))
            
            if l_board:
                # Add Top 3 Curves
                colors = ["#00ffcc", "#3498db", "#9b59b6"]
                for i, row in enumerate(comp_df.head(3).itertuples()):
                    # Find matches in results
                    s_dat = next(x for x in l_board if x["strategy"] == row.Strategy)
                    fig_bt.add_trace(go.Scatter(x=s_dat["dates_arr"], y=s_dat["equity_curve"], name=f"Rank {i+1}: {row.Strategy}", line=dict(color=colors[i], width=2 if i>0 else 3.5)))
            else:
                fig_bt.add_trace(go.Scatter(x=r["dates_arr"], y=r["equity_curve"], name=r["strategy"], line=dict(color="#00ffcc", width=3)))
            
            fig_bt.update_layout(template="plotly_dark", height=450, yaxis_title="Equity (€)", hovermode="x unified", legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig_bt, use_container_width=True)

            with st.expander("📋 View Trade Log"):
                st.dataframe(pd.DataFrame(r["trade_log"]), width="stretch", hide_index=True)
        else:
            st.info("👈 Configure your trading rule on the left and click **▶ Run All Strategies & Find Best** to start.")
