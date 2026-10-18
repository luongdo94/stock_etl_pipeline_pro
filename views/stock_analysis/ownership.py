"""Ownership structure and short-squeeze gauge."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.icons import render_header



def render(dd, ctx):
    meta = dd.meta
    # ── OWNERSHIP & SHORT SQUEEZE RISK ──────────────────────────────
    st.markdown("---")
    render_header("search", "Ownership & Short-Squeeze Risk")
    
    def _frac(v):
        """A fraction, or None when the provider did not report it (0 is a real value; unknown is not 0)."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f == f else None

    inst_own, insider_own = _frac(meta.get("inst_ownership")), _frac(meta.get("insider_ownership"))
    short_pct, short_ratio = _frac(meta.get("short_percent_of_float")), _frac(meta.get("short_ratio"))

    col_own1, col_own2 = st.columns([1, 1])
    with col_own1:
        if inst_own is None and insider_own is None:
            st.info("Ownership is not reported for this listing.")
        else:
            known = {"Institutions": inst_own, "Insiders": insider_own}
            labels = [k for k, v in known.items() if v is not None]
            values = [known[k] for k in labels]
            colors = ['#00d2ff', '#3a7bd5'][:0] + [c for k, c in zip(known, ['#00d2ff', '#3a7bd5']) if known[k] is not None]
            if inst_own is not None and insider_own is not None:          # the remainder is only known when both are
                labels.append("Public/other")
                values.append(max(0.0, 1.0 - inst_own - insider_own))
                colors.append('rgba(255,255,255,0.05)')
            fig_own = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.65)])
            fig_own.update_traces(hoverinfo='label+percent', textinfo='none', marker=dict(colors=colors, line=dict(color='#0d0e14', width=2)))
            fig_own.update_layout(
                title=dict(text="Corporate Ownership Structure", font=dict(size=18)),
                template="plotly_dark",
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            centre = (f"{(inst_own + insider_own) * 100:.1f}%<br><b>Inst. + insiders</b>"
                      if inst_own is not None and insider_own is not None else "<b>partly<br>reported</b>")
            fig_own.add_annotation(text=centre, x=0.5, y=0.5, font_size=18, showarrow=False)
            st.plotly_chart(fig_own, use_container_width=True)
            if inst_own is None or insider_own is None:
                st.caption("Institutions: " + ("n/a" if inst_own is None else f"{inst_own:.1%}")
                           + " · Insiders: " + ("n/a" if insider_own is None else f"{insider_own:.1%}")
                           + " — one source is not reported, so no remainder is shown.")

    with col_own2:
        if short_pct is None:
            st.info("Short interest is not reported for this listing (not 0%).")
        else:
            squeeze_color = "#e74c3c" if short_pct > 0.15 else "#f39c12" if short_pct > 0.05 else "#2ecc71"

            fig_short = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = short_pct * 100,
                number = {'suffix': "%", 'font': {'size': 45, 'color': squeeze_color}},
                title = {'text': "Short % of Float (Squeeze Risk)", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [None, max(30, (short_pct*100)+5)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': squeeze_color},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 5], 'color': "rgba(46, 204, 113, 0.15)"},
                        {'range': [5, 15], 'color': "rgba(243, 156, 18, 0.15)"},
                        {'range': [15, 100], 'color': "rgba(231, 76, 60, 0.15)"}],
                }
            ))
            fig_short.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_short, use_container_width=True)

        ratio_txt = "N/A" if short_ratio is None else f"{short_ratio:.1f} days"
        st.markdown(f"<p style='text-align:center; color:#bbb; font-size:1rem;'>Short Ratio (Days to Cover): <b>{ratio_txt}</b></p>", unsafe_allow_html=True)

    _render_insider_activity(dd)


_INSIDER_COLOR = {"NET BUY": "#2ecc71", "NET SELL": "#e74c3c", "NO TRADES": "#8899aa", "NO DATA": "#667788"}


def _render_insider_activity(dd):
    """6-month insider buying / selling (provider summary) and the latest Form 4 open-market trades of this ticker."""
    ins = dd.scores.get("insider") or {}
    status = ins.get("status") or "NO DATA"
    st.markdown("**Insider activity — last 6 months**")
    if status == "NO DATA":
        st.caption("No insider-trading data for this listing (the provider covers US filers; zero is not reported as "
                   "'no trades').")
        return
    c1, c2, c3 = st.columns(3)
    net_pct = ins.get("net_pct")
    c1.markdown(f"<div style='color:{_INSIDER_COLOR[status]};font-size:1.3rem;font-weight:800;'>{status}</div>"
                f"<div style='color:#8899aa;font-size:0.75rem;'>net "
                f"{'n/a' if net_pct is None else f'{net_pct:+.2f}% of insider holdings'}</div>", unsafe_allow_html=True)
    c2.metric("Shares bought", f"{ins.get('buys') or 0:,.0f}")
    c3.metric("Shares sold", f"{ins.get('sells') or 0:,.0f}")
    st.caption("Open-market buying by several insiders is informative; selling is often routine (taxes, diversification, "
               "planned 10b5-1 sales). Not part of the scores until the Track Record shows it adds information.")
    try:
        from services.db import load_insider_transactions
        tx = load_insider_transactions(ticker=dd.ticker, days=180, limit=15)
    except Exception:
        tx = pd.DataFrame()
    if tx is not None and not tx.empty:
        with st.expander(f"Latest Form 4 transactions ({len(tx)})"):
            cols = [c for c in ("transaction_date", "insider_name", "position", "transaction_type", "shares", "value")
                    if c in tx.columns]
            st.dataframe(tx[cols], hide_index=True, width="stretch")
