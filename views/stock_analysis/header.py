"""Layer 1 — title, Decision Summary and 52-week range meter."""
import pandas as pd
import streamlit as st

from etl.utils import compute_score_details
from services.db import load_track_record
from services.user_store import load_portfolio_from_db
from ui.decision_panel import render_decision_panel
from views.stock_analysis.layout import layer_banner


def render(dd, ctx):
    meta, prices_full = dd.meta, ctx["prices_full"]
    company = meta.get("company", dd.ticker)
    if pd.isna(company):
        company = dd.ticker
    st.markdown(f"#### {company} ({dd.ticker}) — {meta['sector']} - €{dd.cur_p:.2f}")
    st.markdown("---")
    layer_banner(1, "Structural context", "#3498db", top=10, bottom=0)

    def _holdings_value():
        _pf = load_portfolio_from_db()
        _last = prices_full.sort_values("date").groupby("ticker")["price_close"].last()
        return {t: v.get("shares", 0) * float(_last.get(t, 0)) for t, v in _pf.items()}

    render_decision_panel(
        ticker=dd.ticker, meta=meta, price=float(dd.cur_p),
        price_date=pd.to_datetime(dd.df_deep["date"].iloc[-1]).date(),
        stop_loss=dd.stop_loss, vin=dd.vin, relval=dd.relval,
        missing=compute_score_details(dd.meta_enriched)["missing"],
        next_earnings=dd.next_earnings, quality=dd.ai_score,
        snapshots=load_track_record(), prices=prices_full,
        holdings_loader=_holdings_value, companies=ctx["companies_full"])

    pos, lo, hi = dd.w52_pos, dd.tm["w52_lo"], dd.tm["w52_hi"]
    zone = "Near Low" if pos < 20 else ("Near High" if pos > 80 else "Mid-Range")
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1);
                border-radius:10px; padding:14px 20px; margin-bottom:10px;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
            <span style='color:#999; font-size:0.75rem; font-weight:600; text-transform:uppercase;'>52-Week Range</span>
            <span style='color:#fff; font-size:0.85rem; font-weight:700;'>{zone} &nbsp;|&nbsp; Position: {pos:.0f}%</span>
        </div>
        <div style='display:flex; align-items:center; gap:10px;'>
            <span style='color:#e74c3c; font-size:0.85rem; white-space:nowrap;'>Low: €{lo:.2f}</span>
            <div style='flex:1; background:rgba(255,255,255,0.1); border-radius:4px; height:10px; position:relative;'>
                <div style='width:{pos:.1f}%; height:100%; background:linear-gradient(90deg,#e74c3c,#f1c40f,#2ecc71); border-radius:4px;'></div>
                <div style='position:absolute; top:-3px; left:{pos:.1f}%; transform:translateX(-50%);
                            width:14px; height:14px; background:#fff; border-radius:50%; border:2px solid #3498db;'></div>
            </div>
            <span style='color:#2ecc71; font-size:0.85rem; white-space:nowrap;'>High: €{hi:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
