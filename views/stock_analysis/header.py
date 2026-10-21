"""Layer 1 — title, Decision Summary and 52-week range meter."""
import pandas as pd
import streamlit as st

from services.db import load_ticker_provenance, load_track_record
from ui.provenance import render_confidence_line
from services.user_store import read_portfolio_status
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
        """({ticker: EUR value}, cash EUR) — the panel sizes against NAV = holdings + cash."""
        _pf, _cash, _, _ok = read_portfolio_status()
        if not _ok:
            st.warning("Your portfolio could not be read. Try again by reopening Overview.")
            return None
        _last = prices_full.sort_values("date").groupby("ticker")["price_close"].last()
        # a holding without a price is held but of UNKNOWN value (None), not worth €0
        def _val(t, v):
            px = _last.get(t)
            return None if px is None or pd.isna(px) else v.get("shares", 0) * float(px)
        return {t: _val(t, v) for t, v in _pf.items()}, _cash

    render_decision_panel(
        ticker=dd.ticker, meta=meta, price=float(dd.cur_p),
        price_date=dd.extra["latest_date"],               # the latest close, not the chart's end
        stop_loss=dd.stop_loss, vin=dd.vin, relval=dd.relval,
        missing=dd.scores["missing"], scores=dd.scores,
        next_earnings=dd.next_earnings, quality=dd.ai_score,
        snapshots=load_track_record(), prices=prices_full,
        holdings_loader=_holdings_value, companies=ctx["companies_full"],
        signal_conflict=dd.extra.get("signal_conflict"), decision=dd.extra.get("decision"))
    render_confidence_line(load_ticker_provenance(dd.ticker), dd.vin)

    pos, lo, hi = dd.w52_pos, dd.tm["w52_lo"], dd.tm["w52_hi"]
    zone = "Near Low" if pos < 20 else ("Near High" if pos > 80 else "Mid-Range")
    st.markdown(f"""
    <div style='background:var(--secondary-background-color, transparent); border:1px solid rgba(128,128,128,0.25);
                border-radius:10px; padding:14px 20px; margin-bottom:10px;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
            <span style='color:inherit; font-size:0.75rem; font-weight:600; text-transform:uppercase;'>52-Week Range</span>
            <span style='color:inherit; font-size:0.85rem; font-weight:700;'>{zone} &nbsp;|&nbsp; Position: {pos:.0f}%</span>
        </div>
        <div style='display:flex; align-items:center; gap:10px;'>
            <span style='color:#e74c3c; font-size:0.85rem; white-space:nowrap;'>Low: €{lo:.2f}</span>
            <div style='flex:1; background:var(--secondary-background-color, transparent); border-radius:4px; height:10px; position:relative;'>
                <div style='width:{pos:.1f}%; height:100%; background:linear-gradient(90deg,#e74c3c,#f1c40f,#2ecc71); border-radius:4px;'></div>
                <div style='position:absolute; top:-3px; left:{pos:.1f}%; transform:translateX(-50%);
                            width:14px; height:14px; background:#fff; border-radius:50%; border:2px solid #3498db;'></div>
            </div>
            <span style='color:#2ecc71; font-size:0.85rem; white-space:nowrap;'>High: €{hi:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
