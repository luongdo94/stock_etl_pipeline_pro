"""
Decision Summary + Valuation sections of the Stock Analysis tab, and the alert inbox.
All numbers come from core.valuation / core.decision / core.track_record; this module only renders.
"""
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from core import decision as dec
from core import valuation as val
from core.valuation import risk_free_from_macro, valuation_inputs  # noqa: F401 (re-exported)
from core.portfolio_risk import candidate_impact
from core.track_record import evidence_status

_STANCE_COLORS = {"BUY CANDIDATE": "#2ecc71", "HOLD / WATCH": "#f1c40f",
                  "AVOID / TRIM": "#e74c3c", "NOT ENOUGH DATA": "#8899aa"}
_CONF_COLORS = {"HIGH": "#2ecc71", "MEDIUM": "#f1c40f", "LOW": "#e74c3c"}


def _f(x) -> Optional[float]:
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_evidence(snapshots: pd.DataFrame, prices: pd.DataFrame) -> dict:
    return evidence_status(snapshots, prices)


def render_decision_panel(*, ticker, meta, price, price_date, stop_loss, vin, relval, missing,
                          next_earnings, quality, snapshots, prices, holdings_loader, companies):
    ev = _cached_evidence(snapshots, prices[["date", "ticker", "price_close"]])
    d = dec.build_decision(
        price=price, base_value=vin["base"], bear_value=vin["bear"], bull_value=vin["bull"], stop_loss=stop_loss,
        currency=meta.get("currency"), country=meta.get("country"),
        dividend_yield_pct=_f(meta.get("dividend_yield_pct")), missing_metrics=missing,
        price_date=price_date, fundamentals_date=_to_date(meta.get("info_updated_at")),
        next_earnings=next_earnings, track_record_ok=ev["ok"], quality_score=quality,
        valuation_reliable=vin["reliable"], valuation_note=vin["note"])

    sc, cc = _STANCE_COLORS.get(d.stance, "#8899aa"), _CONF_COLORS[d.confidence]
    fmt = lambda v, s="%": f"{v:+.1f}{s}" if v is not None else "N/A"
    st.markdown(f"""
    <div style='border:1px solid {sc}55; border-left:5px solid {sc}; border-radius:10px;
                padding:14px 18px; margin:6px 0 14px 0; background:rgba(255,255,255,0.03);'>
      <div style='display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px;'>
        <div><div style='color:#8899aa; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;'>Decision Summary</div>
             <div style='color:{sc}; font-size:1.6rem; font-weight:900;'>{d.stance}</div>
             <div style='color:{cc}; font-size:0.8rem; font-weight:700;'>Confidence: {d.confidence}</div></div>
        <div style='display:grid; grid-template-columns:repeat(4, auto); gap:6px 22px; text-align:right;'>
          <div style='color:#8899aa; font-size:0.7rem;'>Expected (net)</div>
          <div style='color:#8899aa; font-size:0.7rem;'>Downside</div>
          <div style='color:#8899aa; font-size:0.7rem;'>Reward / Risk</div>
          <div style='color:#8899aa; font-size:0.7rem;'>Suggested size</div>
          <div style='color:#fff; font-weight:800;'>{fmt(d.net_expected_return_pct)}</div>
          <div style='color:#e74c3c; font-weight:800;'>{("-" + format(d.downside_pct, ".1f") + "%") if d.downside_pct else "N/A"}</div>
          <div style='color:#fff; font-weight:800;'>{f"{d.reward_risk:.1f}x" if d.reward_risk is not None else ("no upside" if (d.net_expected_return_pct or 0) <= 0 else "N/A")}</div>
          <div style='color:#fff; font-weight:800;'>{f"{d.position['size_pct']:.1f}% of portfolio" if d.position['size_pct'] else "—"}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    for w in d.warnings:
        st.warning(w, icon="⏳")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Why**")
        for r in d.reasons:
            st.markdown(f"- {r}")
        if relval["percentiles"]:
            parts = [f"{k.replace('_', ' ')} {v:.0f}th pct" for k, v in relval["percentiles"].items()]
            st.markdown(f"- Relative to {relval['group']} ({relval['n_peers']} peers; 0 = cheapest): " + ", ".join(parts))
        if relval["pe_vs_5y"]:
            st.markdown(f"- P/E is {relval['pe_vs_5y']:.2f}× its own 5-year average.")
        if vin["implied_growth"] is not None:
            st.markdown(f"- The price implies **{vin['implied_growth']:+.1%}** starting FCF growth "
                        f"(model base case: {vin['growth']:+.1%}).")
    with c2:
        st.markdown("**What would make this wrong (sell discipline)**")
        for r in d.invalidation:
            st.markdown(f"- {r}")
        if d.confidence_notes:
            st.markdown("**Confidence limited by**")
            for n in d.confidence_notes:
                st.markdown(f"- {n}")
    if d.position["size_pct"]:
        cap = " (capped)" if d.position["capped"] else ""
        st.caption(f"Size = {dec.load_rules()['risk']['account_risk_pct']}% portfolio risk ÷ "
                   f"{d.position['stop_distance_pct']:.1f}% stop distance{cap}. Costs assumed in "
                   f"config/decision_rules.yaml (commission, FX spread, dividend withholding).")
        holdings_value = holdings_loader() if holdings_loader else None   # lazy: Supabase call
        imp = candidate_impact(prices, holdings_value, ticker, d.position["size_pct"], companies) \
            if holdings_value else None
        if imp:
            corr = f"{imp['correlation']:.2f}" if imp["correlation"] is not None else "n/a"
            st.info(f"Portfolio fit: correlation with current holdings **{corr}** · "
                    f"{imp['sector']} weight {imp['sector_weight_before']:.0f}% → {imp['sector_weight_after']:.0f}% · "
                    f"{imp['currency']} exposure {imp['currency_weight_before']:.0f}% → {imp['currency_weight_after']:.0f}% · "
                    f"largest sector after: {imp['largest_sector_after'][0]} {imp['largest_sector_after'][1]:.0f}%")
    st.caption("Decision support for your own judgement — not investment advice. "
               f"Signal evidence: {ev['label']}")
    return d


def render_valuation_section(*, meta, price, vin, relval):
    """Replaces the old one-size DCF: company-anchored inputs, scenarios, sensitivity, reverse DCF."""
    from ui.icons import render_header
    st.markdown("---")
    render_header("gem", "Intrinsic Valuation (FCFE DCF) — Scenarios & Sensitivity")
    if not vin["fcfe"] or vin["fcfe"] <= 0 or not vin["shares"]:
        st.info("⚠️ " + (vin["note"] or "No positive free cash flow — a cash-flow valuation is not meaningful.")
                + " Use the relative valuation and quality pillars instead.")
        return
    if not vin["reliable"] and vin["note"]:
        st.warning("⚠️ " + vin["note"])
    st.caption(f"Free cash flow is after interest, so it is discounted at the cost of equity "
               f"(CAPM: rf {vin['risk_free']:.1%} + Blume-adjusted β×5% ERP) and debt is not subtracted again. "
               f"Base cash flow: {vin['fcf_source']}. Starting growth anchored on: "
               f"{', '.join(vin['growth_sources'])}; it fades to the terminal rate over {val.EXPLICIT_YEARS} years.")
    c1, c2, c3 = st.columns(3)
    g = c1.number_input("Starting FCF growth (%)", value=round(vin["growth"] * 100, 1), step=1.0,
                        key=f"dcf_g_{meta.get('ticker')}") / 100
    tg = c2.number_input("Terminal growth (%)", value=val.TERMINAL_GROWTH * 100, step=0.5,
                         key=f"dcf_tg_{meta.get('ticker')}") / 100
    r = c3.number_input("Cost of equity (%)", value=round(vin["cost_of_equity"] * 100, 1), step=0.5,
                        key=f"dcf_r_{meta.get('ticker')}") / 100
    if r <= tg:
        st.warning("Cost of equity must exceed terminal growth.")
        return
    scen = val.dcf_scenarios(vin["fcfe"], vin["shares"], g, r, tg)
    cols = st.columns(3)
    for col, key, color in zip(cols, ("bear", "base", "bull"), ("#e74c3c", "#3498db", "#2ecc71")):
        s = scen[key]
        mos = s.value_per_share / price - 1 if s.value_per_share else None
        col.markdown(f"""<div style='border-top:3px solid {color}; background:rgba(255,255,255,0.03);
            border-radius:8px; padding:10px 12px;'>
            <div style='color:#8899aa; font-size:0.7rem; text-transform:uppercase;'>{key} · g {s.growth:+.0%} · r {s.discount_rate:.1%}</div>
            <div style='color:#fff; font-size:1.5rem; font-weight:800;'>€{s.value_per_share:,.2f}</div>
            <div style='color:{color}; font-weight:700;'>{mos:+.0%} vs price · {val.valuation_verdict(mos)}</div></div>""",
                     unsafe_allow_html=True)
    implied = val.reverse_dcf_growth(price, vin["fcfe"], vin["shares"], r, tg)
    if implied is not None:
        st.markdown(f"**Reverse DCF:** at €{price:,.2f} the market prices in **{implied:+.1%}** starting growth "
                    f"(your base case {g:+.1%}). Buying only makes sense if you believe growth beats that.")
    growths = [g - 0.06, g - 0.03, g, g + 0.03, g + 0.06]
    rates = [x for x in (r - 0.02, r - 0.01, r, r + 0.01, r + 0.02) if x > tg]
    table = val.sensitivity_table(vin["fcfe"], vin["shares"], growths, rates, tg)
    st.markdown("**Sensitivity — value per share (rows: starting growth, columns: cost of equity)**")
    st.dataframe(table.style.format("€{:,.2f}", na_rep="—")
                 .map(lambda v: "color:#2ecc71" if v and v >= price * (1 + val.REQUIRED_MARGIN_OF_SAFETY)
                      else ("color:#e74c3c" if v and v < price else "")),
                 width="stretch")
    st.caption(f"Required margin of safety for 'UNDERVALUED': {val.REQUIRED_MARGIN_OF_SAFETY:.0%}. "
               "Excess cash is not in the warehouse and is not added (conservative for cash-rich firms).")


def render_alert_inbox(items: list, title: str = "🔔 Alerts & Sell-Discipline Inbox"):
    if not items:
        return
    with st.expander(f"{title} ({len(items)})", expanded=True):
        icons = {"THESIS INVALIDATED": "🔴", "TARGET REACHED": "🎯", "AT INTRINSIC VALUE": "💎",
                 "ENTRY ZONE": "🟢", "EARNINGS SOON": "⏳", "RULE": "🔔"}
        for it in items:
            st.markdown(f"{icons.get(it['kind'], '•')} **{it['kind']}** — {it['message']}")


def _to_date(x) -> Optional[date]:
    try:
        t = pd.to_datetime(x)
        return None if pd.isna(t) else t.date()
    except (TypeError, ValueError):
        return None
