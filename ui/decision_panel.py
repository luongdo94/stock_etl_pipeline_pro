"""
Decision Summary + Valuation sections of the Stock Analysis tab, and the alert inbox.
All numbers come from core.valuation / core.decision / core.track_record; this module only renders.
"""
import pandas as pd
import streamlit as st

from core import decision as dec
from core import valuation as val
from core.valuation import risk_free_from_macro, valuation_inputs  # noqa: F401 (re-exported)
from core.portfolio_risk import candidate_impact
from core.rating import quality_tier, value_tier
from core.track_record import evidence_status

_STANCE_COLORS = {"BUY CANDIDATE": "#2ecc71", "HOLD / WATCH": "#f1c40f",
                  "AVOID / TRIM": "#e74c3c", "NOT ENOUGH DATA": "#8899aa"}
_CONF_COLORS = {"HIGH": "#2ecc71", "MEDIUM": "#f1c40f", "LOW": "#e74c3c"}


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_evidence(snapshots: pd.DataFrame, prices: pd.DataFrame) -> dict:
    return evidence_status(snapshots, prices)


def _score_tile(title, sub, score, colour):
    shown = "N/A" if score is None else f"{score:.0f}"
    return (f"<div style='flex:1; min-width:150px; background:rgba(255,255,255,0.03); padding:10px 12px; "
            f"border-radius:8px; border-top:3px solid {colour};'>"
            f"<div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>{title}</div>"
            f"<div style='font-weight:900; font-size:1.3em; color:{colour};'>{shown}"
            f"<span style='font-size:0.55em; color:#667;'> /100</span></div>"
            f"<div style='font-size:0.65em; color:#778;'>{sub}</div></div>")


def _rev_sub(scores):
    up, down = scores.get("rev_up"), scores.get("rev_down")
    counts = f" · {int(up)}↑ / {int(down)}↓ (30d)" if up is not None and down is not None else ""
    return f"30-day change in analysts' EPS estimates{counts}; context only"


def render_score_strip(scores):
    """Quality / Value / Momentum tiles + red flags. `scores` = dict(quality, value, momentum, flags, coverage)."""
    q, v, m = scores.get("quality"), scores.get("value"), scores.get("momentum")
    q_label, q_col = quality_tier(q) if q is not None else ("n/a", "#8899aa")
    v_label, v_col = value_tier(v) if v is not None else ("n/a", "#8899aa")
    m_col = "#8899aa" if m is None else ("#2ecc71" if m >= 60 else ("#e74c3c" if m < 40 else "#f1c40f"))
    cov = scores.get("coverage")
    rv = scores.get("revisions")
    rv_col = "#8899aa" if rv is None else ("#2ecc71" if rv >= 60 else ("#e74c3c" if rv < 40 else "#f1c40f"))
    tiles = (_score_tile("Quality — the business", f"{q_label} · a floor for any BUY", q, q_col)
             + _score_tile("Value — the price", f"{v_label} · vs sector peers, no analyst forecasts", v, v_col)
             + _score_tile("Momentum — timing only", "12-1 month return (local currency) + trend", m, m_col)
             + _score_tile("Revisions — estimate changes", _rev_sub(scores), rv, rv_col))
    st.markdown(f"<div style='display:flex; gap:10px; flex-wrap:wrap; margin-bottom:8px;'>{tiles}</div>",
                unsafe_allow_html=True)
    if scores.get("flags"):
        st.warning(f"Red flags: {scores['flags']}", icon="🚩")
    if cov is not None and cov < 60:
        st.caption(f"⚠️ Only {cov:.0f}% of the scoring inputs are available — scores are pulled toward neutral.")


def render_decision_panel(*, ticker, meta, price, price_date, stop_loss, vin, relval, missing,
                          next_earnings, quality, snapshots, prices, holdings_loader, companies,
                          scores=None):
    ev = _cached_evidence(snapshots, prices[["date", "ticker", "price_close"]])
    d = dec.decide(price=price, vin=vin, stop_loss=stop_loss, meta=meta,
                   scores={**(scores or {}), "quality": quality, "missing": missing},
                   price_date=price_date, next_earnings=next_earnings, track_record_ok=ev["ok"])

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
          <div style='color:#fff; font-weight:800;'>{f"{d.reward_risk:.1f}x" if d.reward_risk is not None else ("no upside" if (d.net_expected_return_pct is not None and d.net_expected_return_pct <= 0) else "N/A")}</div>
          <div style='color:#fff; font-weight:800;'>{f"{d.position['size_pct']:.1f}% of portfolio" if d.position['size_pct'] else "—"}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if scores:
        render_score_strip(scores)
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


def _render_financial_valuation(meta, price, vin):
    """Banks, insurers, financial services: justified P/B (residual income) instead of a cash-flow DCF."""
    from ui.icons import render_header
    st.markdown("---")
    render_header("gem", "Intrinsic Valuation (Justified P/B) — Banks & Insurers")
    if vin["note"]:
        st.warning("⚠️ " + vin["note"])
    if not vin["book_per_share"]:
        st.info("Book value per share is not available, so a P/B-based value cannot be computed.")
        return
    st.caption(f"{vin['model_note']} Rates: {vin['currency']} risk-free {vin['risk_free']:.1%} ({vin['rate_source']}), "
               f"cost of equity {vin['cost_of_equity']:.1%}, long-run growth {vin['terminal_growth']:.1%}.")
    c1, c2, c3 = st.columns(3)
    roe = c1.number_input("Sustainable ROE (%)", value=round((vin["roe"] or 0) * 100, 1), step=0.5,
                          key=f"pb_roe_{meta.get('ticker')}") / 100
    tg = c2.number_input("Long-run growth (%)", value=round(vin["terminal_growth"] * 100, 2), step=0.25,
                         key=f"pb_g_{meta.get('ticker')}") / 100
    r = c3.number_input("Cost of equity (%)", value=round(vin["cost_of_equity"] * 100, 1), step=0.5,
                        key=f"pb_r_{meta.get('ticker')}") / 100
    if r <= tg:
        st.warning("Cost of equity must exceed long-run growth.")
        return
    bv = vin["book_per_share"]
    cols = st.columns(3)
    for col, (key, d_roe, d_r, color) in zip(cols, (("bear", -0.03, +0.01, "#e74c3c"), ("base", 0.0, 0.0, "#3498db"),
                                                    ("bull", +0.03, -0.01, "#2ecc71"))):
        value = val.justified_pb_value(bv, max(roe + d_roe, 0), max(r + d_r, tg + 0.01), tg)
        mos = value / price - 1 if value else None
        body = (f"€{value:,.2f}</div><div style='color:{color}; font-weight:700;'>{mos:+.0%} vs price · "
                f"{val.valuation_verdict(mos)}") if value else "n/a</div><div style='color:#8899aa;'>ROE not above growth"
        col.markdown(f"""<div style='border-top:3px solid {color}; background:rgba(255,255,255,0.03);
            border-radius:8px; padding:10px 12px;'>
            <div style='color:#8899aa; font-size:0.7rem; text-transform:uppercase;'>{key} · ROE {max(roe + d_roe, 0):.1%} · r {max(r + d_r, tg + 0.01):.1%}</div>
            <div style='color:#fff; font-size:1.5rem; font-weight:800;'>{body}</div></div>""", unsafe_allow_html=True)
    implied = tg + (price / bv) * (r - tg)
    st.markdown(f"**Reverse model:** at €{price:,.2f} (book €{bv:,.2f}, P/B {price / bv:.2f}x) the market prices in a perpetual "
                f"ROE of **{implied:.1%}** (your base case {roe:.1%}).")
    rois = [roe - 0.04, roe - 0.02, roe, roe + 0.02, roe + 0.04]
    rates = [x for x in (r - 0.02, r - 0.01, r, r + 0.01, r + 0.02) if x > tg]
    table = pd.DataFrame({f"{x:.1%}": [val.justified_pb_value(bv, max(o, 0), x, tg) for o in rois] for x in rates},
                         index=[f"{o:.1%}" for o in rois])
    st.markdown("**Sensitivity — value per share (rows: sustainable ROE, columns: cost of equity)**")
    st.dataframe(table.style.format("€{:,.2f}", na_rep="—")
                 .map(lambda v: "color:#2ecc71" if v and v >= price * (1 + val.REQUIRED_MARGIN_OF_SAFETY)
                      else ("color:#e74c3c" if v and v < price else "")), width="stretch")


def render_valuation_section(*, meta, price, vin, relval):
    """Replaces the old one-size DCF: company-anchored inputs, scenarios, sensitivity, reverse DCF."""
    from ui.icons import render_header
    if vin.get("model") == "justified_pb":
        return _render_financial_valuation(meta, price, vin)
    st.markdown("---")
    render_header("gem", "Intrinsic Valuation (FCFE DCF) — Scenarios & Sensitivity")
    if not vin["fcfe"] or vin["fcfe"] <= 0 or not vin["shares"]:
        st.info("⚠️ " + (vin["note"] or "No positive free cash flow — a cash-flow valuation is not meaningful.")
                + " Use the relative valuation and quality pillars instead.")
        return
    if not vin["reliable"] and vin["note"]:
        st.warning("⚠️ " + vin["note"])
    st.caption(f"Free cash flow is after interest, so it is discounted at the cost of equity "
               f"(CAPM: {vin['currency']} risk-free {vin['risk_free']:.1%} [{vin['rate_source']}] + Blume-adjusted β×"
               f"{val.EQUITY_RISK_PREMIUM:.0%} ERP) and debt is not subtracted again. "
               f"Base cash flow: {vin['fcf_source']}. Starting growth anchored on: "
               f"{', '.join(vin['growth_sources'])}; it fades to the terminal rate over {val.EXPLICIT_YEARS} years.")
    c1, c2, c3 = st.columns(3)
    g = c1.number_input("Starting FCF growth (%)", value=round(vin["growth"] * 100, 1), step=1.0,
                        key=f"dcf_g_{meta.get('ticker')}") / 100
    tg = c2.number_input("Terminal growth (%)", value=round(vin["terminal_growth"] * 100, 2), step=0.25,
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


