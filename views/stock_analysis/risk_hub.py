"""Layer 3 — LLM risk audit, news sentiment (FinBERT) and quality radar."""
from datetime import datetime, timezone
import os

import plotly.graph_objects as go
import streamlit as st


from views.stock_analysis.layout import layer_banner

from etl.llm_parser import analyze_risk_with_llm
from services.ai import get_finbert_pipeline, get_unified_verdict
from ui.icons import render_header


def _known(v):
    """A macro value for the LLM prompt: the number, or "N/A" when missing / NaN (0 is a real value)."""
    from core.market_regime import _finite
    f = _finite(v)
    return "N/A" if f is None else f


def _saved_context(rec, ticker, snapshot):
    saved = rec.get("snapshot")
    when = rec.get("extracted_at") or rec.get("fetched_at") or "unknown"
    if isinstance(when, str) and len(when) == 8 and when.count(":") == 2:
        when = f"date unknown (recorded time {when})"
    label = f"{rec.get('ticker', ticker)} · snapshot {saved or 'unknown'} · created/fetched {when}"
    if saved is not None and snapshot is not None and saved != snapshot:
        label += " · from a previous snapshot"
    elif saved is None or snapshot is None:
        label += " · snapshot provenance unknown"
    return label


def _render_news(meta, ticker, snapshot=None):
    """Headline sentiment, fetched when the user asks (it used to run on every page open: a request to Google News, and
    the ~440 MB FinBERT model on first use). The result is kept in the session per ticker."""
    from services.news import fetch_news_sentiment
    key = f"news_sent::{ticker}"
    rec = st.session_state.get(key)
    label = "Refresh headlines" if rec else "Load headlines and sentiment"
    if st.button(label, key=f"news_btn_{ticker}",
                 help="Reads recent headlines from Google News and scores them with FinBERT. The first use downloads "
                      "the model (~440 MB). Headline sentiment is context only — it does not feed the Decision."):
        with st.spinner(f"Reading headlines for {meta.get('company', ticker)}..."):
            rec = fetch_news_sentiment(meta.get("company", ticker), ticker, get_pipe=get_finbert_pipeline)
        rec = dict(rec, ticker=ticker, snapshot=snapshot)
        st.session_state[key] = rec
    if not rec:
        st.caption("Not loaded — headlines and their sentiment are fetched only when you ask.")
        return
    st.caption("Google News / FinBERT · " + _saved_context(rec, ticker, snapshot))
    if rec["error"] and not rec["items"]:
        st.caption(f"News feed unavailable: {rec['error']}")
        return
    if not rec["items"]:
        st.info("No recent news found for this ticker.")
        return
    if rec["scored"]:
        colour = {"BULLISH": "#2ecc71", "BEARISH": "#e74c3c", "NEUTRAL": "#f1c40f"}[rec["mood"]]
        st.markdown(f"<div style='margin-bottom:12px;padding:8px 12px;background:var(--secondary-background-color, transparent);border-radius:6px;border-left:3px solid {colour};font-size:0.85rem;'><b style='color:{colour};'>{rec['mood']}</b> &nbsp;·&nbsp; FinBERT: {rec['avg']:+.2f} on {len(rec['items'])} headlines</div>", unsafe_allow_html=True)
    else:
        st.caption("Headlines are listed without a sentiment score (the model was not available).")
    st.caption(f"Fetched {rec['fetched_at']:%d %b %H:%M}. Headlines are context, not an input of the Decision.")
    with st.popover(f"View {len(rec['items'])} Detailed Headlines", width="stretch"):
        st.markdown("### Recent Headlines")
        for it in rec["items"]:
            sentiment = {"POSITIVE": "Positive", "NEGATIVE": "Negative", "NEUTRAL": "Neutral"}.get(it["label"], "Unscored")
            with st.expander(f"{sentiment}: {it['title'][:70]}..."):
                st.caption(f"**Source:** {it['source']} | **Date:** {it['published']}")
                st.markdown(f"[Read Article ↗]({it['link']})")


def render(dd, ctx):
    _dxy_pct = ctx["_dxy_pct"]
    _r1 = dd.r1
    _r2 = dd.r2
    _s1 = dd.s1
    _s2 = dd.s2
    _stop_loss = dd.stop_loss
    _vix_val = ctx["_vix_val"]
    _w52_pos = dd.w52_pos
    ai_score = dd.ai_score
    cur_p = dd.cur_p
    deep_ticker = dd.ticker
    _full = ctx["prices_full"][ctx["prices_full"]["ticker"] == dd.ticker].sort_values("date")

    def _latest(col):
        v = _full[col].dropna() if col in _full.columns else _full.iloc[0:0]
        return round(float(v.iloc[-1]), 2) if len(v) else "N/A"
    earnings_surprise_full = ctx["earnings_surprise_full"]
    macro = ctx["macro"]
    meta = dd.meta
    meta_enriched = dd.meta_enriched
    p_sm = dd.sm["signal"]
    regime = ctx["regime"]
    layer_banner(3, "Risk intelligence hub", "#9b59b6", bottom=0)
    # ── RISK INTELLIGENCE HUB: Full-Width Top, then Split View ─────
    render_header("zap", "AI news and risk review", level="####")
    st.caption("LLM narrative: it summarises the quantitative signals and news above in words. "
               "It is not an independent signal — the Decision Summary does not count it as a vote.")
    st.caption("An on-request summary of news and quantitative context. Refer to the Decision Summary for the current recommendation.")

    # ── PART A (Full-Width): Audit Button + Cockpit + Conflict Banner ─
    _uv_data = st.session_state.get(f"unified_verdict_{deep_ticker}")
    if _uv_data:
        st.caption("Saved language-model review · " + _saved_context(_uv_data, deep_ticker, ctx.get("data_version")))

    # Initialize safe defaults to prevent NameErrors in later blocks
    _nlp_score, _nlp_sent = 0, "N/A"
    _is_conflict, _ai_score_snap, _audit_time = False, 0, ""
    _unified_report = ""

    if _uv_data:
        _nlp_score     = _uv_data.get("nlp_score", 0)
        _nlp_sent      = _uv_data.get("nlp_sentiment", "Neutral")
        _is_conflict   = _uv_data.get("is_conflict", False)
        _ai_score_snap = _uv_data.get("ai_score_snap", 0)
        _audit_time    = _uv_data.get("extracted_at", "")
        _unified_report = _uv_data.get("report", "")

    # UI: Language selector + Audit Button
    col_lang, col_btn = st.columns([1, 4])
    with col_lang:
        llm_language = st.selectbox("Language", ["English", "Vietnamese"], index=0, label_visibility="collapsed")
    with col_btn:
        run_audit_btn = st.button("Run AI news and risk review", type="primary", use_container_width=True)

    if run_audit_btn:
        with st.spinner(f"Scanning news for {meta['company']}..."):
            # Build macro context string from live dashboard values
            _vix_str = f"{_vix_val:.1f}" if isinstance(_vix_val, (int, float)) else "N/A"
            _dxy_str = f"DXY {'+' if _dxy_pct >= 0 else ''}{_dxy_pct:.2f}%" if isinstance(_dxy_pct, (int, float)) else ""
            _llm_macro_ctx = f"{regime} | VIX={_vix_str} | {_dxy_str}".strip(" |")

            # Pre-compute enrichment fields needed for both quant_context and unified_metrics
            _es_for_llm = "N/A"
            if not earnings_surprise_full.empty:
                _es_tmp = earnings_surprise_full[
                    earnings_surprise_full["ticker"] == deep_ticker
                ].sort_values("quarter_date", ascending=False).head(2)
                if not _es_tmp.empty:
                    _es_parts = []
                    for _, _er in _es_tmp.iterrows():
                        _act = _er.get("eps_actual", None)
                        _est = _er.get("eps_estimate", None)
                        _pct = _er.get("surprise_pct", None)
                        _pd  = str(_er.get("quarter_date", ""))[:7]
                        if _act is not None and _est is not None:
                            _beat = "BEAT" if (_act > _est) else "MISS"
                            _es_parts.append(f"{_pd}: {_beat} ({'+' if _pct and _pct > 0 else ''}{round(_pct, 1) if _pct else 'N/A'}%)")
                    _es_for_llm = " | ".join(_es_parts) if _es_parts else "N/A"

            _debt_ebitda_for_llm = "N/A"
            try:
                _ebitda_v = float(meta.get("ebitda") or 0)
                _debt_v   = float(meta.get("total_debt") or 0)
                if _ebitda_v > 0:
                    _debt_ebitda_for_llm = round(_debt_v / _ebitda_v, 2)
            except Exception:
                pass

            _llm_quant_ctx = {
                "quant_score":       ai_score,
                "rsi":               round(float(dd.rsi), 1),   # was read from a key that never exists → always 50
                "z_score":           round(float(dd.z_score), 2),
                "w52_pos":           round(_w52_pos, 1),
                "smart_money":       p_sm,
                "debt_ebitda":       _debt_ebitda_for_llm,
                "earnings_surprise": _es_for_llm,
            }
            llm_res = analyze_risk_with_llm(deep_ticker, meta['company'], macro_context=_llm_macro_ctx, quant_context=_llm_quant_ctx, language=llm_language)
            if llm_res.get("error"):
                st.error(f"NLP Error: {llm_res['error'][:80]}")
            else:
                _cohere_key_ra = (
                    os.environ.get("COHERE_API_KEY", "")
                    or st.session_state.get("cohere_api_key", "")
                )
                if _cohere_key_ra:

                    _unified_metrics = {
                        **meta_enriched,
                        "ticker":        deep_ticker,
                        "company":       meta.get("company", deep_ticker),
                        "sector":        meta.get("sector", "N/A"),
                        "ai_score":      ai_score,
                        # the computed Decision is the ONLY rating the LLM may state (audit F-06)
                        "decision": (lambda _d, _v: None if _d is None else {
                            "stance": _d.stance, "confidence": _d.confidence, "reliable": _v.get("reliable"),
                            "base": _v.get("base"), "bear": _v.get("bear"), "bull": _v.get("bull"),
                            "expected_value": _v.get("expected_value"), "horizon_irr": _d.horizon_irr_pct,
                            "reasons": _d.reasons})(dd.extra.get("decision"), dd.vin),
                        "decision_stance": getattr(dd.extra.get("decision"), "stance", "N/A"),

                        "price":         cur_p,
                        "market_regime": regime,
                        "smart_money":   p_sm,
                        "support_s1":    round(_s1, 2),
                        "support_s2":    round(_s2, 2),
                        "resistance_r1": round(_r1, 2),
                        "resistance_r2": round(_r2, 2),
                        "stop_loss_technical": round(_stop_loss, 2),
                        # "current" = the latest bar of the full history, not the sidebar chart's end
                        "ma_20_current": _latest("ma_20"),
                        "ma_50_current": _latest("ma_50"),
                        # unknown → "N/A", a real 0 (flat SPY) stays 0
                        "vix_current":   _known(((macro or {}).get("VIX") or {}).get("val")),
                        "spy_trend":     _known(((macro or {}).get("SPY") or {}).get("pct")),
                        # --- Enrichments ---
                        "forward_pe":           float(meta.get("forward_pe") or 0) or "N/A",
                        "debt_ebitda":          _debt_ebitda_for_llm,
                        "earnings_surprise_summary": _es_for_llm,
                        "net_payout_yield_pct": float(meta.get("net_payout_yield_pct") or 0) or "N/A",
                        "w52_pos":              round(_w52_pos, 1),
                        # New: needed for FCF Yield computation in get_unified_verdict
                        "free_cashflow":        meta.get("free_cashflow"),
                        "market_cap":           meta.get("market_cap"),
                    }
                    with st.spinner("Synthesizing CIO Unified Verdict..."):
                        _unified_report, _cio_prompt_debug = get_unified_verdict(_cohere_key_ra, _unified_metrics, llm_res, language=llm_language)
                    try:
                        _qs = int(float(ai_score))
                    except (ValueError, TypeError):
                        _qs = 0
                    _ns = llm_res.get("red_flag_score", 0)
                    _nst = llm_res.get("sentiment", "Neutral")
                    _conflict = (
                        (_qs >= 65 and (_ns >= 55 or _nst in ["Negative", "Critical"])) or
                        (_qs < 45  and (_ns <= 25  and _nst == "Positive"))
                    )
                    st.session_state[f"unified_verdict_{deep_ticker}"] = {
                        "report":            _unified_report,
                        "nlp_insights":      llm_res.get("key_insights", []),
                        "nlp_sentiment":     _nst,
                        "nlp_score":         _ns,
                        "extracted_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "ticker":            deep_ticker,
                        "snapshot":          ctx.get("data_version"),
                        "is_conflict":       _conflict,
                        "ai_score_snap":     _qs,
                        # Debug prompts for in-UI inspection
                        "cio_prompt_debug":  _cio_prompt_debug,
                        "risk_prompt_debug": llm_res.get("_prompt_debug", ""),
                    }
                    st.rerun()




    st.markdown("---")

    # ── AI RISK COCKPIT ───────────────────
    render_header("zap", "AI news and risk review (on request)", level="####")
    st.caption("News tone and headline risk (NLP). The Quality score itself is in the score tiles at the top of the page.")

    if not _uv_data:
        st.markdown("""
        <div style='text-align:center; padding:40px 20px; color:inherit; background:var(--secondary-background-color, transparent); border-radius:10px; border:1px dashed rgba(255,255,255,0.1); margin-top:20px;'>
            <div style='font-size:2.5rem; filter:grayscale(1); opacity:0.3; margin-bottom:15px;'></div>
            <div style='font-size:0.85rem;'>Click <b>'Run AI news and risk review'</b> above<br>to generate the CIO Deep Dive and Risk Scorecard.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── AI Risk Cockpit: 3-Column Premium Scorecard (Full-Width) ──
    if _uv_data:
        _pulse_style = "border: 1px solid rgba(230,126,34,0.6); box-shadow: 0 0 15px rgba(230,126,34,0.15); border-left: 4px solid #e67e22;" if _is_conflict else "border: 1px solid rgba(255,255,255,0.08); border-left: 4px solid #444;"
        _r_color = "#e74c3c" if _nlp_score >= 60 else "#f39c12" if _nlp_score >= 30 else "#2ecc71"
        _s_color = "#0284C7" if _nlp_sent == "Positive" else "#e74c3c" if _nlp_sent in ["Negative", "Critical"] else "#8899aa"

        st.markdown(f"""
<div style='
    display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap:12px; margin:16px 0 10px 0;
    background:var(--secondary-background-color, transparent);
    backdrop-filter: blur(8px);
    padding: 1px; border-radius: 12px;
    {_pulse_style}
'>
    <!-- Card 2: News Sentiment -->
    <div style='padding:15px; background:var(--secondary-background-color, transparent); border-radius:10px;'>
        <div style='font-size:0.65rem; color:inherit; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>News Tone</div>
        <div style='display:flex; align-items:center; gap:8px;'>
            <span style='font-size:1.2rem; font-weight:600; color:{_s_color};'>{_nlp_sent}</span>
        </div>
        <div style='margin-top:8px; font-size:0.7rem; color:inherit;'>Sentiment in the saved review</div>
    </div>
    <!-- Card 3: Risk Exposure -->
    <div style='padding:15px; background:var(--secondary-background-color, transparent); border-radius:10px;'>
        <div style='font-size:0.65rem; color:inherit; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Risk Exposure</div>
        <div style='display:flex; align-items:baseline; gap:6px;'>
            <span style='font-size:1.4rem; font-weight:700; color:{_r_color};'>{_nlp_score}</span>
            <span style='font-size:0.75rem; color:inherit;'>/100</span>
        </div>
        <div style='width:100%; height:3px; background:var(--secondary-background-color, transparent); border-radius:2px; margin-top:8px;'>
            <div style='width:{_nlp_score}%; height:100%; background:{_r_color}; border-radius:2px;'></div>
        </div>
    </div>
</div>
<div style='font-size:0.62rem; color:inherit; text-align:right; margin-bottom:12px; letter-spacing:0.5px;'>
    Saved review: {_audit_time} &nbsp;&middot;&nbsp; <b>Stored result</b>
</div>
""", unsafe_allow_html=True)

        # ── Signal Conflict Banner (full-width, only when diverging) ──
        if _is_conflict:
            if _ai_score_snap >= 65:
                if _nlp_sent in ["Negative", "Critical"]:
                    _conf_dir = f"Strong Quality ({_ai_score_snap}/100), but News Tone is distinctly <b>{_nlp_sent.upper()}</b>. The market may penalize the stock soon."
                else:
                    _conf_dir = f"Strong Quality ({_ai_score_snap}/100), but Risk Exposure is elevated (<b>Red Flag: {_nlp_score}/100</b>). Monitor for potential headline shocks."
            else:
                _conf_dir = f"Weak Quality ({_ai_score_snap}/100), but News Tone is <b>POSITIVE</b>. Beware of a temporary, sentiment-driven rally."
            st.markdown(f"""
    <div style='display:flex; align-items:flex-start; gap:14px; margin:8px 0; padding:14px 18px; background:linear-gradient(90deg,rgba(230,126,34,0.14),rgba(231,76,60,0.08)); border:1px solid rgba(230,126,34,0.55); border-left:4px solid #e67e22; border-radius:10px;'>
<span style='font-size:1.4rem; line-height:1; padding-top:2px;'></span>
<div>
    <div style='color:#e67e22; font-weight:900; font-size:0.75rem; text-transform:uppercase; letter-spacing:2px; margin-bottom:4px;'>News vs Quality Conflict</div>
    <div style='color:inherit; font-size:0.85rem; line-height:1.5;'>{_conf_dir}</div>
</div>
    </div>
    """, unsafe_allow_html=True)

    # ── PART B: Full-Width Narrative & 50/50 Split (News | Radar) ──────
    st.markdown("<hr style='border:0; height:1px; background:var(--secondary-background-color, transparent); margin:24px 0;'>", unsafe_allow_html=True)

    # Detailed Audit Narrative (Full Width, only when audit has been run)
    if _uv_data:
        st.markdown("<div style='color:#3498db; font-size:1.1rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; border-bottom:1px solid rgba(52,152,219,0.3); padding-bottom:6px;'>CIO AI Deep Dive & Scenario Analysis</div>", unsafe_allow_html=True)
        st.info(_uv_data.get("report", ""))
        _show_insights = _uv_data.get("nlp_insights", []) if isinstance(_uv_data, dict) else []
        if _show_insights:
            with st.expander("Raw Evidence: News Signals Analyzed"):
                for insight in _show_insights:
                    st.markdown(f"- {insight}")

        # ── DEBUG: Raw Prompts sent to Cohere ──────────────────────
        _cio_p   = _uv_data.get("cio_prompt_debug", "")
        _risk_p  = _uv_data.get("risk_prompt_debug", "")
        if _cio_p or _risk_p:
            with st.expander("Debug: Raw Prompts Sent to Cohere"):
                if _risk_p:
                    st.markdown("**Stage 1 — Risk Audit Prompt (News Parser)**")
                    st.code(_risk_p, language="markdown")
                if _cio_p:
                    st.markdown("**Stage 2 — CIO Verdict Prompt (Unified Synthesis)**")
                    st.code(_cio_p, language="markdown")

        st.markdown("<br>", unsafe_allow_html=True)

    qual_col, quant_col = st.columns([1, 1])

    with qual_col:

        # ── NEWS FEED (on request: Google News + FinBERT) ────────────────────────────────────────
        st.markdown("<div style='color:#f39c12; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-top:16px; margin-bottom:8px; border-bottom:1px solid rgba(243,156,18,0.3); padding-bottom:6px;'>Market Sentiment (FinBERT)</div>", unsafe_allow_html=True)
        _render_news(meta, deep_ticker, ctx.get("data_version"))


    with quant_col:

        # Radar of the sub-scores behind Quality (and the Value / Momentum scores); unknown axes are left out
        _cmp = dd.scores.get("components") or {}
        def _avg(keys):
            v = [_cmp[k] for k in keys if _cmp.get(k) is not None]
            return sum(v) / len(v) if v else None
        _axes = {
            "Returns on capital": _avg(["q_roc", "q_roe"]),
            "Margins": _avg(["q_op_margin", "q_gross_margin", "q_fcf_margin", "q_net_margin"]),
            "Growth & stability": _avg(["q_growth_stability"]),
            "Balance sheet": _avg(["q_balance_sheet"]),
            "Cash conversion": _avg(["q_earnings_quality"]),
            "Value (vs peers)": dd.scores.get("value"),
            "Momentum (timing)": dd.scores.get("momentum"),
        }
        _axes = {k: round(v, 1) for k, v in _axes.items() if v is not None}
        _radar_labels = list(_axes)
        _radar_vals_closed = list(_axes.values()) + list(_axes.values())[:1]
        _radar_labels_closed = _radar_labels + _radar_labels[:1]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=_radar_vals_closed,
            theta=_radar_labels_closed,
            fill="toself",
            fillcolor="rgba(2,132,199,0.12)",
            line=dict(color="#0284C7", width=2),
            name=deep_ticker
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 100], autorange=False,
                    tickfont=dict(size=9)
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            showlegend=False,
            height=290,
            margin=dict(t=20, b=10, l=40, r=40),
        )
        with st.container():
            st.plotly_chart(fig_radar, use_container_width=True)
