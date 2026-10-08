"""Layer 3 — LLM risk audit, news sentiment (FinBERT) and quality radar."""
from datetime import datetime
import os

import plotly.graph_objects as go
import streamlit as st

from core.rating import quality_tier

from views.stock_analysis.layout import layer_banner

from etl.llm_parser import analyze_risk_with_llm
from etl.utils import compute_score_details
from services.ai import get_finbert_pipeline, get_unified_verdict
from ui.icons import SVG_ICONS, render_header



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
    df_deep = dd.df_deep
    earnings_surprise_full = ctx["earnings_surprise_full"]
    macro = ctx["macro"]
    meta = dd.meta
    meta_enriched = dd.meta_enriched
    p_sm = dd.sm["signal"]
    regime = ctx["regime"]
    layer_banner(3, "Risk intelligence hub", "#9b59b6", bottom=0)
    # ── RISK INTELLIGENCE HUB: Full-Width Top, then Split View ─────
    render_header("zap", "AI Investment Intelligence: Unified Risk Audit", level="####")
    st.caption("🧠 LLM narrative: it summarises the quantitative signals and news above in words. "
               "It is not an independent signal — the Decision Summary does not count it as a vote.")
    st.caption("A multi-dimensional synthesis of Qualitative (NLP News) and Quantitative (Fundamental Pillars) risk factors to provide a unified investment verdict.")

    # ── PART A (Full-Width): Audit Button + Cockpit + Conflict Banner ─
    _uv_data = st.session_state.get(f"unified_verdict_{deep_ticker}")
    
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
        run_audit_btn = st.button("Run Real-Time AI Risk Audit", type="primary", use_container_width=True)

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

                        "price":         cur_p,
                        "market_regime": regime,
                        "smart_money":   p_sm,
                        "support_s1":    round(_s1, 2),
                        "support_s2":    round(_s2, 2),
                        "resistance_r1": round(_r1, 2),
                        "resistance_r2": round(_r2, 2),
                        "stop_loss_technical": round(_stop_loss, 2),
                        "ma_20_current": round(float(df_deep["ma_20"].iloc[-1]), 2) if "ma_20" in df_deep.columns and not df_deep["ma_20"].isna().all() else "N/A",
                        "ma_50_current": round(float(df_deep["ma_50"].iloc[-1]), 2) if "ma_50" in df_deep.columns and not df_deep["ma_50"].isna().all() else "N/A",
                        "vix_current":   (macro or {}).get("VIX", {}).get("val", "N/A"),
                        "spy_trend":     (macro or {}).get("SPY", {}).get("pct", 0),
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
                        "extracted_at":      datetime.now().strftime("%H:%M:%S"),
                        "is_conflict":       _conflict,
                        "ai_score_snap":     _qs,
                        # Debug prompts for in-UI inspection
                        "cio_prompt_debug":  _cio_prompt_debug,
                        "risk_prompt_debug": llm_res.get("_prompt_debug", ""),
                    }
                    st.rerun()




    st.markdown("---")
    
    # ── AI RISK COCKPIT ───────────────────
    render_header("zap", "Real-Time AI Risk Audit", level="####")
    st.caption("Integrated NLP sentiment and Quantitative pillar breakdown from fundamental data.")
    
    if not _uv_data:
        st.markdown("""
        <div style='text-align:center; padding:40px 20px; color:#666; background:rgba(255,255,255,0.02); border-radius:10px; border:1px dashed rgba(255,255,255,0.1); margin-top:20px;'>
            <div style='font-size:2.5rem; filter:grayscale(1); opacity:0.3; margin-bottom:15px;'>🔍</div>
            <div style='font-size:0.85rem;'>Click <b>'Run Real-Time AI Risk Audit'</b> above<br>to generate the CIO Deep Dive and Risk Scorecard.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── AI Risk Cockpit: 3-Column Premium Scorecard (Full-Width) ──
    if _uv_data:
        _pulse_style = "border: 1px solid rgba(230,126,34,0.6); box-shadow: 0 0 15px rgba(230,126,34,0.15); border-left: 4px solid #e67e22;" if _is_conflict else "border: 1px solid rgba(255,255,255,0.08); border-left: 4px solid #444;"
        _q_color = quality_tier(_ai_score_snap)[1]
        _r_color = "#e74c3c" if _nlp_score >= 60 else "#f39c12" if _nlp_score >= 30 else "#2ecc71"
        _s_color = "#00ffcc" if _nlp_sent == "Positive" else "#e74c3c" if _nlp_sent in ["Negative", "Critical"] else "#8899aa"

        st.markdown(f"""
<div style='
    display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap:12px; margin:16px 0 10px 0;
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(8px);
    padding: 1px; border-radius: 12px;
    {_pulse_style}
'>
    <!-- Card 1: Quant Health -->
    <div style='padding:15px; background:rgba(255,255,255,0.01); border-radius:10px;'>
        <div style='font-size:0.65rem; color:#8899aa; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>{SVG_ICONS["chart"]} Quant Health</div>
        <div style='display:flex; align-items:baseline; gap:6px;'>
            <span style='font-size:1.4rem; font-weight:700; color:white;'>{_ai_score_snap}</span>
            <span style='font-size:0.75rem; color:#666;'>/100</span>
        </div>
        <div style='width:100%; height:3px; background:rgba(255,255,255,0.05); border-radius:2px; margin-top:8px;'>
            <div style='width:{_ai_score_snap}%; height:100%; background:{_q_color}; border-radius:2px;'></div>
        </div>
    </div>
    <!-- Card 2: News Sentiment -->
    <div style='padding:15px; background:rgba(255,255,255,0.01); border-radius:10px;'>
        <div style='font-size:0.65rem; color:#8899aa; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>{SVG_ICONS["globe"]} News Tone</div>
        <div style='display:flex; align-items:center; gap:8px;'>
            <span style='font-size:1.2rem; font-weight:600; color:{_s_color};'>{_nlp_sent}</span>
        </div>
        <div style='margin-top:8px; font-size:0.7rem; color:#666;'>Extracting sentiment from latest financial headlines</div>
    </div>
    <!-- Card 3: Risk Exposure -->
    <div style='padding:15px; background:rgba(255,255,255,0.01); border-radius:10px;'>
        <div style='font-size:0.65rem; color:#8899aa; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>{SVG_ICONS["risk"]} Risk Exposure</div>
        <div style='display:flex; align-items:baseline; gap:6px;'>
            <span style='font-size:1.4rem; font-weight:700; color:{_r_color};'>{_nlp_score}</span>
            <span style='font-size:0.75rem; color:#666;'>/100</span>
        </div>
        <div style='width:100%; height:3px; background:rgba(255,255,255,0.05); border-radius:2px; margin-top:8px;'>
            <div style='width:{_nlp_score}%; height:100%; background:{_r_color}; border-radius:2px;'></div>
        </div>
    </div>
</div>
<div style='font-size:0.62rem; color:#556677; text-align:right; margin-bottom:12px; letter-spacing:0.5px;'>
    SYNCHRONIZED AUDIT TIMESTAMP: {_audit_time} &nbsp;&middot;&nbsp; <b>DYNAMIC OVERLAY V12.1</b>
</div>
""", unsafe_allow_html=True)

        # ── Signal Conflict Banner (full-width, only when diverging) ──
        if _is_conflict:
            if _ai_score_snap >= 65:
                if _nlp_sent in ["Negative", "Critical"]:
                    _conf_dir = f"Strong Quant Health ({_ai_score_snap}/100), but News Tone is distinctly <b>{_nlp_sent.upper()}</b>. The market may penalize the stock soon."
                else:
                    _conf_dir = f"Strong Quant Health ({_ai_score_snap}/100), but Risk Exposure is elevated (<b>Red Flag: {_nlp_score}/100</b>). Monitor for potential headline shocks."
            else:
                _conf_dir = f"Weak Quant Health ({_ai_score_snap}/100), but News Tone is <b>POSITIVE</b>. Beware of a temporary, sentiment-driven rally."
            st.markdown(f"""
    <div style='display:flex; align-items:flex-start; gap:14px; margin:8px 0; padding:14px 18px; background:linear-gradient(90deg,rgba(230,126,34,0.14),rgba(231,76,60,0.08)); border:1px solid rgba(230,126,34,0.55); border-left:4px solid #e67e22; border-radius:10px;'>
<span style='font-size:1.4rem; line-height:1; padding-top:2px;'>⚠️</span>
<div>
    <div style='color:#e67e22; font-weight:900; font-size:0.75rem; text-transform:uppercase; letter-spacing:2px; margin-bottom:4px;'>⚡ Signal Conflict Detected</div>
    <div style='color:#ddd; font-size:0.85rem; line-height:1.5;'>{_conf_dir}</div>
</div>
    </div>
    """, unsafe_allow_html=True)

    # ── PART B: Full-Width Narrative & 50/50 Split (News | Radar) ──────
    st.markdown("<hr style='border:0; height:1px; background:rgba(255,255,255,0.08); margin:24px 0;'>", unsafe_allow_html=True)
    
    # Detailed Audit Narrative (Full Width, only when audit has been run)
    if _uv_data:
        st.markdown("<div style='color:#3498db; font-size:1.1rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; border-bottom:1px solid rgba(52,152,219,0.3); padding-bottom:6px;'>🧠 CIO AI Deep Dive & Scenario Analysis</div>", unsafe_allow_html=True)
        st.info(_uv_data.get("report", ""))
        _show_insights = _uv_data.get("nlp_insights", []) if isinstance(_uv_data, dict) else []
        if _show_insights:
            with st.expander("🧩 Raw Evidence: News Signals Analyzed"):
                for insight in _show_insights:
                    st.markdown(f"- {insight}")

        # ── DEBUG: Raw Prompts sent to Cohere ──────────────────────
        _cio_p   = _uv_data.get("cio_prompt_debug", "")
        _risk_p  = _uv_data.get("risk_prompt_debug", "")
        if _cio_p or _risk_p:
            with st.expander("🔍 Debug: Raw Prompts Sent to Cohere"):
                if _risk_p:
                    st.markdown("**📰 Stage 1 — Risk Audit Prompt (News Parser)**")
                    st.code(_risk_p, language="markdown")
                if _cio_p:
                    st.markdown("**🧠 Stage 2 — CIO Verdict Prompt (Unified Synthesis)**")
                    st.code(_cio_p, language="markdown")

        st.markdown("<br>", unsafe_allow_html=True)

    qual_col, quant_col = st.columns([1, 1])

    with qual_col:

        # ── NEWS FEED (Auto-load, FinBERT Sentiment) ─────────────────
        st.markdown("<div style='color:#f39c12; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-top:16px; margin-bottom:8px; border-bottom:1px solid rgba(243,156,18,0.3); padding-bottom:6px;'>📰 Market Sentiment (FinBERT)</div>", unsafe_allow_html=True)
        try:
            import feedparser
            import urllib.parse
            # Prefer searching by company name to avoid ticker clash with US stocks
            _q = urllib.parse.quote(f"{meta.get('company', deep_ticker)} stock when:7d")
            _rss_url = f"https://news.google.com/rss/search?q={_q}&hl=en-US&gl=US&ceid=US:en"
            _feed = feedparser.parse(_rss_url)
            _news_items = _feed.entries[:10]
            if _news_items:
                _pipe = get_finbert_pipeline()
                _titles = [item.get("title", "").split(" - ")[0] for item in _news_items]
                _sent_scores = []
                
                # Pre-calculate sentiment to show mood OUTSIDE popover
                if _pipe:
                    _results = _pipe(_titles)
                    for _res in _results:
                        _lbl = _res['label'].upper()
                        _sc = _res['score']
                        _sent_scores.append(_sc if _lbl == 'POSITIVE' else (-_sc if _lbl == 'NEGATIVE' else 0))
                    
                    if _sent_scores:
                        _avg_sent = sum(_sent_scores) / len(_sent_scores)
                        _mood_lbl = "🚀 BULLISH" if _avg_sent > 0.1 else ("📉 BEARISH" if _avg_sent < -0.1 else "😴 NEUTRAL")
                        _mood_color = "#2ecc71" if _avg_sent > 0.1 else ("#e74c3c" if _avg_sent < -0.1 else "#f1c40f")
                        st.markdown(f"<div style='margin-bottom:12px;padding:8px 12px;background:rgba(255,255,255,0.04);border-radius:6px;border-left:3px solid {_mood_color};font-size:0.85rem;'><b style='color:{_mood_color};'>{_mood_lbl}</b> &nbsp;·&nbsp; FinBERT: {_avg_sent:+.2f}</div>", unsafe_allow_html=True)
                
                # Popover for details
                with st.popover(f"View {len(_news_items)} Detailed Headlines", width="stretch"):
                    st.markdown("### 📰 Recent Headlines")
                    if _pipe:
                        for _i, _res in enumerate(_results):
                            _lbl = _res['label'].upper()
                            _sc = _res['score']
                            _icon = "🟢" if _lbl == 'POSITIVE' else ("🔴" if _lbl == 'NEGATIVE' else "⚪")
                            _entry = _news_items[_i]
                            with st.expander(f"{_icon} {_titles[_i][:70]}..."):
                                st.caption(f"**Source:** {_entry.get('source', {}).get('title', 'Google News')} | **Date:** {_entry.get('published', 'N/A')}")
                                st.markdown(f"[Read Article ↗]({_entry.get('link')})")
                    else:
                        for _entry in _news_items[:5]:
                            _title = _entry.get("title", "").split(" - ")[0]
                            with st.expander(f"📰 {_title[:70]}..."):
                                st.caption(f"**Date:** {_entry.get('published', 'N/A')}")
                                st.markdown(f"[Read Article ↗]({_entry.get('link')})")
            else:
                st.info("No recent news found for this ticker.")
        except Exception as _e:
            st.caption(f"⚠️ News feed unavailable: {str(_e)[:60]}")


    with quant_col:
        
        # Build radar from score_details
        _radar_sd = compute_score_details(meta_enriched)
        _radar_breakdown = _radar_sd.get("breakdown", {})
        _sector_lc = meta.get("sector", "").lower() if meta.get("sector") else ""
        _TECH_SECTORS = {
            "ai & data", "design software", "ecommerce", "fintech",
            "platform software", "semiconductor tools", "semiconductors", "technology",
            "consumer electronics", "cybersecurity", "data storage", "digital advertising",
            "enterprise hardware", "it services", "media & entertainment", "networking",
            "saas", "social media", "telecom",
        }
        _is_tech = _sector_lc in _TECH_SECTORS
        _max_pts = {
            "Valuation":       20,
            "Profitability":   30 if _is_tech else 25,
            "Fin. Health":     15,
            "Net Yield":       5  if _is_tech else 10,
            "Momentum":        15,   # v4.0: Context & Momentum cap reduced from 25 → 15
            "Analyst Est.":    10,   # v4.0: Analyst Estimates cap increased from 5 → 10
            "Rev. Growth":     5,    # v4.0: Revenue Consistency pillar (new)
        }
        _pillar_keys = {
            "Valuation":       "Valuation",
            "Profitability":   "Profitability",
            "Fin. Health":     "Financial Health",
            "Net Yield":       "Net Payout Yield",
            "Momentum":        "Context & Momentum",
            "Analyst Est.":    "Analyst Estimates",
            "Rev. Growth":     "Revenue Consistency",
        }
        _radar_labels = list(_max_pts.keys())
        _radar_vals   = [
            round((_radar_breakdown.get(_pillar_keys[k], 0) / _max_pts[k]) * 100, 1)
            for k in _radar_labels
        ]
        # Close the polygon
        _radar_labels_closed = _radar_labels + [_radar_labels[0]]
        _radar_vals_closed   = _radar_vals   + [_radar_vals[0]]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=_radar_vals_closed,
            theta=_radar_labels_closed,
            fill="toself",
            fillcolor="rgba(0,255,204,0.08)",
            line=dict(color="#00ffcc", width=2),
            name=deep_ticker
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 100], autorange=False,
                    tickfont=dict(size=9, color="#666"),
                    gridcolor="rgba(255,255,255,0.06)",
                    linecolor="rgba(255,255,255,0.08)"
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color="#bbb"),
                    gridcolor="rgba(255,255,255,0.06)"
                )
            ),
            showlegend=False,
            template="plotly_dark",
            height=290,
            margin=dict(t=20, b=10, l=40, r=40),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        with st.container():
            st.plotly_chart(fig_radar, use_container_width=True)


