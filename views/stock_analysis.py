"""View: 🔬 Stock Analysis"""
from datetime import date
from datetime import datetime
import json
import os

from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.levels import get_tactical_metrics
from core.rating import compute_institutional_rating
from core.smart_money import get_sm_spirit_unified_v2
from core.symbols import get_tv_symbol
from core.valuation import relative_valuation
from etl.llm_parser import analyze_risk_with_llm
from etl.utils import compute_score
from etl.utils import compute_score_details
from services.ai import get_finbert_pipeline, get_unified_verdict
from services.db import get_db_connection, load_track_record
from services.market_data import get_forex_rates
from services.user_store import load_portfolio_from_db, load_watchlist, save_watchlist
from ui.components import render_metric_row
from ui.decision_panel import render_decision_panel, render_valuation_section, valuation_inputs
from ui.icons import SVG_ICONS, render_header


def render(ctx):
    """Render the 🔬 Stock Analysis tab. ctx is the app globals() dict."""
    _action_map = ctx['_action_map']
    _dxy_pct = ctx['_dxy_pct']
    _vix_val = ctx['_vix_val']
    annual_fin = ctx['annual_fin']
    companies_full = ctx['companies_full']
    current_universe = ctx['current_universe']
    earnings_cal = ctx['earnings_cal']
    earnings_surprise_full = ctx['earnings_surprise_full']
    format_ticker = ctx['format_ticker']
    hist_fcf_full = ctx['hist_fcf_full']
    hist_fcf_q_full = ctx['hist_fcf_q_full']
    indices_list = ctx['indices_list']
    m_df = ctx['m_df']
    macro = ctx['macro']
    prices = ctx['prices']
    prices_full = ctx['prices_full']
    quarterly_fin = ctx['quarterly_fin']
    regime = ctx['regime']
    spy_prices = ctx['spy_prices']
    render_header("search", "Single Stock Deep Dive")
    if current_universe:
        # Persist selection across reruns via session_state
        # Pre-fill with active_ticker if deep_ticker_selector not yet set
        if "deep_ticker_selector" not in st.session_state:
            _default_deep = st.session_state.get("active_ticker", None)
            if _default_deep and _default_deep not in current_universe:
                _default_deep = None
            st.session_state["deep_ticker_selector"] = _default_deep

        deep_ticker = st.selectbox(
            "Select Asset to Analyze:",
            current_universe,
            placeholder="Search and Select an Asset...",
            format_func=format_ticker,
            key="deep_ticker_selector"
            # Removed index parameter because Streamlit automatically uses Session State for widgets with a key
        )
        # Sync back so active_ticker stays aligned
        if deep_ticker:
            st.session_state.active_ticker = deep_ticker
            
        if deep_ticker:
            _meta_df = companies_full[companies_full["ticker"] == deep_ticker]
            if _meta_df.empty:
                st.warning(f"⚠️ No fundamental data found for **{deep_ticker}** in the warehouse. Please run the pipeline to fetch data.", icon="⚠️")
                st.stop()
            meta = _meta_df.iloc[0]
            df_deep = prices[prices["ticker"] == deep_ticker].sort_values("date")
            if df_deep.empty:
                st.warning(f"⚠️ No price history found for **{deep_ticker}**. Please run the pipeline first.", icon="⚠️")
                st.stop()
            df_fin = annual_fin[annual_fin["ticker"] == deep_ticker].sort_values("year", ascending=False)
            
            target_p = meta.get('target_mean_price', 0)
            cur_p = df_deep['price_close'].iloc[-1]
            upside = ((target_p / cur_p) - 1) * 100 if (target_p and cur_p) else 0
            # Stale target detection: target > 3x current price → analyst target is outdated
            if abs(upside) > 100 and cur_p > 0 and abs(target_p / cur_p) > 3:
                upside = 0  # Treat as stale (reverse split, crash, FX mismatch)
            upside = max(-100, min(100, upside))
            
            # --- SUMMARY STRIP (High Density) ---
            company_name = meta.get('company', deep_ticker)
            if pd.isna(company_name): company_name = deep_ticker
            st.markdown(f"#### {company_name} ({deep_ticker}) — {meta['sector']} - €{cur_p:.2f}")
            
            # --- Pre-compute values used in the grid ---
            z_score = df_deep['price_z_score'].iloc[-1] if 'price_z_score' in df_deep.columns else 0
            if pd.isna(z_score): z_score = 0
            if z_score > 2:    z_status = "🚨 EXTREME OVERBOUGHT"
            elif z_score > 1:  z_status = "⚠️ OVEREXTENDED"
            elif z_score < -2: z_status = "💎 DEEP VALUE"
            elif z_score < -1: z_status = "🟢 UNDERVALUED"
            else:              z_status = "🔵 MEAN REVERTING"

            # --- Enrich meta with latest technicals for the scoring engine ---
            latest_tech = df_deep.iloc[-1]
            meta_enriched = meta.to_dict()
            
            # Ensure numeric safety for core fields
            for col in ['pe_ratio', 'peg_ratio', 'price_to_book', 'roe', 'fcf_margin', 'dividend_yield_pct']:
                val = meta_enriched.get(col)
                try:
                    meta_enriched[col] = float(val) if pd.notnull(val) else None
                except:
                    meta_enriched[col] = None

            meta_enriched['rsi'] = float(latest_tech.get('rsi', 50))
            meta_enriched['ma_signal'] = str(latest_tech.get('ma_signal', 'NEUTRAL'))
            meta_enriched['price_z_score'] = float(z_score)
            meta_enriched['upside_pct'] = float(upside)
            
            # ── AI SCORING ────────────────────────────────────────────────────────
            # Use m_df["Quality"] as the SINGLE canonical score source across all tabs.
            # Fallback to compute_score(meta_enriched) only if ticker is missing from screener.
            _m_quality_row = m_df[m_df["Ticker"] == deep_ticker]
            if not _m_quality_row.empty:
                ai_score = float(_m_quality_row.iloc[0]["Quality"])
            else:
                ai_score = compute_score(meta_enriched)
            ai_action = _action_map.get(deep_ticker, "HOLD / NEUTRAL")

            if ai_score >= 70:    ai_color, ai_icon = "#00ffcc", "🚀"
            elif ai_score >= 55:  ai_color, ai_icon = "#2ecc71", "✅"
            elif ai_score >= 35:  ai_color, ai_icon = "#f1c40f", "🟡"
            else:                 ai_color, ai_icon = "#e74c3c", "🔴"

            st.markdown("---")




            st.markdown("<div style='margin-top:10px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #3498db; color:#3498db; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 1: STRUCTURAL CONTEXT</div>", unsafe_allow_html=True)
            # ── TRADING CONTEXT (TOP of page) — 52-Week Range & Strategic Plan ──
            # v4.0 thresholds aligned with redistributed pillar weights
            if ai_score >= 65:    ai_color, ai_icon = "#00ffcc", "🚀"
            elif ai_score >= 50:  ai_color, ai_icon = "#2ecc71", "✅"
            elif ai_score >= 38:  ai_color, ai_icon = "#f1c40f", "🟡"
            else:                  ai_color, ai_icon = "#e74c3c", "⚠️"

            # Quality tier badge
            if ai_score >= 65:   p_qual, p_qual_c = "ELITE", "#00ffcc"
            elif ai_score >= 50: p_qual, p_qual_c = "SOLID", "#2ecc71"
            elif ai_score >= 38: p_qual, p_qual_c = "FAIR",  "#f1c40f"
            else:                p_qual, p_qual_c = "WEAK",  "#e74c3c"
            # All tactical values computed by the shared helper (identical formula to Screener)
            # Levels and the 52-week range always use the FULL price history — `df_deep` follows the
            # sidebar horizon (1M → "52-week high" was the 1-month high) and is only used for display.
            _df_levels = prices_full[prices_full["ticker"] == deep_ticker].sort_values("date")
            _tm        = get_tactical_metrics(_df_levels, cur_p, analyst_target=target_p)
            _kinds     = _tm.get("kinds", {})
            _s1        = _tm["s1"]
            _s2        = _tm["s2"]
            _s3        = _tm["s3"]
            _r1        = _tm["r1"]
            _r2        = _tm["r2"]
            _r3        = _tm["r3"]
            _rsi_val   = _tm["rsi"]
            _ma_sig    = str(latest_tech.get("ma_signal", meta.get("ma_signal", "NEUTRAL")))
            _w52_pos   = _tm["w52_pos"]
            _w52_hi    = _tm["w52_hi"]
            _w52_lo    = _tm["w52_lo"]
            _w52_zone  = "Near Low" if _w52_pos < 20 else ("Near High" if _w52_pos > 80 else "Mid-Range")
            _stop_loss = _tm["stop_loss"]

            # TP1: honour AI Ensemble target if already computed, otherwise use standard formula
            _global_ai_target = st.session_state.get(f"ai_target_for_de_{deep_ticker}")
            _tp1 = float(_global_ai_target) if _global_ai_target is not None else _tm["tp1"]
            _tp2 = max(target_p, _tm["tp2"]) if target_p > 0 else _tm["tp2"]

            # ── DECISION SUMMARY (valuation → expected return, risk, size, sell rules) ──
            _vin = valuation_inputs(meta, cur_p, hist_fcf_full, deep_ticker, macro, annual_fin)
            _relval = relative_valuation(companies_full, deep_ticker)
            _next_er = None
            _er = earnings_cal[earnings_cal["ticker"] == deep_ticker] if not earnings_cal.empty else earnings_cal
            if not _er.empty:
                _future = pd.to_datetime(_er["earnings_date"]).dt.date
                _future = _future[_future >= date.today()]
                _next_er = _future.min() if not _future.empty else None

            def _holdings_value():
                _pf = load_portfolio_from_db()
                _last = prices_full.sort_values("date").groupby("ticker")["price_close"].last()
                return {t: v.get("shares", 0) * float(_last.get(t, 0)) for t, v in _pf.items()}

            render_decision_panel(
                ticker=deep_ticker, meta=meta, price=float(cur_p),
                price_date=pd.to_datetime(df_deep["date"].iloc[-1]).date(),
                stop_loss=_stop_loss, vin=_vin, relval=_relval,
                missing=compute_score_details(meta_enriched)["missing"],
                next_earnings=_next_er, quality=ai_score,
                snapshots=load_track_record(), prices=prices_full,
                holdings_loader=_holdings_value, companies=companies_full)

            # 52-Week Position Meter
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1);
                        border-radius:10px; padding:14px 20px; margin-bottom:10px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span style='color:#999; font-size:0.75rem; font-weight:600; text-transform:uppercase;'>52-Week Range</span>
                    <span style='color:#fff; font-size:0.85rem; font-weight:700;'>{_w52_zone} &nbsp;|&nbsp; Position: {_w52_pos:.0f}%</span>
                </div>
                <div style='display:flex; align-items:center; gap:10px;'>
                    <span style='color:#e74c3c; font-size:0.85rem; white-space:nowrap;'>Low: €{_w52_lo:.2f}</span>
                    <div style='flex:1; background:rgba(255,255,255,0.1); border-radius:4px; height:10px; position:relative;'>
                        <div style='width:{_w52_pos:.1f}%; height:100%; background:linear-gradient(90deg,#e74c3c,#f1c40f,#2ecc71); border-radius:4px;'></div>
                        <div style='position:absolute; top:-3px; left:{_w52_pos:.1f}%; transform:translateX(-50%);
                                    width:14px; height:14px; background:#fff; border-radius:50%; border:2px solid #3498db;'></div>
                    </div>
                    <span style='color:#2ecc71; font-size:0.85rem; white-space:nowrap;'>High: €{_w52_hi:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


            st.markdown("<div style='margin-top:35px; margin-bottom:-10px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #e67e22; color:#e67e22; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 2: TACTICAL EXECUTION MATRIX</div>", unsafe_allow_html=True)
            # ── UNIFIED DECISION SUPPORT MATRIX (ACTION LAYER) ────────────────
            render_header("activity", "360° Signal Matrix (inputs to the Decision Summary above)")
            
            # PILLAR 1: TECHNICAL TREND
            if _ma_sig == "BULLISH" and _rsi_val < 65:
                p_trend, p_trend_c = "BULLISH", "#2ecc71"
            elif _ma_sig == "BULLISH" and _rsi_val >= 65:
                p_trend, p_trend_c = "EXTENDED", "#f1c40f"
            elif _ma_sig == "BEARISH" and _rsi_val <= 35:
                p_trend, p_trend_c = "OVERSOLD", "#f1c40f"
            else:
                p_trend, p_trend_c = "BEARISH", "#e74c3c"
                
            # PILLAR 2: QUALITY
            if ai_score >= 70: p_qual, p_qual_c = "ELITE", "#00ffcc"
            elif ai_score >= 55: p_qual, p_qual_c = "SOLID", "#2ecc71"
            elif ai_score >= 40: p_qual, p_qual_c = "FAIR", "#f1c40f"
            else: p_qual, p_qual_c = "POOR", "#e74c3c"
            
            # PILLAR 3: VALUATION — Sector-Aware Multi-factor (PEG + P/E + upside)
            _peg_v = float(meta_enriched.get("peg_ratio") or 0)
            _pe_v  = float(meta_enriched.get("pe_ratio")  or 0)
            
            # 🏆 EXPERT: Sector-Specific Dynamic Thresholds
            _sector_str = str(meta.get("sector", "")).lower()
            _is_growth  = any(s in _sector_str for s in ["tech", "semi", "software", "cloud", "ai", "comm"])
            
            # Dynamic cutoff levels (Growth stocks carry premium multiples)
            _pe_cheap_limit = 28.0 if _is_growth else 18.0
            _pe_expensive_limit = 65.0 if _is_growth else 42.0
            _peg_expensive_limit = 3.5 if _is_growth else 2.5
            _peg_cheap_limit = 1.2 if _is_growth else 0.8

            _val_expensive  = (_pe_v > _pe_expensive_limit and _pe_v > 0) or (_peg_v > _peg_expensive_limit and _peg_v > 0)
            _val_cheap      = (upside > 15) and (_peg_v < _peg_cheap_limit or _pe_v < _pe_cheap_limit) and _pe_v > 0
            _val_premium_ok = (upside > 8) and (ai_score >= 60) and (_peg_v < 2.8 or _pe_v < (55 if _is_growth else 35))
            _val_compounder = (upside > 5) and (ai_score >= 50) and (not _val_expensive)
            _val_fair       = (upside > 0) and (not _val_expensive)

            if _val_cheap:
                p_val, p_val_c = "UNDERVALUED", "#2ecc71"
            elif _val_premium_ok:
                p_val, p_val_c = "PREMIUM / JUSTIFIED", "#3498db"
            elif _val_compounder:
                p_val, p_val_c = "FAIR FOR QUALITY", "#3498db"
            elif _val_fair:
                p_val, p_val_c = "FAIR VS SECTOR", "#f1c40f"
            elif _val_expensive:
                p_val, p_val_c = "EXPENSIVE / PREMIUM", "#e67e22"
            elif _pe_v < 0:
                p_val, p_val_c = "SPECULATIVE / RISK", "#e74c3c"
            else:
                p_val, p_val_c = "AVERAGE", "#95a5a6"
            
            # PILLAR 4: RISK
            if _w52_pos > 80: p_risk, p_risk_c = "ELEVATED", "#e74c3c"
            elif _w52_pos < 20: p_risk, p_risk_c = "LOW RISK", "#2ecc71"
            else: p_risk, p_risk_c = "MODERATE", "#f1c40f"
            
            # PILLAR 5: CONVICTION (rr_score = raw r1 target, same as Screener)
            _rr = _tm["rr_score"]
            if _rr > 2.5: p_conv, p_conv_c = "HIGH", "#00ffcc"
            elif _rr > 1.2: p_conv, p_conv_c = "MEDIUM", "#2ecc71"
            else: p_conv, p_conv_c = "LOW", "#e74c3c"
            
            # PILLAR 6: SMART MONEY (Unified v6.0 - Always use full history for path-dependent OBV)
            df_sm_raw = prices_full[prices_full["ticker"] == deep_ticker]
            sm_result = get_sm_spirit_unified_v2(df_sm_raw, sector=str(meta.get("sector", "Unknown")))
            p_sm = sm_result["signal"]
            p_sm_strength = sm_result["strength"]
            p_sm_layer = sm_result["layer"]
            p_sm_c = "#2ecc71" if p_sm == "ACCUMULATION" else "#e74c3c"
            
            # ── MASTER POSITIONING LOGIC ──────────────────────────────────────
            # We still call compute_institutional_rating to derive pillar colours
            # (p_trend_c, p_val_c, etc.) for the UI matrix.
            # BUT the final Action label is ALWAYS read from _action_map (m_df),
            # which is the Single Source of Truth — identical to the Screener tab.
            _rating = compute_institutional_rating(
                ai_score   = ai_score,
                ma_sig     = _ma_sig,
                latest_rsi = _rsi_val,
                upside     = float(upside),
                pe_v       = float(meta_enriched.get("forward_pe") or meta_enriched.get("pe_ratio") or 0),
                peg_v      = float(meta_enriched.get("peg_ratio") or 0),
                sector     = str(meta.get("sector", "")),
                w52_pos    = _w52_pos,
                rr         = _tm["rr_score"],   # scoring uses raw r1 target
                sm_status  = p_sm,
                sm_strength = p_sm_strength,
                sm_layer   = p_sm_layer
            )
            # Action label: Always use the fresh rating calculated above to reflect the latest logic
            act_str = _rating["action_label"]
            # Colour is derived from the canonical label — NOT from the local engine score
            _colour_map = {
                "STRONG BUY":          "#00ffcc",
                "BUY / ACCUMULATE":    "#2ecc71",
                "HOLD / NEUTRAL":      "#3498db",
                "REDUCE / UNDERPERFORM": "#e67e22",
                "SELL / AVOID":        "#e74c3c",
            }
            act_color = _colour_map.get(act_str, _rating["action_color"])
            # Override only p_val_c (valuation is always from the rating engine)
            # p_trend_c is NOT overridden here — it is set correctly at lines 2920-2927
            p_val_c = _rating["p_val_c"]

            # ── Action description text (context-aware) ──────────────────────
            if act_str == "STRONG BUY":
                act_desc = f"Optimal alignment of quantitative pillars. High structural conviction. Ideal entry zone between €{_s1:.2f} and €{cur_p:.2f}."
            elif act_str == "BUY / ACCUMULATE":
                act_desc = f"Institutional-grade asset consolidating. Momentum is neutralizing. Support holds near €{_s1:.2f}."
            elif act_str == "SELL / AVOID" and p_trend_c == "#e74c3c" and _rating["p_val_c"] == "#e74c3c":
                act_desc = "Negative trend synergy with poor valuation metrics. Risk/Reward is heavily skewed to the downside."
            elif act_str == "SELL / AVOID":
                act_desc = "Significant fundamental and technical breakdown detected. Focus on capital preservation."
            elif act_str == "HOLD / NEUTRAL" and _rating["p_qual_c"] in ["#2ecc71", "#00ffcc"]:
                act_desc = "Elite asset currently overextended or expensive. Wait for a healthy structural pullback before deployment."
            elif act_str == "REDUCE / UNDERPERFORM":
                if _rsi_val > 70:
                    act_desc = f"Locally overbought (RSI: {_rsi_val:.1f}). Momentum is peaking. Tactical risk is elevated. Consider locking profits."
                elif p_sm_c == "#e74c3c":
                    act_desc = f"Institutional Distribution detected (Smart Money is exiting). Despite low RSI ({_rsi_val:.1f}), the flow is negative. Avoid catching falling knives."
                else:
                    act_desc = "Technical structure weakening. Momentum divergence detected. Reduce exposure to preserve capital."
            else:
                act_desc = "Mixed signals across pillars. System lacks execution conviction. Monitor for structural breakout or mean reversion."

            def hex_to_rgb(hex_str):
                h = hex_str.lstrip('#')
                return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"

            bg_rgb = hex_to_rgb(act_color)


            # --- R/R DIAGNOSTIC EXPLAINER (Dynamic for all Risk/Reward states) ---
            _rr_section_html = ""
            _risk_gap  = cur_p - _stop_loss
            _risk_pct  = (_risk_gap / cur_p * 100) if cur_p > 0 else 0
            _rwrd_gap  = _tp1 - cur_p
            _rwrd_pct  = (_rwrd_gap / cur_p * 100) if cur_p > 0 else 0

            if p_conv_c == "#e74c3c":  # R/R is LOW  (<1.2x)
                _b1 = (f"Risk/Reward is {_rr:.2f}x — the stop loss at \u20ac{_stop_loss:.2f} risks \u20ac{_risk_gap:.2f} ({_risk_pct:.1f}%) while TP1 at \u20ac{_tp1:.2f} only offers \u20ac{_rwrd_gap:.2f} ({_rwrd_pct:.1f}%) upside. A ratio below 1.2x is considered unfavorable for new entries.")
                if _rsi_val > 65: _b2 = (f"RSI is elevated at {_rsi_val:.1f} — overbought momentum increases the probability of a pullback before reaching TP1, reducing effective reward potential.")
                elif _w52_pos > 75: _b2 = (f"Price is at {_w52_pos:.0f}% of its 52-week range — proximity to annual highs compresses remaining upside and increases downside risk if resistance holds.")
                elif _pe_v > 35 and _pe_v > 0: _b2 = (f"P/E of {_pe_v:.1f}x signals premium valuation — limited margin of safety amplifies the downside if earnings disappoint, worsening the R/R profile.")
                else: _b2 = (f"Technical structure shows limited near-term catalysts: current price \u20ac{cur_p:.2f} is close to TP1, suggesting most of the move may already be priced in.")
                _b3 = (f"To improve the setup, consider waiting for a pullback toward \u20ac{(_s1 * 0.97):.2f}\u2013\u20ac{_s1:.2f} (support zone), which would widen the reward-to-risk ratio above 2x.")
                _bullet_items = "".join([f"<li style='margin-bottom:7px; line-height:1.55;'>{b}</li>" for b in [_b1, _b2, _b3]])
                _rr_section_html = f"<div style='margin-top:14px; padding:14px 16px; background:rgba(231,76,60,0.07); border:1px solid rgba(231,76,60,0.25); border-radius:8px;'><div style='font-size:0.7em; color:#e74c3c; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;'>Why Risk/Reward is LOW</div><ul style='margin:0; padding-left:18px; color:#ccc; font-size:0.82em;'>{_bullet_items}</ul></div>"
            elif p_conv_c == "#2ecc71":  # R/R is MEDIUM (1.2x – 2.5x)
                _b1 = (f"Risk/Reward is {_rr:.2f}x — acceptable but not yet asymmetric. The setup risks \u20ac{_risk_gap:.2f} ({_risk_pct:.1f}%) for a potential gain of \u20ac{_rwrd_gap:.2f} ({_rwrd_pct:.1f}%). A ratio between 1.2x and 2.5x supports a partial position, not full deployment.")
                if _rsi_val < 45 and _w52_pos < 50: _b2 = (f"Supportive setup: RSI at {_rsi_val:.1f} (non-overbought) and price at {_w52_pos:.0f}% of its 52-week range reduces near-term downside pressure and leaves room for momentum to develop toward TP1.")
                elif ai_score >= 60: _b2 = (f"Quality score of {ai_score:.0f}/100 underpins the thesis — a fundamentally strong asset with acceptable technicals. The R/R is constrained by entry timing rather than structural weakness.")
                else: _b2 = (f"The setup is balanced: price at {_w52_pos:.0f}% of its 52-week range with RSI at {_rsi_val:.1f}. No extreme conditions exist to strongly favour bulls or bears — the market is in a discovery phase.")
                _b3 = (f"Execution tip: initiate a 50% position near current levels and reserve the remaining allocation for a pullback toward \u20ac{(_s1 * 0.98):.2f}\u2013\u20ac{_s1:.2f}, which would push the blended R/R above 2x.")
                _bullet_items = "".join([f"<li style='margin-bottom:7px; line-height:1.55;'>{b}</li>" for b in [_b1, _b2, _b3]])
                _rr_section_html = f"<div style='margin-top:14px; padding:14px 16px; background:rgba(46,204,113,0.07); border:1px solid rgba(46,204,113,0.25); border-radius:8px;'><div style='font-size:0.7em; color:#2ecc71; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;'>Why Risk/Reward is MEDIUM</div><ul style='margin:0; padding-left:18px; color:#ccc; font-size:0.82em;'>{_bullet_items}</ul></div>"
            else:  # R/R is HIGH (>2.5x)
                _b1 = (f"Risk/Reward is {_rr:.2f}x — strongly asymmetric. TP1 at \u20ac{_tp1:.2f} offers \u20ac{_rwrd_gap:.2f} ({_rwrd_pct:.1f}%) upside while the stop at \u20ac{_stop_loss:.2f} limits downside to \u20ac{_risk_gap:.2f} ({_risk_pct:.1f}%). A ratio above 2.5x represents a high-conviction, institutionally sound entry.")
                if _w52_pos < 25: _b2 = (f"Price is at {_w52_pos:.0f}% of its 52-week range — near structural lows with significant runway to the upside.")
                elif _rsi_val < 40: _b2 = (f"RSI at {_rsi_val:.1f} signals oversold conditions — historically, mean-reversion from these levels boosts the probability of reaching TP1.")
                else: _b2 = (f"The stop loss at \u20ac{_stop_loss:.2f} is anchored near key technical support, structurally minimizing the risk side while the reward window to TP1 at \u20ac{_tp1:.2f} remains wide open.")
                _b3 = (f"Execution: this setup supports full position sizing. Consider entering between \u20ac{_s1:.2f}\u2013\u20ac{cur_p:.2f} with a hard stop at \u20ac{_stop_loss:.2f}. If price breaks above \u20ac{_tp1:.2f}, reassess TP2 at \u20ac{_tp2:.2f}.")
                _bullet_items = "".join([f"<li style='margin-bottom:7px; line-height:1.55;'>{b}</li>" for b in [_b1, _b2, _b3]])
                _rr_section_html = f"<div style='margin-top:14px; padding:14px 16px; background:rgba(0,255,204,0.06); border:1px solid rgba(0,255,204,0.25); border-radius:8px;'><div style='font-size:0.7em; color:#00ffcc; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;'>Why Risk/Reward is HIGH</div><ul style='margin:0; padding-left:18px; color:#ccc; font-size:0.82em;'>{_bullet_items}</ul></div>"

            # RENDER UNIFIED UI MATRIX
            st.markdown(f"""
            <div style='background:rgba(10,15,25,0.6); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:20px; margin-bottom:25px;'>
                <div style='display:flex; justify-content:space-between; text-align:center; margin-bottom:20px; flex-wrap:wrap; gap:10px;'>
                    <div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; border-top:3px solid {p_trend_c}; min-width:14%'>
                        <div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>Technical Trend</div>
                        <div style='font-weight:900; font-size:0.9em; color:{p_trend_c}; margin-top:8px;'>{p_trend}</div>
                    </div>
                    <div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; border-top:3px solid {p_qual_c}; min-width:14%'>
                        <div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>Quality</div>
                        <div style='font-weight:900; font-size:0.9em; color:{p_qual_c}; margin-top:8px;'>{p_qual}</div>
                    </div>
                    <div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; border-top:3px solid {p_val_c}; min-width:14%'>
                        <div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>Valuation</div>
                        <div style='font-weight:900; font-size:0.9em; color:{p_val_c}; margin-top:8px;'>{p_val}</div>
                    </div>
                    <div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; border-top:3px solid {_rating["p_sm_c"]}; min-width:14%'>
                        <div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>Smart Money</div>
                        <div style='font-weight:900; font-size:0.9em; color:{_rating["p_sm_c"]}; margin-top:8px;'>{_rating["sm_label"]}</div>
                        <div style='font-size:0.65em; color:#888; margin-top:4px;'>Strength: {p_sm_strength}/100</div>
                        <div style='font-size:0.6em; color:#666; margin-top:2px;'>Points: {_rating["sm_points"]:.2f}</div>
                    </div>
                    <div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; border-top:3px solid {p_risk_c}; min-width:14%'>
                        <div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>Risk (52w)</div>
                        <div style='font-weight:900; font-size:0.9em; color:{p_risk_c}; margin-top:8px;'>{p_risk}</div>
                    </div>
                    <div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; border-top:3px solid {p_conv_c}; min-width:14%'>
                        <div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>Risk/Reward</div>
                        <div style='font-weight:900; font-size:0.9em; color:{p_conv_c}; margin-top:8px;'>{p_conv}</div>
                    </div>
                </div>
                <div style='background:rgba({bg_rgb},0.12); border-left:6px solid {act_color}; padding:20px; border-radius:8px; box-shadow:0 4px 15px rgba(0,0,0,0.3);'>
                    <div style='font-size:0.75em; color:#bbb; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px;'>Signal (technical + quality) — the recommendation is the Decision Summary</div>
                    <div style='font-size:1.6em; font-weight:900; color:{act_color}; margin-bottom:8px; text-shadow: 0px 2px 10px rgba({bg_rgb}, 0.5);'>{act_str}</div>
                    <div style='color:#e0e0e0; font-size:1.0em; line-height:1.5; margin-bottom:15px;'>{act_desc}</div>
                    <hr style='border:0; height:1px; background:linear-gradient(90deg, rgba(255,255,255,0.15), transparent); margin-bottom:15px;'>
                    <div style='display:flex; justify-content:space-between; font-family:"Courier New", monospace; font-size:0.95em; background:rgba(0,0,0,0.4); padding:12px; border-radius:6px;'>
                        <span style='color:#2ecc71;'><b>ENTRY:</b> €{_s1:.2f} ➔ €{cur_p:.2f}</span>
                        <span style='color:#e74c3c;'><b>STOP LOSS:</b> €{_stop_loss:.2f}</span>
                        <span style='color:#3498db;'><b>TARGET:</b> €{_tp1:.2f} (R/R: {_rr:.1f}x)</span>
                    </div>
                    {_rr_section_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:35px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #9b59b6; color:#9b59b6; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 3: RISK INTELLIGENCE HUB</div>", unsafe_allow_html=True)
            # ── RISK INTELLIGENCE HUB: Full-Width Top, then Split View ─────
            render_header("zap", "AI Investment Intelligence: Unified Risk Audit", level="####")
            st.caption("🧠 LLM narrative: it summarises the quantitative signals and news above in words. "
                       "It is not an independent signal — the Decision Summary does not count it as a vote.")
            st.caption("A multi-dimensional synthesis of Qualitative (NLP News) and Quantitative (Fundamental Pillars) risk factors to provide a unified investment verdict.")

            # ── PART A (Full-Width): Audit Button + Cockpit + Conflict Banner ─
            _uv_data = st.session_state.get(f"unified_verdict_{deep_ticker}")
            
            # Initialize safe defaults to prevent NameErrors in later blocks
            _nlp_score, _nlp_sent, _nlp_insights = 0, "N/A", []
            _is_conflict, _ai_score_snap, _audit_time = False, 0, ""
            _unified_report = ""
            
            if _uv_data:
                _nlp_score     = _uv_data.get("nlp_score", 0)
                _nlp_sent      = _uv_data.get("nlp_sentiment", "Neutral")
                _nlp_insights  = _uv_data.get("nlp_insights", [])
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
                        "rsi":               round(float(meta_enriched.get("RSI (14)", 50)), 1),
                        "z_score":           round(float(meta_enriched.get("Z-Score", 0)), 2),
                        "w52_pos":           round(_w52_pos, 1),
                        "smart_money":       p_sm,
                        "debt_ebitda":       _debt_ebitda_for_llm,
                        "earnings_surprise": _es_for_llm,
                    }
                    llm_res = analyze_risk_with_llm(deep_ticker, meta['company'], macro_context=_llm_macro_ctx, quant_context=_llm_quant_ctx, language=llm_language)
                    if llm_res.get("error"):
                        st.error(f"NLP Error: {llm_res['error'][:80]}")
                    else:
                        nlp_score     = llm_res.get("red_flag_score", 0)
                        nlp_sentiment = llm_res.get("sentiment", "Neutral")
                        nlp_reco      = llm_res.get("recommendation", "N/A")
                        nlp_insights  = llm_res.get("key_insights", [])
                        nlp_category  = llm_res.get("risk_category", "None")
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
                                "vix_current":   macro.get("VIX", {}).get("val", "N/A") if 'macro' in locals() else "N/A",
                                "spy_trend":     macro.get("SPY", {}).get("pct", 0) if 'macro' in locals() else 0,
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
                _q_color = "#00ffcc" if _ai_score_snap >= 65 else "#f1c40f" if _ai_score_snap >= 50 else "#e74c3c"
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
                    
                    # Score summary under radar
                    _q_score = ai_score
                    _q_pct   = f"{_q_score}/100"
                    _q_color = ai_color
                    st.markdown(f"""
                    
                    """, unsafe_allow_html=True)



            st.markdown("---")
            st.markdown("<div style='margin-top:35px; margin-bottom:15px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #e74c3c; color:#e74c3c; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 4: DEEP DIAGNOSTICS & RAW DATA</div>", unsafe_allow_html=True)
            render_header("activity", "Diagnostic Metrics Portfolio")

            _card_style = "background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 4px 4px 4px;margin-bottom:4px;"
            _header_style = "color:#aabbcc;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;padding:0 8px 6px 8px;"
            
            # Missing values must read "N/A" — not "nan%", and not "0.0%" painted red as if it were a fact
            def _num(v):
                try:
                    v = float(v)
                    return v if v == v else None
                except (TypeError, ValueError):
                    return None

            def _txt(v, fmt):
                return "N/A" if v is None else fmt.format(v)

            with st.container():
                kcol1, kcol2, kcol3, kcol4, kcol5, kcol6 = st.columns(6)

                with kcol1:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Valuation & Size</div>", unsafe_allow_html=True)
                    _mc = meta.get('market_cap'); m_cap = float(_mc) if not pd.isna(_mc) and _mc else 0.0
                    if m_cap >= 1e12: m_cap_txt = f"€{m_cap/1e12:.2f}T"
                    elif m_cap >= 1e9: m_cap_txt = f"€{m_cap/1e9:.1f}B"
                    else: m_cap_txt = f"€{m_cap/1e6:.0f}M"
                    
                    render_metric_row("Market Cap", m_cap_txt)
                    fwd_pe_txt = f"Fwd: {meta.get('forward_pe', 0):.1f}" if pd.notnull(meta.get('forward_pe')) and meta.get('forward_pe', 0) > 0 else ""
                    pe_val = f"{meta['pe_ratio']:.1f}" if pd.notnull(meta['pe_ratio']) else "N/A"
                    render_metric_row("P/E", pe_val, delta=fwd_pe_txt)
                    
                    peg_raw = meta.get('peg_ratio', 0)
                    peg_col = "#2ecc71" if pd.notnull(peg_raw) and 0 < peg_raw <= 1.0 else ("#e74c3c" if pd.notnull(peg_raw) and peg_raw > 2.0 else None)
                    render_metric_row("PEG", f"{peg_raw:.2f}" if pd.notnull(peg_raw) else "N/A", value_color=peg_col)
                    
                    ev_raw = _num(meta.get('ev_to_ebitda'))
                    ev_col = None if ev_raw is None else ("#2ecc71" if 0 < ev_raw <= 10 else ("#e74c3c" if ev_raw > 20 else None))
                    render_metric_row("EV/EBITDA", _txt(ev_raw, "{:.2f}x"), value_color=ev_col, help_text="🟢 <10x (Value) | 🔴 >20x (Expensive)")

                    ps_val = _num(meta.get('price_to_sales'))
                    ps_col = None if ps_val is None else ("#2ecc71" if 0 < ps_val <= 2 else ("#e74c3c" if ps_val > 10 else None))
                    render_metric_row("Price/Sales", _txt(ps_val, "{:.2f}x"), value_color=ps_col, help_text="🟢 < 2x (Cheap) | 🔴 > 10x (Expensive)")
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol2:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Profit & Returns</div>", unsafe_allow_html=True)
                    
                    div_val = _num(meta.get('dividend_yield_pct'))
                    div_col = "#2ecc71" if div_val is not None and div_val > 4 else None
                    render_metric_row("Div Yield", _txt(div_val, "{:.2f}%"), value_color=div_col, help_text="🟢 > 4% (High Yielding)")

                    # Net Payout = Div + Buybacks
                    net_payout = _num(meta.get('net_payout_yield_pct'))
                    bb_yield   = _num(meta.get('buyback_yield_pct'))
                    render_metric_row("Net Payout", _txt(net_payout, "{:.2f}%"),
                                      delta=f"BB: {bb_yield:.1f}%" if bb_yield is not None else None)

                    _roe = _num(meta.get('roe'))
                    roe_raw = _roe * 100 if _roe is not None else None
                    roe_col = None if roe_raw is None else ("#2ecc71" if roe_raw >= 15 else ("#e74c3c" if roe_raw < 5 else None))
                    render_metric_row("ROE", _txt(roe_raw, "{:.1f}%"), value_color=roe_col, help_text="🟢 > 15% (Strong Profitability) | 🔴 < 5% (Poor)")

                    _gm = _num(meta.get('gross_margin'))
                    gm_val = _gm * 100 if _gm is not None else None
                    gm_col = None if gm_val is None else ("#2ecc71" if gm_val >= 40 else ("#e74c3c" if gm_val < 10 else None))
                    render_metric_row("Gross Margin", _txt(gm_val, "{:.1f}%"), value_color=gm_col, help_text="🟢 > 40% (Wide Moat) | 🔴 < 10% (Thin Margin)")

                    _om = _num(meta.get('operating_margin'))
                    op_val = _om * 100 if _om is not None else None
                    op_col = None if op_val is None else ("#2ecc71" if op_val >= 15 else ("#e74c3c" if op_val < 5 else None))
                    render_metric_row("Op Margin", _txt(op_val, "{:.1f}%"), value_color=op_col)

                    fcf_m = _num(meta.get('fcf_margin'))
                    render_metric_row("FCF Margin", _txt(fcf_m, "{:.1f}%"), value_color="#2ecc71" if (fcf_m or 0) > 15 else None)

                    _rg = _num(meta.get('revenue_growth'))
                    rev_growth = _rg * 100 if _rg is not None else None
                    rev_col = None if rev_growth is None else ("#2ecc71" if rev_growth > 20 else ("#e74c3c" if rev_growth < 0 else None))
                    render_metric_row("Rev Growth (last qtr YoY)", _txt(rev_growth, "{:.1f}%"), value_color=rev_col,
                                      help_text="Latest quarter vs the same quarter a year ago (Yahoo) — noisy for cyclical businesses")
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol3:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Solvency</div>", unsafe_allow_html=True)
                    debt_eq_raw = meta.get('debt_to_equity', 0)
                    if pd.notnull(debt_eq_raw) and debt_eq_raw != 0:
                        debt_eq_txt = f"{(debt_eq_raw / 100.0):.2f}x"
                    else:
                        _tot_debt = meta.get('total_debt', 0)
                        debt_eq_txt = "N/A (Neg Equity)" if pd.notnull(_tot_debt) and _tot_debt > 0 else "0.00x"
                    
                    curr_rat  = meta.get('current_ratio', 0)
                    if pd.isna(curr_rat): curr_rat = 0
                    quick_rat = meta.get('quick_ratio', 0)
                    if pd.isna(quick_rat): quick_rat = 0
                    
                    # Liquidity Status Colors
                    c_col = "#2ecc71" if curr_rat > 1.5 else ("#e74c3c" if curr_rat < 1.0 else "#f39c12")
                    q_col = "#2ecc71" if quick_rat > 1.0 else ("#e74c3c" if quick_rat < 0.7 else "#f39c12")
                    
                    _ebitda_val = 0.0
                    try: _ebitda_val = float(meta.get('ebitda', 0) or 0)
                    except (TypeError, ValueError): pass
                    _debt_val = 0.0
                    try: _debt_val = float(meta.get('total_debt', 0) or 0)
                    except (TypeError, ValueError): pass
                    debt_ebitda = (_debt_val / _ebitda_val) if _ebitda_val > 0 else 0
                    
                    de_col = "#2ecc71" if 0 < debt_ebitda < 3.0 else ("#e74c3c" if debt_ebitda >= 5.0 else "#f39c12")

                    render_metric_row("Debt/Eq", debt_eq_txt)
                    render_metric_row("Debt/EBITDA", f"{debt_ebitda:.2f}x" if debt_ebitda > 0 else "N/A", value_color=de_col)
                    render_metric_row("Current Ratio", f"{curr_rat:.2f}", value_color=c_col)
                    render_metric_row("Quick Ratio",   f"{quick_rat:.2f}", value_color=q_col)
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol4:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Risk & Volume</div>", unsafe_allow_html=True)
                    beta_val = meta.get('beta', 1.0)
                    if pd.notnull(beta_val) and beta_val != 0:
                        beta_col = "#e74c3c" if beta_val > 1.5 else ("#3498db" if beta_val < 0.8 else None)
                        render_metric_row("Beta", f"{beta_val:.2f}", value_color=beta_col, help_text="🔴 > 1.5 (High Volatility) | 🔵 < 0.8 (Defensive)")
                    else:
                        render_metric_row("Beta", "N/A")
                    
                    _io = _num(meta.get('inst_ownership'))
                    inst = _io * 100 if _io is not None else None
                    inst_col = None if inst is None else ("#2ecc71" if inst > 60 else ("#e74c3c" if inst < 10 else None))
                    render_metric_row("Inst Own", _txt(inst, "{:.0f}%"), value_color=inst_col, help_text="🟢 > 60% (Strong Institutional Backing)")

                    _sf = _num(meta.get('short_percent_of_float'))
                    short_val = _sf * 100 if _sf is not None else None
                    short_col = None if short_val is None else ("#e74c3c" if short_val > 10 else ("#2ecc71" if short_val <= 2 else None))
                    render_metric_row("Short Float", _txt(short_val, "{:.1f}%"), value_color=short_col, help_text="🔴 > 10% (Squeeze Risk) | 🟢 < 2% (Safe)")
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol5:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Price & Context</div>", unsafe_allow_html=True)
                    _tgt = _num(target_p)
                    render_metric_row("Analyst Target", _txt(_tgt if _tgt else None, "€{:.2f}"),
                                      delta=upside if _tgt else None, is_pct=True)
                    
                    pe_5y_avg    = meta.get('pe_5y_avg', 0)
                    pe_cur       = meta.get('pe_ratio', 0)
                    pe_delta     = ((pe_cur / pe_5y_avg) - 1) * 100 if pe_5y_avg > 0 and pe_cur > 0 else 0
                    
                    render_metric_row("5Y Avg P/E",    f"{pe_5y_avg:.1f}" if pe_5y_avg > 0 else "N/A", delta=pe_delta, is_pct=True, color_invert=True)
                    
                    zs_col = "#2ecc71" if z_score < -1 else ("#e74c3c" if z_score > 1.5 else None)
                    render_metric_row("Z-Score (5Y)",  f"{z_score:.2f}", value_color=zs_col)
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol6:
                    # ── EARNINGS CALENDAR (v13.0) ──
                    e_row = earnings_cal[earnings_cal['ticker'] == deep_ticker]
                    e_header = _header_style
                    if not e_row.empty:
                        e_date = e_row.iloc[0]['earnings_date']
                        if pd.notnull(e_date):
                            # Handle both Timestamp and date objects safely
                            e_date_obj = e_date.date() if hasattr(e_date, 'date') else e_date
                            days_to_e = (e_date_obj - date.today()).days
                            if 0 <= days_to_e <= 7:
                                e_header = e_header.replace("#aabbcc", "#f39c12") # Highlight upcoming
                                e_date_str = f"⚠️ {e_date_obj.strftime('%b %d')}"
                            else:
                                e_date_str = e_date_obj.strftime('%b %d, %y')
                        else:
                            e_date_str = "TBD"
                        
                        _e_ccy = meta.get('currency', 'USD')
                        _e_fx = get_forex_rates(target="EUR", source=_e_ccy)
                        
                        _raw_eps = e_row.iloc[0]['eps_avg']
                        eps_est = float(_raw_eps) * _e_fx if pd.notnull(_raw_eps) else None
                        
                        _raw_rev = e_row.iloc[0]['rev_avg']
                        rev_est = float(_raw_rev) * _e_fx if pd.notnull(_raw_rev) else None
                    else:
                        e_date_str = "N/A"
                        eps_est = None
                        rev_est = None

                    st.markdown(f"<div style='{_card_style}'><div style='{e_header}'>Earnings & Events</div>", unsafe_allow_html=True)
                    render_metric_row("Report Date", e_date_str)
                    render_metric_row("EPS Est",     f"€{eps_est:.2f}" if pd.notnull(eps_est) else "N/A")
                    
                    if pd.notnull(rev_est) and rev_est > 0:
                        if rev_est >= 1e9: rev_txt = f"€{rev_est/1e9:.1f}B"
                        else: rev_txt = f"€{rev_est/1e6:.0f}M"
                    else:
                        rev_txt = "N/A"
                    render_metric_row("Revenue Est", rev_txt)
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<div style='margin-top:35px; margin-bottom:15px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #e74c3c; color:#e74c3c; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 4: TECHNICAL & CHARTING INTELLIGENCE</div>", unsafe_allow_html=True)

            # Main Technical Chart (Full Width)

            fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                     vertical_spacing=0.05, 
                                     row_heights=[0.7, 0.3])
            
            fig_tech.add_trace(go.Candlestick(
                x=df_deep['date'],
                open=df_deep['price_open'], high=df_deep['price_high'],
                low=df_deep['price_low'], close=df_deep['price_close'],
                name="Price",
                increasing=dict(line=dict(color='#00e676', width=1), fillcolor='rgba(0,230,118,0.85)'),
                decreasing=dict(line=dict(color='#ff5252', width=1), fillcolor='rgba(255,82,82,0.85)')
            ), row=1, col=1)
            
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_20'], name='MA20', line=dict(color='#FFB300', width=1.5)), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_50'], name='MA50', line=dict(color='#40C4FF', width=1.5)), row=1, col=1)
            # 🏆 EXPERT: MA200 (Long-term trend anchor)
            if 'ma_200' in df_deep.columns:
                fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_200'], name='MA200', line=dict(color='#E040FB', width=2.5)), row=1, col=1)
            
            # Support/Resistance → Scatter traces (appear in legend, not as annotations).
            # Each label says where the level comes from; ATR projections are grey and dotted so they
            # are never mistaken for real support/resistance.
            dates_range = df_deep['date'].tolist()
            df_deep['rsi'] = df_deep['rsi'] if 'rsi' in df_deep.columns else _rsi_val  # RSI from get_tactical_metrics
            _zw = _tm.get("zone_width", 0.0)
            for _lvl, _val, _col, _w in (("S1", _s1, "#2ecc71", 1.2), ("R1", _r1, "#e74c3c", 1.2),
                                          ("S2", _s2, "#27ae60", 1.6), ("R2", _r2, "#c0392b", 1.6),
                                          ("S3", _s3, "#1b5e20", 2.4), ("R3", _r3, "#b71c1c", 2.4)):
                _src = _kinds.get(_lvl.lower(), "")
                _proj = _src.startswith("projected")
                fig_tech.add_trace(go.Scatter(
                    x=[dates_range[0], dates_range[-1]], y=[_val, _val],
                    name=f"{_lvl} {'Support' if _lvl[0] == 'S' else 'Resistance'} €{_val:.2f} · {_src or 'zone'}",
                    mode='lines',
                    line=dict(color='rgba(160,160,160,0.6)' if _proj else _col, width=1 if _proj else _w,
                              dash='dot' if _proj else ('dot' if _lvl.endswith('1') else 'dash')),
                    opacity=0.85), row=1, col=1)
                # real S1/R1 zones are bands (±½ zone width), not single prices
                if _lvl in ("S1", "R1") and not _proj and _zw > 0:
                    fig_tech.add_hrect(y0=_val * (1 - _zw / 2), y1=_val * (1 + _zw / 2), line_width=0,
                                       fillcolor=_col, opacity=0.08, row=1, col=1)
            # 📈 AUTOMATED TRENDLINE (Linear Regression)
            # Calculate best-fit line for the current price window
            y_data = df_deep['price_close'].values
            x_data = np.arange(len(y_data))
            # Clean NaNs if any
            mask = ~np.isnan(y_data)
            if mask.any():
                slope, intercept = np.polyfit(x_data[mask], y_data[mask], 1)
                trendline_y = slope * x_data + intercept
                fig_tech.add_trace(go.Scatter(
                    x=df_deep['date'], y=trendline_y,
                    name='Linear fit of closes (visible window)',
                    line=dict(color='rgba(255, 215, 0, 0.4)', width=2, dash='dash'),
                    hoverinfo='skip'
                ), row=1, col=1)

            # RSI with overbought/oversold level traces in legend
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['rsi'], name='RSI (14)', line=dict(color='#9b59b6', width=2)), row=2, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[70, 70],
                name='RSI Overbought (70)', mode='lines',
                line=dict(color='rgba(231,76,60,0.5)', width=1, dash='dash'), showlegend=True
            ), row=2, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[30, 30],
                name='RSI Oversold (30)', mode='lines',
                line=dict(color='rgba(46,204,113,0.5)', width=1, dash='dash'), showlegend=True
            ), row=2, col=1)

            fig_tech.update_layout(
                title=dict(text=f"📈 {deep_ticker} — Technical Master Analysis", font=dict(size=20, color='#e8eaf6')),
                height=740,
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
                # Custom premium dark background
                paper_bgcolor='#0d0e14',
                plot_bgcolor='#11121a',
                font=dict(family="Inter, sans-serif", color="#b0bec5"),
                # Grid styling (subtle)
                xaxis=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                    zeroline=False, linecolor='rgba(255,255,255,0.1)'
                ),
                xaxis2=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                    zeroline=False, linecolor='rgba(255,255,255,0.1)',
                    tickprefix='€'
                ),
                yaxis2=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.04)',
                    zeroline=False
                ),
                # Legend → outside right side
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.01,
                    bgcolor="rgba(13,14,20,0.92)",
                    bordercolor="rgba(255,255,255,0.12)",
                    borderwidth=1,
                    font=dict(size=11, color='#cfd8dc'),
                    itemsizing="constant",
                    traceorder="normal"
                ),
                margin=dict(r=180, t=60, l=60, b=40)
            )
            fig_tech.update_yaxes(title_text="Price (€)", row=1, col=1)
            fig_tech.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            tech_tab_2, tech_tab_1 = st.tabs(["📈 Interactive TradingView", "📊 AI Technical Master"])
            
            with tech_tab_1:
                st.plotly_chart(fig_tech, use_container_width=True)
                
            with tech_tab_2:
                import streamlit.components.v1 as components
                tv_symbol = get_tv_symbol(deep_ticker)

                # Smart Money badge above chart
                _sm_color = "#2ecc71" if p_sm == "ACCUMULATION" else "#e74c3c"
                _sm_icon  = "📈" if p_sm == "ACCUMULATION" else "📉"
                _sm_badge_html = f"""
                <div style="display:flex; gap:10px; align-items:center; margin-bottom:8px; flex-wrap:wrap;">
                    <span style="background:{_sm_color}22; border:1px solid {_sm_color}; color:{_sm_color};
                                 padding:4px 12px; border-radius:20px; font-size:0.82rem; font-weight:700;">
                        {_sm_icon} Smart Money: {p_sm} ({p_sm_strength})
                    </span>
                    <span style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.12); color:#aaa;
                                 padding:4px 12px; border-radius:20px; font-size:0.82rem;">
                        Layer: {p_sm_layer}
                    </span>
                    <span style="color:#888; font-size:0.75rem; margin-left:auto;">TradingView Institutional Panel · {tv_symbol}</span>
                </div>"""
                st.markdown(_sm_badge_html, unsafe_allow_html=True)

                # Main layout: chart (left) + side panels (right)
                _tv_col_main, _tv_col_side = st.columns([3, 1])

                with _tv_col_main:
                    # Advanced Chart with pre-loaded indicators
                    components.html(f"""
                    <!-- TradingView Advanced Chart Widget -->
                    <div class="tradingview-widget-container" style="height:660px;width:100%">
                      <div id="tv_adv_chart_{tv_symbol.replace(':','_')}" style="height:660px;width:100%"></div>
                      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                      <script type="text/javascript">
                      new TradingView.widget({{
                        "autosize": true,
                        "symbol": "{tv_symbol}",
                        "interval": "D",
                        "timezone": "Etc/UTC",
                        "theme": "dark",
                        "style": "1",
                        "locale": "en",
                        "enable_publishing": false,
                        "backgroundColor": "rgba(13, 14, 20, 1)",
                        "gridColor": "rgba(255, 255, 255, 0.05)",
                        "withdateranges": true,
                        "hide_top_toolbar": false,
                        "hide_legend": false,
                        "hide_side_toolbar": false,
                        "allow_symbol_change": true,
                        "save_image": true,
                        "show_popup_button": true,
                        "popup_width": "1000",
                        "popup_height": "650",
                        "studies": [
                          "STD;MA%Cross",
                          "STD;RSI",
                          "STD;MACD",
                          "STD;Volume"
                        ],
                        "studies_overrides": {{
                          "moving average cross.first ma length": 50,
                          "moving average cross.second ma length": 200,
                          "rsi.length": 14,
                          "macd.fast length": 12,
                          "macd.slow length": 26,
                          "macd.signal smoothing": 9
                        }},
                        "overrides": {{
                          "paneProperties.background": "rgba(13, 14, 20, 1)",
                          "mainSeriesProperties.candleStyle.upColor": "#26a69a",
                          "mainSeriesProperties.candleStyle.downColor": "#ef5350",
                          "mainSeriesProperties.candleStyle.borderUpColor": "#26a69a",
                          "mainSeriesProperties.candleStyle.borderDownColor": "#ef5350"
                        }},
                        "drawing_access": {{ "type": "all" }},
                        "container_id": "tv_adv_chart_{tv_symbol.replace(':','_')}"
                      }});
                      </script>
                    </div>
                    """, height=680)

                with _tv_col_side:
                    # Panel 1: Technical Analysis Summary (Buy/Sell gauge)
                    components.html(f"""
                    <!-- TradingView Technical Analysis Widget -->
                    <div class="tradingview-widget-container" style="height:310px;">
                      <div class="tradingview-widget-container__widget"></div>
                      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
                      {{
                        "interval": "1D",
                        "width": "100%",
                        "isTransparent": true,
                        "height": "310",
                        "symbol": "{tv_symbol}",
                        "showIntervalTabs": true,
                        "locale": "en",
                        "colorTheme": "dark"
                      }}
                      </script>
                    </div>
                    """, height=320)

                    # Panel 2: Symbol Overview (Financials + Analyst Targets)
                    components.html(f"""
                    <!-- TradingView Symbol Overview Widget -->
                    <div class="tradingview-widget-container" style="height:330px; margin-top:10px;">
                      <div class="tradingview-widget-container__widget"></div>
                      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js" async>
                      {{
                        "symbols": [["{tv_symbol}"]],
                        "chartOnly": false,
                        "width": "100%",
                        "height": "330",
                        "locale": "en",
                        "colorTheme": "dark",
                        "autosize": false,
                        "showVolume": false,
                        "showMA": false,
                        "hideDateRanges": false,
                        "hideMarketStatus": false,
                        "hideSymbolLogo": false,
                        "scalePosition": "right",
                        "scaleMode": "Normal",
                        "fontFamily": "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
                        "fontSize": "10",
                        "noTimeScale": false,
                        "valuesTracking": "1",
                        "changeMode": "price-and-percent",
                        "chartType": "area",
                        "isTransparent": true,
                        "lineWidth": 2,
                        "lineType": 0,
                        "dateRanges": ["1m|1D", "3m|1D", "12m|1W", "60m|1M"]
                      }}
                      </script>
                    </div>
                    """, height=340)

                # ── STOCK PROFILE (Full Width) ────────────────────────────────────────────
                components.html(f"""
                <!-- TradingView Symbol Profile Widget -->
                <div class="tradingview-widget-container">
                  <div class="tradingview-widget-container__widget"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-profile.js" async>
                  {{
                    "width": "100%",
                    "height": "400",
                    "colorTheme": "dark",
                    "isTransparent": true,
                    "symbol": "{tv_symbol}",
                    "locale": "en"
                  }}
                  </script>
                </div>
                """, height=410)

                # Cross-Asset Comparison (Relative Strength vs Benchmark)
                st.markdown("<div style='margin-top:12px; color:#667788; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;'>📊 Relative Strength vs Key Benchmarks</div>", unsafe_allow_html=True)
                _compare_syms = json.dumps([
                    {"symbol": "FOREXCOM:SPXUSD", "position": "SameScale"},
                    {"symbol": "FOREXCOM:NSXUSD", "position": "SameScale"},
                    {"symbol": "XETR:DAX",        "position": "SameScale"},
                ])
                components.html(f"""
                <!-- TradingView Advanced Chart (Relative Strength) -->
                <div class="tradingview-widget-container" style="height:220px;">
                  <div class="tradingview-widget-container__widget"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
                  {{
                    "autosize": true,
                    "symbol": "{tv_symbol}",
                    "interval": "D",
                    "timezone": "Etc/UTC",
                    "theme": "dark",
                    "style": "3",
                    "locale": "en",
                    "backgroundColor": "rgba(0, 0, 0, 0)",
                    "gridColor": "rgba(255, 255, 255, 0.06)",
                    "hide_top_toolbar": true,
                    "hide_legend": false,
                    "save_image": false,
                    "calendar": false,
                    "hide_volume": true,
                    "compare_symbols": {_compare_syms},
                    "studies": [],
                    "height": 220,
                    "width": "100%"
                  }}
                  </script>
                </div>
                """, height=230)

                # ── COMMUNITY IDEAS ──────────────────────────────────────────
                _tv_base_ideas = tv_symbol if tv_symbol else deep_ticker
                _ideas_url = f"https://www.tradingview.com/symbols/{_tv_base_ideas}/ideas/"
                _chart_url  = f"https://www.tradingview.com/chart/?symbol={_tv_base_ideas}"
                _profile_url = f"https://www.tradingview.com/symbols/{_tv_base_ideas}/"
                components.html(f"""
                <div style="margin-top:16px; display:flex; gap:12px; flex-wrap:wrap;">
                  <a href="{_ideas_url}" target="_blank" style="
                    display:inline-flex; align-items:center; gap:8px;
                    padding:12px 22px; border-radius:8px; text-decoration:none;
                    background:rgba(155,89,182,0.15); border:1px solid rgba(155,89,182,0.4);
                    color:#c39bd3; font-size:0.85rem; font-weight:700;
                    font-family:Inter,sans-serif; transition:all 0.2s;
                    letter-spacing:0.5px;">
                    💡 View Community Ideas
                  </a>
                  <a href="{_chart_url}" target="_blank" style="
                    display:inline-flex; align-items:center; gap:8px;
                    padding:12px 22px; border-radius:8px; text-decoration:none;
                    background:rgba(52,152,219,0.15); border:1px solid rgba(52,152,219,0.4);
                    color:#7fb3d3; font-size:0.85rem; font-weight:700;
                    font-family:Inter,sans-serif; letter-spacing:0.5px;">
                    📈 Full Chart
                  </a>
                  <a href="{_profile_url}" target="_blank" style="
                    display:inline-flex; align-items:center; gap:8px;
                    padding:12px 22px; border-radius:8px; text-decoration:none;
                    background:rgba(46,204,113,0.12); border:1px solid rgba(46,204,113,0.35);
                    color:#82e0aa; font-size:0.85rem; font-weight:700;
                    font-family:Inter,sans-serif; letter-spacing:0.5px;">
                    🏢 Symbol Profile
                  </a>
                </div>
                <p style="margin-top:10px; color:#445566; font-size:0.72rem; font-family:Inter,sans-serif;">
                  Opens TradingView in a new tab · Symbol: <b style="color:#667788;">{_tv_base_ideas}</b>
                </p>
                """, height=120)


            # --- HISTORICAL FUNDAMENTAL TRENDS (Dual Axis) ---
            st.markdown("---")
            st.markdown("<div style='margin-top:10px; margin-bottom:15px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #1abc9c; color:#1abc9c; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 5: FUNDAMENTAL TRAJECTORY & VALUATION</div>", unsafe_allow_html=True)
            
            tab_annual, tab_quarterly = st.tabs(["📊 Annual", "📉 Quarterly"])
            
            with tab_annual:
                if not df_fin.empty:
                    df_fin_plot = df_fin.sort_values("year")
                    
                    # Calculate YoY Growth manually to handle negative values properly: (New - Old) / abs(Old)
                    df_fin_plot['rev_growth'] = (df_fin_plot['revenue'] - df_fin_plot['revenue'].shift(1)) / df_fin_plot['revenue'].shift(1).abs() * 100
                    df_fin_plot['eps_growth'] = (df_fin_plot['eps'] - df_fin_plot['eps'].shift(1)) / df_fin_plot['eps'].shift(1).abs() * 100
                    
                    # ── Merge FCF from raw.hist_fcf ──────────────────────────────
                    df_fcf_ticker = pd.DataFrame()
                    if not hist_fcf_full.empty and "ticker" in hist_fcf_full.columns:
                        df_fcf_ticker = hist_fcf_full[hist_fcf_full["ticker"] == deep_ticker].copy()
                    
                    if not df_fcf_ticker.empty:
                        df_fin_plot = df_fin_plot.merge(
                            df_fcf_ticker[["year", "free_cash_flow", "operating_cash_flow"]],
                            on="year", how="left"
                        )
                        # hist_fcf is stored in EUR by the ETL — no conversion here
                        df_fin_plot['fcf_growth'] = (
                            df_fin_plot['free_cash_flow'] - df_fin_plot['free_cash_flow'].shift(1)
                        ) / df_fin_plot['free_cash_flow'].shift(1).abs() * 100
                    else:
                        df_fin_plot['free_cash_flow'] = None
                        df_fin_plot['fcf_growth']     = None
                    
                    # Auto-scale based on max of Revenue and FCF
                    max_val = max(
                        df_fin_plot['revenue'].max(),
                        df_fin_plot['free_cash_flow'].max() if df_fin_plot['free_cash_flow'].notna().any() else 0
                    )
                    scale = 1e9 if max_val >= 1e9 else 1e6
                    unit = "B" if scale == 1e9 else "M"
                    
                    fig_fin = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Text labels for YoY growth
                    rev_text = [f"{v:+.1f}%" if pd.notnull(v) else "" for v in df_fin_plot['rev_growth']]
                    eps_text = [f"{v:+.1f}%" if pd.notnull(v) else "" for v in df_fin_plot['eps_growth']]

                    # Revenue Bar
                    fig_fin.add_trace(
                        go.Bar(
                            x=df_fin_plot['year'], 
                            y=df_fin_plot['revenue']/scale, 
                            name=f"Revenue (€{unit})", 
                            marker_color="rgba(0, 255, 204, 0.6)",
                            text=rev_text,
                            textposition="outside",
                            hovertemplate="<b>Year: %{x}</b><br>Revenue: €%{y:.2f}" + unit + "<br>YoY Growth: %{text}<extra></extra>"
                        ),
                        secondary_y=False
                    )

                    # FCF Bar (if available)
                    if df_fin_plot['free_cash_flow'].notna().any():
                        fcf_text = [f"{v:+.1f}%" if pd.notnull(v) else "" for v in df_fin_plot['fcf_growth']]
                        fig_fin.add_trace(
                            go.Bar(
                                x=df_fin_plot['year'],
                                y=df_fin_plot['free_cash_flow']/scale,
                                name=f"Free Cash Flow (€{unit})",
                                marker_color="rgba(39, 174, 96, 0.75)",
                                text=fcf_text,
                                textposition="outside",
                                hovertemplate="<b>Year: %{x}</b><br>FCF: €%{y:.2f}" + unit + "<br>YoY Growth: %{text}<extra></extra>"
                            ),
                            secondary_y=False
                        )

                    # EPS Line on secondary axis
                    fig_fin.add_trace(
                        go.Scatter(
                            x=df_fin_plot['year'], 
                            y=df_fin_plot['eps'], 
                            name="EPS (€)", 
                            line=dict(color="gold", width=3), 
                            mode="lines+markers+text",
                            text=eps_text,
                            textposition="top center",
                            hovertemplate="<b>Year: %{x}</b><br>EPS: €%{y:.2f}<br>YoY Growth: %{text}<extra></extra>"
                        ),
                        secondary_y=True
                    )
                    
                    # ── Analyst EPS Forecast overlay (Annual) ──────────────────
                    try:
                        with get_db_connection() as _fe_conn2:
                            _fe2 = _fe_conn2.execute(
                                """SELECT eps_est_cur_y, eps_est_next_y,
                                          eps_trend_delta_30d, eps_trend_delta_next_y,
                                          upgrade_ratio_30d
                                   FROM marts.dim_forward_estimates WHERE ticker = ?""",
                                [deep_ticker]
                            ).df()
                        if not _fe2.empty:
                            import datetime as _dt2
                            _cy = _dt2.datetime.now().year
                            _fe2r = _fe2.iloc[0]

                            # FX normalisation: forecast native currency → EUR
                            _ticker_ccy = str(meta.get("currency") or "USD")
                            _fx_rate = get_forex_rates(target="EUR", source=_ticker_ccy)

                            # Revision metrics (also in native currency → convert)
                            _delta_cy = _fe2r.get("eps_trend_delta_30d")
                            _delta_ny = _fe2r.get("eps_trend_delta_next_y")
                            _up_ratio = _fe2r.get("upgrade_ratio_30d")
                            _delta_cy_eur = float(_delta_cy) * _fx_rate if (_delta_cy is not None and pd.notnull(_delta_cy)) else None
                            _delta_ny_eur = float(_delta_ny) * _fx_rate if (_delta_ny is not None and pd.notnull(_delta_ny)) else None
                            _conviction  = round(float(_up_ratio) * 100, 0) if (_up_ratio is not None and pd.notnull(_up_ratio)) else None

                            def _rev_arrow(d):
                                """Return arrow+value string for on-chart label."""
                                if d is None: return ""
                                _a = "\u2191" if d > 0 else "\u2193"
                                return f" {_a}{abs(d):.2f}"

                            _fwd_x, _fwd_y, _fwd_cdata = [], [], []
                            # Bridge from last historical EPS point
                            if not df_fin_plot.empty:
                                _last_hist = df_fin_plot.iloc[-1]
                                _fwd_x.append(int(_last_hist["year"]))
                                _fwd_y.append(float(_last_hist["eps"]))
                                _fwd_cdata.append([float("nan"), float("nan")])
                            for _yr_offset, _col, _delta_eur in [
                                (0, "eps_est_cur_y",  _delta_cy_eur),
                                (1, "eps_est_next_y", _delta_ny_eur)
                            ]:
                                _v = _fe2r.get(_col)
                                if _v is not None and pd.notnull(_v):
                                    _fwd_x.append(_cy + _yr_offset)
                                    _fwd_y.append(float(_v) * _fx_rate)  # → EUR
                                    _fwd_cdata.append([
                                        _delta_eur if _delta_eur is not None else float("nan"),
                                        _conviction if _conviction is not None else float("nan")
                                    ])
                            if len(_fwd_x) > 1:
                                _fc_deltas = [_delta_cy_eur, _delta_ny_eur]
                                _text_labels = [""] + [
                                    f"Est \u20ac{_fwd_y[i+1]:.2f}{_rev_arrow(_fc_deltas[i])}"
                                    for i in range(len(_fwd_x) - 1)
                                ]
                                fig_fin.add_trace(
                                    go.Scatter(
                                        x=_fwd_x, y=_fwd_y,
                                        name="EPS Consensus Forecast",
                                        line=dict(color="#9b59b6", width=2.5, dash="dash"),
                                        mode="lines+markers+text",
                                        marker=dict(size=9, symbol="diamond"),
                                        text=_text_labels,
                                        textposition="top center",
                                        customdata=_fwd_cdata,
                                        hovertemplate=(
                                            "<b>Year: %{x}</b><br>"
                                            "EPS Forecast: \u20ac%{y:.2f}<br>"
                                            "30d Revision: %{customdata[0]:+.2f}<br>"
                                            "Analyst Conviction: %{customdata[1]:.0f}% \u2191"
                                            "<extra></extra>"
                                        )
                                    ),
                                    secondary_y=True
                                )
                                fig_fin.add_vrect(
                                    x0=_fwd_x[1] - 0.4, x1=_fwd_x[-1] + 0.4,
                                    fillcolor="rgba(155,89,182,0.07)",
                                    line_width=0,
                                    annotation_text="Forecast", annotation_position="top left",
                                    annotation_font=dict(size=10, color="#9b59b6")
                                )
                    except Exception:
                        pass  # Silently skip if forward estimates not available

                    fig_fin.update_layout(
                        template="plotly_dark", height=500,
                        margin=dict(l=20, r=20, t=60, b=20),
                        hovermode="x unified",
                        barmode="group",
                        title_text=f"📊 {deep_ticker} Annual Financial Performance (Revenue, FCF & EPS)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    fig_fin.update_yaxes(title_text=f"Amount (€{unit})", secondary_y=False, range=[0, (max_val/scale)*1.3])
                    fig_fin.update_yaxes(title_text="Earnings Per Share (€)", secondary_y=True)
                    
                    st.plotly_chart(fig_fin, use_container_width=True)



                else:
                    st.info("No historical financial data available for this ticker.")
            
            with tab_quarterly:
                if not quarterly_fin.empty:
                    df_fin_q = quarterly_fin[quarterly_fin["ticker"] == deep_ticker].sort_values("report_date")
                    if not df_fin_q.empty:
                        df_fin_q_plot = df_fin_q.copy()

                        # ── Merge FCF from raw.hist_fcf_quarterly ──────────────────
                        df_fcf_q_ticker = pd.DataFrame()
                        if not hist_fcf_q_full.empty and "ticker" in hist_fcf_q_full.columns:
                            df_fcf_q_ticker = hist_fcf_q_full[hist_fcf_q_full["ticker"] == deep_ticker].copy()
                        
                        if not df_fcf_q_ticker.empty:
                            df_fin_q_plot = df_fin_q_plot.merge(
                                df_fcf_q_ticker[["year", "quarter", "free_cash_flow", "operating_cash_flow"]],
                                on=["year", "quarter"], how="left"
                            )
                            # hist_fcf_quarterly is stored in EUR by the ETL — no conversion here
                        else:
                            df_fin_q_plot['free_cash_flow'] = None

                        # Growth basis: YoY (shift 4) if enough history, else QoQ (shift 1) as fallback
                        _n_quarters = len(df_fin_q_plot)
                        _growth_shift = 4 if _n_quarters >= 5 else 1
                        _growth_label = "YoY" if _growth_shift == 4 else "QoQ"

                        def _safe_growth(series, n):
                            prev = series.shift(n)
                            return (series - prev) / prev.abs() * 100

                        df_fin_q_plot['rev_growth'] = _safe_growth(df_fin_q_plot['revenue'], _growth_shift)
                        df_fin_q_plot['eps_growth'] = _safe_growth(df_fin_q_plot['eps'], _growth_shift)
                        if 'free_cash_flow' in df_fin_q_plot.columns and df_fin_q_plot['free_cash_flow'].notna().any():
                            df_fin_q_plot['fcf_growth'] = _safe_growth(df_fin_q_plot['free_cash_flow'], _growth_shift)
                        else:
                            df_fin_q_plot['fcf_growth'] = None
                        
                        # Auto-scale based on max of Revenue and FCF
                        max_val_q = max(
                            df_fin_q_plot['revenue'].max() if not df_fin_q_plot['revenue'].empty else 0,
                            df_fin_q_plot['free_cash_flow'].max() if 'free_cash_flow' in df_fin_q_plot.columns and df_fin_q_plot['free_cash_flow'].notna().any() else 0
                        )
                        scale_q = 1e9 if max_val_q >= 1e9 else 1e6
                        unit_q = "B" if scale_q == 1e9 else "M"
                        
                        # ── Merge Earnings Surprise (Actual vs Estimate) ───────────
                        if not earnings_surprise_full.empty:
                            _es_q = earnings_surprise_full[earnings_surprise_full["ticker"] == deep_ticker].copy()
                            if not _es_q.empty:
                                # Create join keys
                                _es_q["year"] = _es_q["quarter_date"].dt.year
                                _es_q["quarter"] = _es_q["quarter_date"].dt.quarter
                                df_fin_q_plot = df_fin_q_plot.merge(
                                    _es_q[["year", "quarter", "eps_actual", "eps_estimate", "surprise_pct"]],
                                    on=["year", "quarter"], how="left"
                                )
                                # Use eps_actual (Adjusted EPS, already converted to EUR) to match eps_estimate.
                                # Fall back to GAAP eps if eps_actual is missing.
                                df_fin_q_plot["eps_chart"] = df_fin_q_plot["eps_actual"].combine_first(df_fin_q_plot["eps"])
                            else:
                                df_fin_q_plot["eps_chart"] = df_fin_q_plot["eps"]
                        else:
                            df_fin_q_plot["eps_chart"] = df_fin_q_plot["eps"]

                        fig_fin_q = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        rev_text_q = [f"{v:+.1f}%" if pd.notnull(v) else "" for v in df_fin_q_plot['rev_growth']]
                        eps_text_q = [f"{v:+.1f}%" if pd.notnull(v) else "" for v in df_fin_q_plot['eps_growth']]
                        
                        x_labels = df_fin_q_plot['year'].astype(str) + " Q" + df_fin_q_plot['quarter'].astype(str)
                        
                        # Revenue Bar
                        fig_fin_q.add_trace(
                            go.Bar(
                                x=x_labels, 
                                y=df_fin_q_plot['revenue']/scale_q, 
                                name=f"Revenue (€{unit_q})", 
                                marker_color="rgba(0, 204, 255, 0.6)",
                                text=rev_text_q,
                                textposition="outside",
                                hovertemplate="<b>Quarter: %{x}</b><br>Revenue: €%{y:.2f}" + unit_q + "<br>" + _growth_label + " Growth: %{text}<extra></extra>"
                            ),
                            secondary_y=False
                        )

                        # FCF Bar (if available)
                        if 'free_cash_flow' in df_fin_q_plot.columns and df_fin_q_plot['free_cash_flow'].notna().any():
                            fcf_text_q = [f"{v:+.1f}%" if pd.notnull(v) else "" for v in df_fin_q_plot['fcf_growth']]
                            fig_fin_q.add_trace(
                                go.Bar(
                                    x=x_labels, 
                                    y=df_fin_q_plot['free_cash_flow']/scale_q, 
                                    name=f"Free Cash Flow (€{unit_q})", 
                                    marker_color="rgba(39, 174, 96, 0.75)",
                                    text=fcf_text_q,
                                    textposition="outside",
                                    hovertemplate="<b>Period: %{x}</b><br>FCF: €%{y:.2f}" + unit_q + "<br>" + _growth_label + " Growth: %{text}<extra></extra>"
                                ),
                                secondary_y=False
                            )
                        
                        # EPS ACTUAL — Adjusted (Line, primary)
                        fig_fin_q.add_trace(
                            go.Scatter(
                                x=x_labels,
                                y=df_fin_q_plot['eps_chart'],
                                name="Adjusted EPS (Actual)",
                                line=dict(color="orange", width=3),
                                mode="lines+markers",
                                hovertemplate="<b>Quarter: %{x}</b><br>Adjusted EPS: €%{y:.2f}<br>" + _growth_label + " Growth: %{text}<extra></extra>",
                                text=eps_text_q,
                            ),
                            secondary_y=True
                        )

                        # EPS ACTUAL — GAAP (dotted, secondary reference)
                        if df_fin_q_plot['eps'].notna().any():
                            fig_fin_q.add_trace(
                                go.Scatter(
                                    x=x_labels,
                                    y=df_fin_q_plot['eps'],
                                    name="GAAP EPS",
                                    line=dict(color="rgba(231, 76, 60, 0.75)", width=2),
                                    mode="lines+markers",
                                    marker=dict(size=6, symbol="circle-open"),
                                    hovertemplate="<b>Quarter: %{x}</b><br>GAAP EPS: €%{y:.2f}<extra></extra>",
                                ),
                                secondary_y=True
                            )

                        # EPS ESTIMATE (Dashed Line)
                        if "eps_estimate" in df_fin_q_plot.columns and df_fin_q_plot["eps_estimate"].notna().any():
                            # Create surprise labels for hover
                            surprise_text = [f"{v*100:+.1f}% Surprise" if pd.notnull(v) else "No Est." for v in df_fin_q_plot.get('surprise_pct', [])]
                            fig_fin_q.add_trace(
                                go.Scatter(
                                    x=x_labels, 
                                    y=df_fin_q_plot['eps_estimate'], 
                                    name="EPS Estimate", 
                                    line=dict(color="rgba(189, 195, 199, 0.8)", width=2, dash="dash"), 
                                    mode="lines+markers",
                                    hovertemplate="<b>Quarter: %{x}</b><br>EPS Estimate: €%{y:.2f}<br>Result: %{text}<extra></extra>",
                                    text=surprise_text,
                                ),
                                secondary_y=True
                            )
                        
                        # EPS QUARTERLY FORECAST (Analyst Consensus — CQ & NQ)
                        try:
                            with get_db_connection() as _fe_q_conn:
                                _fe_q = _fe_q_conn.execute(
                                    """SELECT eps_est_cur_q, eps_est_next_q,
                                              eps_trend_delta_30d, upgrade_ratio_30d
                                       FROM marts.dim_forward_estimates WHERE ticker = ?""",
                                    [deep_ticker]
                                ).df()
                            if not _fe_q.empty:
                                import datetime as _dtq
                                _now = _dtq.datetime.now()
                                _cur_q  = (_now.month - 1) // 3 + 1
                                _cur_qy = _now.year
                                _next_q = _cur_q + 1 if _cur_q < 4 else 1
                                _next_qy = _cur_qy if _cur_q < 4 else _cur_qy + 1

                                # FX normalisation: forecast native currency → EUR
                                _ticker_ccy_q = str(meta.get("currency") or "USD")
                                _fx_rate_q = get_forex_rates(target="EUR", source=_ticker_ccy_q)

                                # Revision metrics for quarterly (CQ only; NQ has no separate delta)
                                _fe_qr = _fe_q.iloc[0]
                                _qdelta = _fe_qr.get("eps_trend_delta_30d")
                                _qratio = _fe_qr.get("upgrade_ratio_30d")
                                _qdelta_eur = float(_qdelta) * _fx_rate_q if (_qdelta is not None and pd.notnull(_qdelta)) else None
                                _qconviction = round(float(_qratio) * 100, 0) if (_qratio is not None and pd.notnull(_qratio)) else None

                                def _qarrow(d):
                                    if d is None: return ""
                                    _a = "\u2191" if d > 0 else "\u2193"
                                    return f" {_a}{abs(d):.2f}"

                                _fq_labels, _fq_vals, _fq_cdata = [], [], []
                                if not df_fin_q_plot.empty:
                                    _last_q_row = df_fin_q_plot.iloc[-1]
                                    _fq_labels.append(f"{int(_last_q_row['year'])} Q{int(_last_q_row['quarter'])}")
                                    _fq_vals.append(float(_last_q_row["eps_chart"]) if pd.notnull(_last_q_row["eps_chart"]) else None)
                                    _fq_cdata.append([float("nan"), float("nan")])

                                for _qlabel, _qcol, _qd_eur in [
                                    ("CQ (est)", "eps_est_cur_q",  _qdelta_eur),   # Current fiscal quarter
                                    ("NQ (est)", "eps_est_next_q", None),           # Next fiscal quarter
                                ]:
                                    _qv = _fe_qr.get(_qcol)
                                    if _qv is not None and pd.notnull(_qv):
                                        _fq_labels.append(_qlabel)
                                        _fq_vals.append(float(_qv) * _fx_rate_q)  # → EUR
                                        _fq_cdata.append([
                                            _qd_eur if _qd_eur is not None else float("nan"),
                                            _qconviction if _qconviction is not None else float("nan")
                                        ])

                                if len(_fq_labels) > 1 and any(v is not None for v in _fq_vals[1:]):
                                    _fq_texts = [""] + [
                                        f"Est \u20ac{_fq_vals[i]:.2f}{_qarrow(_fq_cdata[i][0]) if not pd.isnull(_fq_cdata[i][0]) else ''}"
                                        if _fq_vals[i] is not None else ""
                                        for i in range(1, len(_fq_labels))
                                    ]
                                    fig_fin_q.add_trace(
                                        go.Scatter(
                                            x=_fq_labels, y=_fq_vals,
                                            name="EPS Quarterly Forecast",
                                            line=dict(color="#9b59b6", width=2.5, dash="dash"),
                                            mode="lines+markers+text",
                                            marker=dict(size=10, symbol="diamond"),
                                            text=_fq_texts,
                                            textposition="top center",
                                            customdata=_fq_cdata,
                                            hovertemplate=(
                                                "<b>Quarter: %{x}</b><br>"
                                                "EPS Forecast: \u20ac%{y:.2f}<br>"
                                                "30d Revision (CY): %{customdata[0]:+.2f}<br>"
                                                "Analyst Conviction: %{customdata[1]:.0f}% \u2191"
                                                "<extra></extra>"
                                            )
                                        ),
                                        secondary_y=True
                                    )
                                    if len(_fq_labels) > 1:
                                        fig_fin_q.add_vrect(
                                            x0=_fq_labels[1], x1=_fq_labels[-1],
                                            fillcolor="rgba(155,89,182,0.08)", line_width=0,
                                            annotation_text="Forecast", annotation_position="top left",
                                            annotation_font=dict(size=10, color="#9b59b6")
                                        )
                        except Exception:
                            pass  # Skip silently if data not available

                        fig_fin_q.update_layout(

                            template="plotly_dark", height=500,
                            margin=dict(l=20, r=20, t=60, b=20),
                            hovermode="x unified",
                            barmode="group",
                            title_text=f"📊 {deep_ticker} Quarterly Financial Performance (Revenue, FCF & EPS) — Growth: {_growth_label}",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        y_range_q = [0, (max_val_q/scale_q)*1.3] if pd.notnull(max_val_q) else None
                        fig_fin_q.update_yaxes(title_text=f"Amount (€{unit_q})", secondary_y=False, range=y_range_q)
                        fig_fin_q.update_yaxes(title_text="Earnings Per Share (€)", secondary_y=True)
                        
                        st.plotly_chart(fig_fin_q, use_container_width=True)

                    else:
                        st.info("No historical quarterly financial data available for this ticker.")
                else:
                    st.info("Quarterly financials warehouse table is empty. Please run the ETL pipeline.")
            


            render_valuation_section(meta=meta, price=float(cur_p), vin=_vin, relval=_relval)

            # ── OWNERSHIP & SHORT SQUEEZE RISK ──────────────────────────────
            st.markdown("---")
            render_header("search", "Smart Money Flow & Short Squeeze Risk")
            
            inst_own = meta.get("inst_ownership", 0)
            insider_own = meta.get("insider_ownership", 0)
            
            inst_own = float(inst_own) if pd.notnull(inst_own) else 0.0
            insider_own = float(insider_own) if pd.notnull(insider_own) else 0.0
            public_own = max(0, 1.0 - inst_own - insider_own)
            
            short_pct = meta.get("short_percent_of_float", 0)
            short_pct = float(short_pct) if pd.notnull(short_pct) else 0.0
            short_ratio = meta.get("short_ratio", 0)
            short_ratio = float(short_ratio) if pd.notnull(short_ratio) else 0.0
            
            col_own1, col_own2 = st.columns([1, 1])
            with col_own1:
                labels = ['Institutions (Smart Money)', 'Insiders', 'Public/Retail Float']
                values = [inst_own, insider_own, public_own]
                colors = ['#00d2ff', '#3a7bd5', 'rgba(255,255,255,0.05)']
                
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
                
                fig_own.add_annotation(text=f"{(inst_own+insider_own)*100:.1f}%<br><b>Locked</b>", x=0.5, y=0.5, font_size=20, showarrow=False)
                st.plotly_chart(fig_own, use_container_width=True)
                
            with col_own2:
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
                
                st.markdown(f"<p style='text-align:center; color:#bbb; font-size:1rem;'>Short Ratio (Days to Cover): <b>{short_ratio:.1f} days</b></p>", unsafe_allow_html=True)


            # ── PEER COMPARISON ────────────────────────────────────────────
            st.markdown("<div style='margin-top:10px; margin-bottom:15px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #f39c12; color:#f39c12; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 6: COMPETITIVE INTELLIGENCE & PEER BENCHMARKING</div>", unsafe_allow_html=True)

            # Smart Peer Matching: Industry-first, then Sector-level fallback
            _ticker_industry = meta.get("industry")
            _ticker_sector   = meta.get("sector")
            _min_peers = 3

            peer_companies = pd.DataFrame()
            _match_level = "Sector"

            if _ticker_industry and "industry" in companies_full.columns:
                peer_companies = companies_full[
                    (companies_full["industry"] == _ticker_industry) &
                    (~companies_full["ticker"].isin(indices_list)) &
                    (companies_full["ticker"] != deep_ticker)
                ].copy()
                if len(peer_companies) >= _min_peers:
                    _match_level = "Industry"

            if len(peer_companies) < _min_peers:
                peer_companies = companies_full[
                    (companies_full["sector"] == _ticker_sector) &
                    (~companies_full["ticker"].isin(indices_list)) &
                    (companies_full["ticker"] != deep_ticker)
                ].copy()
                _match_level = "Sector"

            _peer_group_label = _ticker_industry if (_match_level == "Industry" and _ticker_industry) else _ticker_sector
            render_header("package", f"Peer Comparison — {_peer_group_label} {_match_level}")

            if not peer_companies.empty:
                # ── Merge with latest price data ──────────────────────────────
                peer_prices = prices.sort_values('date').groupby('ticker').tail(1)[['ticker', 'price_close', 'rsi', 'ma_signal']]
                peer_df = peer_companies.merge(peer_prices, on='ticker', how='left')
                peer_df["upside_pct"] = (peer_df["target_mean_price"] / peer_df["price_close"] - 1) * 100

                # ── Compute Net Debt / EBITDA ──────────────────────────────────
                def _net_debt_ebitda(row):
                    td  = row.get("total_debt") or 0
                    eb  = row.get("ebitda")
                    if eb and eb > 0:
                        return td / eb
                    return None

                # ── Compute YoY growth from actual annual_fin data ─────────────
                # Same method as Financial Performance chart: (Y - Y-1) / |Y-1| * 100
                _all_peer_tickers = list(peer_df["ticker"].unique()) + [deep_ticker]
                _af_peers = annual_fin[annual_fin["ticker"].isin(_all_peer_tickers)].copy()

                def _yoy_growth_from_annual(ticker_sym, col):
                    """Latest YoY growth for a column from dim_annual_financials."""
                    t_df = _af_peers[_af_peers["ticker"] == ticker_sym].sort_values("year")
                    if len(t_df) < 2:
                        return None
                    prev = t_df[col].iloc[-2]
                    curr = t_df[col].iloc[-1]
                    if pd.isna(prev) or pd.isna(curr) or prev == 0:
                        return None
                    return (curr - prev) / abs(prev) * 100

                # ── Build unified rows ─────────────────────────────────────────
                def _build_row(row, ticker, is_selected):
                    td   = row.get("total_debt") or 0
                    eb   = row.get("ebitda")
                    rev_growth = _yoy_growth_from_annual(ticker, "revenue")
                    fcf_mgn = row.get("fcf_margin") or 0
                    return {
                        "ticker":           ticker,
                        "company":          row.get("company", ticker),
                        "market_cap":       row.get("market_cap"),
                        # Growth
                        "revenue_growth":   rev_growth,
                        "earnings_growth":  _yoy_growth_from_annual(ticker, "eps"),
                        # Quality / Profitability
                        "gross_margin":     (row.get("gross_margin") or 0) * 100,
                        "operating_margin": (row.get("operating_margin") or 0) * 100,
                        "roe_pct":          (row.get("roe") or 0) * 100,
                        "fcf_margin":       fcf_mgn,
                        "rule_of_40":       (rev_growth or 0) + fcf_mgn if rev_growth is not None else None,
                        # Balance sheet & Risk
                        "net_debt_ebitda":  td / eb if (pd.notna(eb) and eb > 0) else None,
                        "debt_to_equity":   row.get("debt_to_equity"),
                        "dividend_yield":   row.get("dividend_yield_pct"),
                        # Valuation
                        "ev_to_ebitda":     row.get("ev_to_ebitda"),
                        "pe_ratio":         row.get("pe_ratio"),
                        "peg_ratio":        row.get("peg_ratio"),
                        "price_to_sales":   row.get("price_to_sales"),
                        "price_to_book":    row.get("price_to_book"),
                        "is_selected":      is_selected,
                    }

                rows = [_build_row(meta, deep_ticker, True)]
                for _, pr in peer_df.iterrows():
                    rows.append(_build_row(pr, pr["ticker"], False))

                comp_df = pd.DataFrame(rows)

                # ── Dynamic Column Definitions based on Business Type ─────────
                _sector_lower = str(_ticker_sector).lower()
                _industry_lower = str(_ticker_industry).lower()
                
                if any(x in _sector_lower or x in _industry_lower for x in ['software', 'saas', 'cyber', 'ai', 'data', 'technology services', 'it services', 'cloud', 'internet']):
                    b_type = 'saas'
                elif any(x in _sector_lower or x in _industry_lower for x in ['semi', 'industrial', 'manufacturing', 'machinery', 'hardware', 'electronic', 'aerospace', 'auto']):
                    b_type = 'industrial'
                elif any(x in _sector_lower or x in _industry_lower for x in ['consumer', 'retail', 'food', 'beverage', 'apparel', 'leisure', 'staples']):
                    b_type = 'consumer'
                elif any(x in _sector_lower or x in _industry_lower for x in ['bank', 'financial', 'insurance', 'finance', 'capital']):
                    b_type = 'finance'
                else:
                    b_type = 'general'

                base_cols = [
                    ("company",          "Company",           None,         None),
                    ("market_cap",       "Mkt Cap",           None,         None),
                ]

                if b_type == 'saas':
                    _COL_DEFS = base_cols + [
                        ("revenue_growth",   "Rev Growth",        "pct_sign",   "higher"),
                        ("gross_margin",     "Gross Mgn",         "pct",        "higher"),
                        ("operating_margin", "Op Mgn",            "pct",        "higher"),
                        ("fcf_margin",       "FCF Mgn",           "pct",        "higher"),
                        ("rule_of_40",       "Rule of 40",        "pct",        "higher"),
                        ("price_to_sales",   "EV/Sales",          "x1",         "lower"),
                        ("peg_ratio",        "PEG",               "x1",         "lower"),
                    ]
                    _group_spans = [("", 3), ("📈 Growth & Profitability", 5), ("💰 Valuation", 2)]
                elif b_type == 'industrial':
                    _COL_DEFS = base_cols + [
                        ("gross_margin",     "Gross Mgn",         "pct",        "higher"),
                        ("operating_margin", "Op Mgn",            "pct",        "higher"),
                        ("roe_pct",          "ROIC/ROE",          "pct",        "higher"),
                        ("net_debt_ebitda",  "ND/EBITDA",         "x",          "lower"),
                        ("ev_to_ebitda",     "EV/EBITDA",         "x",          "lower"),
                        ("peg_ratio",        "PEG",               "x1",         "lower"),
                    ]
                    _group_spans = [("", 3), ("💎 Operations & Return", 3), ("🏦 Risk", 1), ("💰 Valuation", 2)]
                elif b_type == 'consumer':
                    _COL_DEFS = base_cols + [
                        ("revenue_growth",   "Org Growth",        "pct_sign",   "higher"),
                        ("gross_margin",     "Gross Mgn",         "pct",        "higher"),
                        ("operating_margin", "Op Mgn",            "pct",        "higher"),
                        ("fcf_margin",       "FCF Mgn",           "pct",        "higher"),
                        ("debt_to_equity",   "Debt/Eq",           "x",          "lower"),
                        ("dividend_yield",   "Div Yield",         "pct",        "higher"),
                        ("pe_ratio",         "P/E",               "x",          "lower"),
                    ]
                    _group_spans = [("", 3), ("🛍️ Consumer Metrics", 4), ("🏦 Risk & Yield", 2), ("💰 Valuation", 1)]
                elif b_type == 'finance':
                    _COL_DEFS = base_cols + [
                        ("revenue_growth",   "Rev Growth",        "pct_sign",   "higher"),
                        ("roe_pct",          "ROE",               "pct",        "higher"),
                        ("dividend_yield",   "Div Yield",         "pct",        "higher"),
                        ("price_to_book",    "P/B",               "x1",         "lower"),
                        ("pe_ratio",         "P/E",               "x",          "lower"),
                    ]
                    _group_spans = [("", 3), ("🏦 Financial Metrics", 3), ("💰 Valuation", 2)]
                else: # general
                    _COL_DEFS = base_cols + [
                        ("revenue_growth",   "Rev Growth",        "pct_sign",   "higher"),
                        ("earnings_growth",  "EPS Growth",        "pct_sign",   "higher"),
                        ("gross_margin",     "Gross Mgn",         "pct",        "higher"),
                        ("operating_margin", "Op Mgn",            "pct",        "higher"),
                        ("roe_pct",          "ROE",               "pct",        "higher"),
                        ("fcf_margin",       "FCF Mgn",           "pct",        "higher"),
                        ("net_debt_ebitda",  "ND/EBITDA",         "x",          "lower"),
                        ("ev_to_ebitda",     "EV/EBITDA",         "x",          "lower"),
                        ("pe_ratio",         "P/E",               "x",          "lower"),
                        ("price_to_sales",   "P/S",               "x1",         "lower"),
                    ]
                    _group_spans = [("", 3), ("📈 Growth", 2), ("💎 Profitability", 4), ("🏦 Risk", 1), ("💰 Valuation", 3)]

                # ── Sector Average row ─────────────────────────────────────────
                _numeric_cols = [c[0] for c in _COL_DEFS if c[0] not in ("company", "market_cap")]
                _peer_only = comp_df[~comp_df["is_selected"]]
                avg_vals   = _peer_only[_numeric_cols].mean(numeric_only=True)
                avg_row    = {"ticker": "AVG", "company": f"⊘ {_match_level} Average", "is_selected": False, "market_cap": None}
                for c in _numeric_cols:
                    avg_row[c] = avg_vals.get(c)
                comp_df = pd.concat([comp_df, pd.DataFrame([avg_row])], ignore_index=True)
                comp_df = comp_df.set_index("ticker")
                avg_dict = {c: avg_vals.get(c) for c in _numeric_cols}

                # ── Color coding ───────────────────────────────────────────────
                # Higher = green: growth, margins, roe, fcf
                # Lower  = green: net_debt_ebitda, valuation multiples
                HIGHER_BETTER = {"revenue_growth", "earnings_growth", "gross_margin", "operating_margin", "roe_pct", "fcf_margin", "rule_of_40", "dividend_yield"}
                LOWER_BETTER  = {"net_debt_ebitda", "ev_to_ebitda", "pe_ratio", "price_to_sales", "price_to_book", "debt_to_equity", "peg_ratio"}

                def _cell_bg(val, col_key, is_avg_row):
                    if is_avg_row or val is None: return ""
                    avg = avg_dict.get(col_key)
                    if avg is None or pd.isna(avg): return ""
                    try:
                        fv, fa = float(val), float(avg)
                    except Exception:
                        return ""
                    if col_key in HIGHER_BETTER:
                        if fv > fa * 1.05: return "background:rgba(0,229,160,0.18);"
                        if fv < fa * 0.95: return "background:rgba(255,107,107,0.18);"
                    elif col_key in LOWER_BETTER:
                        if fv <= 0: return ""
                        if fv < fa * 0.95: return "background:rgba(0,229,160,0.18);"
                        if fv > fa * 1.05: return "background:rgba(255,107,107,0.18);"
                    return ""

                def _fmt_cap(v):
                    if v is None or pd.isna(v): return "—"
                    if v >= 1e12: return f"€{v/1e12:.1f}T"
                    if v >= 1e9:  return f"€{v/1e9:.0f}B"
                    return f"€{v/1e6:.0f}M"

                def _fmt_pct(v, sign=False):
                    if v is None or (isinstance(v, float) and pd.isna(v)): return "—"
                    prefix = "+" if sign and float(v) > 0 else ""
                    return f"{prefix}{float(v):.1f}%"

                def _fmt_x(v, decimals=1):
                    if v is None or (isinstance(v, float) and pd.isna(v)): return "—"
                    fv = float(v)
                    if fv <= 0: return "N/A"
                    return f"{fv:.{decimals}f}x"


                group_header = "".join(
                    f"<th colspan='{span}' style='padding:4px 10px; text-align:center; color:#64748b; font-size:0.72rem; border-bottom:1px solid rgba(255,255,255,0.06); font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>{label}</th>"
                    for label, span in _group_spans
                )

                # ── Column header row ──────────────────────────────────────────
                header_cells = "<th style='padding:6px 10px; color:#94a3b8; text-align:left; font-size:0.8rem; white-space:nowrap;'>Ticker</th>" + "".join(
                    f"<th style='padding:6px 10px; color:#94a3b8; text-align:{'left' if k in ('company','market_cap') else 'center'}; font-size:0.8rem; white-space:nowrap;'>{lbl}</th>"
                    for k, lbl, *_ in _COL_DEFS
                )

                # ── Data rows ─────────────────────────────────────────────────
                html_rows = []
                for ticker_idx, row_data in comp_df.iterrows():
                    is_sel = row_data.get("is_selected", False)
                    is_avg = (ticker_idx == "AVG")
                    row_bg = "background:rgba(99,132,255,0.12); font-weight:700;" if is_sel else (
                             "background:rgba(255,255,255,0.04); font-style:italic; font-weight:600;" if is_avg else "")
                    label  = f"★ {ticker_idx}" if is_sel else ticker_idx

                    cells = f"<td style='padding:8px 10px; color:#a78bfa; font-weight:700; white-space:nowrap;'>{label}</td>"
                    for col_key, col_lbl, fmt, direction in _COL_DEFS:
                        val = row_data.get(col_key)
                        bg  = _cell_bg(val, col_key, is_avg)
                        align = "left" if col_key in ("company",) else "center"
                        if col_key == "company":
                            text = str(val)[:28] if val else "—"
                        elif col_key == "market_cap":
                            text = _fmt_cap(val)
                        elif fmt == "pct":
                            text = _fmt_pct(val)
                        elif fmt == "pct_sign":
                            text = _fmt_pct(val, sign=True)
                        elif fmt == "x":
                            text = _fmt_x(val, 1)
                        elif fmt == "x1":
                            text = _fmt_x(val, 2)
                        else:
                            text = str(val) if val is not None else "—"
                        cells += f"<td style='padding:8px 10px; text-align:{align}; {bg} white-space:nowrap;'>{text}</td>"



                # Rebuild clean html rows
                html_rows = []
                for ticker_idx, row_data in comp_df.iterrows():
                    is_sel = row_data.get("is_selected", False)
                    is_avg = (ticker_idx == "AVG")
                    row_bg = "background:rgba(99,132,255,0.12); font-weight:700;" if is_sel else (
                             "background:rgba(255,255,255,0.04); font-style:italic; font-weight:600;" if is_avg else "")
                    label  = f"★ {ticker_idx}" if is_sel else ticker_idx

                    td_ticker = f"<td style='padding:8px 10px; color:#a78bfa; font-weight:700; white-space:nowrap;'>{label}</td>"
                    col_cells = ""
                    for col_key, col_lbl, fmt, direction in _COL_DEFS:
                        val = row_data.get(col_key)
                        bg  = _cell_bg(val, col_key, is_avg)
                        if col_key == "company":
                            text = str(val)[:28] if val else "—"
                            col_cells += f"<td style='padding:8px 10px; text-align:left; {bg}'>{text}</td>"
                        elif col_key == "market_cap":
                            col_cells += f"<td style='padding:8px 10px; text-align:center; {bg}'>{_fmt_cap(val)}</td>"
                        elif fmt == "pct":
                            col_cells += f"<td style='padding:8px 10px; text-align:center; {bg}'>{_fmt_pct(val)}</td>"
                        elif fmt == "pct_sign":
                            col_cells += f"<td style='padding:8px 10px; text-align:center; {bg}'>{_fmt_pct(val, sign=True)}</td>"
                        elif fmt in ("x", "x1"):
                            col_cells += f"<td style='padding:8px 10px; text-align:center; {bg}'>{_fmt_x(val, 1 if fmt=='x' else 2)}</td>"
                        else:
                            col_cells += f"<td style='padding:8px 10px; text-align:center; {bg}'>—</td>"
                    html_rows.append(f"<tr style='{row_bg}'>{td_ticker}{col_cells}</tr>")

                html_table = f"""
                <div style="overflow-x:auto; border-radius:12px; border:1px solid rgba(255,255,255,0.08); margin-bottom:12px;">
                <table style="width:100%; border-collapse:collapse; font-size:0.85rem; color:#e2e8f0;">
                  <thead>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">{group_header}</tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">{header_cells}</tr>
                  </thead>
                  <tbody>{"".join(html_rows)}</tbody>
                </table>
                </div>
                <div style="font-size:0.78rem; color:#64748b; margin-top:-4px; margin-bottom:16px;">
                  🟢 = above {_match_level} avg &nbsp;|&nbsp; 🔴 = below avg &nbsp;|&nbsp; ★ = selected ticker
                </div>
                """
                st.markdown(html_table, unsafe_allow_html=True)

            else:
                st.info(f"No peers found in the **{_peer_group_label}** {_match_level} to compare with.")

            st.markdown("---")
            render_header("activity", f"Performance Alpha (Cumulative % vs SPY)")
            
            df_ticker_ret = df_deep.set_index('date')['price_close']
            df_spy_ret = spy_prices.set_index('date')['price_close']
            common_dates = df_ticker_ret.index.intersection(df_spy_ret.index)
            if not common_dates.empty:
                ticker_cum = (df_ticker_ret.loc[common_dates] / df_ticker_ret.loc[common_dates].iloc[0] - 1) * 100
                spy_cum = (df_spy_ret.loc[common_dates] / df_spy_ret.loc[common_dates].iloc[0] - 1) * 100
            else:
                ticker_cum = pd.Series()
                spy_cum = pd.Series()

            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=common_dates, y=ticker_cum, name=f"{deep_ticker} (%)", line=dict(color="#3498db", width=3)))
            fig_rel.add_trace(go.Scatter(x=common_dates, y=spy_cum, name="SPY (%)", line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dot")))
            fig_rel.update_layout(template="plotly_dark", height=450, yaxis_title="Return (%)", hovermode="x unified", margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig_rel, use_container_width=True)

            st.markdown("<div style='margin-top:35px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #2ecc71; color:#2ecc71; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 7: PORTFOLIO IDEA MANAGEMENT</div>", unsafe_allow_html=True)
            # --- WATCHLIST QUICK SAVE WORKFLOW ---
            with st.expander("📥 📝 Save Idea to Watchlist Pipeline", expanded=False):
                with st.form(f"quick_save_form_{deep_ticker}"):
                    st.write("**Idea Management & Catalyst Tracking**")
                    _wl_col1, _wl_col2 = st.columns(2)
                    with _wl_col1:
                        # Auto-suggest status based on Logic
                        _s_index = 1 if act_str.startswith("🔥") or "ACCUMULATE" in act_str else 0
                        opt_status = st.selectbox("Status", ["🔵 PENDING", "🟢 ACTIVE", "🟡 REVIEW", "🔴 INVALIDATED", "⚫ CLOSED"], index=_s_index)
                        opt_thesis = st.text_area("Investment Thesis (Why buy/hold?)", value=act_desc, height=110)
                    with _wl_col2:
                        opt_catalyst = st.text_input("Upcoming Catalyst (Earnings, FDA, Macro, etc.)", placeholder="e.g. Q4 Earnings expected positive...")
                        
                        _kcol1, _kcol2, _kcol3 = st.columns(3)
                        with _kcol1: opt_entry = st.number_input("Entry (€)", value=float(_s1), step=1.0)
                        with _kcol2: opt_inval = st.number_input("Inval / Stop (€)", value=float(_stop_loss), step=1.0)
                        with _kcol3: opt_tp = st.number_input("Take Profit (€)", value=float(_tp1), step=1.0)

                        opt_erd = meta.get("next_earnings_date", "TBD")
                        if pd.isna(opt_erd): opt_erd = "TBD"
                        st.caption(f"Next Earnings: **{opt_erd}**")
                        
                    if st.form_submit_button("💾 Save Candidate to Watchlist", type="primary"):
                        try:
                            wl_df = load_watchlist()
                            # Delete existing to overwrite
                            wl_df = wl_df[wl_df["Ticker"] != deep_ticker]
                            
                            new_row = pd.DataFrame([{
                                "Ticker": deep_ticker,
                                "Status": opt_status,
                                "Thesis": opt_thesis,
                                "Catalyst": opt_catalyst,
                                "Entry Target": round(opt_entry, 2),
                                "Invalidation Level": round(opt_inval, 2),
                                "Take Profit": round(opt_tp, 2),
                                "Next Earnings": str(opt_erd),
                                "Added Date": pd.Timestamp.now().strftime("%Y-%m-%d")
                            }])
                            wl_df = pd.concat([wl_df, new_row], ignore_index=True)
                            save_watchlist(wl_df)
                            st.success(f"✅ Successfully added **{deep_ticker}** to Watchlist Pipeline!")
                        except Exception as e:
                            st.error(f"Error saving to watchlist: {e}")

            st.markdown("---")

            # ── LAYER 5b: ANALYST EXPECTATIONS (forward-looking) ──
            st.markdown("---")
            st.markdown("<div style='margin-top:10px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #9b59b6; color:#9b59b6; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>🔭 LAYER 5: ANALYST EXPECTATIONS (FORWARD-LOOKING)</div>", unsafe_allow_html=True)

            _fe_df = pd.DataFrame()
            _fe_err_msg = ""
            try:
                with get_db_connection() as _fe_conn:
                    _fe_df = _fe_conn.execute("""
                        SELECT * FROM marts.dim_forward_estimates WHERE ticker = ?
                    """, [deep_ticker]).df()
            except Exception as _fe_err:
                _fe_err_msg = str(_fe_err)

            if _fe_df.empty:
                if _fe_err_msg:
                    st.warning(f"⚠️ Forward estimates error: `{_fe_err_msg}`")
                else:
                    st.info("🔭 No analyst estimates available for this ticker yet. Run the full ETL pipeline to populate forward estimates.")
            else:
                _fe = _fe_df.iloc[0]

                def _fv(col, default=None):
                    v = _fe.get(col)
                    return float(v) if v is not None and pd.notnull(v) else default

                # ── ROW 1: Forward Valuation KPIs ──────────────────────────────────
                kc1, kc2, kc3, kc4, kc5 = st.columns(5)

                # Forward P/E: computed on-the-fly — cur_price (EUR) ÷ eps_est_next_y (native→EUR)
                # NOT read from DB because ETL stored it with a EUR/USD currency mismatch.
                _fwd_pe = None
                _fe_fxr = 1.0
                try:
                    _fe_ccy   = str(meta.get("currency") or "USD")
                    _fe_fxr   = get_forex_rates(target="EUR", source=_fe_ccy)
                    _eps_ny_eur = (_fv("eps_est_next_y") or 0) * _fe_fxr
                    _fe_price_eur = _fv("cur_price")   # already EUR from fct_daily_returns
                    if _eps_ny_eur and _eps_ny_eur > 0 and _fe_price_eur and _fe_price_eur > 0:
                        _fwd_pe = round(_fe_price_eur / _eps_ny_eur, 2)
                except Exception:
                    _fwd_pe = None

                try: _trail_pe = float(meta.get("pe_ratio")) if pd.notnull(meta.get("pe_ratio")) else None
                except: _trail_pe = None
                # Yahoo's consensus growth fields are sometimes garbage (Sony: revenue +1508%);
                # anything beyond ±300% is treated as missing rather than displayed as fact.
                def _plausible(x):
                    return x if (x is not None and abs(x) <= 3.0) else None
                _eps_g_cy  = _plausible(_fv("eps_growth_cur_y"))
                _eps_g_ny  = _plausible(_fv("eps_growth_next_y"))
                _rev_g_cy  = _plausible(_fv("rev_growth_cur_y"))
                _rev_g_ny  = _plausible(_fv("rev_growth_next_y"))
                _n_analysts     = _fv("n_analysts_cur_y")
                _upgrades       = _fv("upgrades_30d")
                _downgrades     = _fv("downgrades_30d")
                _revision_momentum = _fe.get("revision_momentum", "NEUTRAL")

                with kc1:
                    _pe_color  = "#2ecc71" if (_fwd_pe and _trail_pe and _fwd_pe < _trail_pe) else "#f1c40f"
                    _pe_str    = f"{_fwd_pe:.1f}x" if _fwd_pe else "N/A"
                    _delta_str = f"vs Trail {_trail_pe:.1f}x" if _trail_pe else ""
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                                border-radius:8px; padding:12px; text-align:center;'>
                        <div style='color:#8899aa; font-size:0.65rem; text-transform:uppercase;'>Forward P/E (NTM)</div>
                        <div style='color:{_pe_color}; font-size:1.4rem; font-weight:900; font-family:monospace;'>{_pe_str}</div>
                        <div style='color:#556677; font-size:0.65rem;'>{_delta_str}</div>
                    </div>""", unsafe_allow_html=True)

                with kc2:
                    _epsg_color = "#2ecc71" if _eps_g_cy and _eps_g_cy > 0 else "#e74c3c"
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                                border-radius:8px; padding:12px; text-align:center;'>
                        <div style='color:#8899aa; font-size:0.65rem; text-transform:uppercase;'>EPS Growth (CY)</div>
                        <div style='color:{_epsg_color}; font-size:1.4rem; font-weight:900; font-family:monospace;'>{f"{_eps_g_cy*100:+.1f}%" if _eps_g_cy is not None else "N/A"}</div>
                        <div style='color:#556677; font-size:0.65rem;'>NTY: {f"{_eps_g_ny*100:+.1f}%" if _eps_g_ny is not None else "N/A"}</div>
                    </div>""", unsafe_allow_html=True)

                with kc3:
                    _revg_color = "#2ecc71" if _rev_g_cy and _rev_g_cy > 0 else "#e74c3c"
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                                border-radius:8px; padding:12px; text-align:center;'>
                        <div style='color:#8899aa; font-size:0.65rem; text-transform:uppercase;'>Rev Growth (CY)</div>
                        <div style='color:{_revg_color}; font-size:1.4rem; font-weight:900; font-family:monospace;'>{f"{_rev_g_cy*100:+.1f}%" if _rev_g_cy is not None else "N/A"}</div>
                        <div style='color:#556677; font-size:0.65rem;'>NTY: {f"{_rev_g_ny*100:+.1f}%" if _rev_g_ny is not None else "N/A"}</div>
                    </div>""", unsafe_allow_html=True)

                with kc4:
                    _mom_color = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#f1c40f"}.get(_revision_momentum, "#f1c40f")
                    _mom_icon  = {"POSITIVE": "⬆️", "NEGATIVE": "⬇️", "NEUTRAL": "➡️"}.get(_revision_momentum, "➡️")
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03); border:1px solid {_mom_color}44;
                                border-radius:8px; padding:12px; text-align:center;'>
                        <div style='color:#8899aa; font-size:0.65rem; text-transform:uppercase;'>Revision Signal</div>
                        <div style='color:{_mom_color}; font-size:1.2rem; font-weight:900;'>{_mom_icon} {_revision_momentum}</div>
                        <div style='color:#556677; font-size:0.65rem;'>{int(_upgrades or 0)}↑ / {int(_downgrades or 0)}↓ (30d)</div>
                    </div>""", unsafe_allow_html=True)

                with kc5:
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                                border-radius:8px; padding:12px; text-align:center;'>
                        <div style='color:#8899aa; font-size:0.65rem; text-transform:uppercase;'>Analyst Coverage</div>
                        <div style='color:#e8eaf6; font-size:1.4rem; font-weight:900; font-family:monospace;'>{int(_n_analysts) if _n_analysts else "N/A"}</div>
                        <div style='color:#556677; font-size:0.65rem;'>analysts covering</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

                # ── ROW 2: Charts ────────────────────────────────────────────────────
                ch_l, ch_r = st.columns([3, 2])

                with ch_l:
                    _hist_eps = annual_fin[annual_fin["ticker"] == deep_ticker].sort_values("year")[["year", "eps"]].dropna()
                    import datetime as _dt
                    _cur_yr = _dt.datetime.now().year
                    _fwd_periods = []
                    for _lbl, _col, _offset in [("CY", "eps_est_cur_y", 0), ("NTY", "eps_est_next_y", 1)]:
                        _v = _fv(_col)
                        if _v is not None:
                            _fwd_periods.append({"year": _cur_yr + _offset, "eps": _v * _fe_fxr})

                    fig_eps = go.Figure()
                    if not _hist_eps.empty:
                        fig_eps.add_trace(go.Bar(
                            x=_hist_eps["year"].tolist(), y=_hist_eps["eps"].tolist(),
                            name="Historical EPS", marker_color="#3498db", opacity=0.85
                        ))
                    if _fwd_periods:
                        fig_eps.add_trace(go.Bar(
                            x=[p["year"] for p in _fwd_periods], y=[p["eps"] for p in _fwd_periods],
                            name="Consensus Estimate", marker_color="#9b59b6",
                            opacity=0.75, marker_pattern_shape="/"
                        ))
                    fig_eps.update_layout(
                        template="plotly_dark", height=260,
                        margin=dict(l=0, r=0, t=30, b=0),
                        title=dict(text="EPS Trajectory (Historical + Consensus Forecast)",
                                   font=dict(size=12, color="#8899aa"), x=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                    xanchor="right", x=1, font=dict(size=10)),
                        barmode="group", yaxis_title="EPS (€)",
                        xaxis=dict(tickmode="linear", dtick=1)
                    )
                    st.plotly_chart(fig_eps, use_container_width=True)

                with ch_r:
                    _up    = int(_upgrades or 0)
                    _down  = int(_downgrades or 0)
                    _total = _up + _down
                    _up_pct   = (_up / _total * 100) if _total > 0 else 0
                    _down_pct = (_down / _total * 100) if _total > 0 else 0
                    _eps_delta_30d = _fv("eps_trend_delta_30d")
                    _delta_color = "#2ecc71" if _eps_delta_30d and _eps_delta_30d > 0 else "#e74c3c"
                    _delta_sign  = "+" if _eps_delta_30d and _eps_delta_30d > 0 else ""

                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                                border-radius:8px; padding:16px; height:260px; box-sizing:border-box;'>
                        <div style='color:#8899aa; font-size:0.7rem; font-weight:700;
                                    text-transform:uppercase; margin-bottom:14px;'>Analyst Revision Sentiment (30d)</div>
                        <div style='display:flex; justify-content:space-between;
                                    font-size:0.72rem; color:#8899aa; margin-bottom:4px;'>
                            <span>⬆️ Upgrades: <b style='color:#2ecc71;'>{_up}</b></span>
                            <span>⬇️ Downgrades: <b style='color:#e74c3c;'>{_down}</b></span>
                        </div>
                        <div style='background:rgba(255,255,255,0.08); border-radius:4px;
                                    height:14px; overflow:hidden; display:flex;'>
                            <div style='width:{_up_pct:.0f}%; background:#2ecc71; border-radius:4px 0 0 4px;'></div>
                            <div style='width:{_down_pct:.0f}%; background:#e74c3c; border-radius:0 4px 4px 0;'></div>
                        </div>
                        <div style='margin-top:20px; border-top:1px solid rgba(255,255,255,0.07); padding-top:14px;'>
                            <div style='color:#8899aa; font-size:0.65rem; text-transform:uppercase; margin-bottom:6px;'>EPS Estimate Drift (30d)</div>
                            <div style='color:{_delta_color}; font-size:1.6rem; font-weight:900; font-family:monospace;'>
                                {f"{_delta_sign}{_eps_delta_30d:+.3f}" if _eps_delta_30d is not None else "N/A"}
                            </div>
                            <div style='color:#556677; font-size:0.65rem; margin-top:2px;'>
                                {"Analysts raising estimates ↑" if _eps_delta_30d and _eps_delta_30d > 0 else "Analysts cutting estimates ↓" if _eps_delta_30d and _eps_delta_30d < 0 else "Estimates stable"}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
