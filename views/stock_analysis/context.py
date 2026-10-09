"""Everything the deep-dive sections share for one ticker, computed once per rerun."""
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import streamlit as st

from core.levels import get_tactical_metrics
from core.rating import compute_institutional_rating
from core.smart_money import get_sm_spirit_unified_v2
from core.valuation import relative_valuation
from ui.decision_panel import valuation_inputs


@dataclass
class DeepDive:
    ticker: str
    meta: pd.Series
    meta_enriched: dict
    df_deep: pd.DataFrame          # prices in the sidebar horizon (display only)
    df_fin: pd.DataFrame
    cur_p: float
    target_p: float
    upside: float
    z_score: float
    ai_score: float                # Quality 0-100 (core/scoring.py)
    scores: dict                   # quality / value / momentum / flags / coverage / missing / components
    tm: dict                       # core.levels.get_tactical_metrics on the FULL history
    ma_sig: str
    tp1: float
    tp2: float
    vin: dict
    relval: dict
    next_earnings: object
    sm: dict
    rating: dict
    act_str: str
    act_color: str
    act_desc: str
    extra: dict = field(default_factory=dict)

    # short aliases for the ladder — used everywhere
    @property
    def s1(self): return self.tm["s1"]
    @property
    def s2(self): return self.tm["s2"]
    @property
    def s3(self): return self.tm["s3"]
    @property
    def r1(self): return self.tm["r1"]
    @property
    def r2(self): return self.tm["r2"]
    @property
    def r3(self): return self.tm["r3"]
    @property
    def stop_loss(self): return self.tm["stop_loss"]
    @property
    def rsi(self): return self.tm["rsi"]
    @property
    def w52_pos(self): return self.tm["w52_pos"]
    @property
    def kinds(self): return self.tm.get("kinds", {})


def build(ctx, deep_ticker):
    """Return the DeepDive for `deep_ticker`, or stop the page with a warning when data is missing."""
    companies_full, prices, prices_full = ctx["companies_full"], ctx["prices"], ctx["prices_full"]
    annual_fin, m_df, earnings_cal = ctx["annual_fin"], ctx["m_df"], ctx["earnings_cal"]

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

    z_score = df_deep['price_z_score'].iloc[-1] if 'price_z_score' in df_deep.columns else 0
    if pd.isna(z_score): z_score = 0

    # --- Enrich meta with latest technicals for the scoring engine ---
    latest_tech = df_deep.iloc[-1]
    meta_enriched = meta.to_dict()
    for col in ['pe_ratio', 'peg_ratio', 'price_to_book', 'roe', 'fcf_margin', 'dividend_yield_pct']:
        val = meta_enriched.get(col)
        try:
            meta_enriched[col] = float(val) if pd.notnull(val) else None
        except (TypeError, ValueError):
            meta_enriched[col] = None
    meta_enriched['rsi'] = float(latest_tech.get('rsi', 50))
    meta_enriched['ma_signal'] = str(latest_tech.get('ma_signal', 'NEUTRAL'))
    meta_enriched['price_z_score'] = float(z_score)
    meta_enriched['upside_pct'] = float(upside)

    # The screener table is the single score source across tabs (percentiles need the whole universe)
    _row = m_df[m_df["Ticker"] == deep_ticker]
    _num = lambda k: (float(_row.iloc[0][k]) if (not _row.empty and pd.notnull(_row.iloc[0][k])) else None)  # noqa: E731
    scores = {
        "quality": _num("Quality"), "value": _num("Value"), "momentum": _num("Momentum"),
        "coverage": _num("Coverage (%)"),
        "flags": str(_row.iloc[0]["Flags"]) if not _row.empty else "",
        "missing": list(_row.iloc[0]["Missing"]) if not _row.empty else [],
        "components": dict(_row.iloc[0]["Components"]) if not _row.empty else {},
    }
    ai_score = scores["quality"] if scores["quality"] is not None else 50.0

    # Levels and the 52-week range always use the FULL price history — `df_deep` follows the
    # sidebar horizon (1M → "52-week high" was the 1-month high) and is only used for display.
    _df_levels = prices_full[prices_full["ticker"] == deep_ticker].sort_values("date")
    tm = get_tactical_metrics(_df_levels, cur_p, analyst_target=target_p)
    ma_sig = str(latest_tech.get("ma_signal", meta.get("ma_signal", "NEUTRAL")))

    # TP1: honour AI Ensemble target if already computed, otherwise use standard formula
    _global_ai_target = st.session_state.get(f"ai_target_for_de_{deep_ticker}")
    tp1 = float(_global_ai_target) if _global_ai_target is not None else tm["tp1"]
    tp2 = max(target_p, tm["tp2"]) if target_p > 0 else tm["tp2"]

    vin = valuation_inputs(meta, cur_p, ctx["hist_fcf_full"], deep_ticker, ctx["macro"], annual_fin)
    relval = relative_valuation(companies_full, deep_ticker)
    next_er = None
    _er = earnings_cal[earnings_cal["ticker"] == deep_ticker] if not earnings_cal.empty else earnings_cal
    if not _er.empty:
        _future = pd.to_datetime(_er["earnings_date"]).dt.date
        _future = _future[_future >= date.today()]
        next_er = _future.min() if not _future.empty else None

    # Smart money on the full history (OBV is path-dependent)
    sm = get_sm_spirit_unified_v2(_df_levels, sector=str(meta.get("sector", "Unknown")))
    rating = compute_institutional_rating(
        ai_score=ai_score, ma_sig=ma_sig, latest_rsi=tm["rsi"], upside=float(upside),
        pe_v=float(meta_enriched.get("forward_pe") or meta_enriched.get("pe_ratio") or 0),
        peg_v=float(meta_enriched.get("peg_ratio") or 0), sector=str(meta.get("sector", "")),
        w52_pos=tm["w52_pos"], rr=tm["rr_score"], sm_status=sm["signal"],
        sm_strength=sm["strength"], sm_layer=sm["layer"], value_score=scores["value"])
    act_str = rating["action_label"]
    from core.signal_matrix import ACTION_COLOURS, action_description
    act_color = ACTION_COLOURS.get(act_str, rating["action_color"])
    act_desc = action_description(act_str, rating, s1=tm["s1"], price=float(cur_p), rsi=tm["rsi"],
                                  sm_signal=sm["signal"])

    return DeepDive(ticker=deep_ticker, meta=meta, meta_enriched=meta_enriched, df_deep=df_deep,
                    df_fin=df_fin, cur_p=cur_p, target_p=target_p, upside=upside, z_score=z_score,
                    ai_score=ai_score, scores=scores, tm=tm, ma_sig=ma_sig, tp1=tp1, tp2=tp2, vin=vin, relval=relval,
                    next_earnings=next_er, sm=sm, rating=rating, act_str=act_str, act_color=act_color,
                    act_desc=act_desc)
