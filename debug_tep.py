import pandas as pd
import sys
import os

# Ensure app is in path
sys.path.append('.')
from app import load_data, get_master_screener_data, get_tactical_metrics, compute_institutional_rating, compute_score, get_sm_spirit_unified_v2

def debug_ticker(target_ticker):
    print(f"--- Debugging {target_ticker} ---")
    prices_full, companies_full, monthly_full, annual_fin, quarterly_fin, earnings_cal, dq_warnings, hist_fcf_full, hist_fcf_q_full, etl_audit, total_universe_size, earnings_surprise_full = load_data()
    
    _meta_df = companies_full[companies_full["ticker"] == target_ticker]
    if _meta_df.empty:
        print(f"Ticker {target_ticker} not found.")
        return
    
    meta = _meta_df.iloc[0]
    df_deep = prices_full[prices_full["ticker"] == target_ticker].sort_values("date")
    
    cur_p = df_deep['price_close'].iloc[-1]
    target_p = meta.get('target_mean_price', 0)
    upside = ((target_p / cur_p) - 1) * 100 if target_p > 0 else 0

    latest_tech = df_deep.iloc[-1]
    meta_enriched = meta.to_dict()
    for col in ['pe_ratio', 'peg_ratio', 'price_to_book', 'roe', 'fcf_margin', 'dividend_yield_pct']:
        val = meta_enriched.get(col)
        try:
            meta_enriched[col] = float(val) if pd.notnull(val) else None
        except:
            meta_enriched[col] = None

    z_score = df_deep['price_z_score'].iloc[-1] if 'price_z_score' in df_deep.columns else 0
    if pd.isna(z_score): z_score = 0

    meta_enriched['rsi'] = float(latest_tech.get('rsi', 50))
    meta_enriched['ma_signal'] = str(latest_tech.get('ma_signal', 'NEUTRAL'))
    meta_enriched['price_z_score'] = float(z_score)
    meta_enriched['upside_pct'] = float(upside)
    
    ai_score_dd = compute_score(meta_enriched)
    
    _tm = get_tactical_metrics(df_deep, cur_p)
    _ma_sig = str(latest_tech.get("ma_signal", meta.get("ma_signal", "NEUTRAL")))
    _rsi_val = _tm["rsi"]
    _w52_pos = _tm["w52_pos"]
    
    p_sm = get_sm_spirit_unified_v2(df_deep)
    
    dd_pe_v = float(meta_enriched.get("forward_pe") or meta_enriched.get("pe_ratio") or 0)
    
    rating_dd = compute_institutional_rating(
        ai_score   = ai_score_dd,
        ma_sig     = _ma_sig,
        latest_rsi = _rsi_val,
        upside     = float(upside),
        pe_v       = dd_pe_v,
        peg_v      = float(meta_enriched.get("peg_ratio") or 0),
        sector     = str(meta.get("sector", "")),
        w52_pos    = _w52_pos,
        rr         = _tm["rr_score"],
        sm_status  = p_sm
    )
    
    print("=== DEEP DIVE INPUTS ===")
    print(f"ai_score: {ai_score_dd}")
    print(f"ma_sig: {_ma_sig}")
    print(f"latest_rsi: {_rsi_val}")
    print(f"upside: {float(upside)}")
    print(f"pe_v: {dd_pe_v}")
    print(f"peg_v: {float(meta_enriched.get('peg_ratio') or 0)}")
    print(f"sector: {str(meta.get('sector', ''))}")
    print(f"w52_pos: {_w52_pos}")
    print(f"rr: {_tm['rr_score']}")
    print(f"sm_status: {p_sm}")
    print(f"--> DD Result Action: {rating_dd['action_label']}, pts: {rating_dd['pts']}")
    print(f"--> DD pe_v calculation details: forward_pe={meta_enriched.get('forward_pe')}, pe_ratio={meta_enriched.get('pe_ratio')}")
    
    print("\n=== SCREENER INPUTS ===")
    # Now simulate Screener inside get_master_screener_data
    score_input = meta.to_dict()
    score_input['rsi'] = float(latest_tech.get('rsi', 50))
    score_input['ma_signal'] = str(latest_tech.get('ma_signal', 'NEUTRAL'))
    score_input['price_z_score'] = float(latest_tech.get('price_z_score', 0))
    score_input['upside_pct'] = float(upside)
    for col in ['pe_ratio', 'peg_ratio', 'price_to_book', 'roe', 'fcf_margin', 'dividend_yield_pct']:
        val = score_input.get(col)
        try: score_input[col] = float(val) if pd.notnull(val) else None
        except: score_input[col] = None
        
    ai_score_scr = compute_score(score_input)
    
    scr_pe_v = float(score_input.get('forward_pe') or score_input.get('pe_ratio') or 0)
    scr_ma_sig = str(latest_tech.get('ma_signal', 'NEUTRAL'))
    
    _tm_scr = get_tactical_metrics(
        df_deep,
        cur_p,
        analyst_target=float(meta.get('target_mean_price') or 0)
    )
    
    rating_scr = compute_institutional_rating(
        ai_score   = ai_score_scr,
        ma_sig     = scr_ma_sig,
        latest_rsi = _tm_scr["rsi"],
        upside     = float(upside),
        pe_v       = scr_pe_v,
        peg_v      = float(score_input.get('peg_ratio') or 0),
        sector     = str(meta.get('sector', '')),
        w52_pos    = _tm_scr["w52_pos"],
        rr         = _tm_scr["rr_score"],
        sm_status  = p_sm
    )
    
    print(f"ai_score: {ai_score_scr}")
    print(f"ma_sig: {scr_ma_sig}")
    print(f"latest_rsi: {_tm_scr['rsi']}")
    print(f"upside: {float(upside)}")
    print(f"pe_v: {scr_pe_v}")
    print(f"peg_v: {float(score_input.get('peg_ratio') or 0)}")
    print(f"sector: {str(meta.get('sector', ''))}")
    print(f"w52_pos: {_tm_scr['w52_pos']}")
    print(f"rr: {_tm_scr['rr_score']}")
    print(f"sm_status: {p_sm}")
    print(f"--> SCR Result Action: {rating_scr['action_label']}, pts: {rating_scr['pts']}")
    print(f"--> SCR pe_v calculation details: forward_pe={score_input.get('forward_pe')}, pe_ratio={score_input.get('pe_ratio')}")


if __name__ == "__main__":
    debug_ticker("TEP.PA")
