"""Master screener table: Quality score, institutional rating, flow and tactical levels per ticker.
Pure (no Streamlit) so the ETL can snapshot exactly what the dashboard shows."""
import pandas as pd

from core.levels import get_tactical_metrics
from core.rating import compute_institutional_rating
from core.smart_money import get_sm_spirit_unified_v2
from core.decision import decide, load_rules, to_date
from core.valuation import uses_book_model, valuation_inputs
from core.scoring import score_universe
from etl.utils import clean_upside_pct


def build_screener_table(_companies_df, _prices_df, _quarterly_fin, _annual_fin,
                         _hist_fcf=None, _macro=None, _estimates=None,
                         track_record_ok=False) -> pd.DataFrame:
    """
    One row per investable ticker. Two distinct outputs:
      - "Signal"   (column "Action"): 6-pillar technical + quality composite — an INPUT
      - "Decision": the single recommendation, from core.decision.decide — the same
                    function and inputs as the Decision Summary in the Stock Analysis tab
    """
    _rules = load_rules()
    _track_record_ok = bool(track_record_ok)
    _hist_fcf = _hist_fcf if _hist_fcf is not None else pd.DataFrame()
    # Quality / Value / Momentum for the whole universe at once (percentiles need the peer group)
    _scores = score_universe(_companies_df, _annual_fin, _prices_df, hist_fcf=_hist_fcf, estimates=_estimates)
    # Exclude non-investable instruments: indices & volatility measures
    _non_equities = {"^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"}
    _non_equity_sectors = {"Benchmark", "Volatility"}
    screener_rows = []

    # ── Pre-compute 2-quarter consecutive momentum for EPS & Revenue ────────────
    # Logic: label = 'Accelerating' if both q[-1] and q[-2] QoQ growth > +10%,
    #                'Decelerating' if both < -10%, else 'Neutral'
    # Uses QoQ (quarter-over-quarter) because "2 consecutive quarters" is inherently QoQ —
    # comparing Q3->Q4->Q1 in sequence, not the same quarter of the prior year (YoY).
    def _two_quarter_momentum(ticker, qoq_col, threshold=10.0):
        """Returns 'Accelerating', 'Decelerating', or 'Neutral'.
        Uses QoQ growth rates for 2 most recent consecutive quarters.
        """
        t_q = _quarterly_fin[_quarterly_fin['ticker'] == ticker].sort_values('report_date', ascending=False)
        if len(t_q) < 2:
            return 'Neutral'
        vals = t_q[qoq_col].dropna().head(2).tolist()
        if len(vals) < 2:
            return 'Neutral'
        if vals[0] > threshold and vals[1] > threshold:
            return 'Accelerating'
        if vals[0] < -threshold and vals[1] < -threshold:
            return 'Decelerating'
        return 'Neutral'

    # Build lookups by ticker (QoQ — consecutive quarter growth)
    _eps_mom_lookup = {
        t: _two_quarter_momentum(t, 'eps_growth_qoq_pct')
        for t in _companies_df['ticker'].unique()
    }
    _rev_mom_lookup = {
        t: _two_quarter_momentum(t, 'revenue_growth_qoq_pct')
        for t in _companies_df['ticker'].unique()
    }
    
    for _, row in _companies_df.iterrows():
        ticker = row['ticker']
        # Skip indices and volatility
        if ticker in _non_equities: continue
        if str(row.get('sector', '')).strip() in _non_equity_sectors: continue
        ticker_prices = _prices_df[_prices_df['ticker'] == ticker].sort_values('date')
        if ticker_prices.empty or ticker not in _scores.index: continue
        
        # RSI (Pre-calculated in transform.py)
        latest_rsi = ticker_prices["rsi"].iloc[-1] if not ticker_prices.empty else 50
        
        cur_p = ticker_prices["price_close"].iloc[-1]
        upside = clean_upside_pct(row.get("target_mean_price"), cur_p, row.get("avg_5y_price"))
        
        if len(ticker_prices) >= 2:
            prev_p = ticker_prices["price_close"].iloc[-2]
            chg_1d = ((cur_p / prev_p) - 1) * 100 if prev_p > 0 else 0
        else:
            chg_1d = 0
            
        mcap = row.get("market_cap", 0)
        mcap_b = (mcap / 1e9) if pd.notnull(mcap) and mcap > 0 else 0
        
        latest_p = ticker_prices.iloc[-1]
        # ── SCORES (core/scoring.py): Quality, Value, Momentum ────────────────────────
        _sc = _scores.loc[ticker]
        ai_score = float(_sc["quality"]) if pd.notnull(_sc["quality"]) else 50.0     # no data = neutral
        value_score = float(_sc["value"]) if pd.notnull(_sc["value"]) else None
        momentum = float(_sc["momentum"]) if pd.notnull(_sc["momentum"]) else float("nan")
        _cov = min(_sc["quality_coverage"], _sc["value_coverage"])

        # ── Decision (single recommendation; identical logic to the Stock Analysis panel) ──
        ma_sig = str(latest_p.get('ma_signal', 'NEUTRAL'))
        pe_v   = float(row.get('forward_pe') or row.get('pe_ratio') or 0)
        peg_v  = float(row.get('peg_ratio') or 0)

        # Technical levels (same formula as Deep Dive): the stop is one input of the Decision's thesis stop
        _tm = get_tactical_metrics(
            ticker_prices,
            cur_p,
            analyst_target=float(row.get('target_mean_price') or 0)
        )
        _vin = valuation_inputs(row, float(cur_p), _hist_fcf, ticker, _macro or {}, _annual_fin)
        _dec = decide(
            price=float(cur_p), vin=_vin, stop_loss=_tm["stop_loss"], meta=row,
            scores={"quality": ai_score, "value": value_score, "flags": _sc["flags"], "coverage": _cov,
                    "missing": _sc["missing"]},
            price_date=to_date(latest_p.get("date")), track_record_ok=_track_record_ok, rules=_rules)
        _mos = (_vin["base"] / float(cur_p) - 1) * 100 if (_vin["base"] and _vin["reliable"]) else None

        # Volume flow (Unified v6.0 with sector awareness)
        sm_result = get_sm_spirit_unified_v2(ticker_prices, sector=str(row.get('sector', 'Unknown')))
        sm_spirit = sm_result["signal"]
        sm_strength = sm_result["strength"]
        sm_layer = sm_result["layer"]

        # ── Signal: context for the Decision (reward/risk is the Decision's own) ──
        _rating = compute_institutional_rating(
            ai_score   = ai_score,
            ma_sig     = ma_sig,
            latest_rsi = _tm["rsi"],
            upside     = float(upside),
            pe_v       = pe_v,
            peg_v      = peg_v,
            sector     = str(row.get('sector', '')),
            w52_pos    = _tm["w52_pos"],
            rr         = _dec.reward_risk,
            sm_status  = sm_spirit,
            sm_strength = sm_strength,
            sm_layer   = sm_layer,
            value_score = value_score,
            decision_stance = _dec.stance,
        )
        action_label = _rating["action_label"]

        # Additional metrics
        div_yield = float(row.get('dividend_yield_pct', 0)) if pd.notnull(row.get('dividend_yield_pct')) else 0
        # Unknown stays NaN (blank in the table) — sentinels like 999 / 99 / 0 used to be filtered
        # and ranked as if they were real values (every stock without a forward P/E vanished from
        # the scanner because 999 > the "Max Forward P/E" slider default).
        fcf_margin = float(row.get('fcf_margin')) if pd.notnull(row.get('fcf_margin')) else float("nan")
        
        # Safe Financial Metrics (Handling pd.NA)
        eb_val = row.get('ebitda')
        td_val = row.get('total_debt')
        ebitda = float(eb_val) if pd.notnull(eb_val) else None
        total_debt = float(td_val) if pd.notnull(td_val) else None
        if total_debt is not None and total_debt <= 0:
            debt_ebitda = 0.0                                   # debt-free
        elif total_debt is not None and ebitda is not None and ebitda > 0:
            debt_ebitda = min(total_debt / ebitda, 99)
        else:
            debt_ebitda = float("nan")                          # unknown, or debt with EBITDA ≤ 0 (n/m)

        ev_eb_val = row.get('ev_to_ebitda')
        ev_ebitda = float(ev_eb_val) if pd.notnull(ev_eb_val) else float("nan")

        roe_raw = row.get('roe')
        roe_val = (float(roe_raw) * 100) if pd.notnull(roe_raw) else float("nan")
        net_payout = row.get('net_payout_yield_pct', 0) or 0
        vol_30d = row.get('volatility_30d', 0) or 0
        short_pct = (row.get('short_percent_of_float', 0) * 100) if pd.notnull(row.get('short_percent_of_float')) else 0

        # Cash-flow and leverage ratios are meaningless for banks/insurers (deposits, float)
        if uses_book_model(row, _hist_fcf, ticker):
            fcf_margin, debt_ebitda = float("nan"), float("nan")

        screener_rows.append({
            "Ticker": ticker,
            "Company": row['company'],
            "Sector": row['sector'],
            "Decision": _dec.stance,
            "Confidence": _dec.confidence,
            "MoS (%)": round(_mos, 0) if _mos is not None else float("nan"),   # numeric column → blank, not "None"
            "Action": action_label,
            "Quality": ai_score,
            "Value": value_score if value_score is not None else float("nan"),
            "Momentum": momentum,
            "Revisions": float(_sc["revisions"]) if pd.notnull(_sc["revisions"]) else float("nan"),
            "ADV (EUR M)": round(float(_sc["adv_eur"]) / 1e6, 1) if pd.notnull(_sc["adv_eur"]) else float("nan"),
            "Coverage (%)": float(_cov) if pd.notnull(_cov) else 0.0,
            "Flags": _sc["flags"],
            "Missing": _sc["missing"],
            "Components": _sc["components"],
            "Upside (%)": round(upside, 1),
            "1D Chg (%)": round(chg_1d, 2),
            "Price": cur_p,
            "MCap (B)": round(mcap_b, 1),
            "RSI (14)": round(latest_rsi, 1),
            "Z-Score": round(ticker_prices['price_z_score'].iloc[-1] if 'price_z_score' in ticker_prices.columns else 0, 2),
            "Smart Money": sm_spirit,
            "vs MA200 (%)": round(ticker_prices['pct_from_ma200'].iloc[-1] if 'pct_from_ma200' in ticker_prices.columns else 0, 1),
            "Yield (%)": round(div_yield, 2),
            "Net Payout (%)": round(net_payout, 2),
            "FCF Margin (%)": round(fcf_margin, 1),
            "ROE (%)": round(roe_val, 1),
            "P/E (Fwd)": round(float(row['forward_pe']), 1) if pd.notnull(row.get('forward_pe')) else float("nan"),
            "EV/EBITDA": round(ev_ebitda, 1),
            "PEG": round(float(row['peg_ratio']), 2) if pd.notnull(row.get('peg_ratio')) else float("nan"),
            "Debt/EBITDA": round(debt_ebitda, 2),
            "Vol 30D (%)": round(vol_30d, 1) if vol_30d else 0,
            "Short %": round(short_pct, 1),
            "Trend": latest_p.get('ma_signal', 'NEUTRAL'),
            "Region": row['region'],
            "EPS Momentum": _eps_mom_lookup.get(ticker, 'Neutral'),
            "Rev Momentum": _rev_mom_lookup.get(ticker, 'Neutral'),
        })
        
    return pd.DataFrame(screener_rows)
