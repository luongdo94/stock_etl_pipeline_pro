"""Master screener table: Quality score, institutional rating, flow and tactical levels per ticker.
Pure (no Streamlit) so the ETL can snapshot exactly what the dashboard shows."""
import pandas as pd

from core.decision import NO_DATA as NO_DATA_STANCE

from core.levels import get_tactical_metrics
from core.rating import compute_institutional_rating, verdict
from core.smart_money import get_volume_flow
from core.decision import BUY, RESEARCH, decide, load_rules, to_date, with_conflict
from core.valuation import uses_book_model, valuation_inputs
from core.scoring import score_universe
from etl.utils import clean_upside_pct


def insider_activity(row) -> dict:
    """Insider buying / selling over 6 months from the provider's net share purchase activity:
    {status: NET BUY / NET SELL / NET FLAT / NO TRADES / PARTIAL DATA / NO DATA, net_pct (% of insider holdings),
    net_shares, buys, sells, as_of (when the summary was fetched)}. A missing summary line stays unknown.
    All-zero rows are "no data" (most non-US listings): the transform's NEUTRAL label cannot tell them apart."""
    def num(k):
        v = row.get(k)
        return float(v) if v is not None and pd.notnull(v) else None
    buys, sells, net = num("insider_purchases_6m"), num("insider_sales_6m"), num("insider_net_shares_6m")
    pb, ps = num("insider_pct_buy"), num("insider_pct_sell")
    as_of = row.get("insider_extracted_at")
    as_of = as_of if as_of is not None and pd.notnull(as_of) else None
    if not any(x for x in (buys, sells, net, pb, ps)):
        return {"status": "NO DATA", "net_pct": None, "net_shares": None, "buys": None, "sells": None, "as_of": as_of}
    if net is None:                                      # one side of the summary is missing: no net verdict
        status = "PARTIAL DATA"
    elif net > 0:
        status = "NET BUY"
    elif net < 0:
        status = "NET SELL"
    else:                                                # bought and sold the same number of shares
        status = "NET FLAT" if (buys or sells) else "NO TRADES"
    net_pct = (pb - ps) * 100 if pb is not None and ps is not None else None     # both shares known, or N/A
    return {"status": status, "net_pct": net_pct, "net_shares": net, "buys": buys, "sells": sells, "as_of": as_of}


def _by_ticker(df) -> dict:
    """{ticker: that ticker's rows} in original order (df.iloc of the group positions); {} when there is nothing."""
    if df is None or df.empty or "ticker" not in df.columns:
        return {}
    return {t: df.iloc[idx] for t, idx in df.groupby("ticker", sort=False, observed=True).indices.items()}


def build_screener_table(_companies_df, _prices_df, _quarterly_fin, _annual_fin,
                         _hist_fcf=None, _macro=None, _estimates=None,
                         track_record_ok=False, _earnings_cal=None, _sec_annual_history=None) -> pd.DataFrame:
    """
    One row per investable ticker. Two distinct outputs:
      - "Signal"   (column "Action"): 6-pillar technical + quality composite — an INPUT
      - "Decision": the single recommendation, from core.decision.decide — the same
                    function and inputs as the Decision Summary in the Stock Analysis tab
    """
    _sec_by = _by_ticker(_sec_annual_history)
    _rules = load_rules()
    _track_record_ok = bool(track_record_ok)
    _hist_fcf = _hist_fcf if _hist_fcf is not None else pd.DataFrame()
    # Quality / Value / Momentum for the whole universe at once (percentiles need the peer group)
    _scores = score_universe(_companies_df, _annual_fin, _prices_df, hist_fcf=_hist_fcf, estimates=_estimates)
    # Exclude non-investable instruments: indices & volatility measures
    _non_equities = {"^VIX", "SPY", "^GSPC", "^DJI", "^IXIC", "^TNX", "DX-Y.NYB", "^IRX", "CL=F", "GC=F"}
    _non_equity_sectors = {"Benchmark", "Volatility"}
    screener_rows = []

    # ── Pre-compute 2-quarter consecutive momentum for EPS & Revenue ────────────
    # Logic: label = 'Accelerating' if both q[-1] and q[-2] QoQ growth > +10%,
    #                'Decelerating' if both < -10%, else 'Neutral'
    # Uses QoQ (quarter-over-quarter) because "2 consecutive quarters" is inherently QoQ —
    # comparing Q3->Q4->Q1 in sequence, not the same quarter of the prior year (YoY).
    def _two_quarter_momentum(ticker, qoq_col, threshold=10.0):
        """Returns 'Accelerating', 'Decelerating', 'Neutral' (both quarters known), or 'N/A' (not known).
        Uses QoQ growth rates for 2 most recent consecutive quarters.
        """
        if (_quarterly_fin is None or qoq_col not in _quarterly_fin.columns
                or (getattr(_quarterly_fin, "attrs", {}) or {}).get("error")):
            return 'N/A'                                   # quarterly data missing or unreadable (K7-02)
        t_q = _q_by.get(ticker, _quarterly_fin.iloc[:0]).sort_values('report_date', ascending=False)
        if len(t_q) < 2:
            return 'N/A'                                   # fewer than two quarters: unknown, not neutral
        # E3: the TWO MOST RECENT quarters, both with a QoQ figure. dropna() used to skip a gap and pair two
        # non-consecutive readings; QoQ is NULL unless the previous quarter is adjacent, so two non-null
        # latest values are three consecutive quarters.
        vals = t_q[qoq_col].head(2).tolist()
        if len(vals) < 2 or any(pd.isna(v) for v in vals):
            return 'N/A'
        if vals[0] > threshold and vals[1] > threshold:
            return 'Accelerating'
        if vals[0] < -threshold and vals[1] < -threshold:
            return 'Decelerating'
        return 'Neutral'

    # PERF-03: each issuer's rows come from a partition built once instead of masking the whole frame per issuer.
    # Same rows, order, index and dtypes as the old `frame[frame.ticker == t]`, and the same sorts are applied after;
    # a ticker without rows still gets the full frame (financial helpers then filter it as before).
    _px_by, _fcf_by, _ann_by, _q_by = (_by_ticker(f) for f in (_prices_df, _hist_fcf, _annual_fin, _quarterly_fin))

    # Build lookups by ticker (QoQ — consecutive quarter growth). These used to read the YoY columns while the label,
    # presets and docstring promised QoQ (audit R3-E). QoQ is seasonal: read it with the fiscal calendar in mind.
    _eps_mom_lookup = {
        t: _two_quarter_momentum(t, 'eps_growth_qoq_pct')
        for t in _companies_df['ticker'].unique()
    }
    _rev_mom_lookup = {
        t: _two_quarter_momentum(t, 'revenue_growth_qoq_pct')
        for t in _companies_df['ticker'].unique()
    }

    # Next earnings date per ticker: the scanner's Decision gets the same blackout gate as the deep dive
    _next_er = {}
    if _earnings_cal is not None and not _earnings_cal.empty and {"ticker", "earnings_date"} <= set(_earnings_cal.columns):
        _today = pd.Timestamp.today().normalize()
        _er = _earnings_cal[["ticker", "earnings_date"]].copy()
        _er["earnings_date"] = pd.to_datetime(_er["earnings_date"], errors="coerce")
        _er = _er[_er["earnings_date"] >= _today]
        _next_er = {t: d.date() for t, d in _er.groupby("ticker")["earnings_date"].min().items()}
    
    for _, row in _companies_df.iterrows():
        ticker = row['ticker']
        # Skip indices and volatility
        if ticker in _non_equities: continue
        if str(row.get('sector', '')).strip() in _non_equity_sectors: continue
        ticker_prices = _px_by.get(ticker, _prices_df.iloc[:0]).sort_values('date')
        _t_fcf, _t_ann = _fcf_by.get(ticker, _hist_fcf), _ann_by.get(ticker, _annual_fin)
        if ticker_prices.empty or ticker not in _scores.index:
            # K7-01: kept as a row the engine cannot judge (its own NO_DATA stance, no figures), so "Needs data" and
            # the universe count show it instead of the stock silently vanishing
            screener_rows.append({"Ticker": ticker, "Company": row.get("company", ticker), "Sector": row.get("sector"),
                                  "Region": row.get("region"), "Decision": NO_DATA_STANCE, "Action": "N/A",
                                  "Verdict": "N/A", "Confidence": "LOW",
                                  "Why": "no price history on file" if ticker_prices.empty else "not scored",
                                  "Missing": ["price history"] if ticker_prices.empty else ["scores"],
                                  "_beyond_model": False,
                                  # R7-01: empty containers, not NaN (readers call dict() / .get() on them)
                                  "Components": {}, "Flags": "", "_valuation": {}, "_metrics": {}, "_reasons": [],
                                  "_warnings": [], "Conflicts": "", "EPS Momentum": "N/A", "Rev Momentum": "N/A",
                                  # the columns the app and the snapshot read, so a table of placeholders only is complete
                                  "Price": float("nan"), "Quality": float("nan"), "Value": float("nan"),
                                  "Momentum": float("nan"), "Revisions": float("nan"), "Upside (%)": float("nan"),
                                  "RSI (14)": float("nan"), "MCap (B)": float("nan"), "Smart Money": "N/A",
                                  "Trend": "N/A"})
            continue
        
        # RSI (Pre-calculated in transform.py)
        latest_rsi = ticker_prices["rsi"].iloc[-1] if not ticker_prices.empty else 50
        
        cur_p = ticker_prices["price_close"].iloc[-1]
        upside = clean_upside_pct(row.get("target_mean_price"), cur_p, row.get("avg_5y_price"), unknown=float("nan"))
        
        if len(ticker_prices) >= 2:
            prev_p = ticker_prices["price_close"].iloc[-2]
            chg_1d = ((cur_p / prev_p) - 1) * 100 if prev_p > 0 else float("nan")
        else:
            chg_1d = float("nan")                        # one close: no daily change is known
            
        mcap = row.get("market_cap", 0)
        mcap_b = (mcap / 1e9) if pd.notnull(mcap) and mcap > 0 else float("nan")      # unknown, not "€0.0B"
        
        latest_p = ticker_prices.iloc[-1]
        # ── SCORES (core/scoring.py): Quality, Value, Momentum ────────────────────────
        _sc = _scores.loc[ticker]
        quality_raw = float(_sc["quality"]) if pd.notnull(_sc["quality"]) else None
        ai_score = quality_raw if quality_raw is not None else 50.0   # display / Signal only — the Decision gets quality_raw
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
        _vin = valuation_inputs(row, float(cur_p), _t_fcf, ticker, _macro or {}, _t_ann, prices=ticker_prices,
                                sec_annual_history=_sec_by.get(ticker))
        _dec = decide(
            price=float(cur_p), vin=_vin, stop_loss=_tm["stop_loss"], meta=row,
            scores={"quality": quality_raw, "value": value_score, "flags": _sc["flags"], "coverage": _cov,
                    "missing": _sc["missing"], "momentum": None if pd.isna(momentum) else momentum,
                    "adv_eur": _sc["adv_eur"] if pd.notnull(_sc["adv_eur"]) else None},
            price_date=to_date(latest_p.get("date")), next_earnings=_next_er.get(ticker),
            track_record_ok=_track_record_ok, rules=_rules)
        _mos = (_vin["base"] / float(cur_p) - 1) * 100 if (_vin["base"] and _vin["reliable"]) else None

        # Volume flow (Unified v6.0 with sector awareness)
        sm_result = get_volume_flow(ticker_prices, sector=str(row.get('sector', 'Unknown')))
        sm_spirit = sm_result["signal"]
        sm_strength = sm_result["strength"]
        sm_layer = sm_result["layer"]

        # ── Signal: context for the Decision (reward/risk is the Decision's own) ──
        def _signal(stance):
            return compute_institutional_rating(
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
                decision_stance = stance,
            )
        _rating = _signal(_dec.stance)
        if _rating["clamped"] and _dec.stance in (BUY, RESEARCH):
            # BUY vs an UNFAVOURABLE set-up: a review state, not a relabelled signal (R3-E)
            _dec = with_conflict(_dec, f"timing context is {_rating['raw_label']} (trend / value against "
                                       f"the valuation case)")
            _rating = _signal(_dec.stance)
        action_label = _rating["action_label"]

        # Additional metrics
        div_yield = float(row.get('dividend_yield_pct')) if pd.notnull(row.get('dividend_yield_pct')) else float("nan")
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
        _np = row.get('net_payout_yield_pct')
        net_payout = float(_np) if pd.notnull(_np) else float("nan")                  # unknown, not a 0 % payout
        vol_30d = row.get('volatility_30d', 0) or 0
        short_pct = (row.get('short_percent_of_float') * 100) if pd.notnull(row.get('short_percent_of_float')) else float("nan")

        # measured values behind the Quality sub-scores (shown next to them); insider net activity over 6 months
        _met = dict(_sc["metrics"]) if isinstance(_sc.get("metrics"), dict) else {}
        _nde = _met.get("net_debt_ebitda")
        roic_pct = _met.get("roic_pct") if _met.get("roic_is_roic") else None
        _ins = insider_activity(row)
        # reverse DCF: the starting FCF growth the price assumes (DCF only; the bank model has no growth to invert)
        _implied = _vin.get("implied_growth") if _vin.get("model") == "dcf" and (_vin.get("fcfe") or 0) > 0 else None
        _exp_irr = (_vin.get("irr") or {}).get("expected") if _vin.get("reliable") else None

        # Cash-flow and leverage ratios are meaningless for banks/insurers (deposits, float)
        if uses_book_model(row, _t_fcf, ticker):
            fcf_margin, debt_ebitda = float("nan"), float("nan")
            roic_pct, _nde = None, None
            _met = {**_met, "fcf_conversion": None, "net_debt_ebitda": None, "roic_pct": None, "fcf_yield_pct": None}

        screener_rows.append({
            "Ticker": ticker,
            "Company": row['company'],
            "Sector": row['sector'],
            "Peer group": _sc.get("peer_group", ""),        # who the Quality / Value percentiles were ranked against
            "Decision": _dec.stance,
            "Confidence": _dec.confidence,
            "MoS (%)": round(_mos, 0) if _mos is not None else float("nan"),   # numeric column → blank, not "None"
            "Action": action_label,
            "Verdict": verdict(_dec.stance, action_label),
            "Quality": quality_raw if quality_raw is not None else float("nan"),   # unknown stays visibly unknown
            "Conflicts": "; ".join(_dec.conflicts),
            # what the Decision says about itself — shown, never re-derived by a view: the first reason, then the lists
            "Why": str(_dec.reasons[0]) if _dec.reasons else "",
            "_beyond_model": bool(_dec.beyond_model),
            "_reasons": [str(r) for r in _dec.reasons], "_warnings": [str(w) for w in _dec.warnings],
            "_confidence_notes": [str(n) for n in _dec.confidence_notes], "_invalidation": [str(i) for i in _dec.invalidation],
            "Value": value_score if value_score is not None else float("nan"),
            "Momentum": momentum,
            "Revisions": float(_sc["revisions"]) if pd.notnull(_sc["revisions"]) else float("nan"),
            # raw inputs of the shared sizing rule (core.decision.size_limits) for portfolio rebalances (R4-F02)
            "_adv_eur": float(_sc["adv_eur"]) if pd.notnull(_sc["adv_eur"]) else float("nan"),
            "_stop_distance_pct": _dec.downside_pct if _dec.downside_pct is not None else float("nan"),
            "_thesis_stop": _dec.stop if _dec.stop is not None else float("nan"),   # recorded once at opening (P1/5)
            "ADV (EUR M)": round(float(_sc["adv_eur"]) / 1e6, 1) if pd.notnull(_sc["adv_eur"]) else float("nan"),
            "Coverage (%)": float(_cov) if pd.notnull(_cov) else float("nan"),         # unknown, not 0 % observable
            "Flags": _sc["flags"],
            "Missing": _sc["missing"],
            "Components": _sc["components"],
            "_quality_raw": quality_raw if quality_raw is not None else float("nan"),
            "_valuation": {k: _vin.get(k) for k in ("model", "base", "bear", "bull", "expected_value", "fcfe", "growth",
                                                     "cost_of_equity", "terminal_growth", "terminal_share", "reliable",
                                                     "fcfe_adjustments", "currency")},
            "Upside (%)": round(upside, 1),
            "1D Chg (%)": round(chg_1d, 2),
            "Price": cur_p,
            "MCap (B)": round(mcap_b, 1),
            "RSI (14)": round(latest_rsi, 1),
            "Z-Score": round(ticker_prices['price_z_score'].iloc[-1], 2) if 'price_z_score' in ticker_prices.columns else float("nan"),
            "Smart Money": sm_spirit if sm_spirit != "N/A" else f"N/A ({sm_result['reason']})",
            "vs MA200 (%)": round(ticker_prices['pct_from_ma200'].iloc[-1], 1) if 'pct_from_ma200' in ticker_prices.columns else float("nan"),
            "Yield (%)": round(div_yield, 2),
            "Net Payout (%)": round(net_payout, 2),
            "FCF Margin (%)": round(fcf_margin, 1),
            "ROE (%)": round(roe_val, 1),
            "P/E (Fwd)": round(float(row['forward_pe']), 1) if pd.notnull(row.get('forward_pe')) else float("nan"),
            "EV/EBITDA": round(ev_ebitda, 1),
            "PEG": round(float(row['peg_ratio']), 2) if pd.notnull(row.get('peg_ratio')) else float("nan"),
            "Debt/EBITDA": round(debt_ebitda, 2),
            "ROIC (%)": round(roic_pct, 1) if roic_pct is not None else float("nan"),
            "Net Debt/EBITDA": round(_nde, 2) if _nde is not None and _nde < 99 else float("nan"),
            "FCF Conversion (x)": round(_met["fcf_conversion"], 2) if _met.get("fcf_conversion") is not None else float("nan"),
            "Insider 6M (%)": round(_ins["net_pct"], 2) if _ins["net_pct"] is not None else float("nan"),
            # valuation every reader can compare: what the price yields, and what growth it already assumes
            "FCF Yield (%)": round(_met["fcf_yield_pct"], 1) if _met.get("fcf_yield_pct") is not None else float("nan"),
            "Implied Growth (%)": round(_implied * 100, 1) if _implied is not None else float("nan"),
            # same basis on both sides: the model's history-anchored starting FCF growth vs the starting FCF growth the
            # price implies, in the same DCF (it used to subtract FCF growth from a REVENUE CAGR)
            "Growth Gap (pp)": (round((_vin["growth"] - _implied) * 100, 1)
                                if _implied is not None and _vin.get("growth") is not None else float("nan")),
            "Exp. Return (%/yr)": round(_exp_irr * 100, 1) if _exp_irr is not None else float("nan"),
            "Next ER (days)": (_next_er[ticker] - pd.Timestamp.today().date()).days if ticker in _next_er else float("nan"),
            "_metrics": _met,
            "_insider": _ins,
            "Vol 30D (%)": round(vol_30d, 1) if vol_30d else 0,
            "Short %": round(short_pct, 1),
            "Trend": latest_p.get('ma_signal', 'NEUTRAL'),
            "Region": row['region'],
            "EPS Momentum": _eps_mom_lookup.get(ticker, 'N/A'),      # R7-09: not computed is unknown, not Neutral
            "Rev Momentum": _rev_mom_lookup.get(ticker, 'N/A'),
        })
        
    return pd.DataFrame(screener_rows)
