"""Layer 4 — diagnostic metric cards."""
from datetime import date

import pandas as pd
import streamlit as st

from views.stock_analysis.layout import layer_banner

from services.market_data import get_forex_rates
from ui.components import render_metric_row
from ui.icons import render_header



def render(dd, ctx):
    deep_ticker = dd.ticker
    earnings_cal = ctx["earnings_cal"]
    meta = dd.meta
    target_p = dd.target_p
    upside = dd.upside
    z_score = dd.z_score
    st.markdown("---")
    layer_banner(4, "Deep diagnostics & raw data", "#e74c3c")
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
        kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)

        with kcol1:
            st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Valuation & Size</div>", unsafe_allow_html=True)
            _mc = meta.get('market_cap'); m_cap = float(_mc) if not pd.isna(_mc) and _mc else 0.0
            if m_cap >= 1e12: m_cap_txt = f"€{m_cap/1e12:.2f}T"
            elif m_cap >= 1e9: m_cap_txt = f"€{m_cap/1e9:.1f}B"
            else: m_cap_txt = f"€{m_cap/1e6:.0f}M"
            
            render_metric_row("Market Cap", m_cap_txt)
            pe_val = f"{meta['pe_ratio']:.1f}" if pd.notnull(meta['pe_ratio']) else "N/A"
            render_metric_row("P/E (trailing)", pe_val, help_text="Forward P/E is in the Analyst expectations section")
            
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
            st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Risk & Solvency</div>", unsafe_allow_html=True)
            debt_eq_raw = meta.get('debt_to_equity', 0)
            if pd.notnull(debt_eq_raw) and debt_eq_raw != 0:
                debt_eq_txt = f"{(debt_eq_raw / 100.0):.2f}x"
            else:
                _tot_debt = meta.get('total_debt', 0)
                debt_eq_txt = "N/A (Neg Equity)" if pd.notnull(_tot_debt) and _tot_debt > 0 else "0.00x"
            
            curr_rat  = _num(meta.get('current_ratio'))
            quick_rat = _num(meta.get('quick_ratio'))
            
            # Liquidity Status Colors
            c_col = None if curr_rat is None else ("#2ecc71" if curr_rat > 1.5 else ("#e74c3c" if curr_rat < 1.0 else "#f39c12"))
            q_col = None if quick_rat is None else ("#2ecc71" if quick_rat > 1.0 else ("#e74c3c" if quick_rat < 0.7 else "#f39c12"))
            
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
            render_metric_row("Current Ratio", _txt(curr_rat, "{:.2f}"), value_color=c_col)
            render_metric_row("Quick Ratio",   _txt(quick_rat, "{:.2f}"), value_color=q_col)
            beta_val = meta.get('beta', 1.0)
            if pd.notnull(beta_val) and beta_val != 0:
                beta_col = "#e74c3c" if beta_val > 1.5 else ("#3498db" if beta_val < 0.8 else None)
                render_metric_row("Beta", f"{beta_val:.2f}", value_color=beta_col, help_text="🔴 > 1.5 (High Volatility) | 🔵 < 0.8 (Defensive)")
            else:
                render_metric_row("Beta", "N/A")
            st.markdown("</div>", unsafe_allow_html=True)

        with kcol4:
            st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Price & Context</div>", unsafe_allow_html=True)
            _tgt = _num(target_p)
            render_metric_row("Analyst Target (not in Decision)", _txt(_tgt if _tgt else None, "€{:.2f}"),
                              delta=upside if _tgt else None, is_pct=True)
            
            pe_5y_avg    = meta.get('pe_5y_avg', 0)
            pe_cur       = meta.get('pe_ratio', 0)
            pe_delta     = ((pe_cur / pe_5y_avg) - 1) * 100 if pe_5y_avg > 0 and pe_cur > 0 else 0
            
            render_metric_row("5Y Avg P/E",    f"{pe_5y_avg:.1f}" if pe_5y_avg > 0 else "N/A", delta=pe_delta, is_pct=True, color_invert=True)
            
            zs_col = "#2ecc71" if z_score < -1 else ("#e74c3c" if z_score > 1.5 else None)
            render_metric_row("Z-Score (5Y)",  f"{z_score:.2f}", value_color=zs_col)
            st.markdown("</div>", unsafe_allow_html=True)

        with kcol5:
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
