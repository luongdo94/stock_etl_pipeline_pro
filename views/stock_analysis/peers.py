"""Peer comparison table."""

import pandas as pd
import streamlit as st

from views.stock_analysis.layout import layer_banner

from ui.icons import render_header



def render(dd, ctx):
    annual_fin = ctx["annual_fin"]
    companies_full = ctx["companies_full"]
    deep_ticker = dd.ticker
    indices_list = ctx["indices_list"]
    meta = dd.meta
    prices = ctx["prices"]
    # ── PEER COMPARISON ────────────────────────────────────────────
    layer_banner(8, "Competitive intelligence & peer benchmarking", "#f39c12", top=10)

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
                ("price_to_sales",   "P/S",               "x1",         "lower"),
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

        # ── Data rows ──
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
