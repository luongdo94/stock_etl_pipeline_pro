"""Peer comparison table."""

import html
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.sticky import sticky_choice
from ui.components import phone_safe_chart
from views.stock_analysis.layout import layer_banner

from ui.icons import render_header



def render(dd, ctx):
    annual_fin = ctx["annual_fin"]
    companies_full = ctx["companies_full"]
    deep_ticker = dd.ticker
    indices_list = ctx["indices_list"]
    meta = dd.meta
    prices = ctx.get("prices_full", ctx["prices"])
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

        # ── Source-currency, period-aware growth from the annual mart ─
        _all_peer_tickers = list(peer_df["ticker"].unique()) + [deep_ticker]
        _af_peers = annual_fin[annual_fin["ticker"].isin(_all_peer_tickers)].copy()

        def _yoy_growth_from_annual(ticker_sym, col):
            """Latest YoY growth for a column from dim_annual_financials."""
            t_df = _af_peers[_af_peers["ticker"] == ticker_sym].sort_values("year")
            if t_df.empty or col not in t_df:
                return None
            value = t_df[col].iloc[-1]
            return None if pd.isna(value) else value

        # ── Build unified rows ─────────────────────────────────────────
        def _build_row(row, ticker, is_selected):
            def _val(k):                                      # None when missing (0 is a value, unknown is not 0)
                v = row.get(k)
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return None
                return f if f == f else None
            td, eb = _val("total_debt"), _val("ebitda")
            rev_growth = _yoy_growth_from_annual(ticker, "revenue_growth_pct")
            fcf_mgn = _val("fcf_margin")
            _pct100 = lambda k: None if _val(k) is None else _val(k) * 100          # noqa: E731
            return {
                "ticker":           ticker,
                "company":          row.get("company", ticker),
                "market_cap":       row.get("market_cap"),
                # Growth
                "revenue_growth":   rev_growth,
                "earnings_growth":  _yoy_growth_from_annual(ticker, "eps_growth_pct"),
                # Quality / Profitability
                "gross_margin":     _pct100("gross_margin"),
                "operating_margin": _pct100("operating_margin"),
                "roe_pct":          _pct100("roe"),
                "fcf_margin":       fcf_mgn,
                "rule_of_40":       rev_growth + fcf_mgn if rev_growth is not None and fcf_mgn is not None else None,
                # Balance sheet & Risk
                "debt_ebitda":  td / eb if (td is not None and eb is not None and eb > 0) else None,
                "debt_to_equity":   None if _val("debt_to_equity") is None else _val("debt_to_equity") / 100,
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
        # whole words: "ai" must not match retail / railroads / airlines
        _words = set(re.findall(r"[a-z]+", f"{_sector_lower} {_industry_lower}"))

        if _words & {"ai", "data"} or any(x in _sector_lower or x in _industry_lower for x in ['software', 'saas', 'cyber', 'technology services', 'it services', 'cloud', 'internet']):
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
            _group_spans = [("", 3), ("Growth & Profitability", 5), ("Valuation", 2)]
        elif b_type == 'industrial':
            _COL_DEFS = base_cols + [
                ("gross_margin",     "Gross Mgn",         "pct",        "higher"),
                ("operating_margin", "Op Mgn",            "pct",        "higher"),
                ("roe_pct",          "ROE",               "pct",        "higher"),
                ("debt_ebitda",  "Debt/EBITDA",         "x",          "lower"),
                ("ev_to_ebitda",     "EV/EBITDA",         "x",          "lower"),
                ("peg_ratio",        "PEG",               "x1",         "lower"),
            ]
            _group_spans = [("", 3), ("Operations & Return", 3), ("Risk", 1), ("Valuation", 2)]
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
            _group_spans = [("", 3), ("Consumer Metrics", 4), ("Risk & Yield", 2), ("Valuation", 1)]
        elif b_type == 'finance':
            _COL_DEFS = base_cols + [
                ("revenue_growth",   "Rev Growth",        "pct_sign",   "higher"),
                ("roe_pct",          "ROE",               "pct",        "higher"),
                ("dividend_yield",   "Div Yield",         "pct",        "higher"),
                ("price_to_book",    "P/B",               "x1",         "lower"),
                ("pe_ratio",         "P/E",               "x",          "lower"),
            ]
            _group_spans = [("", 3), ("Financial Metrics", 3), ("Valuation", 2)]
        else: # general
            _COL_DEFS = base_cols + [
                ("revenue_growth",   "Rev Growth",        "pct_sign",   "higher"),
                ("earnings_growth",  "EPS Growth",        "pct_sign",   "higher"),
                ("gross_margin",     "Gross Mgn",         "pct",        "higher"),
                ("operating_margin", "Op Mgn",            "pct",        "higher"),
                ("roe_pct",          "ROE",               "pct",        "higher"),
                ("fcf_margin",       "FCF Mgn",           "pct",        "higher"),
                ("debt_ebitda",  "Debt/EBITDA",         "x",          "lower"),
                ("ev_to_ebitda",     "EV/EBITDA",         "x",          "lower"),
                ("pe_ratio",         "P/E",               "x",          "lower"),
                ("price_to_sales",   "P/S",               "x1",         "lower"),
            ]
            _group_spans = [("", 3), ("Growth", 2), ("Profitability", 4), ("Risk", 1), ("Valuation", 3)]

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
        # Lower  = green: debt_ebitda, valuation multiples
        HIGHER_BETTER = {"revenue_growth", "earnings_growth", "gross_margin", "operating_margin", "roe_pct", "fcf_margin", "rule_of_40", "dividend_yield"}
        LOWER_BETTER  = {"debt_ebitda", "ev_to_ebitda", "pe_ratio", "price_to_sales", "price_to_book", "debt_to_equity", "peg_ratio"}

        def _cell_bg(val, col_key, is_avg_row):
            if is_avg_row or val is None: return ""
            avg = avg_dict.get(col_key)
            if avg is None or pd.isna(avg): return ""
            try:
                fv, fa = float(val), float(avg)
            except Exception:
                return ""
            diff, tolerance = fv - fa, 0.05 * abs(fa)
            if col_key in HIGHER_BETTER:
                if diff > tolerance: return "background:rgba(0,229,160,0.18);"
                if diff < -tolerance: return "background:rgba(255,107,107,0.18);"
            elif col_key in LOWER_BETTER:
                if fv <= 0: return ""
                if diff < -tolerance: return "background:rgba(0,229,160,0.18);"
                if diff > tolerance: return "background:rgba(255,107,107,0.18);"
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
            f"<th colspan='{span}' style='padding:4px 10px; text-align:center; color:inherit; font-size:0.72rem; border-bottom:1px solid rgba(128,128,128,0.25); font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>{label}</th>"
            for label, span in _group_spans
        )

        # ── Column header row ──────────────────────────────────────────
        header_cells = "<th style='padding:6px 10px; color:inherit; text-align:left; font-size:0.8rem; white-space:nowrap;'>Ticker</th>" + "".join(
            f"<th style='padding:6px 10px; color:inherit; text-align:{'left' if k in ('company','market_cap') else 'center'}; font-size:0.8rem; white-space:nowrap;'>{lbl}</th>"
            for k, lbl, *_ in _COL_DEFS
        )

        # ── Data rows ──
        html_rows = []
        for ticker_idx, row_data in comp_df.iterrows():
            is_sel = row_data.get("is_selected", False)
            is_avg = (ticker_idx == "AVG")
            row_bg = "background:rgba(99,132,255,0.12); font-weight:700;" if is_sel else (
                     "background:var(--secondary-background-color, transparent); font-style:italic; font-weight:600;" if is_avg else "")
            label  = f"{ticker_idx} (selected)" if is_sel else ticker_idx

            td_ticker = f"<td style='padding:8px 10px; color:#a78bfa; font-weight:700; white-space:nowrap;'>{label}</td>"
            col_cells = ""
            for col_key, col_lbl, fmt, direction in _COL_DEFS:
                val = row_data.get(col_key)
                bg  = _cell_bg(val, col_key, is_avg)
                comparison = ""
                if is_sel and col_key in _numeric_cols:
                    avg = avg_dict.get(col_key)
                    if pd.isna(val) or avg is None or pd.isna(avg) or (fmt in ("x", "x1") and val <= 0):
                        comparison = " · N/A"
                    else:
                        comparison = (" · Above peer average" if val > avg else
                                      " · Below peer average" if val < avg else " · At peer average")
                if col_key == "company":
                    text = html.escape(str(val)[:28]) if val else "—"
                    col_cells += f"<td style='padding:8px 10px; text-align:left; {bg}'>{text}</td>"
                elif col_key == "market_cap":
                    col_cells += f"<td style='padding:8px 10px; text-align:center; {bg}'>{_fmt_cap(val)}</td>"
                elif fmt == "pct":
                    col_cells += f"<td style='padding:8px 10px; text-align:center; {bg}'>{_fmt_pct(val)}{comparison}</td>"
                elif fmt == "pct_sign":
                    col_cells += f"<td style='padding:8px 10px; text-align:center; {bg}'>{_fmt_pct(val, sign=True)}{comparison}</td>"
                elif fmt in ("x", "x1"):
                    col_cells += f"<td style='padding:8px 10px; text-align:center; {bg}'>{'0.0x' if col_key == 'debt_to_equity' and val == 0 else _fmt_x(val, 1 if fmt=='x' else 2)}{comparison}</td>"
                else:
                    col_cells += f"<td style='padding:8px 10px; text-align:center; {bg}'>—</td>"
            html_rows.append(f"<tr style='{row_bg}'>{td_ticker}{col_cells}</tr>")

        html_table = f"""
        <div style="overflow-x:auto; border-radius:12px; border:1px solid rgba(128,128,128,0.25); margin-bottom:12px;">
        <table style="width:100%; border-collapse:collapse; font-size:0.85rem; color:inherit;">
          <thead>
            <tr style="border-bottom:1px solid rgba(128,128,128,0.25);">{group_header}</tr>
            <tr style="border-bottom:1px solid rgba(128,128,128,0.25);">{header_cells}</tr>
          </thead>
          <tbody>{"".join(html_rows)}</tbody>
        </table>
        </div>
        <div style="font-size:0.78rem; color:inherit; margin-top:-4px; margin-bottom:16px;">
          Selected ticker cells state Above / Below / At peer average; N/A means unavailable.
        </div>
        """
        st.markdown(html_table, unsafe_allow_html=True)

        # ── Peer Metric Dot Plot (C18) ──────────────────────────────────
        _col_label_map = {c[0]: c[1] for c in _COL_DEFS}
        chosen_metric = sticky_choice(
            "Select metric for peer distribution",
            _numeric_cols,
            _numeric_cols[0] if _numeric_cols else "roe_pct",
            "sa_peer_dotplot_metric",
            format_func=lambda k: _col_label_map.get(k, k)
        )
        if not chosen_metric or chosen_metric not in _numeric_cols:
            chosen_metric = _numeric_cols[0] if _numeric_cols else None
        if chosen_metric:
            # same N/A rule as the table: a multiple at or below 0 is not interpretable
            _fmt = {c[0]: c[2] for c in _COL_DEFS}.get(chosen_metric)
            _known = comp_df[chosen_metric].notna() & (comp_df.index != "AVG")
            if _fmt in ("x", "x1"):
                _known &= comp_df[chosen_metric] > 0
            peers_only_rows = comp_df[(comp_df.index != "AVG") & (~comp_df["is_selected"])]
            n_total_peers = len(peers_only_rows)
            valid_peers = peers_only_rows[_known[peers_only_rows.index]]
            n_valid_peers = len(valid_peers)
            n_missing_peers = n_total_peers - n_valid_peers
            if n_valid_peers < 4:
                st.caption(f"Dot plot requires ≥ 4 peers with data ({n_valid_peers} available, {n_missing_peers} missing).")
            else:
                if n_missing_peers > 0:
                    st.caption(f"{n_missing_peers} of {n_total_peers} peers missing data for {_col_label_map.get(chosen_metric, chosen_metric)}; omitted from plot.")
                plot_rows = comp_df[_known].sort_values(chosen_metric, ascending=True)
                peers_plot = plot_rows[~plot_rows["is_selected"]]
                sel_plot = plot_rows[plot_rows["is_selected"]]
                fig_dots = go.Figure()
                if not peers_plot.empty:
                    fig_dots.add_trace(go.Scatter(
                        x=peers_plot[chosen_metric],
                        y=peers_plot.index,
                        mode="markers",
                        name="Peers",
                        marker=dict(color="#64748b", size=10),
                        hovertemplate="<b>%{y}</b>: %{x}<extra></extra>"
                    ))
                if not sel_plot.empty:
                    fig_dots.add_trace(go.Scatter(
                        x=sel_plot[chosen_metric],
                        y=sel_plot.index,
                        mode="markers",
                        name=f"{deep_ticker} (Selected)",
                        marker=dict(color="#38bdf8", size=14, line=dict(color="#0284c7", width=2)),
                        hovertemplate="<b>%{y} (selected)</b>: %{x}<extra></extra>"
                    ))
                avg_val = avg_dict.get(chosen_metric)
                if avg_val is not None and not pd.isna(avg_val):
                    fig_dots.add_vline(
                        x=float(avg_val),
                        line_dash="dash",
                        line_color="#f59e0b",
                        line_width=1.5,
                        annotation_text=f"Peer Avg: {avg_val:.1f}",
                        annotation_position="top right"
                    )
                fig_dots.update_layout(
                    title=dict(text=f"Peer Distribution — {_col_label_map.get(chosen_metric, chosen_metric)}", font=dict(size=14)),
                    height=280,
                    xaxis=dict(zeroline=True, zerolinecolor="rgba(128,128,128,0.2)"),
                    yaxis=dict(autorange="reversed"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                    margin=dict(l=10, r=20, t=40, b=45)
                )
                st.plotly_chart(phone_safe_chart(fig_dots), use_container_width=True)

    else:
        st.info(f"No peers found in the **{_peer_group_label}** {_match_level} to compare with.")
