"""Annual / quarterly revenue, FCF and EPS trajectory with consensus overlay."""

from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.stock_analysis.layout import layer_banner

from services.db import get_db_connection
from services.market_data import get_forex_rates



def render(dd, ctx):
    deep_ticker = dd.ticker
    df_fin = dd.df_fin
    earnings_surprise_full = ctx["earnings_surprise_full"]
    hist_fcf_full = ctx["hist_fcf_full"]
    hist_fcf_q_full = ctx["hist_fcf_q_full"]
    meta = dd.meta
    quarterly_fin = ctx["quarterly_fin"]
    # --- HISTORICAL FUNDAMENTAL TRENDS (Dual Axis) ---
    st.markdown("---")
    layer_banner(6, "Fundamental trajectory & valuation", "#1abc9c", top=10)
    
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
            # negative revenue/FCF bars were clipped by an axis forced to start at 0
            _min_amt = min(df_fin_plot['revenue'].min(),
                           df_fin_plot['free_cash_flow'].min() if df_fin_plot['free_cash_flow'].notna().any() else 0)
            _min_amt = 0 if pd.isna(_min_amt) else _min_amt
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
            
            fig_fin.update_yaxes(title_text=f"Amount (€{unit})", secondary_y=False, range=[min(0, _min_amt / scale * 1.3), (max_val/scale)*1.3])
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
                
                _min_q = min(df_fin_q_plot['revenue'].min(),
                             df_fin_q_plot['free_cash_flow'].min() if 'free_cash_flow' in df_fin_q_plot.columns and df_fin_q_plot['free_cash_flow'].notna().any() else 0)
                _min_q = 0 if pd.isna(_min_q) else _min_q
                y_range_q = [min(0, _min_q / scale_q * 1.3), (max_val_q/scale_q)*1.3] if pd.notnull(max_val_q) else None
                fig_fin_q.update_yaxes(title_text=f"Amount (€{unit_q})", secondary_y=False, range=y_range_q)
                fig_fin_q.update_yaxes(title_text="Earnings Per Share (€)", secondary_y=True)
                
                st.plotly_chart(fig_fin_q, use_container_width=True)

            else:
                st.info("No historical quarterly financial data available for this ticker.")
        else:
            st.info("Quarterly financials warehouse table is empty. Please run the ETL pipeline.")
