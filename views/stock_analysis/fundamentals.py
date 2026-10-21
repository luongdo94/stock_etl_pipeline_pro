"""Annual / quarterly revenue, FCF and EPS trajectory with consensus overlay."""

from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.stock_analysis.layout import layer_banner

from services.db import load_ticker_provenance
from ui.provenance import pit_caption
from services.market_data import stored_fx_rate



class _NoFx(Exception):
    """The EUR rate of the listing's currency is unknown: the analyst forecast overlays are left out."""


def _estimates_of(ctx, ticker, cols):
    """The analyst-estimate row(s) of `ticker` from the loaded snapshot (ctx), limited to `cols`. A column the warehouse
    does not have raises KeyError like the SQL it replaces did (the callers sit in try blocks)."""
    fe = ctx.get("forward_estimates_full")
    if fe is None or fe.empty or "ticker" not in fe.columns:
        return pd.DataFrame(columns=cols)
    return fe.loc[fe["ticker"] == ticker, cols]


_OPTIONAL_EST_COLS = ("upgrades_30d", "downgrades_30d", "_loaded_at")
_UP, _DOWN = "\u2191", "\u2193"


def _with_optional(ctx, cols):
    """`cols` plus the optional estimate columns this snapshot has (revision counts, fetch time)."""
    fe = ctx.get("forward_estimates_full")
    have = set(fe.columns) if fe is not None else set()
    return cols + [c for c in _OPTIONAL_EST_COLS if c in have]


def _up_share(row):
    """Share of the 30-day FY EPS revisions that were up, in % - only when BOTH counts are known and not both zero
    (a missing downgrade count is unknown, not 0, so it can never make the share 100%)."""
    up, down = row.get("upgrades_30d"), row.get("downgrades_30d")
    if up is None or down is None or pd.isna(up) or pd.isna(down) or (up + down) <= 0:
        return None
    return round(float(up) / (float(up) + float(down)) * 100, 0)


def _snapshot_note(row):
    ts = row.get("_loaded_at")
    return f"snapshot {pd.Timestamp(ts):%Y-%m-%d}" if ts is not None and pd.notna(ts) else "snapshot date unknown"


def _rev_text(d):
    """On-chart revision label: arrow and size; an unchanged estimate says so (0 is not a cut)."""
    if d is None or pd.isna(d):
        return ""
    if abs(d) < 0.005:
        return " (unchanged 30d)"
    arrow = _UP if d > 0 else _DOWN
    return f" {arrow}{abs(d):.2f}"


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
    
    tab_annual, tab_quarterly = st.tabs(["Annual", "Quarterly"])
    
    with tab_annual:
        if not df_fin.empty:
            df_fin_plot = df_fin.sort_values("year")
            
            # The mart checks periods and removes FX moves from business growth.
            df_fin_plot['rev_growth'] = df_fin_plot.get('revenue_growth_pct')
            df_fin_plot['eps_growth'] = df_fin_plot.get('eps_growth_pct')
            
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
                ) / df_fin_plot['free_cash_flow'].shift(1).abs().replace(0, float('nan')) * 100
                dates = pd.to_datetime(df_fin_plot.get('report_date', pd.Series(pd.NaT, index=df_fin_plot.index)))
                df_fin_plot['fcf_growth'] = df_fin_plot['fcf_growth'].where(dates.diff().dt.days.between(350, 380))
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
                        hovertemplate="<b>Year: %{x}</b><br>FCF (EUR; growth includes FX): €%{y:.2f}" + unit + "<br>YoY Growth: %{text}<extra></extra>"
                    ),
                    secondary_y=False
                )

            # EPS Line on secondary axis
            fig_fin.add_trace(
                go.Scatter(
                    x=df_fin_plot['year'], 
                    y=df_fin_plot['eps'], 
                    name="Basic EPS (€)",
                    line=dict(color="gold", width=3),  
                    mode="lines+markers+text",
                    text=eps_text,
                    textposition="top center",
                    hovertemplate="<b>Year: %{x}</b><br>Basic EPS: €%{y:.2f}<br>YoY Growth: %{text}<extra></extra>"
                ),
                secondary_y=True
            )
            
            # ── Analyst EPS Forecast overlay (Annual) ──────────────────
            try:
                # the estimates of the snapshot this page was built from (ctx), not a second read of the warehouse
                _fe2 = _estimates_of(ctx, deep_ticker, _with_optional(ctx, ['eps_est_cur_y', 'eps_est_next_y',
                                                                         'eps_trend_delta_30d', 'eps_trend_delta_next_y']))
                if not _fe2.empty and not df_fin_plot.empty:
                    # The provider gives "current / next fiscal year" without the period end: the estimates sit right
                    # after the last REPORTED year, never on the machine clock's year (which shifts on 1 January).
                    _cy = int(df_fin_plot.iloc[-1]["year"]) + 1
                    _fe2r = _fe2.iloc[0]
                    _snap = _snapshot_note(_fe2r)

                    # FX normalisation: forecast native currency → EUR
                    _ticker_ccy = str(meta.get("currency") or "USD")
                    _fx_rate = stored_fx_rate(_ticker_ccy)
                    if _fx_rate is None:                       # unknown rate: no overlay in a wrong currency
                        raise _NoFx(_ticker_ccy)

                    # Revision metrics (also in native currency → convert)
                    _delta_cy = _fe2r.get("eps_trend_delta_30d")
                    _delta_ny = _fe2r.get("eps_trend_delta_next_y")
                    _delta_cy_eur = float(_delta_cy) * _fx_rate if (_delta_cy is not None and pd.notnull(_delta_cy)) else None
                    _delta_ny_eur = float(_delta_ny) * _fx_rate if (_delta_ny is not None and pd.notnull(_delta_ny)) else None
                    _conviction = _up_share(_fe2r)

                    _fwd_x, _fwd_y, _fwd_cdata, _fwd_label, _fwd_delta = [], [], [], [], []
                    # Bridge from the last reported EPS point, only when it is known (None used to drop the overlay)
                    _last_hist = df_fin_plot.iloc[-1]
                    if pd.notnull(_last_hist["eps"]):
                        _fwd_x.append(int(_last_hist["year"]))
                        _fwd_y.append(float(_last_hist["eps"]))
                        _fwd_cdata.append([float("nan"), float("nan"), "Basic EPS (reported)"])
                        _fwd_label.append("")
                        _fwd_delta.append(None)
                    for _yr_offset, _col, _delta_eur, _name in [
                        (0, "eps_est_cur_y",  _delta_cy_eur, "Current FY est."),
                        (1, "eps_est_next_y", _delta_ny_eur, "Next FY est.")
                    ]:
                        _v = _fe2r.get(_col)
                        if _v is not None and pd.notnull(_v):
                            _fwd_x.append(_cy + _yr_offset)
                            _fwd_y.append(float(_v) * _fx_rate)  # → EUR
                            _fwd_cdata.append([
                                _delta_eur if _delta_eur is not None else float("nan"),
                                _conviction if _conviction is not None else float("nan"),
                                f"{_name} · {_snap} · period end unverified"
                            ])
                            _fwd_label.append(_name)
                            _fwd_delta.append(_delta_eur)       # the revision of THIS point, not by position
                    if any(_fwd_label):
                        _text_labels = [f"Est \u20ac{y:.2f}{_rev_text(d)}" if l else ""
                                        for y, l, d in zip(_fwd_y, _fwd_label, _fwd_delta)]
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
                                    "<b>%{customdata[2]}</b><br>"
                                    "EPS: \u20ac%{y:.2f}<br>"
                                    "30d revision: %{customdata[0]:+.2f}<br>"
                                    "Up share of 30d FY revisions: %{customdata[1]:.0f}%"
                                    "<extra></extra>"
                                )
                            ),
                            secondary_y=True
                        )
                        _first_est = next(x for x, l in zip(_fwd_x, _fwd_label) if l)
                        fig_fin.add_vrect(
                            x0=_first_est - 0.4, x1=_fwd_x[-1] + 0.4,
                            fillcolor="rgba(155,89,182,0.07)",
                            line_width=0,
                            annotation_text="Forecast", annotation_position="top left",
                            annotation_font=dict(size=10, color="#9b59b6")
                        )
            except _NoFx as _e:
                st.caption(f"Analyst forecast overlay not shown: no EUR rate for {_e} right now.")
            except KeyError:
                pass  # an older warehouse without these estimate columns: no overlay
            except Exception as _e:
                st.caption(f"Analyst forecast overlay could not be drawn ({type(_e).__name__}).")

            fig_fin.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=60, b=20),
                hovermode="x unified",
                barmode="group",
                title_text=f"{deep_ticker} Annual Financial Performance (Revenue, FCF & EPS)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig_fin.update_yaxes(title_text=f"Amount (€{unit})", secondary_y=False, range=[min(0, _min_amt / scale * 1.3), (max_val/scale)*1.3])
            fig_fin.update_yaxes(title_text="Earnings Per Share (€)", secondary_y=True)
            
            st.plotly_chart(fig_fin, use_container_width=True)
            st.caption("Years are the calendar year of each fiscal year-end; missing years are omitted, not zero. Amounts "
                       "in EUR. Revenue and basic EPS growth in the reporting currency (a moving exchange rate is not "
                       "growth); FCF growth in EUR, so it includes currency moves.")
            _pit = pit_caption(load_ticker_provenance(deep_ticker), "annual")
            if _pit:
                st.caption(_pit)



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

                # Two statements in one calendar quarter (a fiscal calendar ending mid-month, e.g. AZO / DPZ) make the
                # (year, quarter) key ambiguous: their FCF cannot be matched to one period, so it is left blank.
                _dup = df_fin_q_plot.duplicated(['year', 'quarter'], keep=False)
                df_fin_q_plot.loc[_dup, 'free_cash_flow'] = None
                _growth_label = "YoY"
                df_fin_q_plot['rev_growth'] = df_fin_q_plot.get('revenue_growth_yoy_pct')
                df_fin_q_plot['eps_growth'] = df_fin_q_plot.get('eps_growth_yoy_pct')
                _fcf_by_q = df_fin_q_plot[~_dup].set_index(['year', 'quarter'])['free_cash_flow']
                prev = _fcf_by_q.reindex(
                    pd.MultiIndex.from_arrays([df_fin_q_plot['year'] - 1, df_fin_q_plot['quarter']])).to_numpy()
                prev = pd.Series(prev, index=df_fin_q_plot.index, dtype=float).replace(0, float('nan'))
                df_fin_q_plot['fcf_growth'] = (df_fin_q_plot['free_cash_flow'] - prev) / prev.abs() * 100
                
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
                        # the provider's reported EPS (the figure the estimate is compared with), EUR. A quarter
                        # without it stays blank: GAAP EPS has its own line, a silent switch would mix two bases.
                        df_fin_q_plot["eps_chart"] = df_fin_q_plot["eps_actual"]
                    else:
                        df_fin_q_plot["eps_chart"] = float("nan")
                else:
                    df_fin_q_plot["eps_chart"] = float("nan")

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
                            hovertemplate="<b>Period: %{x}</b><br>FCF (EUR; growth includes FX): €%{y:.2f}" + unit_q + "<br>" + _growth_label + " Growth: %{text}<extra></extra>"
                        ),
                        secondary_y=False
                    )
                
                # EPS ACTUAL — as reported to analysts by the provider (Line, primary)
                if df_fin_q_plot['eps_chart'].notna().any():
                    fig_fin_q.add_trace(
                        go.Scatter(
                            x=x_labels,
                            y=df_fin_q_plot['eps_chart'],
                            name="EPS actual (Yahoo)",
                            line=dict(color="orange", width=3),
                            mode="lines+markers",
                            hovertemplate="<b>Quarter: %{x}</b><br>EPS actual (Yahoo): €%{y:.2f}<extra></extra>",
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
                            hovertemplate="<b>Quarter: %{x}</b><br>GAAP EPS: €%{y:.2f}<br>YoY Growth: %{text}<extra></extra>",
                            text=eps_text_q,
                        ),
                        secondary_y=True
                    )

                # EPS ESTIMATE (Dashed Line)
                if "eps_estimate" in df_fin_q_plot.columns and df_fin_q_plot["eps_estimate"].notna().any():
                    # Create surprise labels for hover
                    surprise_text = [f"{v*100:+.1f}% Surprise" if pd.notnull(v) else "Surprise N/A" for v in df_fin_q_plot.get('surprise_pct', [])]
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
                    # the estimates of the snapshot this page was built from (ctx), not a second read of the warehouse
                    _fe_q = _estimates_of(ctx, deep_ticker, _with_optional(ctx, ['eps_est_cur_q', 'eps_est_next_q',
                                                                              'eps_trend_delta_30d']))
                    if not _fe_q.empty:

                        # FX normalisation: forecast native currency → EUR
                        _ticker_ccy_q = str(meta.get("currency") or "USD")
                        _fx_rate_q = stored_fx_rate(_ticker_ccy_q)
                        if _fx_rate_q is None:
                            raise _NoFx(_ticker_ccy_q)

                        # The only revision the provider gives here is the CURRENT FISCAL YEAR's: it is shown in the
                        # hover as such, never as an arrow on the quarter (a FY revision is not a quarterly one).
                        _fe_qr = _fe_q.iloc[0]
                        _fy_delta = _fe_qr.get("eps_trend_delta_30d")
                        _fy_delta_eur = (float(_fy_delta) * _fx_rate_q if (_fy_delta is not None and pd.notnull(_fy_delta))
                                         else float("nan"))
                        _qconviction = _up_share(_fe_qr)
                        _qsnap = _snapshot_note(_fe_qr)

                        _fq_labels, _fq_vals, _fq_cdata = [], [], []
                        if not df_fin_q_plot.empty:
                            _last_q_row = df_fin_q_plot.iloc[-1]
                            _fq_labels.append(f"{int(_last_q_row['year'])} Q{int(_last_q_row['quarter'])}")
                            _fq_vals.append(float(_last_q_row["eps_chart"]) if pd.notnull(_last_q_row["eps_chart"]) else None)
                            _fq_cdata.append([float("nan"), float("nan"), "EPS actual (Yahoo)"])

                        for _qlabel, _qcol in [("Current Q est.", "eps_est_cur_q"), ("Next Q est.", "eps_est_next_q")]:
                            _qv = _fe_qr.get(_qcol)
                            if _qv is not None and pd.notnull(_qv):
                                _fq_labels.append(_qlabel)
                                _fq_vals.append(float(_qv) * _fx_rate_q)  # → EUR
                                _fq_cdata.append([_fy_delta_eur,
                                                  _qconviction if _qconviction is not None else float("nan"),
                                                  f"{_qlabel} · {_qsnap} · period end unverified"])

                        if len(_fq_labels) > 1 and any(v is not None for v in _fq_vals[1:]):
                            _fq_texts = [""] + [f"Est \u20ac{_fq_vals[i]:.2f}" if _fq_vals[i] is not None else ""
                                                for i in range(1, len(_fq_labels))]
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
                                        "<b>%{customdata[2]}</b><br>"
                                        "EPS: \u20ac%{y:.2f}<br>"
                                        "Current FY revision, 30d (not quarterly): %{customdata[0]:+.2f}<br>"
                                        "Up share of 30d FY revisions: %{customdata[1]:.0f}%"
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
                except _NoFx as _e:
                    st.caption(f"Analyst forecast overlay not shown: no EUR rate for {_e} right now.")
                except KeyError:
                    pass  # an older warehouse without these estimate columns: no overlay
                except Exception as _e:
                    st.caption(f"Analyst forecast overlay could not be drawn ({type(_e).__name__}).")

                fig_fin_q.update_layout(

                    height=500,
                    margin=dict(l=20, r=20, t=60, b=20),
                    hovermode="x unified",
                    barmode="group",
                    title_text=f"{deep_ticker} Quarterly Financial Performance (Revenue, FCF & EPS) — Growth: {_growth_label}",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                _min_q = min(df_fin_q_plot['revenue'].min(),
                             df_fin_q_plot['free_cash_flow'].min() if 'free_cash_flow' in df_fin_q_plot.columns and df_fin_q_plot['free_cash_flow'].notna().any() else 0)
                _min_q = 0 if pd.isna(_min_q) else _min_q
                y_range_q = [min(0, _min_q / scale_q * 1.3), (max_val_q/scale_q)*1.3] if pd.notnull(max_val_q) else None
                fig_fin_q.update_yaxes(title_text=f"Amount (€{unit_q})", secondary_y=False, range=y_range_q)
                fig_fin_q.update_yaxes(title_text="Earnings Per Share (€)", secondary_y=True)
                
                st.plotly_chart(fig_fin_q, use_container_width=True)
                st.caption("Labels are the calendar quarter of each period end, not the company's fiscal quarter; missing "
                           "quarters are omitted (adjacent labels may skip a quarter). Amounts in EUR. Revenue and GAAP EPS "
                           "growth year on year in the reporting currency; FCF growth in EUR, so it includes currency "
                           "moves. Estimates: current / next quarter as the provider labels them, period end unverified.")
                _pit_q = pit_caption(load_ticker_provenance(deep_ticker), "quarterly")
                if _pit_q:
                    st.caption(_pit_q)

            else:
                st.info("No historical quarterly financial data available for this ticker.")
        elif quarterly_fin.attrs.get("error"):
            st.warning(f"Quarterly financials could not be read ({quarterly_fin.attrs['error']}). Reload to try again.")
        else:
            st.info("Quarterly financials warehouse table is empty. Please run the ETL pipeline.")
