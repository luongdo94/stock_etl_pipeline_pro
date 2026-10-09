"""Analyst expectations: forward P/E, consensus growth, revisions."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.stock_analysis.layout import layer_banner

from services.db import get_db_connection
from services.market_data import get_forex_rates



def render(dd, ctx):
    annual_fin = ctx["annual_fin"]
    deep_ticker = dd.ticker
    meta = dd.meta
    st.markdown("---")

    st.markdown("---")
    layer_banner(7, "Analyst expectations (forward-looking)", "#9b59b6", top=10, bottom=0)

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
        kc1, kc2, kc3, kc4 = st.columns(4)

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
            # 30d change of the consensus EPS, reported in the listing currency → EUR like every other amount
            _eps_delta_30d = _fv("eps_trend_delta_30d")
            _eps_delta_30d = _eps_delta_30d * _fe_fxr if _eps_delta_30d is not None else None
            _delta_color = "#2ecc71" if _eps_delta_30d and _eps_delta_30d > 0 else "#e74c3c"

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
                        {f"€{_eps_delta_30d:+.3f}" if _eps_delta_30d is not None else "N/A"}
                    </div>
                    <div style='color:#556677; font-size:0.65rem; margin-top:2px;'>
                        {"Analysts raising estimates ↑" if _eps_delta_30d and _eps_delta_30d > 0 else "Analysts cutting estimates ↓" if _eps_delta_30d and _eps_delta_30d < 0 else "Estimates stable"}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

