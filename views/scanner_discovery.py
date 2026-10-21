"""External discovery: TradingView's own screener, fetched only when Scan is pressed.

Separate from the local Scanner on purpose: the results are not in the warehouse snapshot, are not scored by this app and
have no Decision. A real 0 (P/E, ROE, growth, RSI…) is shown as 0; only a missing value is N/A.
"""
import pandas as pd
import streamlit as st

from services.market_data import get_forex_rates


def render():
    with st.expander("Live TradingView Global Screener (On-Demand)", expanded=False):
        st.write("Fetch data directly from TradingView's servers (14 markets). External and unpinned: it is not part of the "
                 "warehouse snapshot, the stocks are not scored here and there is no Decision for them. Nothing is fetched until you press Scan.")
        
        tv_presets = {
            "💎 Value + Pullback (Custom)": {
                "description": "Undervalued stocks (P/E < 15, P/B < 1.5) of large, profitable companies (ROE > 10%). Currently experiencing short-term oversold conditions (RSI < 40) but remaining above long-term support (MA200). Ideal for buy-the-dip setups.",
                "filter": [
                    {"left": "price_earnings_ttm", "operation": "in_range", "right": [0, 15]},
                    {"left": "price_book_fq", "operation": "less", "right": 1.5},
                    {"left": "market_cap_basic", "operation": "greater", "right": 5000000000},
                    {"left": "return_on_equity", "operation": "greater", "right": 10},
                    {"left": "RSI", "operation": "less", "right": 40},
                    {"left": "close", "operation": "greater", "right": "SMA200"}
                ],
                "sort": {"sortBy": "RSI", "sortOrder": "asc"}
            },
            "🌱 GARP (Growth at Reasonable Price)": {
                "description": "Strong growth companies (Revenue > 10%, ROE > 15%) trading at extremely reasonable valuations (PEG < 2) with low debt. Peter Lynch's preferred screening strategy.",
                "filter": [
                    {"left": "price_earnings_growth_ttm", "operation": "less", "right": 2},
                    {"left": "return_on_equity", "operation": "greater", "right": 15},
                    {"left": "total_revenue_yoy_growth_ttm", "operation": "greater", "right": 10},
                    {"left": "market_cap_basic", "operation": "greater", "right": 1000000000},
                    {"left": "debt_to_equity", "operation": "less", "right": 1.5}
                ],
                "sort": {"sortBy": "price_earnings_growth_ttm", "sortOrder": "asc"}
            },
            "⚡ Breakout Scanner": {
                "description": "High momentum super-stocks: Golden Cross confirmed (MA50 > MA200), price trading above MA50, RSI in the strength zone (50-75), accompanied by breakout volume.",
                "filter": [
                    {"left": "SMA50", "operation": "greater", "right": "SMA200"},
                    {"left": "RSI", "operation": "in_range", "right": [50, 75]},
                    {"left": "close", "operation": "greater", "right": "SMA50"},
                    {"left": "volume", "operation": "greater", "right": 500000},
                    {"left": "market_cap_basic", "operation": "greater", "right": 500000000}
                ],
                "sort": {"sortBy": "RSI", "sortOrder": "desc"}
            },
            "🛡️ Quality Compounders": {
                "description": "Cash-printing machines with massive competitive moats: Gross Margin > 40%, ROIC > 15%, strong Free Cash Flow generation, and strict debt control. Perfect for long-term compounding.",
                "filter": [
                    {"left": "gross_margin", "operation": "greater", "right": 40},
                    {"left": "return_on_invested_capital", "operation": "greater", "right": 15},
                    {"left": "free_cash_flow_margin_ttm", "operation": "greater", "right": 15},
                    {"left": "debt_to_equity", "operation": "less", "right": 1.5},
                    {"left": "total_revenue_yoy_growth_ttm", "operation": "greater", "right": 5}
                ],
                "sort": {"sortBy": "return_on_invested_capital", "sortOrder": "desc"}
            },
            "💰 High Yield Dividend": {
                "description": "Cash-flow focused dividend stocks: Consistent yield > 3%, positive dividend growth rate, backed by strong Free Cash Flow (FCF Margin > 10%) to sustain future payouts.",
                "filter": [
                    {"left": "dividend_yield_recent", "operation": "greater", "right": 3},
                    {"left": "dividends_paid_growth_yoy", "operation": "greater", "right": 2},
                    {"left": "free_cash_flow_margin_ttm", "operation": "greater", "right": 10},
                    {"left": "market_cap_basic", "operation": "greater", "right": 1000000000}
                ],
                "sort": {"sortBy": "dividend_yield_recent", "sortOrder": "desc"}
            }
        }
        tv_markets = [
            "america", "vietnam", "uk", "germany", "france", "japan", 
            "hongkong", "china", "australia", "canada", "india", "brazil", "taiwan", "korea"
        ]
        
        col_preset, col_market = st.columns([3, 1])
        with col_preset:
            selected_tv_preset = st.selectbox("Select TradingView Strategy:", list(tv_presets.keys()),
                                            format_func=lambda value: value.split(" ", 1)[1])
        with col_market:
            selected_tv_market = st.selectbox("Market:", tv_markets, index=0)
            
        st.info(f"**Strategy:** {tv_presets[selected_tv_preset]['description']}")
        
        if st.button(f"Scan {selected_tv_market.upper()} Market (Live)", type="primary"):
            with st.spinner(f"Calling TradingView API for {selected_tv_market.upper()}..."):
                import requests
                url = f"https://scanner.tradingview.com/{selected_tv_market}/scan"
                payload = {
                    "filter": tv_presets[selected_tv_preset]["filter"],
                    "options": {"lang": "en"},
                    "markets": [selected_tv_market],
                    "symbols": {"query": {"types": []}, "tickers": []},
                    "columns": ["name", "description", "sector", "close", "price_earnings_ttm", "price_book_fq", "RSI", "market_cap_basic", "return_on_equity", "total_revenue_yoy_growth_ttm", "earnings_per_share_diluted_yoy_growth_ttm"],
                    "sort": tv_presets[selected_tv_preset]["sort"],
                    "range": [0, 20]
                }
                try:
                    r = requests.post(url, json=payload, timeout=10)
                    r.raise_for_status()                       # an HTTP error is an error, not "no stocks matched"
                    data = r.json()
                    if not isinstance(data, dict) or "data" not in data:
                        raise ValueError("unexpected response (no 'data' field)")
                    results = []
                    market_currency_map = {
                        "america": "USD", "vietnam": "VND", "uk": "GBP", "germany": "EUR",
                        "france": "EUR", "japan": "JPY", "hongkong": "HKD", "china": "CNY",
                        "australia": "AUD", "canada": "CAD", "india": "INR", "brazil": "BRL",
                        "taiwan": "TWD", "korea": "KRW"
                    }
                    local_currency = market_currency_map.get(selected_tv_market.lower(), "USD")
                    local_to_eur = get_forex_rates("EUR", local_currency)      # None = unknown: no EUR amounts, not parity
                    
                    for d in (data.get('data') or []):
                        f = d['d']
                        price_local = f[3]
                        price_eur = price_local * local_to_eur if price_local is not None and local_to_eur else None
                        mcap_local = f[7]
                        mcap_eur = (mcap_local * local_to_eur) / 1e9 if mcap_local is not None and local_to_eur else None
                        
                        results.append({
                            "Symbol": d['s'].split(':')[-1],
                            "Company": f[1],
                            "Sector": f[2] if f[2] else "N/A",
                            "Price (€)": f"€{price_eur:.2f}" if price_eur is not None else "N/A",
                            "P/E": round(f[4], 1) if f[4] is not None else "N/A",
                            "P/B": round(f[5], 1) if f[5] is not None else "N/A",
                            "RSI (14)": round(f[6], 1) if f[6] is not None else "N/A",
                            "Market Cap (€)": f"€{mcap_eur:.1f}B" if mcap_eur is not None else "N/A",
                            "ROE (%)": f"{round(f[8], 1)}%" if f[8] is not None else "N/A",
                            "Rev Growth YoY (%)": f"{round(f[9], 1)}%" if f[9] is not None else "N/A",
                            "EPS Growth YoY (%)": f"{round(f[10], 1)}%" if f[10] is not None else "N/A"
                        })
                    if results:
                        tv_df = pd.DataFrame(results)
                        st.dataframe(tv_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No stocks matched the criteria currently.")
                except Exception as e:
                    st.error(f"Error fetching from TradingView: {e}")
