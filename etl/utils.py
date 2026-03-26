# etl/utils.py
import pandas as pd
import duckdb
import os

def get_rich_email_content(db_path):
    """
    Query DuckDB and generate a mobile-friendly HTML table for the success email.
    """
    conn = duckdb.connect(db_path, read_only=True)
    
    # 1. Fetch Data
    companies = conn.execute("SELECT * FROM marts.dim_companies").df()
    latest_prices = conn.execute("""
        SELECT ticker, price_close, ma_signal 
        FROM marts.fct_daily_returns 
        WHERE date = (SELECT MAX(date) FROM marts.fct_daily_returns)
    """).df()
    conn.close()
    
    # 2. Pre-process for scoring
    df = companies.merge(latest_prices, on="ticker", how="inner")
    df["upside_pct"] = (df["target_mean_price"] / df["price_close"] - 1) * 100
    df["upside_pct"] = df["upside_pct"].fillna(0)
    df["recommendation_key"] = df["recommendation_key"].fillna("none").astype(str).str.replace("_", " ").str.title()
    
    def compute_score(row):
        score = 0
        # 1. Valuation (P/E & PEG) - 25 pts
        pe = row.get("pe_ratio", 999) if not pd.isna(row.get("pe_ratio")) else 999
        peg = row.get("peg_ratio", 99) if not pd.isna(row.get("peg_ratio")) else 99
        if pe < 15: score += 10
        elif pe < 25: score += 5
        if peg < 1.0: score += 15
        elif peg < 1.8: score += 7
        
        # 2. Profitability (FCF Margin & ROE) - 25 pts
        fcf = row.get("fcf_margin", 0) if not pd.isna(row.get("fcf_margin")) else 0
        roe = row.get("roe", 0) if not pd.isna(row.get("roe")) else 0
        roe_pct = roe * 100
        if fcf > 20: score += 15
        elif fcf > 10: score += 7
        if roe_pct > 15: score += 10
        elif roe_pct > 8: score += 5
        
        # 3. Efficiency (EV/EBITDA) - 15 pts
        eve = row.get("ev_to_ebitda", 99) if not pd.isna(row.get("ev_to_ebitda")) else 99
        if eve < 10: score += 15
        elif eve < 18: score += 7
        
        # 4. Dividends & Yield - 10 pts
        yld = row.get("dividend_yield_pct", 0) if not pd.isna(row.get("dividend_yield_pct")) else 0
        if yld > 3: score += 10
        elif yld > 1: score += 5

        # 5. Technical Trend - 15 pts
        sig = row.get("ma_signal", "NEUTRAL")
        if sig == "BULLISH": score += 15
        elif sig == "NEUTRAL": score += 7
        
        # 6. Wall St Target - 10 pts
        upside = row.get("upside_pct", 0)
        if upside > 15: score += 10
        elif upside > 5: score += 5
        
        return min(score, 100)

    df["score"] = df.apply(compute_score, axis=1)
    
    def get_action(score):
        if score >= 75: return "🚀 STRONG BUY"
        if score >= 60: return "✅ BUY"
        if score >= 40: return "🟡 HOLD"
        return "🔴 SELL"
        
    df["action"] = df["score"].apply(get_action)
    df = df.sort_values("score", ascending=False).head(12) # Top 12 for professional report

    # 3. Build HTML Table
    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; color: #333; max-width: 700px; border: 1px solid #eee; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <div style="background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 25px;">
            <h2 style="margin: 0; font-size: 20px;">🏙️ Elite Pro Diagnostic Morning Report</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 14px;">Market Scan: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>
        
        <div style="padding: 20px;">
            <p style="margin: 0 0 15px 0; font-size: 15px;">Targeting the <b>Top 12 Quantitative Signals</b> for today's session:</p>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Ticker</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Price</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">PEG</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Yield</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Trend</th>
                        <th style="padding: 12px; text-align: left; font-size: 13px;">Action</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for _, row in df.iterrows():
        trend_color = "#27ae60" if row['ma_signal'] == "BULLISH" else "#7f8c8d"
        html += f"""
                    <tr style="border-bottom: 1px solid #f0f0f0;">
                        <td style="padding: 12px;"><b>{row['ticker']}</b></td>
                        <td style="padding: 12px;">${row['price_close']:.2f}</td>
                        <td style="padding: 12px;">{row['peg_ratio'] if not pd.isna(row['peg_ratio']) else 'N/A'}</td>
                        <td style="padding: 12px;">{row['dividend_yield_pct']}%</td>
                        <td style="padding: 12px; color: {trend_color}; font-weight: 600;">{row['ma_signal']}</td>
                        <td style="padding: 12px; font-size: 12px;"><b>{row['action']}</b></td>
                    </tr>
        """
        
    html += """
                </tbody>
            </table>
            
            <div style="margin-top: 25px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;">
                <a href="http://localhost:8501" style="display: inline-block; background-color: #2a5298; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: 600;">🚀 Launch Deep-Dive Dashboard</a>
            </div>
        </div>
        <div style="background-color: #f8f9fa; padding: 15px; font-size: 11px; color: #999; text-align: center;">
            Elite Pro Diagnostic Engine v2.5 | DuckDB Warehouse | Automation by Airflow
        </div>
    </div>
    """
    return html
