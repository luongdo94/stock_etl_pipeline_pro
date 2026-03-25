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
    df["upside_pct"] = (df["target_mean_price"] / df["price_close"] - 1) * 100
    df["upside_pct"] = df["upside_pct"].fillna(0)
    df["recommendation_key"] = df["recommendation_key"].fillna("none").astype(str).str.replace("_", " ").str.title()
    
    def compute_score(row):
        score = 0
        # 1. Valuation (P/E & P/B) - 20 pts
        pe = row.get("pe_ratio", 999) if not pd.isna(row.get("pe_ratio")) else 999
        pb = row.get("price_to_book", 99) if not pd.isna(row.get("price_to_book")) else 99
        if pe < 15: score += 10
        elif pe < 25: score += 5
        if pb < 3: score += 10
        elif pb < 5: score += 5
        
        # 2. Profitability (FCF Margin & ROE) - 30 pts
        fcf = row.get("fcf_margin", 0) if not pd.isna(row.get("fcf_margin")) else 0
        roe = row.get("roe", 0) if not pd.isna(row.get("roe")) else 0
        roe_pct = roe * 100
        if fcf > 20: score += 15
        elif fcf > 10: score += 7
        if roe_pct > 15: score += 15
        elif roe_pct > 8: score += 7
        
        # 3. Health (Debt/EBITDA) - 15 pts
        debt = row.get("total_debt", 0) if not pd.isna(row.get("total_debt")) else 0
        ebitda = row.get("ebitda", 0) if not pd.isna(row.get("ebitda")) else 0
        ratio = debt / ebitda if ebitda > 0 else 999
        if ratio < 2: score += 15
        elif ratio < 4: score += 7
        
        # 4. Dividends - 15 pts
        yld = row.get("dividend_yield_pct", 0) if not pd.isna(row.get("dividend_yield_pct")) else 0
        if yld > 3: score += 15
        elif yld > 1: score += 7

        # 5. Technical Trend & Beta - 15 pts
        sig = row.get("ma_signal", "NEUTRAL")
        beta = row.get("beta", 99) if not pd.isna(row.get("beta")) else 99
        if sig == "BULLISH": score += 10
        elif sig == "NEUTRAL": score += 5
        if beta < 1.2: score += 5
        
        # 6. Wall St Consensus & Target - 20 pts
        upside = row.get("upside_pct", 0)
        consensus = str(row.get("recommendation_key", "")).lower()
        if upside > 15: score += 10
        elif upside > 5: score += 5
        if "buy" in consensus: score += 10
        
        return min(score, 100)

    df["score"] = df.apply(compute_score, axis=1)
    
    def get_action(score):
        if score >= 75: return "🚀 STRONG BUY"
        if score >= 60: return "✅ BUY"
        if score >= 40: return "🟡 HOLD"
        return "🔴 SELL"
        
    df["action"] = df["score"].apply(get_action)
    df = df.sort_values("score", ascending=False).head(10) # Top 10 for email

    # 3. Build HTML Table
    html = f"""
    <div style="font-family: sans-serif; color: #333; max-width: 600px;">
        <h2 style="color: #2c3e50;">📊 AI Market Report (Mobile Snapshot)</h2>
        <p>Data has been updated on <b>{pd.Timestamp.now().strftime('%d/%m/%Y')}</b>. Here are the top AI recommendations:</p>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                    <th style="padding: 10px; text-align: left;">Ticker</th>
                    <th style="padding: 10px; text-align: left;">Price</th>
                    <th style="padding: 10px; text-align: left;">Score</th>
                    <th style="padding: 10px; text-align: left;">Action</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in df.iterrows():
        html += f"""
                <tr style="border-bottom: 1px solid #eee; background-color: #ffffff;">
                    <td style="padding: 10px;"><b>{row['ticker']}</b></td>
                    <td style="padding: 10px;">${row['price_close']:.2f}</td>
                    <td style="padding: 10px;">{row['score']}</td>
                    <td style="padding: 10px;">{row['action']}</td>
                </tr>
        """
        
    html += """
            </tbody>
        </table>
        <p style="margin-top: 20px; font-size: 14px; color: #777;">
            View details at: <a href="http://localhost:8501">Dashboard Streamlit Layer</a>
        </p>
    </div>
    """
    return html
