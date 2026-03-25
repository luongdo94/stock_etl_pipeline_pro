# etl/utils.py
import pandas as pd
import duckdb
import os

def get_rich_email_content(db_path, dashboard_url="http://localhost:8501"):
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
    
    # 2. Merge and Score (Simplified version of app.py logic)
    df = companies.merge(latest_prices, on="ticker", how="inner")
    
    def compute_score(row):
        score = 0
        pe = row.get("pe_ratio", 999)
        if pe < 15: score += 10
        elif pe < 25: score += 5
        
        roe = row.get("roe", 0) * 100
        if roe > 15: score += 15
        elif roe > 8: score += 7
        
        fcf = row.get("fcf_margin", 0)
        if fcf > 15: score += 15
        
        sig = row.get("ma_signal", "NEUTRAL")
        if sig == "BULLISH": score += 10
        
        consensus = str(row.get("recommendation_key", "")).lower()
        if "buy" in consensus: score += 15
        
        # Upside
        price = row.get("price_close", 1)
        target = row.get("target_mean_price", 0)
        upside = (target / price - 1) * 100 if target > 0 else 0
        if upside > 10: score += 20
        
        return min(score, 100)

    df["score"] = df.apply(compute_score, axis=1)
    df["action"] = df["score"].apply(lambda s: "🚀 STRONG BUY" if s >= 75 else ("✅ BUY" if s >= 60 else ("🟡 HOLD" if s >= 40 else "🔴 SELL")))
    df = df.sort_values("score", ascending=False).head(10) # Top 10 for email

    # 3. Build HTML Table
    html = f"""
    <div style="font-family: sans-serif; color: #333; max-width: 600px;">
        <h2 style="color: #2c3e50;">📊 AI Market Report (Mobile Snapshot)</h2>
        <p>Đã cập nhật dữ liệu ngày <b>{pd.Timestamp.now().strftime('%d/%m/%Y')}</b>. Dưới đây là các khuyến nghị AI hàng đầu:</p>
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
            Xem chi tiết biểu đồ tại: <a href="{dashboard_url}">{dashboard_url}</a>
        </p>
    </div>
    """
    return html
