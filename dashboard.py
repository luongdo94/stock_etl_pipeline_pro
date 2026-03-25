"""
dashboard.py — Interactive stock analytics dashboard using Plotly.
Reads directly from the DuckDB warehouse and opens charts in the browser.

Usage:
    python c:\\etl_pipeline\\dashboard.py
"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DB_PATH = os.path.join(ROOT, "warehouse", "stock_dw.duckdb")
conn = duckdb.connect(DB_PATH, read_only=True)

# Read data and handle Benchmark (SPY)
prices_full = conn.execute("""
    SELECT f.date, f.ticker, d.company, d.sector, d.region,
           f.price_close, f.daily_return_pct, f.volume,
           f.ma_20, f.ma_50, f.ma_signal, f.pct_from_52w_high,
           f.is_volume_spike, f.cap_category
    FROM marts.fct_daily_returns f
    LEFT JOIN marts.dim_companies d USING (ticker)
    ORDER BY f.date
""").df()

companies_full = pd.read_sql("SELECT * FROM marts.dim_companies", conn)
companies = companies_full[companies_full["ticker"] != "SPY"].copy()

spy_prices = prices_full[prices_full["ticker"] == "SPY"].copy()
prices = prices_full[prices_full["ticker"] != "SPY"].copy()

monthly = pd.read_sql("SELECT * FROM marts.agg_monthly_performance", conn)
monthly = monthly[monthly["ticker"] != "SPY"].copy()

conn.close()

# Color palette per ticker
COLORS = {
    "NVDA": "#76b900", "MSFT": "#00a4ef", "AAPL": "#555555",
    "GOOGL": "#fbbc04", "AMZN": "#ff9900", "SAP": "#0070f2",
    "ASML": "#e10000", "BABA": "#ff6900", "BAIDU": "#2932e1",
}

# ── CHART 1: Normalized price performance (base = 100) ───────────────────────
fig1 = go.Figure()
tickers = prices["ticker"].unique()
# Sort tickers by market cap to highlight the biggest
top_caps = prices.groupby("ticker")["price_close"].last().sort_values(ascending=False).index[:5]

for ticker in sorted(tickers):
    df = prices[prices["ticker"] == ticker].copy().sort_values("date")
    df["normalized"] = df["price_close"] / df["price_close"].iloc[0] * 100
    
    # Show only top 5 by default, hide the rest in legend to reduce immediate clutter
    is_visible = True if ticker in top_caps else 'legendonly'
    
    fig1.add_trace(go.Scatter(
        x=df["date"], y=df["normalized"],
        name=ticker,
        visible=is_visible,
        line=dict(width=2),
        hovertemplate=f"<b>{ticker}</b><br>Date: %{{x}}<br>Normalized: %{{y:.1f}}<extra></extra>"
    ))

if not spy_prices.empty:
    spy_prices = spy_prices.sort_values("date")
    spy_prices["normalized"] = spy_prices["price_close"] / spy_prices["price_close"].iloc[0] * 100
    fig1.add_trace(go.Scatter(
        x=spy_prices["date"], y=spy_prices["normalized"],
        name="<b>S&P 500 (SPY)</b>",
        line=dict(color="white", width=4, dash="dot"),
        hovertemplate="<b>S&P 500 (SPY)</b><br>Date: %{x}<br>Normalized: %{y:.1f}<extra></extra>"
    ))

fig1.update_layout(
    title=dict(text="📈 Normalized Price Performance vs S&P 500 Benchmark - <i>Click legend to show more</i>", font=dict(size=20)),
    xaxis_title="Date", yaxis_title="Normalized Price",
    hovermode="x unified",
    template="plotly_dark",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    height=600,
    margin=dict(r=150) # give space for legend
)
fig1.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)

# ── CHART 2: Monthly returns heatmap ─────────────────────────────────────────
pivot = monthly.pivot_table(
    index="ticker", columns=monthly["month"].astype(str).str[:7],
    values="monthly_return", aggfunc="sum"
)
fig2 = go.Figure(go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale="RdYlGn",
    zmid=0,
    text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
    texttemplate="%{text}",
    textfont=dict(size=10),
    hovertemplate="<b>%{y}</b><br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
    colorbar=dict(title="Return %"),
))
fig2.update_layout(
    title=dict(text="🗓️ Monthly Return Heatmap (%)", font=dict(size=20)),
    template="plotly_dark",
    height=max(400, len(pivot) * 25 + 150),
    xaxis=dict(tickangle=-45),
)

# ── CHART 3: Latest MA Signal + Distance from 52w High ───────────────────────
latest = prices[prices["date"] == prices["date"].max()].copy()
latest = latest.sort_values("pct_from_52w_high", ascending=True)

signal_colors = {"BULLISH": "#00c853", "BEARISH": "#ff1744", "NEUTRAL": "#ffd600"}
bar_colors = [signal_colors.get(s, "#888") for s in latest["ma_signal"]]

fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Distance from 52-Week High (%)", "MA Signal (MA20 vs MA50)"),
    horizontal_spacing=0.12,
)
fig3.add_trace(go.Bar(
    x=latest["pct_from_52w_high"], y=latest["ticker"],
    orientation="h", marker_color=bar_colors,
    text=[f"{v:.1f}%" for v in latest["pct_from_52w_high"]],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>From 52w High: %{x:.2f}%<extra></extra>",
    showlegend=False,
), row=1, col=1)

fig3.add_trace(go.Bar(
    x=latest["ma_20"] - latest["ma_50"], y=latest["ticker"],
    orientation="h",
    marker_color=[signal_colors.get(s, "#888") for s in latest["ma_signal"]],
    text=latest["ma_signal"],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>MA20-MA50: %{x:.2f}<extra></extra>",
    showlegend=False,
), row=1, col=2)

# Legend markers
for signal, color in signal_colors.items():
    fig3.add_trace(go.Bar(x=[None], y=[None], name=signal,
                          marker_color=color, showlegend=True))
fig3.update_layout(
    title=dict(text="🎯 Technical Signal Dashboard (Latest Day)", font=dict(size=20)),
    template="plotly_dark", height=max(450, len(latest) * 25 + 150),
    legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0.4),
)

# ── CHART 4: Risk vs Return scatter ──────────────────────────────────────────
risk_return = monthly.groupby("ticker").agg(
    avg_return=("monthly_return", "mean"),
    volatility=("volatility", "mean"),
).reset_index().merge(companies[["ticker", "company", "sector"]], on="ticker")

fig4 = px.scatter(
    risk_return,
    x="volatility", y="avg_return",
    color="sector", size=[40] * len(risk_return),
    text="ticker",
    title="⚖️ Risk vs Return (Avg Monthly Return vs Volatility)",
    labels={"volatility": "Volatility (Std Dev of Daily Returns)",
            "avg_return": "Avg Monthly Return (%)"},
    template="plotly_dark",
    height=480,
    color_discrete_sequence=px.colors.qualitative.Bold,
    hover_data={"company": True, "sector": True},
)
fig4.update_traces(textposition="top center", marker=dict(size=14, opacity=0.85))
fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

# ── CHART 5: Volume Timeline by Sector ───────────────────────────────────────
last_90 = prices[prices["date"] >= prices["date"].max() - pd.Timedelta(days=90)].copy()
sector_volume = last_90.groupby(["date", "sector"])["volume"].sum().reset_index()

fig5 = px.bar(
    sector_volume, x="date", y=sector_volume["volume"] / 1e6, color="sector",
    title="📊 Aggregated Trading Volume by Sector — Last 90 Days (Millions)",
    labels={"y": "Volume (M shares)", "date": "Date", "sector": "Sector"},
    template="plotly_dark", height=450,
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig5.update_layout(
    barmode="stack",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# ── CHART 6: Company Margins Breakdown ───────────────────────────────────────
margins = companies[["ticker", "company", "gross_margin", "operating_margin", "fcf_margin"]].dropna().sort_values("gross_margin", ascending=True)

fig6 = go.Figure()
fig6.add_trace(go.Bar(
    y=margins["ticker"], x=margins["gross_margin"] * 100,
    name="Gross Margin (%)", marker_color="#3498db", orientation='h'
))
fig6.add_trace(go.Bar(
    y=margins["ticker"], x=margins["operating_margin"] * 100,
    name="Operating Margin (%)", marker_color="#f1c40f", orientation='h'
))
fig6.add_trace(go.Bar(
    y=margins["ticker"], x=margins["fcf_margin"],
    name="FCF Margin (%)", marker_color="#2ecc71", orientation='h'
))

fig6.update_layout(
    title=dict(text="💰 Company Profitability Margins", font=dict(size=20)),
    barmode="group",
    xaxis_title="Margin (%)",
    template="plotly_dark", height=max(500, len(margins) * 35 + 100),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig6.add_vline(x=0, line_dash="solid", line_color="gray", opacity=0.8)

# ── CHART 7: Financial Health — Debt vs EBITDA vs Free Cash Flow ─────────────
# Scatter bubble chart: X=EBITDA, Y=Debt, Size=FCF, Color=Sector
health = companies[["ticker", "company", "sector", "total_debt", "ebitda", "free_cashflow"]].dropna()
health["free_cashflow"] = health["free_cashflow"].apply(lambda x: max(x, 0)) # Prevent negative sizes

fig7 = px.scatter(
    health,
    x=health["ebitda"] / 1e9, y=health["total_debt"] / 1e9,
    size=health["free_cashflow"] / 1e9 + 1, # +1 to ensure visibility
    color="sector",
    text="ticker",
    title="🏢 Financial Health: Debt vs EBITDA (Bubble Size = Free Cash Flow)",
    labels={"x": "EBITDA ($ Billions)", "y": "Total Debt ($ Billions)"},
    template="plotly_dark", height=500,
    color_discrete_sequence=px.colors.qualitative.Pastel,
    hover_data={"company": True, "sector": True}
)
fig7.update_traces(textposition="top center", marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))

# ── CHART 8: Valuation Metrics (P/E and EPS) ─────────────────────────────────
valuation = companies[["ticker", "company", "pe_ratio", "forward_pe", "trailing_eps", "forward_eps"]].dropna().sort_values("pe_ratio", ascending=True)

fig8 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Price-to-Earnings (P/E) Ratio", "Earnings Per Share (EPS)"),
    horizontal_spacing=0.12,
)

# P/E Subplot
fig8.add_trace(go.Bar(
    y=valuation["ticker"], x=valuation["pe_ratio"],
    name="Trailing P/E", marker_color="#9b59b6", orientation='h'
), row=1, col=1)
fig8.add_trace(go.Bar(
    y=valuation["ticker"], x=valuation["forward_pe"],
    name="Forward P/E", marker_color="#8e44ad", orientation='h'
), row=1, col=1)

# EPS Subplot
fig8.add_trace(go.Bar(
    y=valuation["ticker"], x=valuation["trailing_eps"],
    name="Trailing EPS ($)", marker_color="#1abc9c", orientation='h'
), row=1, col=2)
fig8.add_trace(go.Bar(
    y=valuation["ticker"], x=valuation["forward_eps"],
    name="Forward EPS ($)", marker_color="#16a085", orientation='h'
), row=1, col=2)

fig8.update_layout(
    title=dict(text="⚖️ Valuation & Earnings (Trailing vs Forward)", font=dict(size=20)),
    barmode="group",
    template="plotly_dark", height=max(500, len(valuation) * 35 + 100),
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
)

# ── CHART 9: AI Recommendation Engine ────────────────────────────────────────
# Compute a heuristic score (0-100)
latest = prices[prices["date"] == prices["date"].max()].copy()
reco_df = companies.merge(latest[["ticker", "ma_signal", "price_close"]], on="ticker", how="left")

# Calculate Analyst Upside
reco_df["upside_pct"] = (reco_df["target_mean_price"] / reco_df["price_close"] - 1) * 100
reco_df["upside_pct"] = reco_df["upside_pct"].fillna(0)
reco_df["recommendation_key"] = reco_df["recommendation_key"].fillna("none").astype(str).str.replace("_", " ").str.title()

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

    # 5. Technical Trend & Beta - 15 pts (Scaled down)
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

reco_df["score"] = reco_df.apply(compute_score, axis=1)

def get_action(score):
    if score >= 75: return "STRONG BUY 🌟"
    if score >= 60: return "BUY 🟢"
    if score >= 40: return "HOLD 🟡"
    return "SELL 🔴"

reco_df["action"] = reco_df["score"].apply(get_action)
reco_df = reco_df.sort_values("score", ascending=False)
reco_df["upside_str"] = reco_df["upside_pct"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")

fig9 = go.Figure(data=[go.Table(
    header=dict(values=["<b>Ticker</b>", "<b>Company</b>", "<b>Wall St Consensus</b>", "<b>Price Target Upside</b>", "<b>AI Score</b>", "<b>Recommendation</b>"],
                fill_color='rgba(40, 40, 40, 1)', font=dict(color='white', size=14), align='left'),
    cells=dict(values=[reco_df["ticker"], reco_df["company"], reco_df["recommendation_key"], reco_df["upside_str"],
                       reco_df["score"], reco_df["action"]],
               fill_color='rgba(20, 20, 20, 1)', font=dict(color='white', size=13), align='left', height=30)
)])

fig9.update_layout(
    title=dict(text="🤖 AI Buy/Hold/Sell Recommendations (Heuristic Model)", font=dict(size=20)),
    template="plotly_dark", height=max(400, len(reco_df) * 35 + 100),
    margin=dict(l=0, r=0, t=50, b=0)
)

# ── CHART 10: Quality & Valuation (ROE vs P/B) ───────────────────────────────
qv = companies[["ticker", "company", "sector", "price_to_book", "roe", "market_cap", "dividend_yield_pct"]].dropna()

fig10 = px.scatter(
    qv, x="price_to_book", y=qv["roe"] * 100,
    size="market_cap", color="sector", hover_name="ticker",
    hover_data={"company": True, "dividend_yield_pct": ':.2f'},
    labels={"price_to_book": "Price-to-Book (P/B)", "y": "Return on Equity (ROE %)", "dividend_yield_pct": "Yield %"},
    title="💎 Quality vs Valuation: Return on Equity (ROE) vs Price-to-Book (P/B)",
    template="plotly_dark", height=600,
)
fig10.add_hline(y=15, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Good ROE (>15%)")
fig10.add_vline(x=3, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Expensive P/B (>3)")

fig10.update_traces(textposition="top center", marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))



# ── Combine into single HTML dashboard ───────────────────────────────────────
OUTPUT_PATH = os.path.join(ROOT, "dashboard.html")

def generate_html_report():
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stock Analytics Dashboard</title>
  <style>
    body { background: #111; color: #eee; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }
    h1 { text-align: center; color: #fff; font-size: 28px; margin-bottom: 4px; }
    .subtitle { text-align: center; color: #888; margin-bottom: 30px; }
    .chart { background: #1a1a2e; border-radius: 12px; padding: 10px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
  </style>
</head>
<body>
  <h1>📊 Stock Market Analytics Dashboard</h1>
  <p class="subtitle">ETL Pipeline — DuckDB Warehouse · Generated {date}</p>
""".replace("{date}", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")))

        # Logical Sequence
        for fig in [fig9, fig1, fig2, fig10, fig8, fig6, fig7, fig3, fig4, fig5]:
            f.write(f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>\n')

        f.write("</body></html>")
    print(f"✅ Dashboard saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_html_report()
    
    # Only open in browser if not running in a headless/CI/Airflow environment
    if "AIRFLOW_HOME" not in os.environ and "GITHUB_ACTIONS" not in os.environ:
        import webbrowser
        webbrowser.open(f"file:///{OUTPUT_PATH.replace(chr(92), '/')}")
        print("🌐 Opening in browser...")
