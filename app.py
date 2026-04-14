"""
app.py — Interactive stock analytics dashboard using Streamlit.
Reads directly from the DuckDB warehouse and opens charts in the browser.

Usage:
    python c:\\etl_pipeline\\app.py
"""
# ── SESSION ACTIVE: 2026-04-09 ───────────────────────────────────────────────
import sys
import os
import json
import logging
from datetime import timedelta, date
import streamlit as st
import auth  
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import duckdb
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load environment variables from .env (COHERE_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on system env vars

import contextlib
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
from etl.llm_parser import analyze_risk_with_llm
from etl.utils import compute_score, compute_fmi_live


# ── COHERE AI INTELLIGENCE ENGINE ───────────────────────────────────────────
def get_cohere_insight(api_key: str, metrics: dict) -> str:
    """Generates an institutional-grade stock analysis report using Cohere Command-R+."""
    try:
        import cohere
        co = cohere.ClientV2(api_key=api_key)

        def _fmt(v, decimals=2, suffix=""):
            if v is None or v == "N/A":
                return "N/A"
            try:
                return f"{float(v):.{decimals}f}{suffix}"
            except Exception:
                return str(v)

        ticker    = metrics.get("ticker", "N/A")
        company   = metrics.get("company", ticker)
        sector    = metrics.get("sector", "N/A")
        ai_score  = metrics.get("ai_score", "N/A")
        fmi_score = metrics.get("fmi_score", "N/A")
        fmi_label = metrics.get("fmi_label", "N/A")
        action    = metrics.get("action", "N/A")
        price     = metrics.get("price", "N/A")
        upside    = metrics.get("upside_pct", 0)
        rsi       = metrics.get("rsi", 50)
        ma_signal = metrics.get("ma_signal", "N/A")
        z_score   = metrics.get("price_z_score", "N/A")
        pe        = metrics.get("pe_ratio", "N/A")
        peg       = metrics.get("peg_ratio", "N/A")
        pb        = metrics.get("price_to_book", "N/A")
        roe       = metrics.get("roe", "N/A")
        fcf       = metrics.get("fcf_margin", "N/A")
        div_yield = metrics.get("dividend_yield_pct", 0)
        beta      = metrics.get("beta", "N/A")
        consensus = metrics.get("recommendation_key", "N/A")
        regime    = metrics.get("market_regime", "NEUTRAL")
        w52_pos   = metrics.get("w52_pos", "N/A")
        target_p  = metrics.get("target_mean_price", "N/A")

        try:
            rsi_note = "— Overbought territory" if float(rsi) > 70 else "— Oversold territory" if float(rsi) < 30 else "— Neutral zone"
        except Exception:
            rsi_note = ""

        prompt = f"""You are a senior equity analyst at a top-tier investment bank (Goldman Sachs, J.P. Morgan level).
Analyze the following stock data and produce a concise, professional investment report.

## Stock Data: {ticker} ({company})
- **Sector**: {sector}
- **Current Price**: \u20ac{_fmt(price)}
- **AI Quality Score**: {ai_score}/100
- **Fundamental Momentum Index (FMI)**: {fmi_score}/100 ({fmi_label})
- **Analyst Recommendation**: {action} | **Consensus**: {consensus}
- **Analyst Price Target**: \u20ac{_fmt(target_p)} (Implied Upside: {_fmt(upside, 1)}%)
- **52-Week Position**: {_fmt(w52_pos, 0)}% of range

### Technical Indicators
- **RSI (14)**: {_fmt(rsi, 1)} {rsi_note}
- **MA Trend Signal**: {ma_signal}
- **Price Z-Score**: {_fmt(z_score, 2)} (deviation from 60-day mean)

### Valuation & Profitability
- **P/E**: {_fmt(pe, 1)}x | **PEG**: {_fmt(peg, 2)} | **P/B**: {_fmt(pb, 2)}x
- **ROE**: {_fmt(roe, 1, '%')} | **FCF Margin**: {_fmt(fcf, 1, '%')} | **Dividend Yield**: {_fmt(div_yield, 2, '%')}
- **Beta**: {_fmt(beta, 2)} | **Market Regime**: {regime}

---
Write a structured analysis with exactly these THREE sections:

### 1. \U0001f3af Investment Verdict
One clear paragraph (3-5 sentences). State the overall investment thesis, Buy/Hold/Sell, and primary driver.

### 2. \U0001f4ca Technical & Fundamental Analysis
One paragraph (3-5 sentences). Analyze the interplay between technical signals and the fundamental picture. Highlight any divergence or confirmation.

### 3. \u26a0\ufe0f Key Risks & Catalysts
- Risk 1: (specific, quantitative)
- Risk 2: (specific, quantitative)
- Catalyst 1: (specific, quantitative)
- Catalyst 2: (specific, quantitative)

Rules: English only. Be direct and decisive. Reference specific data points. Under 300 words."""

        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        return response.message.content[0].text

    except Exception as e:
        err = str(e)
        if "invalid api key" in err.lower() or "unauthorized" in err.lower():
            return "\u274c **Invalid API Key.** Please check your Cohere API Key in the Sidebar."
        elif "rate limit" in err.lower():
            return "\u23f3 **Rate limit reached.** Please wait a moment and try again."
        else:
            return f"\u274c **AI Engine Error:** {err}"


def get_unified_verdict(api_key: str, metrics: dict, nlp_result: dict) -> str:
    """
    Unified Alpha-Risk Intelligence — Chief Investment Officer (CIO) mode.
    Combines quantitative fundamentals + NLP news sentiment into one actionable verdict.
    """
    try:
        import cohere
        co = cohere.ClientV2(api_key=api_key)

        def _f(v, d=2, s=""):
            if v is None or v == "N/A": return "N/A"
            try: return f"{float(v):.{d}f}{s}"
            except: return str(v)

        ticker    = metrics.get("ticker", "N/A")
        company   = metrics.get("company", ticker)
        sector    = metrics.get("sector", "N/A")
        ai_score  = metrics.get("ai_score", "N/A")
        z_score   = metrics.get("price_z_score", 0)
        try:
            z_score = round(float(z_score), 2)
        except (TypeError, ValueError):
            z_score = 0
        price     = metrics.get("price", "N/A")
        upside    = metrics.get("upside_pct", 0)
        rsi       = metrics.get("rsi", 50)
        ma_signal = metrics.get("ma_signal", "N/A")
        pe        = metrics.get("pe_ratio", "N/A")
        peg       = metrics.get("peg_ratio", "N/A")
        fcf       = metrics.get("fcf_margin", "N/A")
        regime    = metrics.get("market_regime", "NEUTRAL")
        target_p  = metrics.get("target_mean_price", "N/A")
        # Price structure data for TP/SL calculation
        high_52w       = metrics.get("price_52w_high", "N/A")
        low_52w        = metrics.get("price_52w_low", "N/A")
        pct_from_ma200 = metrics.get("pct_from_ma200", "N/A")
        try:
            pct_from_ma200 = f"{float(pct_from_ma200):+.1f}%"
        except (TypeError, ValueError):
            pct_from_ma200 = "N/A"
        # Precise technical levels from price action
        support_s1     = metrics.get("support_s1", "N/A")
        support_s2     = metrics.get("support_s2", "N/A")
        resistance_r1  = metrics.get("resistance_r1", "N/A")
        resistance_r2  = metrics.get("resistance_r2", "N/A")
        stop_loss_tech = metrics.get("stop_loss_technical", "N/A")
        ma20_cur       = metrics.get("ma_20_current", "N/A")
        ma50_cur       = metrics.get("ma_50_current", "N/A")

        # NLP data
        nlp_score     = nlp_result.get("red_flag_score", 0)
        nlp_sentiment = nlp_result.get("sentiment", "Neutral")
        nlp_category  = nlp_result.get("risk_category", "None")
        nlp_reco      = nlp_result.get("recommendation", "N/A")
        nlp_insights  = nlp_result.get("key_insights", [])
        nlp_headlines = nlp_result.get("headlines_analyzed", 0)

        # Signal alignment check
        quant_bullish = int(ai_score) >= 65 if str(ai_score).isdigit() else False
        news_bullish  = nlp_score <= 25 and nlp_sentiment in ["Positive"]
        news_bearish  = nlp_score >= 60 or nlp_sentiment in ["Negative", "Critical"]
        if quant_bullish and news_bullish:
            alignment = "CONVERGENCE — Both quantitative and qualitative signals are bullish."
        elif quant_bullish and news_bearish:
            alignment = "DIVERGENCE — Strong fundamentals but negative news sentiment. High risk of surprise downside."
        elif not quant_bullish and news_bullish:
            alignment = "DIVERGENCE — Positive news but weak fundamentals. Rally may be unsustainable."
        else:
            alignment = "ALIGNMENT (BEARISH) — Both quantitative and qualitative signals are weak."

        prompt = f"""You are a Chief Investment Officer (CIO) at a top-tier hedge fund.
You have received both QUANTITATIVE data and QUALITATIVE news intelligence for a stock.
Your task: synthesize both and issue ONE definitive, actionable investment verdict.

## Stock: {ticker} ({company}) | Sector: {sector}

### QUANTITATIVE (Fundamental & Technical)
- AI Quality Score: {ai_score}/100
- Price Z-Score: {z_score:+.2f}σ (deviation from 60-day mean; >+2=Overbought, <-2=Oversold)
- Price: €{_f(price)} | Analyst Target: €{_f(target_p)} | Implied Upside: {_f(upside, 1)}%
- 52W Range: €{_f(low_52w)} – €{_f(high_52w)} | % from MA200: {pct_from_ma200}
- Technical Levels: S1=€{_f(support_s1)} | S2=€{_f(support_s2)} | R1=€{_f(resistance_r1)} | R2=€{_f(resistance_r2)} | Stop=€{_f(stop_loss_tech)}
- Moving Averages: MA20=€{_f(ma20_cur)} | MA50=€{_f(ma50_cur)}
- RSI: {_f(rsi, 1)} | MA Signal: {ma_signal} | P/E: {_f(pe, 1)}x | PEG: {_f(peg, 2)} | FCF: {_f(fcf, 1, '%')}
- Market Regime: {regime}

### QUALITATIVE (News Intelligence — {nlp_headlines} sources analyzed)
- News Red Flag Score: {nlp_score}/100
- Sentiment: {nlp_sentiment} | Risk Category: {nlp_category}
- NLP Recommendation: "{nlp_reco}"
- Key News Signals: {'; '.join(nlp_insights[:3]) if nlp_insights else 'None'}

### SIGNAL ALIGNMENT
{alignment}

---
Write a unified analysis with these FOUR sections:

### CIO Verdict
One decisive paragraph (3-4 sentences). What is your final call? Reference BOTH the quantitative and qualitative data.

### 💡 Signal Convergence Analysis
One paragraph explaining the interplay between the news sentiment and the fundamental data.
If signals DIVERGE, explain which side (news vs fundamentals) you trust more and why.

### 🎯 Actionable Recommendation
State your recommendation — one of: **STRONG BUY / BUY / WATCH & ACCUMULATE / HOLD / REDUCE / AVOID**
Include practical execution context where relevant (entry levels, targets, risk management). Use the technical data provided as reference — apply your own judgment, not mechanical formulas.

### ⚠️ Key Risks to Monitor
3 concise bullet points covering the most material risks.

### 📊 Decision Guidance
Consider these thresholds when forming your view (use your judgment, not hard rules):
- Very weak fundamentals (AI Score < 35) with high news risk (Red Flag ≥ 60) → lean toward REDUCE or AVOID
- Elevated sentiment risk (Red Flag ≥ 70) → exercise caution regardless of fundamentals
- Statistically stretched price (Z-Score > +2.5) with mediocre fundamentals → avoid a BUY call

Rules: English only. Be decisive and direct. Under 400 words total. Start the Actionable Recommendation label on its own line."""

        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
        )
        return response.message.content[0].text

    except Exception as e:
        err = str(e)
        if "invalid api key" in err.lower() or "unauthorized" in err.lower():
            return "❌ **Invalid API Key.** Please check your Cohere API Key."
        elif "rate limit" in err.lower():
            return "⏳ **Rate limit reached.** Please wait a moment and try again."
        else:
            return f"❌ **AI Engine Error:** {err}"


# ── MULTI-CURRENCY NORMALIZATION MATRIX (Target: EUR) ───────────────

@st.cache_data(ttl=1800, show_spinner="🌍 Fetching USD->EUR Rate...")
def get_forex_rates(target="EUR"):
    import yfinance as yf
    try:
        df = yf.download(f"USD{target}=X", period="1d", progress=False)["Close"]
        rate = df.iloc[-1].item() if not df.empty else 1.0
        return float(rate)
    except:
        return 1.0



@st.cache_data(ttl=1800, show_spinner="Fetching Live Macro Data...")
def fetch_macro_data():
    """Fetches real-time SPY, DXY, US10Y and VIX from Yahoo Finance."""
    import logging
    yf.set_tz_cache_location("/tmp/yfinance_tz") # Mute warnings in streamlit
    try:
        # DX-Y.NYB is dollar index, ^TNX is 10 yr treasury yield, ^VIX is volatility
        tickers = "SPY DX-Y.NYB ^TNX ^VIX"
        data = yf.download(tickers, period="5d", interval="1d", progress=False)
        
        # Handling multi-index columns from yfinance 0.2.x+
        if "Close" in data.columns.levels[0]:
            closes = data["Close"]
        else:
            closes = data

        closes = closes.ffill().dropna(how='all')
        if len(closes) < 2: raise ValueError("Not enough macro data rows")
            
        latest = closes.iloc[-1]
        prev = closes.iloc[-2]
        
        results = {}
        for t, col in zip(["SPY", "DXY", "US10Y", "VIX"], ["SPY", "DX-Y.NYB", "^TNX", "^VIX"]):
            if col in closes.columns:
                v_now = float(latest[col])
                v_prev = float(prev[col])
                chg = v_now - v_prev
                pct = (chg / v_prev) * 100 if v_prev != 0 else 0
                results[t] = {"val": v_now, "chg": chg, "pct": pct}
            else:
                results[t] = {"val": 0, "chg": 0, "pct": 0}
        # Apply Forex transformation selectively to SPY (USD -> EUR)
        usdeur_rate = get_forex_rates(target="EUR")
        results["SPY"]["val"] *= usdeur_rate
        # The change value in the UI also needs normalization to match the current price
        # Though pct change is unaffected by constant multiplier
        results["SPY"]["chg"] *= usdeur_rate

        return results
    except Exception as e:
        print("Macro fetch error:", e)
        # Fail-safe static data if Yahoo is 404/Blocked
        return {
            "SPY": {"val": 450.0, "chg": 0.5, "pct": 0.11},
            "DXY": {"val": 103.5, "chg": -0.2, "pct": -0.19},
            "US10Y": {"val": 4.25, "chg": 0.02, "pct": 0.47},
            "VIX": {"val": 14.5, "chg": -0.5, "pct": -3.33}
        }

@st.cache_resource(show_spinner="📥 Loading Institutional NLP Engine (FinBERT ~440MB)...")
def get_finbert_pipeline():
    """Loads the ProsusAI/finbert model for financial-specific sentiment analysis."""
    from transformers import pipeline
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

# ── PREMIUM UI: Institutional SVG Icon Library ──────────────────────────────
SVG_ICONS = {
    "chart": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M12 20V10"></path><path d="M18 20V4"></path><path d="M6 20V16"></path></svg>',
    "globe": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>',
    "search": '<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;vertical-align:middle;opacity:0.7;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',
    "risk": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
    "gem": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M6 3h12l4 6-10 13L2 9z"></path><path d="M11 22 2 9"></path><path d="m11 22 9-13"></path><path d="M6 3 2 9"></path><path d="M18 3l4 6"></path><path d="M11 22V9"></path><path d="M5 9h14"></path><path d="M2 9l4-6"></path><path d="M22 9l-4-6"></path></svg>',
    "calendar": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>',
    "ai": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="M12 8v4"></path><path d="M12 16h.01"></path></svg>',
    "layers": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
    "activity": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
    "package": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="m7.5 4.27 9 5.15"></path><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"></path><path d="m3.3 7 8.7 5 8.7-5"></path><path d="M12 22V12"></path></svg>',
    "brain": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.04-2.44 2.5 2.5 0 0 1-2-2.44 2.5 2.5 0 0 1-2-2.44 2.5 2.5 0 0 1 2-2.44 2.5 2.5 0 0 1 2-2.44 2.5 2.5 0 0 1 2.04-2.44A2.5 2.5 0 0 1 9.5 2Z"></path><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.04-2.44 2.5 2.5 0 0 0 2-2.44 2.5 2.5 0 0 0 2-2.44 2.5 2.5 0 0 0-2-2.44 2.5 2.5 0 0 0-2-2.44 2.5 2.5 0 0 0-2.04-2.44A2.5 2.5 0 0 0 14.5 2Z"></path></svg>',
    "bot": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8" y2="16"></line><line x1="16" y1="16" x2="16" y2="16"></line></svg>',
    "dna": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="m8 15 8-6"></path><path d="m11 18 2-2"></path><path d="m9 12 2-2"></path><path d="m13 10 2-2"></path><path d="m11 6 2-2"></path><path d="M15 22s-4-3-4-8V6s4-3 4-4"></path><path d="M9 22s4-4 4-8V6s-4-4-4-4"></path></svg>',
    "handshake": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="m11 17 2 2 6-7"></path><path d="m18 18 2-2 4-4"></path><path d="m3 10 8-8"></path><path d="m3 14 5-5"></path><path d="m7 18 5-5"></path><path d="m11 22 5-5"></path><path d="m18 10-2-2-6 7"></path></svg>',
    "flask": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M9 2h6"></path><path d="M12 2v14"></path><path d="M12 22a7 7 0 1 0 0-14 7 7 0 0 0 0 14z"></path></svg>',
    "trophy": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path><path d="M4 22h16"></path><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"></path><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"></path><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"></path></svg>',
    "alert": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;vertical-align:middle;opacity:0.8;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><path d="M12 9v4"></path><path d="M12 17h.01"></path></svg>'
}

def render_header(icon_key, text, level="####", color="#e8eaf6"):
    """Renders a premium monochromatic header with an SVG icon."""
    icon_svg = SVG_ICONS.get(icon_key, "")
    html = f"<div style='display:flex; align-items:center; margin-bottom:12px; color:{color};'>" \
           f"{icon_svg}<span style='font-size:1.15rem; font-weight:700; letter-spacing:0.02em;'>{text}</span></div>"
    st.markdown(html, unsafe_allow_html=True)

def get_flat_svg(key: str, size: int = 18, color: str = "currentColor", margin: str = "8px") -> str:
    """
    Returns a resized, recolored SVG string from SVG_ICONS.
    Dynamically injects size, color and margin so icons can be used
    inline at any scale without separate per-size definitions.
    """
    svg = SVG_ICONS.get(key, "")
    if not svg:
        return ""
    import re as _svg_re
    # Inject width / height
    svg = _svg_re.sub(r'width="\d+"', f'width="{size}"', svg)
    svg = _svg_re.sub(r'height="\d+"', f'height="{size}"', svg)
    # Inject color (stroke + fill where currentColor is used)
    svg = svg.replace('stroke="currentColor"', f'stroke="{color}"')
    svg = svg.replace('fill="currentColor"', f'fill="{color}"')
    # Inject margin-right
    svg = _svg_re.sub(r'margin-right:\s*\d+px', f'margin-right:{margin}', svg)
    return svg

def analyze_sentiment_finbert(headlines):
    """Batch processes headlines using FinBERT and returns an average score (-1 to 1)."""
    pipe = get_finbert_pipeline()
    if not pipe or not headlines:
        return 0
    
    results = pipe(headlines)
    scores = []
    for res in results:
        label = res['label'].lower()
        score = res['score']
        # Map: positive -> +score, negative -> -score, neutral -> 0
        if label == 'positive':
            scores.append(score)
        elif label == 'negative':
            scores.append(-score)
        else:
            scores.append(0)
    return np.mean(scores) if scores else 0
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Honest Quant Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Activate Authentication Gateway (Multi-tenant)
auth.require_auth()


# ── SESSION STATE INITIALIZATION ───────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0  # Default to Strategic Overview

# ── PREMIUM GLASSMORPHISM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at top right, #1a1c2c, #0d0e14);
        color: #e0e0e0;
    }
    
    /* Frosted Glass UI Blocks */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.2);
    }
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Header & Label Brightness Fix for Dark Mode */
    h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] p, label p {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.02);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        padding: 0 15px !important;
        border: none !important;
        color: #aaa !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db, #8e44ad) !important;
        color: white !important;
        box-shadow: 0 0 20px rgba(52, 152, 219, 0.4);
        transform: translateY(-2px);
    }

    /* Plotly Charts Container */
    div.stPlotlyChart {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    /* Refined KPI Containers (Symmetry & Integration) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        height: 160px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        padding: 15px !important;
        transition: transform 0.3s ease, border 0.3s ease !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        transform: translateY(-2px);
    }
    .kpi-label { 
        color: #b0b0b0; 
        font-size: 0.85rem; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
        margin-bottom: 6px; 
    }
    .kpi-value { 
        color: #fff; 
        font-size: 1.8rem; 
        font-weight: 700; 
        line-height: 1.1; 
        text-shadow: 0 0 10px rgba(255,255,255,0.2);
    }
    
    /* ── INTEL HUB: DEFINITIVE NEON PILL STYLE ── */
    /* Targetting ALL popover buttons globally for maximum override capability */
    .stPopover {
        display: inline-block !important;
    }
    .stPopover button {
        border-radius: 50px !important;
        background: linear-gradient(135deg, #00d2ff 0%, #9d50bb 100%) !important;
        border: 2px solid rgba(255,255,255,0.7) !important;
        padding: 8px 24px !important;
        height: 44px !important;
        min-width: 140px !important;
        color: white !important;
        font-weight: 900 !important;
        font-family: 'Courier New', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.5), 0 0 30px rgba(157, 80, 187, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }
    
    .stPopover button:hover {
        transform: scale(1.05) translateY(-2px) !important;
        background: linear-gradient(135deg, #00d2ff 25%, #9d50bb 125%) !important;
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.7), 0 0 50px rgba(157, 80, 187, 0.4) !important;
    }

    /* Force text color and font inside all popover pills */
    .stPopover button p {
        color: white !important;
        font-size: 0.9rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
        font-family: 'Courier New', monospace !important;
    }

    /* Remove Streamlit default arrow icon and center the button content */
    .stPopover svg {
        display: none !important;
    }

    /* ── EARNINGS CARDS ── */
    .earning-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .earning-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(0, 210, 255, 0.3);
        transform: translateX(4px);
    }
    .earning-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .earning-ticker {
        font-family: 'Courier New', monospace;
        font-weight: 900;
        color: #00d2ff;
        font-size: 1.1rem;
    }
    .earning-date {
        font-size: 0.75rem;
        color: #8899aa;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .earning-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 8px;
    }
    .earning-m-label {
        font-size: 0.6rem;
        color: #445566;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .earning-m-val {
        font-size: 0.85rem;
        font-weight: 700;
        color: #e8eaf6;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ── UTILITIES ───────────────────────────────────────────────────────────────
def get_rsi_vectorized(df, periods=14):
    """Fast vectorized RSI calculation."""
    close_delta = df['price_close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# ── DATA LOADING ──────────────────────────────────────────────────────────────
DB_PATH = os.path.join(ROOT, "warehouse", "stock_dw.duckdb")
WATCHLIST_PATH = os.path.join(ROOT, "warehouse", "user_watchlist.csv")  # Deprecated local fallback

def load_watchlist():
    cols = ["Ticker", "Status", "Thesis", "Catalyst", "Entry Target", "Invalidation Level", "Take Profit", "Next Earnings", "Added Date"]
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        return pd.DataFrame(columns=cols)
        
    try:
        supabase = auth.get_supabase_client()
        response = supabase.table("stock_watchlist").select("*").eq("user_id", st.session_state["user_id"]).execute()
        
        if not response.data:
            return pd.DataFrame(columns=cols)
            
        df = pd.DataFrame(response.data)
        
        # Render back to Dashboard naming convention
        rename_map = {
            "ticker": "Ticker",
            "status": "Status",
            "thesis": "Thesis",
            "catalyst": "Catalyst",
            "entry_target": "Entry Target",
            "invalidation_level": "Invalidation Level",
            "take_profit": "Take Profit",
            "next_earnings": "Next Earnings",
            "added_date": "Added Date"
        }
        df = df.rename(columns=rename_map)
        
        return df[cols]
    except Exception as e:
        st.sidebar.error(f"⚠️ Data Sync Error (Supabase Load): {e}")
        return pd.DataFrame(columns=cols)

def save_watchlist(df):
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        raise Exception("Authentication required to save data.")
        
    try:
        supabase = auth.get_supabase_client()
        user_id = st.session_state["user_id"]
        
        # Prepare data according to Postgres schema mapping (snake_case)
        records = []
        for _, row in df.iterrows():
            record = {
                "user_id": user_id,
                "ticker": str(row.get("Ticker", "")),
                "status": str(row.get("Status", "🔵 PENDING")),
                "thesis": str(row.get("Thesis", "")),
                "catalyst": str(row.get("Catalyst", "")),
                "entry_target": float(row.get("Entry Target", 0)) if pd.notna(row.get("Entry Target")) and row.get("Entry Target") else None,
                "invalidation_level": float(row.get("Invalidation Level", 0)) if pd.notna(row.get("Invalidation Level")) and row.get("Invalidation Level") else None,
                "take_profit": float(row.get("Take Profit", 0)) if pd.notna(row.get("Take Profit")) and row.get("Take Profit") else None,
                "next_earnings": str(row.get("Next Earnings", "TBD")),
            }
            records.append(record)
            
        # 1. Overwrite (Delete existing records for the logged-in user)
        supabase.table("stock_watchlist").delete().eq("user_id", user_id).execute()
        
        # 2. Insert the entire new Watchlist DataFrame into Supabase
        if records:
            supabase.table("stock_watchlist").insert(records).execute()
            
    except Exception as e:
        raise Exception(f"Failed to sync with Supabase: {e}")

def load_portfolio_from_db():
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        return {}
    try:
        supabase = auth.get_supabase_client()
        response = supabase.table("stock_portfolio").select("ticker, shares, cost_basis").eq("user_id", st.session_state["user_id"]).execute()
        if not response.data:
            return {}
        
        # Parse logic: output format expected by the app is a dict of shares and cost
        # Wait, returning a dict of {"AAPL": {"shares": 10.0, "cost": 150.0}}
        parsed_data = {}
        for row in response.data:
            ticker = row.get("ticker")
            parsed_data[ticker] = {
                "shares": float(row.get("shares", 0)),
                "cost": float(row.get("cost_basis", 0))
            }
        return parsed_data
    except Exception as e:
        st.sidebar.error(f"⚠️ Portfolio Sync Error: {e}")
        return {}

def save_portfolio_to_db(shares_dict, cost_dict):
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        return
    try:
        supabase = auth.get_supabase_client()
        user_id = st.session_state["user_id"]
        
        records = []
        for ticker in shares_dict.keys():
            records.append({
                "user_id": user_id,
                "ticker": ticker,
                "shares": float(shares_dict.get(ticker, 0)),
                "cost_basis": float(cost_dict.get(ticker, 0))
            })
            
        supabase.table("stock_portfolio").delete().eq("user_id", user_id).execute()
        if records:
            supabase.table("stock_portfolio").insert(records).execute()
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to save Portfolio to Cloud: {e}")
@contextlib.contextmanager
def get_db_connection(read_only=False):
    """Database connection context manager with fallback and Hybrid Remote support."""
    is_remote = os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true"
    
    if is_remote:
        # ── HYBRID REMOTE MODE (Parquet over S3/HTTP) ──
        try:
            conn = duckdb.connect(":memory:")
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            
            # S3 Secrets from environment
            s3_key = os.environ.get("S3_ACCESS_KEY_ID")
            s3_secret = os.environ.get("S3_SECRET_ACCESS_KEY")
            s3_endpoint = os.environ.get("S3_ENDPOINT", "").replace("https://", "")
            s3_region = os.environ.get("S3_REGION", "us-east-1")
            bucket = os.environ.get("S3_BUCKET_NAME", "warehouse")
            
            if not all([s3_key, s3_secret, s3_endpoint]):
                st.error("Missing S3 credentials for Remote Mode.")
                raise ValueError("Incomplete S3 configuration.")
            
            conn.execute(f"SET s3_region='{s3_region}';")
            conn.execute(f"SET s3_endpoint='{s3_endpoint}';")
            conn.execute(f"SET s3_access_key_id='{s3_key}';")
            conn.execute(f"SET s3_secret_access_key='{s3_secret}';")
            conn.execute("SET s3_use_ssl=true;")
            conn.execute("SET s3_url_style='path';")
            
            # Create Views to map remote Parquet to local table names
            # Tables to map (synced via etl/supabase_manager.py)
            table_map = {
                "marts.fct_daily_returns": "fct_daily_returns_p*.parquet", # Support sharding
                "marts.dim_companies": "dim_companies.parquet",
                "marts.dq_warnings": "dq_warnings.parquet",
                "marts.etl_audit": "etl_audit.parquet",
                "marts.agg_monthly_performance": "agg_monthly_performance.parquet", # Need to add this to sync
                "marts.dim_annual_financials": "dim_annual_financials.parquet", # Need to add this to sync
                "marts.dim_quarterly_financials": "dim_quarterly_financials.parquet", # Need to add this to sync
                "raw.hist_fcf": "hist_fcf.parquet",
                "raw.hist_fcf_quarterly": "hist_fcf_quarterly.parquet",
                "raw.earnings_calendar": "earnings_calendar.parquet",
                "raw.historical_financials": "historical_financials.parquet",
                "raw.quarterly_financials": "quarterly_financials.parquet",
                "raw.company_info": "company_info.parquet", # Need to add this to sync
            }
            
            conn.execute("CREATE SCHEMA IF NOT EXISTS marts; CREATE SCHEMA IF NOT EXISTS raw;")
            for table, file in table_map.items():
                s3_path = f"s3://{bucket}/{file}"
                conn.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{s3_path}')")
                
            yield conn
            return
        except Exception as e:
            st.error(f"Failed to initialize Remote Mode: {e}")
            raise e
        finally:
            if 'conn' in locals():
                conn.close()

    # ── LOCAL MODE (File-based DuckDB) ──
    possible_paths = [
        DB_PATH,
        os.path.join(ROOT, "warehouse", "stock_demo.duckdb")
    ]
    
    actual_path = None
    for p in possible_paths:
        if os.path.exists(p):
            actual_path = p
            break
            
    if not actual_path:
        wh_dir = os.path.join(ROOT, "warehouse")
        if os.path.exists(wh_dir):
            all_files = os.listdir(wh_dir)
            duck_files = [f for f in all_files if f.endswith(".duckdb")]
            if duck_files:
                actual_path = os.path.join(wh_dir, duck_files[0])
    
    if not actual_path:
        st.error(f"FATAL: Database file not found at {DB_PATH}")
        raise FileNotFoundError(f"Database missing at {DB_PATH}")
        
    conn = duckdb.connect(actual_path, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()

@st.cache_data(ttl=3600, show_spinner="📉 Loading Institutional Data Warehouse...")
def load_data():
    """Load all required data, normalize currencies, and pre-compute técnicos inside cache."""
    with get_db_connection(read_only=True) as conn:
        prices_f = conn.execute("""
            SELECT f.date, f.ticker, d.company, d.sector, d.region,
                   f.price_open, f.price_high, f.price_low, f.price_close, 
                   f.daily_return_pct, f.volume,
                   f.ma_20, f.ma_50, f.ma_200, f.ma_signal, 
                   f.price_z_score, f.pct_from_ma200, f.pct_from_52w_high,
                   f.is_volume_spike, f.cap_category
            FROM marts.fct_daily_returns f
            LEFT JOIN marts.dim_companies d USING (ticker)
            WHERE f.date >= CURRENT_DATE - INTERVAL 15 MONTH
            ORDER BY f.date
        """).df()

        companies_f = conn.execute("""
            SELECT d.*, r.free_cashflow 
            FROM marts.dim_companies d
            LEFT JOIN raw.company_info r USING (ticker)
        """).df()
        monthly_f = conn.execute("SELECT * FROM marts.agg_monthly_performance ORDER BY month, ticker").df()
        annual_f = conn.execute("SELECT * FROM marts.dim_annual_financials").df()
        
        try:
            quarterly_f = conn.execute("SELECT * FROM marts.dim_quarterly_financials").df()
        except Exception:
            quarterly_f = pd.DataFrame(columns=["ticker", "year", "quarter", "report_date", "revenue", "eps"])
            
        try:
            earnings_calendar = conn.execute("SELECT * FROM raw.earnings_calendar").df()
            if not earnings_calendar.empty:
                earnings_calendar["earnings_date"] = pd.to_datetime(earnings_calendar["earnings_date"])
            else:
                # Ensure columns exist even if empty
                earnings_calendar = pd.DataFrame(columns=["ticker", "earnings_date", "eps_avg", "rev_avg"])
        except Exception:
            earnings_calendar = pd.DataFrame(columns=["ticker", "earnings_date", "eps_avg", "rev_avg"])

        try:
            dq_warnings_f = conn.execute("SELECT * FROM marts.dq_warnings ORDER BY is_critical DESC, violations DESC").df()
        except Exception:
            dq_warnings_f = pd.DataFrame()

        try:
            hist_fcf_f = conn.execute("SELECT ticker, year, free_cash_flow, operating_cash_flow FROM raw.hist_fcf ORDER BY ticker, year").df()
        except Exception:
            hist_fcf_f = pd.DataFrame()

        try:
            hist_fcf_q_f = conn.execute("SELECT ticker, year, quarter, free_cash_flow, operating_cash_flow FROM raw.hist_fcf_quarterly ORDER BY ticker, year, quarter").df()
        except Exception:
            hist_fcf_q_f = pd.DataFrame()

        # ── Pipeline Health Data ──
        try:
            etl_audit_f = conn.execute("""
                SELECT status, start_time, rows_processed
                FROM marts.etl_audit 
                ORDER BY start_time DESC 
                LIMIT 1
            """).df()
        except:
            etl_audit_f = pd.DataFrame()

        try:
            total_tickers_f = conn.execute("SELECT COUNT(*) FROM marts.dim_companies").fetchone()[0]
        except:
            total_tickers_f = 0
    # ── PRE-PROCESSING INSIDE CACHE ──
    prices_f["date"] = pd.to_datetime(prices_f["date"])
    monthly_f["month"] = pd.to_datetime(monthly_f["month"])
    prices_f = prices_f.sort_values(['ticker', 'date'])
    
    # Vectorized RSI (only for those missing it or to ensure consistency)
    prices_f['rsi'] = prices_f.groupby('ticker', group_keys=False).apply(lambda x: get_rsi_vectorized(x), include_groups=False)
    


    return (
        prices_f, companies_f, monthly_f, annual_f, quarterly_f, earnings_calendar, 
        dq_warnings_f, hist_fcf_f, hist_fcf_q_f, etl_audit_f, total_tickers_f
    )


# ── ANALYTICS ENGINE: Global Screener Data ──────────────────────────────────
@st.cache_data(ttl=3600)
def compute_institutional_rating(
    ai_score: float,
    ma_sig: str,
    latest_rsi: float,
    upside: float,
    pe_v: float,
    peg_v: float,
    sector: str,
    w52_pos: float,
    rr: float,
) -> dict:
    """
    Unified 5-Pillar Institutional Rating Engine (v13.0).
    Used by BOTH Opportunity Radar Screener and Deep Dive tab to ensure
    consistent Action labels across the entire dashboard.

    Returns:
        dict with keys:
            action_label  (str)  — plain text: STRONG BUY / BUY / HOLD / SELL / REDUCE
            action_color  (str)  — hex color for UI rendering
            p_trend_c, p_qual_c, p_val_c, p_risk_c, p_conv_c  (str)
    """
    # ── PILLAR 1: TECHNICAL TREND ──────────────────────────────────────────
    if ma_sig == "BULLISH" and latest_rsi < 65:
        p_trend_c = "#2ecc71"
    elif ma_sig == "BULLISH" and latest_rsi >= 65:
        p_trend_c = "#f1c40f"   # Extended / overbought in uptrend
    elif ma_sig == "BEARISH" and latest_rsi <= 35:
        p_trend_c = "#f1c40f"   # Oversold in downtrend — caution
    else:
        p_trend_c = "#e74c3c"

    # ── PILLAR 2: QUALITY ────────────────────────────────────────────────
    if ai_score >= 70:   p_qual_c = "#00ffcc"
    elif ai_score >= 55: p_qual_c = "#2ecc71"
    elif ai_score >= 40: p_qual_c = "#f1c40f"
    else:                p_qual_c = "#e74c3c"

    # ── PILLAR 3: VALUATION (Sector-Aware) ──────────────────────────────
    _sector_lc = str(sector or "").lower()
    _is_growth = any(s in _sector_lc for s in ["tech", "semi", "software", "cloud", "ai", "comm"])
    _pe_cheap_limit      = 28.0 if _is_growth else 18.0
    _pe_expensive_limit  = 65.0 if _is_growth else 42.0
    _peg_expensive_limit = 3.5  if _is_growth else 2.5
    _peg_cheap_limit     = 1.2  if _is_growth else 0.8

    _val_expensive  = (pe_v > _pe_expensive_limit and pe_v > 0) or (peg_v > _peg_expensive_limit and peg_v > 0)
    _val_cheap      = (upside > 15) and (peg_v < _peg_cheap_limit or pe_v < _pe_cheap_limit) and pe_v > 0
    _val_premium_ok = (upside > 8) and (ai_score >= 65) and (peg_v < 2.8 or pe_v < (55 if _is_growth else 35))
    _val_compounder = (upside > 5) and (ai_score >= 55) and (not _val_expensive)
    _val_fair       = (upside > 0) and (not _val_expensive)

    if _val_cheap:
        p_val_c = "#2ecc71"
    elif _val_premium_ok or _val_compounder:
        p_val_c = "#3498db"
    elif _val_fair:
        p_val_c = "#f1c40f"
    elif _val_expensive:
        p_val_c = "#e67e22"
    elif pe_v < 0:
        p_val_c = "#e74c3c"
    else:
        p_val_c = "#95a5a6"

    # ── PILLAR 4: RISK (52-Week Position) ───────────────────────────────
    if w52_pos > 80:   p_risk_c = "#e74c3c"
    elif w52_pos < 20: p_risk_c = "#2ecc71"
    else:              p_risk_c = "#f1c40f"

    # ── PILLAR 5: CONVICTION (Risk / Reward) ────────────────────────────
    if rr > 2.5:   p_conv_c = "#00ffcc"
    elif rr > 1.2: p_conv_c = "#2ecc71"
    else:           p_conv_c = "#e74c3c"

    # ── SYNTHESIS: Final Action Label ────────────────────────────────────
    pts = (
        (1 if p_trend_c in ["#2ecc71", "#00ffcc"] else 0) +
        (1 if p_qual_c  in ["#2ecc71", "#00ffcc"] else 0) +
        (1 if p_val_c   in ["#2ecc71", "#00ffcc", "#3498db"] else 0) +
        (1 if p_risk_c  == "#2ecc71" else 0) +
        (1 if p_conv_c  in ["#2ecc71", "#00ffcc"] else 0)
    )

    if pts >= 4 and p_qual_c != "#e74c3c":
        action_label, action_color = "STRONG BUY",          "#00ffcc"
    elif pts >= 3 and p_trend_c == "#f1c40f" and latest_rsi < 45:
        action_label, action_color = "BUY / ACCUMULATE",    "#2ecc71"
    elif p_trend_c == "#e74c3c" and p_val_c == "#e74c3c":
        action_label, action_color = "SELL / AVOID",        "#e74c3c"
    elif pts <= 1 and p_qual_c == "#e74c3c":
        action_label, action_color = "SELL / AVOID",        "#e74c3c"
    elif pts <= 1 and p_qual_c in ["#2ecc71", "#00ffcc"]:
        action_label, action_color = "HOLD / NEUTRAL",      "#f1c40f"
    elif latest_rsi > 70 and pts <= 3:
        action_label, action_color = "REDUCE / UNDERPERFORM","#e67e22"
    else:
        action_label, action_color = "HOLD / NEUTRAL",      "#f1c40f"

    return {
        "action_label":  action_label,
        "action_color":  action_color,
        "p_trend_c":     p_trend_c,
        "p_qual_c":      p_qual_c,
        "p_val_c":       p_val_c,
        "p_risk_c":      p_risk_c,
        "p_conv_c":      p_conv_c,
        "pts":           pts,
    }


def get_tactical_metrics(ticker_prices: "pd.DataFrame", cur_p: float) -> dict:
    """
    Single source of truth for all short/mid-term tactical indicators.
    Called identically by the Screener, Deep Dive, and any future tab.

    Returns a dict containing:
        rsi        — RSI-14 (float, from warehouse-vectorised column if available)
        s1         — 20-day support (lowest low)
        r1         — 20-day resistance (highest high)
        stop_loss  — s1 * 0.96
        tp1        — r1 * 1.05  (5% extension above resistance)
        rr         — reward-to-risk ratio
        w52_pos    — position within 52-week range [0-100]
    """
    # RSI — prefer the warehouse-computed column; fall back to local calc
    if "rsi" in ticker_prices.columns and ticker_prices["rsi"].notna().any():
        rsi_val = float(ticker_prices["rsi"].iloc[-1])
    else:
        delta = ticker_prices["price_close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_series = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

    # Support / Resistance (20-day window)
    s1 = float(ticker_prices["price_low"].tail(20).min())
    r1 = float(ticker_prices["price_high"].tail(20).max())

    # Derived levels (identical formula everywhere)
    stop_loss = s1 * 0.96
    tp1       = r1 * 1.05          # 5% extension above 20-day high

    # Risk/Reward — two variants:
    #   rr_score: uses raw r1 as target (no extension) → feeds rating engine
    #   rr:       uses tp1 = r1*1.05 → display only in Deep Dive trading plan
    risk_dist     = cur_p - stop_loss
    rr_score_dist = r1  - cur_p
    rr_disp_dist  = tp1 - cur_p
    rr_score = (rr_score_dist / risk_dist) if risk_dist > 0 else 0.0
    rr       = (rr_disp_dist  / risk_dist) if risk_dist > 0 else 0.0

    # 52-week position
    df_252  = ticker_prices.tail(252)
    w52_hi  = df_252["price_high"].max()
    w52_lo  = df_252["price_low"].min()
    w52_rng = w52_hi - w52_lo
    w52_pos = ((cur_p - w52_lo) / w52_rng * 100) if w52_rng > 0 else 50.0

    return {
        "rsi":       rsi_val,
        "s1":        s1,
        "r1":        r1,
        "stop_loss": stop_loss,
        "tp1":       tp1,
        "tp2":       r1 * 1.15,    # secondary target (15% above resistance)
        "rr":        rr,           # display only
        "rr_score":  rr_score,     # feeds compute_institutional_rating
        "w52_pos":   w52_pos,
        "w52_hi":    float(w52_hi),
        "w52_lo":    float(w52_lo),
    }

@st.cache_data(ttl=3600)
def get_master_screener_data(_companies_df, _prices_df, _quarterly_fin, _annual_fin):
    # Exclude non-investable instruments: indices & volatility measures
    _non_equities = {"^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"}
    _non_equity_sectors = {"Benchmark", "Volatility"}
    screener_rows = []
    
    for _, row in _companies_df.iterrows():
        ticker = row['ticker']
        # Skip indices and volatility
        if ticker in _non_equities: continue
        if str(row.get('sector', '')).strip() in _non_equity_sectors: continue
        ticker_prices = _prices_df[_prices_df['ticker'] == ticker].sort_values('date')
        if ticker_prices.empty: continue
        
        # RSI (Pre-calculated in transform.py)
        latest_rsi = ticker_prices["rsi"].iloc[-1] if not ticker_prices.empty else 50
        
        cur_p = ticker_prices["price_close"].iloc[-1]
        target_p = row.get("target_mean_price", 0)
        upside = ((target_p / cur_p) - 1) * 100 if target_p > 0 else 0
        
        if len(ticker_prices) >= 2:
            prev_p = ticker_prices["price_close"].iloc[-2]
            chg_1d = ((cur_p / prev_p) - 1) * 100 if prev_p > 0 else 0
        else:
            chg_1d = 0
            
        mcap = row.get("market_cap", 0)
        mcap_b = (mcap / 1e9) if pd.notnull(mcap) and mcap > 0 else 0
        
        # ── AI SCORING (ENRICHED WITH TECHNICALS) ──────────────────────────
        latest_p = ticker_prices.iloc[-1]
        score_input = row.to_dict()
        score_input['rsi'] = float(latest_rsi)
        score_input['ma_signal'] = str(latest_p.get('ma_signal', 'NEUTRAL'))
        score_input['price_z_score'] = float(latest_p.get('price_z_score', 0))
        score_input['upside_pct'] = float(upside)
        
        # Ensure numeric safety for fundamental scores
        for col in ['pe_ratio', 'peg_ratio', 'price_to_book', 'roe', 'fcf_margin', 'dividend_yield_pct']:
            val = score_input.get(col)
            try: score_input[col] = float(val) if pd.notnull(val) else None
            except: score_input[col] = None

        ai_score  = compute_score(score_input)
        
        # 🚀 OPTIMIZATION: Compute FMI instantly using pre-calculated DuckDB variables 
        # instead of dataframe filtering which causes heavy CPU load in loops.
        from etl.utils import compute_fmi_details
        _fmi_res  = compute_fmi_details(row.to_dict())
        fmi_score = _fmi_res["total"]
        fmi_lbl   = _fmi_res["label"]
        
        # ── Unified 5-Pillar Rating (delegates to compute_institutional_rating) ──
        ma_sig = str(latest_p.get('ma_signal', 'NEUTRAL'))
        pe_v   = float(score_input.get('pe_ratio')  or 0)
        peg_v  = float(score_input.get('peg_ratio') or 0)

        # Use shared tactical metrics (same formula as Deep Dive)
        _tm = get_tactical_metrics(ticker_prices, cur_p)

        _rating = compute_institutional_rating(
            ai_score   = ai_score,
            ma_sig     = ma_sig,
            latest_rsi = _tm["rsi"],
            upside     = float(upside),
            pe_v       = pe_v,
            peg_v      = peg_v,
            sector     = str(row.get('sector', '')),
            w52_pos    = _tm["w52_pos"],
            rr         = _tm["rr_score"],   # scoring uses raw r1 target
        )
        action_label = _rating["action_label"]   # plain text — no emoji


        
        # Additional metrics
        div_yield = float(row.get('dividend_yield_pct', 0)) if pd.notnull(row.get('dividend_yield_pct')) else 0
        fcf_margin = float(row.get('fcf_margin', 0)) if pd.notnull(row.get('fcf_margin')) else 0
        
        # Safe Financial Metrics (Handling pd.NA)
        eb_val = row.get('ebitda')
        td_val = row.get('total_debt')
        ebitda = float(eb_val) if pd.notnull(eb_val) else 0
        total_debt = float(td_val) if pd.notnull(td_val) else 0
        
        if ebitda > 0:
            debt_ebitda = min(total_debt / ebitda, 99)
        else:
            debt_ebitda = 99
            
        ev_eb_val = row.get('ev_to_ebitda')
        ev_ebitda = float(ev_eb_val) if pd.notnull(ev_eb_val) else 0
        
        roe_raw = row.get('roe')
        roe_val = (float(roe_raw) * 100) if pd.notnull(roe_raw) else 0
        net_payout = row.get('net_payout_yield_pct', 0) or 0
        vol_30d = row.get('volatility_30d', 0) or 0
        short_pct = (row.get('short_percent_of_float', 0) * 100) if pd.notnull(row.get('short_percent_of_float')) else 0

        screener_rows.append({
            "Ticker": ticker,
            "Company": row['company'],
            "Sector": row['sector'],
            "Action": action_label,
            "Quality": ai_score,
            "Upside (%)": round(upside, 1),
            "1D Chg (%)": round(chg_1d, 2),
            "Price": cur_p,
            "MCap (B)": round(mcap_b, 1),
            "RSI (14)": round(latest_rsi, 1),
            "Z-Score": round(ticker_prices['price_z_score'].iloc[-1] if 'price_z_score' in ticker_prices.columns else 0, 2),
            "vs MA200 (%)": round(ticker_prices['pct_from_ma200'].iloc[-1] if 'pct_from_ma200' in ticker_prices.columns else 0, 1),
            "Yield (%)": round(div_yield, 2),
            "Net Payout (%)": round(net_payout, 2),
            "FCF Margin (%)": round(fcf_margin, 1),
            "ROE (%)": round(roe_val, 1),
            "P/E (Fwd)": round(row.get('forward_pe', 999) or 999, 1),
            "EV/EBITDA": round(ev_ebitda, 1) if ev_ebitda else 0,
            "PEG": round(row.get('peg_ratio', 99) or 99, 2),
            "Debt/EBITDA": round(debt_ebitda, 2),
            "Vol 30D (%)": round(vol_30d, 1) if vol_30d else 0,
            "Short %": round(short_pct, 1),
            "Trend": latest_p.get('ma_signal', 'NEUTRAL'),
            "FMI": fmi_score,
            "FMI Label": fmi_lbl,
            "Region": row['region']
        })
        
    return pd.DataFrame(screener_rows)


def render_sector_health_matrix(m_df: pd.DataFrame):
    """
    Renders a 4-quadrant sector analysis matrix: Valuation (PEG) vs Momentum (Z-Score).
    """
    if m_df.empty:
        st.warning("No data available for Sector Matrix.")
        return

    # 1. Aggregation — Group by Sector
    df_clean = m_df.copy()
    
    # Ensure numeric types
    df_clean['PEG_Num'] = pd.to_numeric(df_clean['PEG'], errors='coerce')
    df_clean['Z_Num'] = pd.to_numeric(df_clean['Z-Score'], errors='coerce')
    df_clean['Upside_Num'] = pd.to_numeric(df_clean['Upside (%)'], errors='coerce')
    
    # We clean PEG to exclude nonsensical negative values or massive outliers for the average
    # Negative PEG usually means negative earnings or negative growth, which breaks the PEG logic
    df_matrix = df_clean[df_clean['PEG_Num'] > 0].copy()
    
    if df_matrix.empty:
        st.info("Insufficient sector data with positive PEG for matrix visualization.")
        return

    sector_stats = df_matrix.groupby('Sector').agg({
        'PEG_Num': 'mean',
        'Z_Num': 'mean',
        'Upside_Num': 'mean',
        'Ticker': 'count'
    }).reset_index()
    
    sector_stats.columns = ['Sector', 'Avg_PEG', 'Avg_ZScore', 'Avg_Upside', 'Count']
    
    # 2. Quadrant Definitions
    # X-axis: PEG (Valuation) — Lower is cheaper
    # Y-axis: Z-Score (Momentum) — Higher is stronger
    
    fig = px.scatter(
        sector_stats, 
        x='Avg_PEG', 
        y='Avg_ZScore',
        size='Count',
        color='Avg_Upside',
        color_continuous_scale='RdYlGn',
        text='Sector',
        labels={'Avg_PEG': 'Valuation (Avg PEG Ratio)', 'Avg_ZScore': 'Momentum (Avg Z-Score)'},
        title="Institutional Sector Matrix: Price vs Value Divergence",
        template="plotly_dark",
        height=600,
        hover_data=['Avg_Upside', 'Count']
    )

    # Calculate pivots (Medians provide better balance than means for quadrants)
    peg_pivot = sector_stats['Avg_PEG'].median()
    z_pivot   = 0  # 0 is the logical neutral point for Z-Score

    fig.add_hline(y=z_pivot, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.add_vline(x=peg_pivot, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    # Quadrant Labels (Positioned in corners)
    # Top-Left: High Momentum, Low PEG
    fig.add_annotation(x=sector_stats['Avg_PEG'].min(), y=sector_stats['Avg_ZScore'].max(), 
                       text="LEADERS (Strong + Fair Value)", showarrow=False, font=dict(color="#2ecc71", size=10), xanchor="left")
    # Top-Right: High Momentum, High PEG
    fig.add_annotation(x=sector_stats['Avg_PEG'].max(), y=sector_stats['Avg_ZScore'].max(), 
                       text="HYPE ZONE (Strong + Expensive)", showarrow=False, font=dict(color="#f1c40f", size=10), xanchor="right")
    # Bottom-Left: Low Momentum, Low PEG
    fig.add_annotation(x=sector_stats['Avg_PEG'].min(), y=sector_stats['Avg_ZScore'].min(), 
                       text="VALUE TRAP / DEEP VALUE", showarrow=False, font=dict(color="#3498db", size=10), xanchor="left")
    # Bottom-Right: Low Momentum, High PEG
    fig.add_annotation(x=sector_stats['Avg_PEG'].max(), y=sector_stats['Avg_ZScore'].min(), 
                       text="LAGGARDS (Weak + Expensive)", showarrow=False, font=dict(color="#e74c3c", size=10), xanchor="right")

    fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='white')))
    fig.update_layout(
        margin=dict(l=20, r=20, b=50, t=50),
        coloraxis_colorbar=dict(title="Avg Upside %"),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Methodology Footer
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.03); padding:15px; border-radius:10px; font-size:0.8rem; border:1px solid rgba(255,255,255,0.1);'>
        <b>Matrix Methodology:</b><br>
        • <b>Vertical Axis (Z-Score):</b> Measures price momentum relative to historical standard deviations. > 0 is strong.<br>
        • <b>Horizontal Axis (PEG):</b> Measures valuation relative to growth. Lower is cheaper. Pivot set at median PEG ({peg_pivot:.2f}).<br>
        • <b>Software Stocks:</b> Currently identifyable in the bottom-left quadrant (Low PEG but Negative Z-Score) — capturing high-conviction "oversold" opportunities.
    </div>
    """, unsafe_allow_html=True)


# Primary Data Load (Cached)
prices_full, companies_full, monthly_full, annual_fin, quarterly_fin, earnings_cal, dq_warnings, hist_fcf_full, hist_fcf_q_full, etl_audit, total_universe_size = load_data()
m_df = get_master_screener_data(companies_full, prices_full, quarterly_fin, annual_fin)


# Shared Global Views (Filtered from the cached full datasets)
all_tickers = sorted(prices_full["ticker"].unique().tolist())
ticker_to_name = dict(zip(companies_full['ticker'], companies_full['company']))

# Clean non-benchmark views
companies = companies_full[companies_full["ticker"] != "SPY"]
spy_prices = prices_full[prices_full["ticker"] == "SPY"]
prices = prices_full[prices_full["ticker"] != "SPY"]
monthly = monthly_full[monthly_full["ticker"] != "SPY"]

def format_ticker(ticker):
    name = ticker_to_name.get(ticker)
    return f"{ticker}: {name}" if name else ticker

# ── UTILITY FUNCTIONS ───────────────────────────────────────────────────────
def render_metric_row(label, value, delta=None, suffix="", is_pct=False, color_invert=False, value_color=None, help_text=None):
    """Render a compact inline KPI row (label | value | delta)."""
    delta_html = ""
    if delta is not None:
        try:
            d_val = float(delta)
            color = ("#e74c3c" if d_val >= 0 else "#2ecc71") if color_invert else ("#2ecc71" if d_val >= 0 else "#e74c3c")
            sign  = "+" if d_val >= 0 else ""
            d_text = f"{sign}{d_val:.1f}%" if is_pct else f"{sign}{d_val:.2f}{suffix}"
            delta_html = f"<span style='color:{color};font-size:0.72rem;font-weight:700;white-space:nowrap;'>{d_text}</span>"
        except:
            delta_html = f"<span style='color:#888;font-size:0.72rem;'>{delta}</span>"

    val_col = value_color if value_color else "#e8eaf6"
    tooltip_attr = f"title='{help_text}'" if help_text else ""
    cursor_style = "cursor:help;" if help_text else ""

    st.markdown(f"""
        <div {tooltip_attr} style='display:flex;align-items:center;flex-wrap:wrap;row-gap:2px;{cursor_style}
                    padding:5px 8px;border-bottom:1px solid rgba(255,255,255,0.05);'>
            <span style='color:#8899aa;font-size:0.72rem;font-weight:600;text-transform:uppercase;
                         letter-spacing:0.04em;white-space:nowrap;margin-right:auto;'>{label}</span>
            <span style='color:{val_col};font-size:0.88rem;font-weight:700;text-align:right;
                         white-space:nowrap;margin-left:8px;'>{value}{suffix}</span>
            <span style='text-align:right;margin-left:8px;'>{delta_html}</span>
        </div>
    """, unsafe_allow_html=True)

def render_metric_tile(label, value, delta=None, suffix="", is_pct=False, color_invert=False, help_text=None):
    """Render a compact standalone KPI card with optional tooltip."""
    delta_html = ""
    if delta is not None:
        try:
            d_val = float(delta)
            color = ("#e74c3c" if d_val >= 0 else "#2ecc71") if color_invert else ("#2ecc71" if d_val >= 0 else "#e74c3c")
            sign  = "+" if d_val >= 0 else ""
            d_text = f"{sign}{d_val:.1f}%" if is_pct else f"{sign}{d_val:.2f}{suffix}"
            delta_html = f"<div style='color:{color};font-size:0.68rem;font-weight:700;margin-top:1px;'>{d_text}</div>"
        except:
            delta_html = f"<div style='color:#888;font-size:0.68rem;'>{delta}</div>"

    tooltip_attr = f"title='{help_text}'" if help_text else ""
    st.markdown(f"""
        <div {tooltip_attr} style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:6px;padding:5px 8px;margin-bottom:5px;text-align:center;cursor:help;'>
            <div style='color:#8899aa;font-size:0.58rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:2px;'>{label}</div>
            <div style='color:#e8eaf6;font-size:0.95rem;font-weight:700;display:flex;align-items:center;justify-content:center;'>{value}{suffix}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

# ── ANALYTICS PRE-COMPUTATION (Scores, Alerts, KPIs) ──────────────────────────
if not prices_full.empty:
    indices_list = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]
    stock_count = prices_full[~prices_full['ticker'].isin(indices_list)]['ticker'].nunique()
else:
    stock_count = 0

# ── Sidebar: Institutional Mission Control ───────────────────────────────────
if not prices_full.empty:
    min_db_date = prices_full["date"].min().date()
    max_db_date = prices_full["date"].max().date()

    # ── Integrated Infrastructure & DQ Pulse (Unified Sidebar) ──
    if not etl_audit.empty:
        last_run = etl_audit.iloc[0]
        st.sidebar.markdown("<div class='sb-section-label'>Infrastructure Engine</div>", unsafe_allow_html=True)
        h_color = "#2ecc71" if last_run['status'] == 'SUCCESS' else "#e74c3c"
        try:
            ls_time = pd.to_datetime(last_run['start_time']).strftime('%b %d, %H:%M')
        except: ls_time = "N/A"
        
        # DQ Summary and detail preparation
        crit_dq = len(dq_warnings[dq_warnings['is_critical']]) if not dq_warnings.empty else 0
        warn_dq = len(dq_warnings[~dq_warnings['is_critical']]) if not dq_warnings.empty else 0
        dq_color = "#2ecc71" if (crit_dq == 0 and warn_dq == 0) else ("#e74c3c" if crit_dq > 0 else "#f1c40f")
        dq_text = "CLEAN" if (crit_dq == 0 and warn_dq == 0) else (f"{crit_dq} CRIT" if crit_dq > 0 else f"{warn_dq} WARN")

        # UI rendering is simplified below

        rows_in = last_run['rows_processed'] if 'rows_processed' in last_run else 0
        pulse_html = f"""
<div style='padding:12px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:8px; margin-bottom:10px;'>
<div style='display:flex; align-items:center; gap:10px;'>
<div style='width:10px; height:10px; border-radius:50%; background:{h_color}; box-shadow:0 0 10px {h_color};'></div>
<div style='flex-grow:1;'>
<div style='font-size:0.75rem; color:#e8eaf6; font-weight:700; line-height:1.1;'>{last_run['status']}</div>
<div style='font-size:0.6rem; color:#8899aa; margin-top:2px;'>Sync: {ls_time}</div>
</div>
<div style='text-align:right;'>
<div style='font-size:0.65rem; color:{dq_color}; font-weight:700; line-height:1.1;'>{dq_text}</div>
<div style='font-size:0.5rem; color:#667788; text-transform:uppercase; letter-spacing:0.04em;'>Integrity</div>
</div>
</div>
</div>""".strip()
        st.sidebar.markdown(pulse_html, unsafe_allow_html=True)
    
    indices = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]

    # ── STICKY CONTEXT: Unified Asset Selection across Tabs ───────────────────
    if 'active_ticker' not in st.session_state:
        st.session_state.active_ticker = "AAPL"

    # ── SIDEBAR CSS ───────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"] { background: #0a0e1a; }
    .sb-section-label {
        font-family: 'Courier New', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.15em;
        color: #445566;
        text-transform: uppercase;
        margin: 14px 0 6px 0;
        border-bottom: 1px solid #1a2233;
        padding-bottom: 4px;
    }
    .sb-macro-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 10px;
        border-radius: 5px;
        margin-bottom: 4px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.05);
        font-family: 'Courier New', monospace;
    }
    .sb-macro-label { font-size: 0.68rem; color: #667788; }
    .sb-macro-val   { font-size: 0.85rem; font-weight: 700; color: #dde4ee; }
    .sb-macro-delta { font-size: 0.68rem; font-weight: 700; }
    .sb-regime-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        font-family: 'Courier New', monospace;
        margin-top: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── LIVE MACRO PULSE (computed later but rendered immediately via placeholder) ─
    _macro_sidebar_placeholder = st.sidebar.empty()

    # ── TIME HORIZON ──────────────────────────────────────────────────────────
    st.sidebar.markdown("<div class='sb-section-label'>Temporal Control</div>", unsafe_allow_html=True)
    horizon_options = ["1D", "1W", "1M", "3M", "6M", "1Y", "YTD", "3Y", "5Y", "ALL", "Custom"]
    selected_horizon = st.sidebar.segmented_control(
        "Horizon",
        options=horizon_options,
        selection_mode="single",
        default="1Y",
        label_visibility="collapsed",
        key="time_horizon_ctrl"
    )
    if not selected_horizon:
        selected_horizon = "1Y"

    # Universal Data Scope (Time only, no Ticker/Sector restriction)
    companies = companies_full.copy()
    prices    = prices_full.copy()
    monthly   = monthly_full.copy()

    # Horizon Logic
    end_date = max_db_date
    if selected_horizon == "1D":  start_date = max_db_date - timedelta(days=1)
    elif selected_horizon == "1W": start_date = max_db_date - timedelta(days=7)
    elif selected_horizon == "1M": start_date = max_db_date - timedelta(days=30)
    elif selected_horizon == "3M": start_date = max_db_date - timedelta(days=90)
    elif selected_horizon == "6M": start_date = max_db_date - timedelta(days=180)
    elif selected_horizon == "YTD": start_date = date(max_db_date.year, 1, 1)
    elif selected_horizon == "3Y": start_date = max_db_date - timedelta(days=1095)
    elif selected_horizon == "5Y": start_date = max_db_date - timedelta(days=1825)
    elif selected_horizon == "ALL": start_date = min_db_date
    elif selected_horizon == "Custom":
        with st.sidebar.expander("Custom Range", expanded=True):
            custom_range = st.date_input(
                "Pick Dates",
                value=(max_db_date - timedelta(days=365), max_db_date),
                min_value=min_db_date,
                max_value=max_db_date
            )
        if isinstance(custom_range, (list, tuple)) and len(custom_range) == 2:
            start_date, end_date = custom_range
        else:
            start_date = custom_range if not isinstance(custom_range, (list, tuple)) else custom_range[0]
            end_date   = max_db_date
    else:
        start_date = max_db_date - timedelta(days=365)

    # Clamp to DB boundaries
    start_date = max(start_date, min_db_date)
    end_date   = min(end_date, max_db_date)

    st.sidebar.caption(f"Range: {start_date:%b %d, %Y}  →  {end_date:%b %d, %Y}")

    # Apply time filters
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)
    prices      = prices[(prices["date"] >= t_start) & (prices["date"] <= t_end)]
    spy_prices  = spy_prices[(spy_prices["date"] >= t_start) & (spy_prices["date"] <= t_end)]
    monthly     = monthly[(monthly["month"] >= t_start) & (monthly["month"] <= t_end)]

    # Exclude indices from analysis tabs
    companies = companies[~companies["ticker"].isin(indices)]
    prices    = prices[~prices["ticker"].isin(indices)]
    monthly   = monthly[~monthly["ticker"].isin(indices)]

    # Current Universe for tab selectors
    current_universe = sorted(prices["ticker"].unique().tolist())
    if not current_universe:
        current_universe = sorted([t for t in all_tickers if t not in indices])


st.sidebar.markdown("---")
st.sidebar.markdown("<div class='sb-section-label'>System Controls</div>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Clear Data Cache", use_container_width=True, help="Force reload all data from DuckDB. Use after running the ETL pipeline."):
    st.cache_data.clear()
    st.rerun()

# ── ANALYTICS PRE-COMPUTATION (Scores, Alerts, KPIs) ──────────────────────────
# This section computes all metrics needed for both the Header and the Tabs

# 1. Movers Calculation (Gainers/Losers) - Optimized
latest_date_all = prices_full['date'].max()
prev_date_all = sorted(prices_full['date'].unique())[-2] if len(prices_full['date'].unique()) > 1 else latest_date_all
indices_list = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]

p_latest_movers = prices_full[(prices_full['date'] == latest_date_all) & (~prices_full['ticker'].isin(indices_list))]
p_prev_movers = prices_full[(prices_full['date'] == prev_date_all) & (~prices_full['ticker'].isin(indices_list))]

movers = p_latest_movers.merge(p_prev_movers[['ticker', 'price_close']], on='ticker', suffixes=('', '_prev'))
movers['chg_24h'] = (movers['price_close'] / movers['price_close_prev'] - 1) * 100
gainers = movers.sort_values('chg_24h', ascending=False).head(5)
losers = movers.sort_values('chg_24h', ascending=True).head(5)

# 2. Quant Intelligence Engine (Scores)
# Import the canonical scoring engine from etl.utils (single source of truth)
from etl.utils import compute_score, compute_score_details, get_macro_regime, apply_macro_adjustment, compute_fmi_score, compute_fmi_details, get_fmi_label, compute_fmi_live

latest_prices_reco = prices_full.sort_values('date').groupby('ticker').tail(1).copy()
# Note: fct_daily_returns has no 'rsi' column — only merge columns that exist
_merge_cols = ["ticker", "ma_signal", "price_close", "price_z_score"]
_merge_cols = [c for c in _merge_cols if c in latest_prices_reco.columns]
reco_df = companies_full.merge(latest_prices_reco[_merge_cols], on="ticker", how="left")
reco_df["upside_pct"] = (reco_df["target_mean_price"] / reco_df["price_close"] - 1) * 100
reco_df["upside_pct"] = reco_df["upside_pct"].fillna(0)
# RSI not in warehouse — use neutral default so other pillars score correctly
reco_df["rsi"] = 50.0

reco_df["score"] = reco_df.apply(compute_score, axis=1)



valid_reco = reco_df[~reco_df['ticker'].isin(indices_list)].dropna(subset=['score', 'market_cap'])
if not valid_reco.empty and valid_reco['market_cap'].sum() > 0:
    market_quality_idx = np.average(valid_reco['score'], weights=valid_reco['market_cap'])
else:
    market_quality_idx = reco_df[~reco_df['ticker'].isin(indices_list)]['score'].mean()

# 3. Hot Signal Analytics
@st.cache_data(ttl=600)
def calc_hot_alerts(df_p, df_reco):
    # Latest data point per ticker
    latest_pts = df_p.sort_values('date').groupby('ticker').tail(1).copy()
    high_52w = df_p.groupby('ticker')['price_close'].rolling(window=252, min_periods=1).max().reset_index()
    latest_highs = high_52w.groupby('ticker').tail(1).rename(columns={'price_close': 'high_52w'})
    avg_vol = df_p.groupby('ticker')['volume'].rolling(window=20, min_periods=1).mean().reset_index()
    latest_vols = avg_vol.groupby('ticker').tail(1).rename(columns={'volume': 'avg_vol_20d'})
    # 3. Hot Signal Analytics (Company Names Integrated)
    alert_df = df_reco[['ticker', 'company', 'score', 'ma_signal', 'rsi']].merge(latest_pts[['ticker', 'price_close', 'volume']], on='ticker')
    alert_df = alert_df.merge(latest_highs[['ticker', 'high_52w']], on='ticker')
    alert_df = alert_df.merge(latest_vols[['ticker', 'avg_vol_20d']], on='ticker')
    alert_df = alert_df[~alert_df['ticker'].isin(indices_list)]
    
    # Merge 24h change from movers
    alert_df = alert_df.merge(movers[['ticker', 'chg_24h']], on='ticker', how='left')
    alert_df['chg_24h'] = alert_df['chg_24h'].fillna(0)
    
    found = []
    for _, r in alert_df.iterrows():
        # --- BUY SIGNALS ---
        if r['volume'] > 2 * r['avg_vol_20d'] and r['avg_vol_20d'] > 0 and r['chg_24h'] > 0:
            found.append({'ticker': r['ticker'], 'name': r['company'], 'type': 'BULLISH VOL', 'color': '#3498db', 'icon': '🔊', 'desc': f"Vol Spike (+{((r['volume']/r['avg_vol_20d'])-1)*100:.0f}%) | Price ↗"})
            
        if r['price_close'] >= 0.98 * r['high_52w']:
             found.append({'ticker': r['ticker'], 'name': r['company'], 'type': '52W PEAK', 'color': '#f1c40f', 'icon': '🏔️', 'desc': f"Price: €{r['price_close']:.2f} (Near High)"})
             
        if r['rsi'] < 35 and r['score'] >= 75:
            found.append({'ticker': r['ticker'], 'name': r['company'], 'type': 'GOLDEN BUY', 'color': '#2ecc71', 'icon': '💎', 'desc': f"RSI: {r['rsi']:.1f} | Score: {r['score']}"})
            
        # --- SELL SIGNALS ---
        if r['rsi'] > 75:
            found.append({'ticker': r['ticker'], 'name': r['company'], 'type': 'EXIT / RISK', 'color': '#ff4b4b', 'icon': '', 'desc': f"Extreme Overbought (RSI: {r['rsi']:.1f})"})
            
        if r['score'] < 35 and r['ma_signal'] == 'BEARISH':
            found.append({'ticker': r['ticker'], 'name': r['company'], 'type': 'BEARISH BLOW', 'color': '#ffa500', 'icon': '', 'desc': f"Weak Fundamentals + Bearish Trend"})
            
        if r['volume'] > 2 * r['avg_vol_20d'] and r['chg_24h'] < -3:
            found.append({'ticker': r['ticker'], 'name': r['company'], 'type': 'PANIC DUMP', 'color': '#d32f2f', 'icon': '', 'desc': f"Heavy Selling | Vol Spike & Price ↘"})
            
    return found

hot_alerts = calc_hot_alerts(prices_full, reco_df)
alert_count = len(hot_alerts)

# ── GLOBAL KPI HEADER (Pure HTML Grid — Guaranteed Symmetry) ─────────────────
macro = fetch_macro_data()

# ── [NEW] MASTER TACTICAL REGIME CALCULATION ────────────────────────────────
# We move the high-fidelity logic here so it's consistent across the whole app
from etl.utils import get_macro_regime
_macro_regime = get_macro_regime(macro)

# Get raw macro values for scoring
_vix_val = macro.get("VIX", {}).get("val", 20)
_dxy_pct = macro.get("DXY", {}).get("pct", 0)
_tnx_chg = macro.get("US10Y", {}).get("chg", 0)

# Get SPY Data & Breadth globally
df_spy_global = prices_full[prices_full["ticker"] == "SPY"].sort_values("date")

# Breadth:
_indices_exclude = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC", "^TNX", "^IRX"]
breadth_data_global = prices_full[
    ~prices_full["ticker"].isin(_indices_exclude) &
    prices_full["ma_50"].notna()
]
breadth_ts_global = (
    breadth_data_global[breadth_data_global["price_close"] > breadth_data_global["ma_50"]]
    .groupby("date")["ticker"].count()
    /
    breadth_data_global.groupby("date")["ticker"].count()
    * 100
).fillna(0).reset_index()
breadth_ts_global.columns = ["date", "breadth_pct"]

latest_spy_global = df_spy_global.iloc[-1] if not df_spy_global.empty else None
latest_breadth_global = breadth_ts_global.iloc[-1]["breadth_pct"] if not breadth_ts_global.empty else 0

conf_score_global = 0
conf_reasons = []

if latest_spy_global is not None:
    if latest_spy_global["price_close"] > latest_spy_global["ma_50"]: 
        conf_score_global += 25
    else:
        conf_reasons.append("SPY < MA50")
        
    if latest_spy_global["price_close"] > latest_spy_global["ma_200"]: 
        conf_score_global += 25
    else:
        conf_reasons.append("SPY < MA200")

if latest_breadth_global > 50: 
    conf_score_global += 30
else:
    conf_reasons.append(f"Weak Breadth ({latest_breadth_global:.0f}%)")

# 3. Volatility Awareness - VIX Explicit (10 pts)
if _vix_val < 20:
    conf_score_global += 10
elif _vix_val < 28:
    conf_score_global += 5
    conf_reasons.append("VIX Elevated")
else:
    conf_reasons.append("VIX Panic (>28)")

# 4. Macro Stability - DXY/TNX Explicit (10 pts)
if _dxy_pct < 0.3 and _tnx_chg < 0.05:
    conf_score_global += 10
else:
    conf_reasons.append("Macro Friction (USD/Rates)")

conf_reason_str = "All indicators bullish." if conf_score_global >= 90 else ", ".join(conf_reasons)

# Master Labels
if conf_score_global >= 75: 
    regime, regime_ui_color = "STRONG BULLISH", "#2ecc71"
    advice = "Market internals are robust with strong trend alignment. Ideal for aggressive growth deployment."
elif conf_score_global >= 50: 
    regime, regime_ui_color = "BULLISH", "#27ae60"
    advice = "Constructive environment. Focus on quality growth and leaders breaking out on volume."
elif conf_score_global >= 35: 
    regime, regime_ui_color = "NEUTRAL / SIDEWAYS", "#f39c12"
    advice = "Trend-less environment. Stick to selective bottom-up picking and range-bound strategies."
else: 
    regime, regime_ui_color = "BEARISH / CAUTION", "#e74c3c"
    advice = "Defensive posture required. Breadth is deteriorating or trend has failed. Focus on capital preservation."

vix_val, vix_delta_html = "N/A", ""
spy_val, spy_delta_html = "N/A", ""

if macro:
    vix = macro["VIX"]["val"]
    dxy_chg = macro["DXY"]["pct"]
    tnx_chg = macro["US10Y"]["chg"]

    # ── MACRO-AWARE SCORE ADJUSTMENT ─────────────────────────────────────
    # Now that we have live macro, apply sector-specific penalty/bonus to scores
    _macro_regime = get_macro_regime(macro)
    if _macro_regime != "NEUTRAL":
        reco_df["score"] = reco_df.apply(
            lambda r: apply_macro_adjustment(r["score"], r.get("sector", ""), _macro_regime), axis=1
        )
        # Recalculate market quality index with macro-adjusted scores
        valid_reco_m = reco_df[~reco_df['ticker'].isin(indices_list)].dropna(subset=['score', 'market_cap'])
        if not valid_reco_m.empty and valid_reco_m['market_cap'].sum() > 0:
            market_quality_idx = np.average(valid_reco_m['score'], weights=valid_reco_m['market_cap'])
        
    # 2. VIX card
    vix_chg = macro["VIX"]["pct"]
    vix_sign = "+" if vix_chg >= 0 else ""
    vix_hud_color = "#e74c3c" if vix_chg >= 0 else "#2ecc71" # VIX up = bad
    vix_delta_html = f'<div class="kpi-delta" style="color:{vix_hud_color}">{vix_sign}{vix_chg:.2f}%</div>'
    vix_val = f"{vix:.2f}"
    
    # 3. SPY card
    spy = macro["SPY"]["val"]
    spy_chg = macro["SPY"]["pct"]
    spy_sign = "+" if spy_chg >= 0 else ""
    spy_hud_color = "#2ecc71" if spy_chg >= 0 else "#e74c3c"
    spy_delta_html = f'<div class="kpi-delta" style="color:{spy_hud_color}">{spy_sign}{spy_chg:.2f}%</div>'
    # ── RENDER SIDEBAR MACRO PULSE (via placeholder created earlier) ──────────
    if macro:
        _spy_v   = macro["SPY"]["val"];  _spy_p   = macro["SPY"]["pct"]
        _vix_v   = macro["VIX"]["val"];  _vix_p   = macro["VIX"]["pct"]
        _tnx_v   = macro["US10Y"]["val"];_tnx_p   = macro["US10Y"]["pct"]
        _dxy_v   = macro["DXY"]["val"];  _dxy_p   = macro["DXY"]["pct"]

        def _sb_delta(pct, invert=False):
            good = "#2ecc71"; bad = "#e74c3c"
            color = (bad if pct >= 0 else good) if invert else (good if pct >= 0 else bad)
            sign  = "+" if pct >= 0 else ""
            return f"<span class='sb-macro-delta' style='color:{color}'>{sign}{pct:.2f}%</span>"

        spy_sema = "🔥 Strong Rally" if _spy_p >= 1 else ("🟢 Advancing" if _spy_p > 0 else ("🔴 Sharp Sell-off" if _spy_p <= -1 else "🟡 Pullback"))
        vix_sema = "🚨 High Panic" if _vix_v >= 25 else ("⚠️ Volatility Elevated" if _vix_v >= 18 else ("🔵 Normal Volatility" if _vix_v > 13 else "😴 Complacent"))
        tnx_sema = "📈 Yields Spiking" if _tnx_p >= 2 else ("↗️ Yields Rising" if _tnx_p > 0 else ("📉 Yields Dropping" if _tnx_p <= -2 else "↘️ Yields Falling"))
        dxy_sema = "🦅 Strong Dollar" if _dxy_p >= 0.5 else ("↗️ Dollar Strengthening" if _dxy_p > 0 else ("🕊️ Weak Dollar" if _dxy_p <= -0.5 else "↘️ Dollar Weakening"))

        _regime_colors = {"🚨 DEFENSIVE (High Risk)": "#e74c3c", "🔥 INFLATIONARY": "#e67e22", "🚀 GROWTH MODE": "#2ecc71", "⚖️ BALANCED": "#f39c12"}
        _rc = _regime_colors.get(regime, "#f39c12")

        _macro_sidebar_placeholder.markdown(f"""
        <div class='sb-section-label'>Live Macro Pulse</div>
        <div class='sb-macro-row' style='display:block;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span class='sb-macro-label'>SPY</span>
                <span class='sb-macro-val'>€{_spy_v:.2f}</span>
                {_sb_delta(_spy_p)}
            </div>
            <div style='font-size:0.55rem; color:#8899aa; text-align:right; margin-top:2px; text-transform:uppercase;'>{spy_sema}</div>
        </div>
        <div class='sb-macro-row' style='display:block;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span class='sb-macro-label'>VIX</span>
                <span class='sb-macro-val'>{_vix_v:.2f}</span>
                {_sb_delta(_vix_p, invert=True)}
            </div>
            <div style='font-size:0.55rem; color:#8899aa; text-align:right; margin-top:2px; text-transform:uppercase;'>{vix_sema}</div>
        </div>
        <div class='sb-macro-row' style='display:block;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span class='sb-macro-label'>US10Y</span>
                <span class='sb-macro-val'>{_tnx_v:.2f}%</span>
                {_sb_delta(_tnx_p, invert=True)}
            </div>
            <div style='font-size:0.55rem; color:#8899aa; text-align:right; margin-top:2px; text-transform:uppercase;'>{tnx_sema}</div>
        </div>
        <div class='sb-macro-row' style='display:block;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span class='sb-macro-label'>DXY</span>
                <span class='sb-macro-val'>{_dxy_v:.2f}</span>
                {_sb_delta(_dxy_p)}
            </div>
            <div style='font-size:0.55rem; color:#8899aa; text-align:right; margin-top:2px; text-transform:uppercase;'>{dxy_sema}</div>
        </div>
        """, unsafe_allow_html=True)

mqi_val = f"{market_quality_idx:.1f}"
mqi_color = "#2ecc71" if market_quality_idx >= 65 else ("#f1c40f" if market_quality_idx >= 45 else "#e74c3c")

# ── MAIN HEADER (Compact — Macro moved to Sidebar) ─────────────────────────
# Split into 2 columns: Title/Stats (L) and Intel Hub (R)
head_l, head_r = st.columns([5, 1])

with head_l:
    st.markdown(f"""
    <div style='display:flex; align-items:center; justify-content:space-between;
                padding:10px 16px; background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.06); border-radius:8px; margin-bottom:0px;'>
        <div>
            <span style='font-size:1.3rem; font-weight:900; color:#e8eaf6; font-family: "Courier New", monospace;'>
                LuongDo | Quant Analytics Workspace
            </span>
            <span style='font-size:0.72rem; color:#556677; margin-left:12px;'>
                {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC &nbsp;|&nbsp; {stock_count} Tickers
            </span>
        </div>
        <div style='display:flex; gap:12px; align-items:center;'>
            <div style='text-align:center;'>
                <div style='font-size:0.6rem; color:#445566; font-family:monospace; text-transform:uppercase; letter-spacing:0.1em;'>Quality Index</div>
                <div style='font-size:1.1rem; font-weight:900; color:{mqi_color}; font-family:"Courier New",monospace;'>{mqi_val}<span style='font-size:0.75rem; color:#667788;'>/100</span></div>
            </div>
            <div style='text-align:center; padding-left:12px; border-left:1px solid #1a2233;'>
                <div style='font-size:0.6rem; color:#445566; font-family:monospace; text-transform:uppercase; letter-spacing:0.1em;'>Market Context</div>
                <div style='font-size:0.8rem; font-weight:700; color:{regime_ui_color}; font-family:"Courier New",monospace;'>{regime}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with head_r:
    # ── INTELLIGENCE HUB (Moved to Header) ──────────────────────────
    total_alerts = alert_count
    today = pd.Timestamp.now().normalize()
    next_week = today + pd.Timedelta(days=7)
    upcoming_count = 0
    if not earnings_cal.empty:
        upcoming_count = len(earnings_cal[
            (earnings_cal["earnings_date"].dt.date >= today.date()) & 
            (earnings_cal["earnings_date"].dt.date <= next_week.date())
        ])
    
    hub_label = f"SIGNAL ({total_alerts + upcoming_count})" if (total_alerts + upcoming_count) > 0 else "SIGNAL"
    
    with st.popover(hub_label, use_container_width=True):
        tab_sig, tab_mov, tab_ern = st.tabs(["SIGNALS", "MOVERS", "EARNINGS"])
        # ... rest of logic remains inside ...
        
        with tab_sig:
            if alert_count > 0 or macro:
                if macro: st.markdown(f"**Macro Advice:** {advice}")
                for a in hot_alerts[:20]:
                    st.markdown(f"**{a['ticker']}** | <span style='color:{a['color']};font-weight:bold;'>[{a['type']}]</span> — `{a['desc']}`", unsafe_allow_html=True)
            else: st.write("No active signals.")

        with tab_mov:
            m_c1, m_c2 = st.columns(2)
            with m_c1:
                st.markdown("##### Gainers")
                for _, r in gainers.iterrows():
                    st.markdown(f"<div style='display:flex; justify-content:space-between; padding:5px; background:rgba(46, 204, 113, 0.1); border-radius:5px; margin-bottom:5px; border-left:4px solid #2ecc71;'><b>{r['ticker']}</b> <span style='color:#2ecc71;'>+{r['chg_24h']:.2f}%</span></div>", unsafe_allow_html=True)
            with m_c2:
                st.markdown("##### Losers")
                for _, r in losers.iterrows():
                    st.markdown(f"<div style='display:flex; justify-content:space-between; padding:5px; background:rgba(231, 76, 60, 0.1); border-radius:5px; margin-bottom:5px; border-left:4px solid #e74c3c;'><b>{r['ticker']}</b> <span style='color:#e74c3c;'>{r['chg_24h']:.2f}%</span></div>", unsafe_allow_html=True)

        with tab_ern:
            if not earnings_cal.empty:
                next_m = today + pd.Timedelta(days=30)
                up_m = earnings_cal[(earnings_cal["earnings_date"].dt.date >= today.date()) & (earnings_cal["earnings_date"].dt.date <= next_m.date())].sort_values("earnings_date")
                if not up_m.empty:
                    # Merge with companies to get full company name
                    up_m = up_m.merge(companies_full[["ticker", "company"]], on="ticker", how="left")
                    
                    for _, r in up_m.iterrows():
                        display_name = r["company"] if pd.notnull(r["company"]) else r["ticker"]
                        e_date = r["earnings_date"].strftime("%b %d")
                        eps_est = f"€{r['eps_avg']:.2f}" if pd.notnull(r['eps_avg']) else "N/A"
                        rev_est = f"€{r['rev_avg']/1e9:.1f}B" if pd.notnull(r['rev_avg']) else "N/A"
                        
                        st.markdown(f"""
                        <div class="earning-card">
                            <div class="earning-header">
                                <span class="earning-ticker" style="font-size:0.9rem; max-width:180px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{display_name}</span>
                                <span class="earning-date">{e_date}</span>
                            </div>
                            <div class="earnings-metrics">
                                <div>
                                    <div class="earning-m-label">EPS Estimate</div>
                                    <div class="earning-m-val">{eps_est}</div>
                                </div>
                                <div style="text-align:right;">
                                    <div class="earning-m-label">Revenue Est</div>
                                    <div class="earning-m-val">{rev_est}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else: st.write("No reports (30d).")
            else: st.write("No data.")

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

st.markdown("---")

# Sync action + reco label from m_df (the Single Source of Truth)
# m_df is keyed by 'Ticker' (display), reco_df by 'ticker' (lowercase)
_action_map = m_df.set_index("Ticker")["Action"].to_dict() if "Ticker" in m_df.columns else {}
reco_df["action"] = reco_df["ticker"].map(_action_map).fillna("HOLD / NEUTRAL")
reco_df = reco_df.sort_values("score", ascending=False)
reco_df["upside_str"] = reco_df["upside_pct"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")

# risk_return is needed by the Overview tab
risk_return = monthly.groupby("ticker").agg(
    avg_return=("monthly_return", "mean"),
    volatility=("volatility", "mean"),
).reset_index().merge(companies[["ticker", "company", "sector"]], on="ticker")

# ── LAYER 6: MAIN TAB EXECUTION ──────────────────────────────────────────────
# define tab labels in Decision Stage workflow order
tab_labels = [
    "1. Market Regime",        # Strategic Overview
    "2. Opportunity Radar",    # Market Scanner
    "3. Qualitative Audit (AI)", # Single Stock Deep Dive
    "4. Quantitative Forecast (ML)", # Predictive Suite
    "5. Backtest Lab",         # Strategy Backtest
    "6. Watchlist",            # Watchlist / Kanban
    "7. Portfolio Builder",    # Portfolio Management
    "8. System Methodology"    # Methodology Docs
]

# To REALLY fix the jumping issue while keeping the modern 'Pills' UI, 
# we use Streamlit's native st.pills with session_state binding.
if 'active_tab' not in st.session_state or st.session_state['active_tab'] not in tab_labels:
    st.session_state['active_tab'] = tab_labels[0]

st.markdown("<p style='color:#8899aa; font-size:0.85rem; font-weight:600; margin-bottom:-10px; margin-top:10px;'>🧭 NAVIGATION CHANNELS — SELECT A MODULE BELOW TO VIEW:</p>", unsafe_allow_html=True)

active_tab = st.pills(
    "Navigation",
    options=tab_labels,
    key="active_tab",
    label_visibility="collapsed"
)

# st.pills allows deselection (returning None), so we default back to the first tab if deselected
if not active_tab:
    active_tab = tab_labels[0]

if active_tab == "1. Market Regime":
    # ── [STEP 2] 6-BLOCK GRID LAYOUT ─────────────────────────────────────────
    # Note: Logic (Step 1) has been moved to global dashboard level for consistency
    
    # ROW 1: HEADERS
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03); padding:20px; border-radius:10px; border-left:5px solid {regime_ui_color};'>
            <span style='color:#8899aa; font-size:0.8rem; font-weight:700; text-transform:uppercase;'>Current Market Regime</span>
            <div style='color:{regime_ui_color}; font-size:2.2rem; font-weight:900;'>{regime}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03); padding:20px; border-radius:10px; border-right:5px solid #3498db;'>
            <span style='color:#8899aa; font-size:0.8rem; font-weight:700; text-transform:uppercase;'>Trend Confidence Score</span>
            <div style='display:flex; align-items:baseline; gap:15px; flex-wrap:wrap;'>
                <div style='color:#fff; font-size:2.2rem; font-weight:900;'>{conf_score_global}%</div>
                <div style='color:#e74c3c; font-size:0.75rem; font-style:italic;'>{ '🚨 ' + conf_reason_str if conf_score_global < 50 else '⚠️ ' + conf_reason_str if conf_score_global < 100 else '✅ ' + conf_reason_str }</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 2: PRIMARY CHARTS
    # Filter SPY and breadth to the selected time horizon
    df_spy_filtered = df_spy_global[(df_spy_global["date"] >= t_start) & (df_spy_global["date"] <= t_end)]
    breadth_filtered = breadth_ts_global[(breadth_ts_global["date"] >= t_start) & (breadth_ts_global["date"] <= t_end)]

    c1, c2 = st.columns(2)
    with c1:
        render_header("activity", "Index Trend (SPY + MA 50/200)")
        if not df_spy_filtered.empty:
            fig_spy = go.Figure()
            fig_spy.add_trace(go.Scatter(
                x=df_spy_filtered["date"], y=df_spy_filtered["price_close"],
                name="SPY Price", mode='lines',
                line=dict(color="#00d4ff", width=2.5)
            ))
            fig_spy.add_trace(go.Scatter(
                x=df_spy_filtered["date"], y=df_spy_filtered["ma_50"],
                name="MA 50", mode='lines',
                line=dict(color="#3498db", width=1.5, dash='dot')
            ))
            fig_spy.add_trace(go.Scatter(
                x=df_spy_filtered["date"], y=df_spy_filtered["ma_200"],
                name="MA 200", mode='lines',
                line=dict(color="#e67e22", width=1.5)
            ))
            fig_spy.update_layout(
                template="plotly_dark", height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Price (USD)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_spy, use_container_width=True)
        else:
            st.info("SPY data not available for the selected period.")

    with c2:
        render_header("dna", "Market Breadth (% Stocks > MA 50)")
        if not breadth_filtered.empty:
            fig_br = go.Figure()
            fig_br.add_trace(go.Scatter(
                x=breadth_filtered["date"], y=breadth_filtered["breadth_pct"],
                name="Breadth %", mode='lines',
                line=dict(color="#2ecc71", width=2),
                fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.08)'
            ))
            # Reference levels
            fig_br.add_hline(y=70, line_dash="dot", line_color="rgba(46, 204, 113, 0.5)",
                             annotation_text="70% Bullish Zone", annotation_position="top left",
                             annotation_font=dict(size=10, color="#2ecc71"))
            fig_br.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                             annotation_text="50% Neutral", annotation_position="top left",
                             annotation_font=dict(size=10, color="#aaa"))
            fig_br.add_hline(y=30, line_dash="dot", line_color="rgba(231, 76, 60, 0.5)",
                             annotation_text="30% Oversold", annotation_position="top left",
                             annotation_font=dict(size=10, color="#e74c3c"))
            fig_br.update_layout(
                template="plotly_dark", height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(title="% Above MA50", range=[0, 100], ticksuffix="%"),
                xaxis_title="", showlegend=False
            )
            st.plotly_chart(fig_br, use_container_width=True)
        else:
            st.info("Calculating breadth history... run pipeline if empty.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 3: HEATMAP & STANCE
    b1, b2 = st.columns(2)
    with b1:
        render_header("globe", "Sector Intelligence")
        
        # ── SECTOR HEATMAP: period_return tied to selected time horizon ──────
        # Use prices (time-filtered) for p_perf
        p_perf = prices.sort_values('date').groupby('ticker')['price_close'].agg(['first', 'last']).reset_index()
        p_perf['first'] = pd.to_numeric(p_perf['first'], errors='coerce').replace(0, 0.001).fillna(0.001)
        p_perf['last']  = pd.to_numeric(p_perf['last'],  errors='coerce').fillna(0.001)
        p_perf['period_return'] = (p_perf['last'] / p_perf['first'] - 1) * 100

        # INNER JOIN: only tickers with data in the selected period.
        # Use companies_full (not reco_df) to avoid stale period_return columns.
        tree_df = companies_full[['ticker', 'company', 'sector', 'region', 'market_cap']].merge(
            p_perf[['ticker', 'period_return']], on='ticker', how='inner'
        )
        # Exclude benchmark indices from heatmap
        tree_df = tree_df[~tree_df['ticker'].isin(["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"])]
        tree_df['period_return'] = pd.to_numeric(tree_df['period_return'], errors='coerce')
        tree_df['period_return'] = tree_df['period_return'].replace([float('inf'), float('-inf')], 0).fillna(0)
        tree_df = tree_df.dropna(subset=['sector', 'ticker'])
        tree_df['cap_bn'] = pd.to_numeric(tree_df['market_cap'], errors='coerce') / 1e9
        tree_df['cap_bn'] = tree_df['cap_bn'].fillna(0.001).replace(0, 0.001)
        
        p_max = max(abs(tree_df['period_return'].min()), abs(tree_df['period_return'].max()), 5)

        # Mini-summary
        if not tree_df.empty and 'sector' in tree_df.columns:
            sector_agg = tree_df.groupby('sector')['period_return'].mean().sort_values(ascending=False)
            if len(sector_agg) >= 2:
                leaders = ", ".join(sector_agg.head(2).index.tolist())
                laggards = ", ".join(sector_agg.tail(2).index.tolist())
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03); padding:10px 15px; border-radius:8px; border-left:3px solid #f1c40f; margin-bottom:12px; font-size:0.85rem;'>
                    <b>Leaders:</b> <span style='color:#2ecc71;'>{leaders}</span> | <b>Laggards:</b> <span style='color:#e74c3c;'>{laggards}</span>
                </div>
                """, unsafe_allow_html=True)

        sec_tabs = st.tabs(["Heatmap", "Top Sectors", "Top Movers", "Health Matrix"])
        
        with sec_tabs[0]:
            # ... (Heatmap code)
            tree_df['return_str'] = tree_df['period_return'].apply(
                lambda x: f"{x:+.2f}%" if pd.notnull(x) else ""
            )
            tree_df['Region'] = tree_df['region'].fillna('Unknown').str.upper()
            
            fig_tree = px.treemap(
                tree_df, path=[px.Constant("Global"), 'Region', 'sector', 'ticker'], values='cap_bn',
                color='period_return', color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                range_color=[-p_max, p_max], template="plotly_dark", height=600,
                custom_data=['return_str', 'Region']
            )
            fig_tree.update_traces(
                texttemplate="%{label}<br>%{customdata[0]}",
                textfont=dict(size=12),
                hovertemplate="<b>%{label}</b><br>Region: %{customdata[1]}<br>Mkt Cap: €%{value:.1f}B<br>Return: %{customdata[0]}<extra></extra>"
            )
            fig_tree.update_layout(margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_tree, use_container_width=True)
            
        with sec_tabs[1]:
            if not tree_df.empty and 'sector' in tree_df.columns:
                fig_sec = px.bar(sector_agg.reset_index(), x='period_return', y='sector', orientation='h', 
                                 color='period_return', color_continuous_scale='RdYlGn', template="plotly_dark", height=600)
                fig_sec.update_traces(texttemplate='%{x:.2f}%', textposition='outside')
                fig_sec.update_layout(margin=dict(r=20, b=0), yaxis={'categoryorder':'total ascending', 'title': None, 'tickmode': 'linear'})
                st.plotly_chart(fig_sec, use_container_width=True)
            
        with sec_tabs[2]:
            if not tree_df.empty:
                top_stocks = tree_df.nlargest(10, 'period_return')
                bot_stocks = tree_df.nsmallest(10, 'period_return')
                movers = pd.concat([top_stocks, bot_stocks]).sort_values('period_return', ascending=True)
                fig_movers = px.bar(movers, x='period_return', y='ticker', orientation='h', 
                                    color='period_return', color_continuous_scale='RdYlGn', template="plotly_dark", height=600)
                fig_movers.update_traces(texttemplate='%{x:.2f}%', textposition='outside')
                fig_movers.update_layout(margin=dict(r=40, b=0), yaxis={'categoryorder':'total ascending', 'title': None, 'tickmode': 'linear', 'dtick': 1})
                st.plotly_chart(fig_movers, use_container_width=True)

        with sec_tabs[3]:
            # Call the new Sector Health Matrix
            render_sector_health_matrix(m_df)

    with b2:
        render_header("package", "Portfolio Stance & Tactical Guide")
        
        # Stance card content
        if conf_score_global >= 70:
            stance, size, bias = "AGGRESSIVE", "80-100%", "Momentum & Growth"
        elif conf_score_global >= 50:
            stance, size, bias = "MODERATE", "50-80%", "Quality Growth"
        elif conf_score_global >= 30:
            stance, size, bias = "DEFENSIVE", "20-50%", "Value & Low Vol"
        else:
            stance, size, bias = "PROTECTIVE", "0-20%", "Cash & Hedging"

        # 🚀 DYNAMIC OVERRIDE: If breadth is weak and macro is Risk-Off, force Defensive stance
        if latest_breadth_global < 50 and _macro_regime == "RISK_OFF" and conf_score_global >= 50:
            stance, size, bias = "DEFENSIVE (Overridden)", "30-50%", "Defensive Value"
            
        st.markdown(f"""
        <div style='background:rgba(20,30,45,0.7); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:25px; height:400px;'>
            <div style='margin-bottom:15px;'>
                <span style='color:#8899aa; font-size:0.75rem; font-weight:700;'>STRATEGIC BIAS</span>
                <div style='color:#fff; font-size:1.4rem; font-weight:800;'>{stance}</div>
            </div>
            <div style='display:flex; gap:20px; margin-bottom:20px;'>
                <div style='flex:1;'>
                    <span style='color:#8899aa; font-size:0.7rem;'>Exposure Size</span>
                    <div style='color:#3498db; font-size:1.1rem; font-weight:700;'>{size}</div>
                </div>
                <div style='flex:1;'>
                    <span style='color:#8899aa; font-size:0.7rem;'>Preferred Factor</span>
                    <div style='color:#2ecc71; font-size:1.1rem; font-weight:700;'>{bias}</div>
                </div>
            </div>
            <div style='border-top:1px solid rgba(255,255,255,0.1); padding-top:15px;'>
                <span style='color:#e74c3c; font-size:0.75rem; font-weight:700;'>⚠️ RISK ALERT</span>
                <p style='color:#cfd8dc; font-size:0.85rem; margin-top:5px;'>
                    {'Watch for failed breakouts in laggard sectors.' if conf_score_global > 50 else 'Focus on capital preservation as breadth deteriorates.'}
                </p>
                <div style='margin-top:15px; background:rgba(52,152,219,0.1); padding:10px; border-radius:6px;'>
                    <span style='color:#3498db; font-size:0.7rem; font-weight:700;'>NEXT ACTION</span>
                    <p style='color:#fff; font-size:0.85rem; margin:0;'>Use <b>Opportunity Radar</b> to scan for high-quality names with positive trend alignment.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")



# ── TAB: SINGLE STOCK ANALYSIS ───────────────────────────────────────────────
if active_tab == "3. Qualitative Audit (AI)":
    render_header("search", "Single Stock Deep Dive")
    if current_universe:
        # Persist selection across reruns via session_state
        # Pre-fill with active_ticker if deep_ticker_selector not yet set
        if "deep_ticker_selector" not in st.session_state:
            _default_deep = st.session_state.get("active_ticker", None)
            if _default_deep and _default_deep not in current_universe:
                _default_deep = None
            st.session_state["deep_ticker_selector"] = _default_deep

        deep_ticker = st.selectbox(
            "Select Asset to Analyze:",
            current_universe,
            placeholder="Search and Select an Asset...",
            format_func=format_ticker,
            key="deep_ticker_selector"
            # Loại bỏ tham số index vì Streamlit tự động dùng Session State cho widget có key
        )
        # Sync back so active_ticker stays aligned
        if deep_ticker:
            st.session_state.active_ticker = deep_ticker
            
        if deep_ticker:
            _meta_df = companies_full[companies_full["ticker"] == deep_ticker]
            if _meta_df.empty:
                st.warning(f"⚠️ No fundamental data found for **{deep_ticker}** in the warehouse. Please run the pipeline to fetch data.", icon="⚠️")
                st.stop()
            meta = _meta_df.iloc[0]
            df_deep = prices[prices["ticker"] == deep_ticker].sort_values("date")
            if df_deep.empty:
                st.warning(f"⚠️ No price history found for **{deep_ticker}**. Please run the pipeline first.", icon="⚠️")
                st.stop()
            df_fin = annual_fin[annual_fin["ticker"] == deep_ticker].sort_values("year", ascending=False)
            
            target_p = meta.get('target_mean_price', 0)
            cur_p = df_deep['price_close'].iloc[-1]
            upside = ((target_p / cur_p) - 1) * 100 if target_p > 0 else 0
            
            # --- SUMMARY STRIP (High Density) ---
            company_name = meta.get('company', deep_ticker)
            if pd.isna(company_name): company_name = deep_ticker
            st.markdown(f"#### {company_name} ({deep_ticker}) — {meta['sector']} - €{cur_p:.2f}")
            
            # --- Pre-compute values used in the grid ---
            z_score = df_deep['price_z_score'].iloc[-1] if 'price_z_score' in df_deep.columns else 0
            if pd.isna(z_score): z_score = 0
            if z_score > 2:    z_status = "🚨 EXTREME OVERBOUGHT"
            elif z_score > 1:  z_status = "⚠️ OVEREXTENDED"
            elif z_score < -2: z_status = "💎 DEEP VALUE"
            elif z_score < -1: z_status = "🟢 UNDERVALUED"
            else:              z_status = "🔵 MEAN REVERTING"

            # --- Enrich meta with latest technicals for the scoring engine ---
            latest_tech = df_deep.iloc[-1]
            meta_enriched = meta.to_dict()
            
            # Ensure numeric safety for core fields
            for col in ['pe_ratio', 'peg_ratio', 'price_to_book', 'roe', 'fcf_margin', 'dividend_yield_pct']:
                val = meta_enriched.get(col)
                try:
                    meta_enriched[col] = float(val) if pd.notnull(val) else None
                except:
                    meta_enriched[col] = None

            meta_enriched['rsi'] = float(latest_tech.get('rsi', 50))
            meta_enriched['ma_signal'] = str(latest_tech.get('ma_signal', 'NEUTRAL'))
            meta_enriched['price_z_score'] = float(z_score)
            meta_enriched['upside_pct'] = float(upside)
            
            # ── AI SCORING ────────────────────────────────────────────────────────
            ai_score = compute_score(meta_enriched)
            ai_action = _action_map.get(deep_ticker, "HOLD / NEUTRAL")
            if ai_score >= 70:    ai_color, ai_icon = "#00ffcc", "🚀"
            elif ai_score >= 55:  ai_color, ai_icon = "#2ecc71", "✅"
            elif ai_score >= 35:  ai_color, ai_icon = "#f1c40f", "🟡"
            else:                 ai_color, ai_icon = "#e74c3c", "🔴"

            st.markdown("---")




            st.markdown("<div style='margin-top:10px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #3498db; color:#3498db; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 1: STRUCTURAL CONTEXT</div>", unsafe_allow_html=True)
            # ── TRADING CONTEXT (TOP of page) — 52-Week Range & Strategic Plan ─
            # All tactical values computed by the shared helper (identical formula to Screener)
            _tm        = get_tactical_metrics(df_deep, cur_p)
            _s1        = _tm["s1"]
            _r1        = _tm["r1"]
            _s2        = float(df_deep["price_low"].tail(50).min())
            _r2        = float(df_deep["price_high"].tail(50).max())
            _rsi_val   = _tm["rsi"]
            _ma_sig    = str(latest_tech.get("ma_signal", meta.get("ma_signal", "NEUTRAL")))
            _w52_pos   = _tm["w52_pos"]
            _w52_hi    = _tm["w52_hi"]
            _w52_lo    = _tm["w52_lo"]
            _w52_zone  = "Near Low" if _w52_pos < 20 else ("Near High" if _w52_pos > 80 else "Mid-Range")
            _stop_loss = _tm["stop_loss"]

            # TP1: honour AI Ensemble target if already computed, otherwise use standard formula
            _global_ai_target = st.session_state.get(f"ai_target_for_de_{deep_ticker}")
            _tp1 = float(_global_ai_target) if _global_ai_target is not None else _tm["tp1"]
            _tp2 = max(target_p, _tm["tp2"]) if target_p > 0 else _tm["tp2"]

            # 52-Week Position Meter
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1);
                        border-radius:10px; padding:14px 20px; margin-bottom:10px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span style='color:#999; font-size:0.75rem; font-weight:600; text-transform:uppercase;'>52-Week Range</span>
                    <span style='color:#fff; font-size:0.85rem; font-weight:700;'>{_w52_zone} &nbsp;|&nbsp; Position: {_w52_pos:.0f}%</span>
                </div>
                <div style='display:flex; align-items:center; gap:10px;'>
                    <span style='color:#e74c3c; font-size:0.85rem; white-space:nowrap;'>Low: €{_w52_lo:.2f}</span>
                    <div style='flex:1; background:rgba(255,255,255,0.1); border-radius:4px; height:10px; position:relative;'>
                        <div style='width:{_w52_pos:.1f}%; height:100%; background:linear-gradient(90deg,#e74c3c,#f1c40f,#2ecc71); border-radius:4px;'></div>
                        <div style='position:absolute; top:-3px; left:{_w52_pos:.1f}%; transform:translateX(-50%);
                                    width:14px; height:14px; background:#fff; border-radius:50%; border:2px solid #3498db;'></div>
                    </div>
                    <span style='color:#2ecc71; font-size:0.85rem; white-space:nowrap;'>High: €{_w52_hi:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


            st.markdown("<div style='margin-top:35px; margin-bottom:-10px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #e67e22; color:#e67e22; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 2: TACTICAL EXECUTION MATRIX</div>", unsafe_allow_html=True)
            # ── UNIFIED DECISION SUPPORT MATRIX (ACTION LAYER) ────────────────
            render_header("activity", "360° Decision & Action Matrix")
            
            # PILLAR 1: TECHNICAL TREND
            if _ma_sig == "BULLISH" and _rsi_val < 65:
                p_trend, p_trend_c = "BULLISH", "#2ecc71"
            elif _ma_sig == "BULLISH" and _rsi_val >= 65:
                p_trend, p_trend_c = "EXTENDED", "#f1c40f"
            elif _ma_sig == "BEARISH" and _rsi_val <= 35:
                p_trend, p_trend_c = "OVERSOLD", "#f1c40f"
            else:
                p_trend, p_trend_c = "BEARISH", "#e74c3c"
                
            # PILLAR 2: QUALITY
            if ai_score >= 70: p_qual, p_qual_c = "ELITE", "#00ffcc"
            elif ai_score >= 55: p_qual, p_qual_c = "SOLID", "#2ecc71"
            elif ai_score >= 40: p_qual, p_qual_c = "FAIR", "#f1c40f"
            else: p_qual, p_qual_c = "POOR", "#e74c3c"
            
            # PILLAR 3: VALUATION — Sector-Aware Multi-factor (PEG + P/E + upside)
            _peg_v = float(meta_enriched.get("peg_ratio") or 0)
            _pe_v  = float(meta_enriched.get("pe_ratio")  or 0)
            
            # 🏆 EXPERT: Sector-Specific Dynamic Thresholds
            _sector_str = str(meta.get("sector", "")).lower()
            _is_growth  = any(s in _sector_str for s in ["tech", "semi", "software", "cloud", "ai", "comm"])
            
            # Dynamic cutoff levels (Growth stocks carry premium multiples)
            _pe_cheap_limit = 28.0 if _is_growth else 18.0
            _pe_expensive_limit = 65.0 if _is_growth else 42.0
            _peg_expensive_limit = 3.5 if _is_growth else 2.5
            _peg_cheap_limit = 1.2 if _is_growth else 0.8

            _val_expensive  = (_pe_v > _pe_expensive_limit and _pe_v > 0) or (_peg_v > _peg_expensive_limit and _peg_v > 0)
            _val_cheap      = (upside > 15) and (_peg_v < _peg_cheap_limit or _pe_v < _pe_cheap_limit) and _pe_v > 0
            _val_premium_ok = (upside > 8) and (ai_score >= 65) and (_peg_v < 2.8 or _pe_v < (55 if _is_growth else 35))
            _val_compounder = (upside > 5) and (ai_score >= 55) and (not _val_expensive)
            _val_fair       = (upside > 0) and (not _val_expensive)

            if _val_cheap:
                p_val, p_val_c = "UNDERVALUED", "#2ecc71"
            elif _val_premium_ok:
                p_val, p_val_c = "PREMIUM / JUSTIFIED", "#3498db"
            elif _val_compounder:
                p_val, p_val_c = "FAIR FOR QUALITY", "#3498db"
            elif _val_fair:
                p_val, p_val_c = "FAIR VS SECTOR", "#f1c40f"
            elif _val_expensive:
                p_val, p_val_c = "EXPENSIVE / PREMIUM", "#e67e22"
            elif _pe_v < 0:
                p_val, p_val_c = "SPECULATIVE / RISK", "#e74c3c"
            else:
                p_val, p_val_c = "AVERAGE", "#95a5a6"
            
            # PILLAR 4: RISK
            if _w52_pos > 80: p_risk, p_risk_c = "ELEVATED", "#e74c3c"
            elif _w52_pos < 20: p_risk, p_risk_c = "LOW RISK", "#2ecc71"
            else: p_risk, p_risk_c = "MODERATE", "#f1c40f"
            
            # PILLAR 5: CONVICTION (rr_score = raw r1 target, same as Screener)
            _rr = _tm["rr_score"]
            if _rr > 2.5: p_conv, p_conv_c = "HIGH", "#00ffcc"
            elif _rr > 1.2: p_conv, p_conv_c = "MEDIUM", "#2ecc71"
            else: p_conv, p_conv_c = "LOW", "#e74c3c"
            
            # ── MASTER POSITIONING LOGIC ──────────────────────────────────────
            # We still call compute_institutional_rating to derive pillar colours
            # (p_trend_c, p_val_c, etc.) for the UI matrix.
            # BUT the final Action label is ALWAYS read from _action_map (m_df),
            # which is the Single Source of Truth — identical to the Screener tab.
            _rating = compute_institutional_rating(
                ai_score   = ai_score,
                ma_sig     = _ma_sig,
                latest_rsi = _rsi_val,
                upside     = float(upside),
                pe_v       = float(meta_enriched.get("pe_ratio")  or 0),
                peg_v      = float(meta_enriched.get("peg_ratio") or 0),
                sector     = str(meta.get("sector", "")),
                w52_pos    = _w52_pos,
                rr         = _tm["rr_score"],   # scoring uses raw r1 target
            )
            # Action label: canonical value from Screener engine (m_df)
            act_str = _action_map.get(deep_ticker, _rating["action_label"])
            # Colour is derived from the canonical label — NOT from the local engine score
            _colour_map = {
                "STRONG BUY":          "#00ffcc",
                "BUY / ACCUMULATE":    "#2ecc71",
                "HOLD / NEUTRAL":      "#3498db",
                "REDUCE / UNDERPERFORM": "#e67e22",
                "SELL / AVOID":        "#e74c3c",
            }
            act_color = _colour_map.get(act_str, _rating["action_color"])
            # Override p_trend_c / p_val_c with engine values so colours are consistent
            p_trend_c = _rating["p_trend_c"]
            p_val_c   = _rating["p_val_c"]

            # ── Action description text (context-aware) ──────────────────────
            if act_str == "STRONG BUY":
                act_desc = f"Optimal alignment of quantitative pillars. High structural conviction. Ideal entry zone between €{_s1:.2f} and €{cur_p:.2f}."
            elif act_str == "BUY / ACCUMULATE":
                act_desc = f"Institutional-grade asset consolidating. Momentum is neutralizing. Support holds near €{_s1:.2f}."
            elif act_str == "SELL / AVOID" and p_trend_c == "#e74c3c" and _rating["p_val_c"] == "#e74c3c":
                act_desc = "Negative trend synergy with poor valuation metrics. Risk/Reward is heavily skewed to the downside."
            elif act_str == "SELL / AVOID":
                act_desc = "Significant fundamental and technical breakdown detected. Focus on capital preservation."
            elif act_str == "HOLD / NEUTRAL" and _rating["p_qual_c"] in ["#2ecc71", "#00ffcc"]:
                act_desc = "Elite asset currently overextended or expensive. Wait for a healthy structural pullback before deployment."
            elif act_str == "REDUCE / UNDERPERFORM":
                act_desc = f"Locally overbought (RSI: {_rsi_val:.1f}). Fundamentals remain solid but tactical risk is elevated. Consider locking profits."
            else:
                act_desc = "Mixed signals across pillars. System lacks execution conviction. Monitor for structural breakout or mean reversion."

            def hex_to_rgb(hex_str):
                h = hex_str.lstrip('#')
                return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"

            bg_rgb = hex_to_rgb(act_color)


            # --- R/R DIAGNOSTIC EXPLAINER (Dynamic for all Risk/Reward states) ---
            _rr_section_html = ""
            _risk_gap  = cur_p - _stop_loss
            _risk_pct  = (_risk_gap / cur_p * 100) if cur_p > 0 else 0
            _rwrd_gap  = _tp1 - cur_p
            _rwrd_pct  = (_rwrd_gap / cur_p * 100) if cur_p > 0 else 0

            if p_conv_c == "#e74c3c":  # R/R is LOW  (<1.2x)
                _b1 = (f"Risk/Reward is {_rr:.2f}x — the stop loss at \u20ac{_stop_loss:.2f} risks \u20ac{_risk_gap:.2f} ({_risk_pct:.1f}%) while TP1 at \u20ac{_tp1:.2f} only offers \u20ac{_rwrd_gap:.2f} ({_rwrd_pct:.1f}%) upside. A ratio below 1.2x is considered unfavorable for new entries.")
                if _rsi_val > 65: _b2 = (f"RSI is elevated at {_rsi_val:.1f} — overbought momentum increases the probability of a pullback before reaching TP1, reducing effective reward potential.")
                elif _w52_pos > 75: _b2 = (f"Price is at {_w52_pos:.0f}% of its 52-week range — proximity to annual highs compresses remaining upside and increases downside risk if resistance holds.")
                elif _pe_v > 35 and _pe_v > 0: _b2 = (f"P/E of {_pe_v:.1f}x signals premium valuation — limited margin of safety amplifies the downside if earnings disappoint, worsening the R/R profile.")
                else: _b2 = (f"Technical structure shows limited near-term catalysts: current price \u20ac{cur_p:.2f} is close to TP1, suggesting most of the move may already be priced in.")
                _b3 = (f"To improve the setup, consider waiting for a pullback toward \u20ac{(_s1 * 0.97):.2f}\u2013\u20ac{_s1:.2f} (support zone), which would widen the reward-to-risk ratio above 2x.")
                _bullet_items = "".join([f"<li style='margin-bottom:7px; line-height:1.55;'>{b}</li>" for b in [_b1, _b2, _b3]])
                _rr_section_html = f"<div style='margin-top:14px; padding:14px 16px; background:rgba(231,76,60,0.07); border:1px solid rgba(231,76,60,0.25); border-radius:8px;'><div style='font-size:0.7em; color:#e74c3c; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;'>Why Risk/Reward is LOW</div><ul style='margin:0; padding-left:18px; color:#ccc; font-size:0.82em;'>{_bullet_items}</ul></div>"
            elif p_conv_c == "#2ecc71":  # R/R is MEDIUM (1.2x – 2.5x)
                _b1 = (f"Risk/Reward is {_rr:.2f}x — acceptable but not yet asymmetric. The setup risks \u20ac{_risk_gap:.2f} ({_risk_pct:.1f}%) for a potential gain of \u20ac{_rwrd_gap:.2f} ({_rwrd_pct:.1f}%). A ratio between 1.2x and 2.5x supports a partial position, not full deployment.")
                if _rsi_val < 45 and _w52_pos < 50: _b2 = (f"Supportive setup: RSI at {_rsi_val:.1f} (non-overbought) and price at {_w52_pos:.0f}% of its 52-week range reduces near-term downside pressure and leaves room for momentum to develop toward TP1.")
                elif ai_score >= 60: _b2 = (f"Quality score of {ai_score:.0f}/100 underpins the thesis — a fundamentally strong asset with acceptable technicals. The R/R is constrained by entry timing rather than structural weakness.")
                else: _b2 = (f"The setup is balanced: price at {_w52_pos:.0f}% of its 52-week range with RSI at {_rsi_val:.1f}. No extreme conditions exist to strongly favour bulls or bears — the market is in a discovery phase.")
                _b3 = (f"Execution tip: initiate a 50% position near current levels and reserve the remaining allocation for a pullback toward \u20ac{(_s1 * 0.98):.2f}\u2013\u20ac{_s1:.2f}, which would push the blended R/R above 2x.")
                _bullet_items = "".join([f"<li style='margin-bottom:7px; line-height:1.55;'>{b}</li>" for b in [_b1, _b2, _b3]])
                _rr_section_html = f"<div style='margin-top:14px; padding:14px 16px; background:rgba(46,204,113,0.07); border:1px solid rgba(46,204,113,0.25); border-radius:8px;'><div style='font-size:0.7em; color:#2ecc71; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;'>Why Risk/Reward is MEDIUM</div><ul style='margin:0; padding-left:18px; color:#ccc; font-size:0.82em;'>{_bullet_items}</ul></div>"
            else:  # R/R is HIGH (>2.5x)
                _b1 = (f"Risk/Reward is {_rr:.2f}x — strongly asymmetric. TP1 at \u20ac{_tp1:.2f} offers \u20ac{_rwrd_gap:.2f} ({_rwrd_pct:.1f}%) upside while the stop at \u20ac{_stop_loss:.2f} limits downside to \u20ac{_risk_gap:.2f} ({_risk_pct:.1f}%). A ratio above 2.5x represents a high-conviction, institutionally sound entry.")
                if _w52_pos < 25: _b2 = (f"Price is at {_w52_pos:.0f}% of its 52-week range — near structural lows with significant runway to the upside.")
                elif _rsi_val < 40: _b2 = (f"RSI at {_rsi_val:.1f} signals oversold conditions — historically, mean-reversion from these levels boosts the probability of reaching TP1.")
                else: _b2 = (f"The stop loss at \u20ac{_stop_loss:.2f} is anchored near key technical support, structurally minimizing the risk side while the reward window to TP1 at \u20ac{_tp1:.2f} remains wide open.")
                _b3 = (f"Execution: this setup supports full position sizing. Consider entering between \u20ac{_s1:.2f}\u2013\u20ac{cur_p:.2f} with a hard stop at \u20ac{_stop_loss:.2f}. If price breaks above \u20ac{_tp1:.2f}, reassess TP2 at \u20ac{_tp2:.2f}.")
                _bullet_items = "".join([f"<li style='margin-bottom:7px; line-height:1.55;'>{b}</li>" for b in [_b1, _b2, _b3]])
                _rr_section_html = f"<div style='margin-top:14px; padding:14px 16px; background:rgba(0,255,204,0.06); border:1px solid rgba(0,255,204,0.25); border-radius:8px;'><div style='font-size:0.7em; color:#00ffcc; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;'>Why Risk/Reward is HIGH</div><ul style='margin:0; padding-left:18px; color:#ccc; font-size:0.82em;'>{_bullet_items}</ul></div>"

            # RENDER UNIFIED UI MATRIX
            st.markdown(f"""
            <div style='background:rgba(10,15,25,0.6); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:20px; margin-bottom:25px;'>
                <div style='display:flex; justify-content:space-between; text-align:center; margin-bottom:20px; flex-wrap:wrap; gap:10px;'>
                    <div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; border-top:3px solid {p_trend_c};'>
                        <div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>Technical Trend</div>
                        <div style='font-weight:900; font-size:0.9em; color:{p_trend_c}; margin-top:8px;'>{p_trend}</div>
                    </div>
                    <div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; border-top:3px solid {p_qual_c};'>
                        <div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>Quality</div>
                        <div style='font-weight:900; font-size:0.9em; color:{p_qual_c}; margin-top:8px;'>{p_qual}</div>
                    </div>
                    <div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; border-top:3px solid {p_val_c};'>
                        <div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>Valuation</div>
                        <div style='font-weight:900; font-size:0.9em; color:{p_val_c}; margin-top:8px;'>{p_val}</div>
                    </div>
                    <div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; border-top:3px solid {p_risk_c};'>
                        <div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>Risk (52w)</div>
                        <div style='font-weight:900; font-size:0.9em; color:{p_risk_c}; margin-top:8px;'>{p_risk}</div>
                    </div>
                    <div style='flex:1; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; border-top:3px solid {p_conv_c};'>
                        <div style='font-size:0.65em; color:#aab; text-transform:uppercase; letter-spacing:1px;'>Risk/Reward</div>
                        <div style='font-weight:900; font-size:0.9em; color:{p_conv_c}; margin-top:8px;'>{p_conv}</div>
                    </div>
                </div>
                <div style='background:rgba({bg_rgb},0.12); border-left:6px solid {act_color}; padding:20px; border-radius:8px; box-shadow:0 4px 15px rgba(0,0,0,0.3);'>
                    <div style='font-size:0.75em; color:#bbb; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px;'>Positioning Hint / Action Layer</div>
                    <div style='font-size:1.6em; font-weight:900; color:{act_color}; margin-bottom:8px; text-shadow: 0px 2px 10px rgba({bg_rgb}, 0.5);'>{act_str}</div>
                    <div style='color:#e0e0e0; font-size:1.0em; line-height:1.5; margin-bottom:15px;'>{act_desc}</div>
                    <hr style='border:0; height:1px; background:linear-gradient(90deg, rgba(255,255,255,0.15), transparent); margin-bottom:15px;'>
                    <div style='display:flex; justify-content:space-between; font-family:"Courier New", monospace; font-size:0.95em; background:rgba(0,0,0,0.4); padding:12px; border-radius:6px;'>
                        <span style='color:#2ecc71;'><b>ENTRY:</b> €{_s1:.2f} ➔ €{cur_p:.2f}</span>
                        <span style='color:#e74c3c;'><b>STOP LOSS:</b> €{_stop_loss:.2f}</span>
                        <span style='color:#3498db;'><b>TARGET:</b> €{_tp1:.2f} (R/R: {_rr:.1f}x)</span>
                    </div>
                    {_rr_section_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:35px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #9b59b6; color:#9b59b6; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 3: RISK INTELLIGENCE HUB</div>", unsafe_allow_html=True)
            # ── RISK INTELLIGENCE HUB: Full-Width Top, then Split View ─────
            render_header("zap", "AI Investment Intelligence: Unified Risk Audit", level="####")
            st.caption("A multi-dimensional synthesis of Qualitative (NLP News) and Quantitative (Fundamental Pillars) risk factors to provide a unified investment verdict.")

            # ── PART A (Full-Width): Audit Button + Cockpit + Conflict Banner ─
            
            if st.button("Run Real-Time AI Risk Audit", type="primary", use_container_width=True):
                with st.spinner(f"Scanning news for {meta['company']}..."):
                    llm_res = analyze_risk_with_llm(deep_ticker, meta['company'])
                    if llm_res.get("error"):
                        st.error(f"NLP Error: {llm_res['error'][:80]}")
                    else:
                        nlp_score     = llm_res.get("red_flag_score", 0)
                        nlp_sentiment = llm_res.get("sentiment", "Neutral")
                        nlp_reco      = llm_res.get("recommendation", "N/A")
                        nlp_insights  = llm_res.get("key_insights", [])
                        nlp_category  = llm_res.get("risk_category", "None")
                        _cohere_key_ra = (
                            os.environ.get("COHERE_API_KEY", "")
                            or st.session_state.get("cohere_api_key", "")
                        )
                        if _cohere_key_ra:
                            _fmi_data_ra = compute_fmi_live(
                                quarterly_fin[quarterly_fin["ticker"] == deep_ticker] if not quarterly_fin.empty else pd.DataFrame(),
                                df_fin
                            )
                            _unified_metrics = {
                                **meta_enriched,
                                "ticker":        deep_ticker,
                                "company":       meta.get("company", deep_ticker),
                                "sector":        meta.get("sector", "N/A"),
                                "ai_score":      ai_score,
                                "fmi_score":     _fmi_data_ra.get("total", "N/A"),
                                "fmi_label":     _fmi_data_ra.get("label", "N/A"),
                                "price":         cur_p,
                                "market_regime": regime,
                                "support_s1":    round(_s1, 2),
                                "support_s2":    round(_s2, 2),
                                "resistance_r1": round(_r1, 2),
                                "resistance_r2": round(_r2, 2),
                                "stop_loss_technical": round(_stop_loss, 2),
                                "ma_20_current": round(float(df_deep["ma_20"].iloc[-1]), 2) if "ma_20" in df_deep.columns and not df_deep["ma_20"].isna().all() else "N/A",
                                "ma_50_current": round(float(df_deep["ma_50"].iloc[-1]), 2) if "ma_50" in df_deep.columns and not df_deep["ma_50"].isna().all() else "N/A",
                            }
                            with st.spinner("Synthesizing CIO Unified Verdict..."):
                                _unified_report = get_unified_verdict(_cohere_key_ra, _unified_metrics, llm_res)
                            _qs = int(ai_score) if str(ai_score).isdigit() else 0
                            _ns = llm_res.get("red_flag_score", 0)
                            _nst = llm_res.get("sentiment", "Neutral")
                            _conflict = (
                                (_qs >= 65 and (_ns >= 55 or _nst in ["Negative", "Critical"])) or
                                (_qs < 45  and (_ns <= 25  and _nst == "Positive"))
                            )
                            st.session_state[f"unified_verdict_{deep_ticker}"] = {
                                "report":        _unified_report,
                                "nlp_insights":  llm_res.get("key_insights", []),
                                "nlp_sentiment": _nst,
                                "nlp_score":     _ns,
                                "extracted_at":  datetime.now().strftime("%H:%M:%S"),
                                "is_conflict":   _conflict,
                                "ai_score_snap": _qs,
                            }
                            st.success("Audit complete — see AI Risk Overlay below.", icon="✅")
            else:
                st.markdown("""
                <div style='text-align:center; padding:30px 20px; color:#666;'>
                    <div style='font-size:2rem; font-weight: 800; font-family: monospace; letter-spacing: -2px;'>NLP</div>
                    <div style='font-size:0.85rem; margin-top:10px;'>Click the button above to scan real-time<br>news headlines and detect hidden risks.</div>
                </div>
                """, unsafe_allow_html=True)




            st.markdown("---")
            
            # ── QUAL vs QUANT: Full-Width 50/50 Split View ───────────────────
            render_header("zap", "Qualitative vs. Quantitative Risk Analysis", level="####")
            st.caption("Left: NLP-powered real-time sentiment from news headlines (Cohere AI). Right: Quantitative pillar breakdown from fundamental data.")
            
            qual_col, quant_col = st.columns([1, 1])
            
            # ── LEFT: NLP Qualitative Audit ──────────────────────────────────
            with qual_col:
                st.markdown("<div style='color:#3498db; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; border-bottom:1px solid rgba(52,152,219,0.3); padding-bottom:6px;'>Qualitative NLP Audit</div>", unsafe_allow_html=True)
                
                if st.button("Run Real-Time AI Risk Audit", type="primary", use_container_width=True):
                    with st.spinner(f"Scanning news for {meta['company']}..."):
                        llm_res = analyze_risk_with_llm(deep_ticker, meta['company'])
                        
                        if llm_res.get("error"):
                            st.error(f"NLP Error: {llm_res['error'][:80]}")
                        else:
                            nlp_score     = llm_res.get("red_flag_score", 0)
                            nlp_sentiment = llm_res.get("sentiment", "Neutral")
                            nlp_reco      = llm_res.get("recommendation", "N/A")
                            nlp_insights  = llm_res.get("key_insights", [])
                            nlp_category  = llm_res.get("risk_category", "None")
                            
                            if nlp_score <= 25:   nlp_border, nlp_badge = "#2ecc71", "LOW RISK"
                            elif nlp_score <= 50: nlp_border, nlp_badge = "#f1c40f", "MODERATE"
                            elif nlp_score <= 75: nlp_border, nlp_badge = "#e67e22", "ELEVATED"
                            else:                 nlp_border, nlp_badge = "#e74c3c", "HIGH RISK"
                            
                            st.markdown(f"""
                            <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px; padding:10px; background:rgba(255,255,255,0.03); border-radius:8px; border-left:3px solid {nlp_border};'>
                                <div style='text-align:center; min-width:55px;'>
                                    <div style='font-size:1.8rem; font-weight:900; color:{nlp_border}; line-height:1;'>{nlp_score}</div>
                                    <div style='font-size:0.6rem; color:#888;'>/100</div>
                                </div>
                                <div>
                                    <div style='font-size:0.75rem; font-weight:700; color:{nlp_border};'>{nlp_badge}</div>
                                    <div style='font-size:0.72rem; color:#aaa;'>Sentiment: <b>{nlp_sentiment}</b> · Category: <b>{nlp_category}</b></div>
                                </div>
                            </div>
                            <div style='font-size:0.8rem; font-style:italic; color:#ddd; border-left:2px solid #3498db; padding-left:8px; margin-bottom:10px;'>"{nlp_reco}"</div>
                            <div style='color:#999; font-size:0.72rem; font-weight:700; margin-bottom:5px;'>KEY INSIGHTS ({llm_res.get("headlines_analyzed", 0)} sources):</div>
                            <ul style='color:#bbb; font-size:0.78rem; line-height:1.5; padding-left:14px; margin:0;'>
                                {"".join([f"<li>{item}</li>" for item in nlp_insights])}
                            </ul>
                            """, unsafe_allow_html=True)
                else:
                    _uv_text, _nlp_insights, _nlp_score = _uv_data, [], 0
                    _nlp_sent, _audit_time, _is_conflict, _ai_score_snap = "N/A", "", False, 0

                # ── AI Risk Cockpit: 3-Column Premium Scorecard (Full-Width) ──
                _pulse_style = "border: 1px solid rgba(230,126,34,0.6); box-shadow: 0 0 15px rgba(230,126,34,0.15); border-left: 4px solid #e67e22;" if _is_conflict else "border: 1px solid rgba(255,255,255,0.08); border-left: 4px solid #444;"
                _q_color = "#00ffcc" if _ai_score_snap >= 70 else "#f1c40f" if _ai_score_snap >= 45 else "#e74c3c"
                _r_color = "#e74c3c" if _nlp_score >= 60 else "#f39c12" if _nlp_score >= 30 else "#2ecc71"
                _s_color = "#00ffcc" if _nlp_sent == "Positive" else "#e74c3c" if _nlp_sent in ["Negative", "Critical"] else "#8899aa"

                st.markdown(f"""
    <div style='
        display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap:12px; margin:16px 0 10px 0;
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(8px);
        padding: 1px; border-radius: 12px;
        {_pulse_style}
    '>
        <!-- Card 1: Quant Health -->
        <div style='padding:15px; background:rgba(255,255,255,0.01); border-radius:10px;'>
            <div style='font-size:0.65rem; color:#8899aa; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>{SVG_ICONS["chart"]} Quant Health</div>
            <div style='display:flex; align-items:baseline; gap:6px;'>
                <span style='font-size:1.4rem; font-weight:700; color:white;'>{_ai_score_snap}</span>
                <span style='font-size:0.75rem; color:#666;'>/100</span>
            </div>
            <div style='width:100%; height:3px; background:rgba(255,255,255,0.05); border-radius:2px; margin-top:8px;'>
                <div style='width:{_ai_score_snap}%; height:100%; background:{_q_color}; border-radius:2px;'></div>
            </div>
        </div>
        <!-- Card 2: News Sentiment -->
        <div style='padding:15px; background:rgba(255,255,255,0.01); border-radius:10px;'>
            <div style='font-size:0.65rem; color:#8899aa; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>{SVG_ICONS["globe"]} News Tone</div>
            <div style='display:flex; align-items:center; gap:8px;'>
                <span style='font-size:1.2rem; font-weight:600; color:{_s_color};'>{_nlp_sent}</span>
            </div>
            <div style='margin-top:8px; font-size:0.7rem; color:#666;'>Extracting sentiment from latest financial headlines</div>
        </div>
        <!-- Card 3: Risk Exposure -->
        <div style='padding:15px; background:rgba(255,255,255,0.01); border-radius:10px;'>
            <div style='font-size:0.65rem; color:#8899aa; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>{SVG_ICONS["risk"]} Risk Exposure</div>
            <div style='display:flex; align-items:baseline; gap:6px;'>
                <span style='font-size:1.4rem; font-weight:700; color:{_r_color};'>{_nlp_score}</span>
                <span style='font-size:0.75rem; color:#666;'>/100</span>
            </div>
            <div style='width:100%; height:3px; background:rgba(255,255,255,0.05); border-radius:2px; margin-top:8px;'>
                <div style='width:{_nlp_score}%; height:100%; background:{_r_color}; border-radius:2px;'></div>
            </div>
        </div>
    </div>
    <div style='font-size:0.62rem; color:#556677; text-align:right; margin-bottom:12px; letter-spacing:0.5px;'>
        SYNCHRONIZED AUDIT TIMESTAMP: {_audit_time} &nbsp;&middot;&nbsp; <b>DYNAMIC OVERLAY V12.1</b>
    </div>
    """, unsafe_allow_html=True)

                # ── Signal Conflict Banner (full-width, only when diverging) ──
                if _is_conflict:
                    if _ai_score_snap >= 65:
                        if _nlp_sent in ["Negative", "Critical"]:
                            _conf_dir = f"Strong Quant Health ({_ai_score_snap}/100), but News Tone is distinctly <b>{_nlp_sent.upper()}</b>. The market may penalize the stock soon."
                        else:
                            _conf_dir = f"Strong Quant Health ({_ai_score_snap}/100), but Risk Exposure is elevated (<b>Red Flag: {_nlp_score}/100</b>). Monitor for potential headline shocks."
                    else:
                        _conf_dir = f"Weak Quant Health ({_ai_score_snap}/100), but News Tone is <b>POSITIVE</b>. Beware of a temporary, sentiment-driven rally."
                    st.markdown(f"""
<div style='display:flex; align-items:flex-start; gap:14px; margin:8px 0; padding:14px 18px; background:linear-gradient(90deg,rgba(230,126,34,0.14),rgba(231,76,60,0.08)); border:1px solid rgba(230,126,34,0.55); border-left:4px solid #e67e22; border-radius:10px;'>
    <span style='font-size:1.4rem; line-height:1; padding-top:2px;'>⚠️</span>
    <div>
        <div style='color:#e67e22; font-weight:900; font-size:0.75rem; text-transform:uppercase; letter-spacing:2px; margin-bottom:4px;'>⚡ Signal Conflict Detected</div>
        <div style='color:#ddd; font-size:0.85rem; line-height:1.5;'>{_conf_dir}</div>
    </div>
</div>
""", unsafe_allow_html=True)

            # ── PART B: 50/50 Split — Narrative (left) | Radar (right) ──────
            st.markdown("<hr style='border:0; height:1px; background:rgba(255,255,255,0.08); margin:24px 0;'>", unsafe_allow_html=True)
            qual_col, quant_col = st.columns([1, 1])

            with qual_col:
                # Detailed Audit Narrative (only when audit has been run)
                if _unified_key in st.session_state:
                    with st.expander("🔍 Detailed CIO Reasoning & Full Audit", expanded=True):
                        st.markdown(
                            f"<div style='background:rgba(255,255,255,0.03); border-left:4px solid #555;"
                            f" border-radius:8px; padding:20px; font-size:0.9rem; line-height:1.7; color:#efefef;'>"
                            + _uv_data.get("report", "") + "</div>",
                            unsafe_allow_html=True
                        )
                        _show_insights = _uv_data.get("nlp_insights", []) if isinstance(_uv_data, dict) else []
                        if _show_insights:
                            st.markdown("---")
                            st.markdown("<div style='font-size:0.75rem; color:#888; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>🧩 Raw Evidence: News Signals Analyzed</div>", unsafe_allow_html=True)
                            for insight in _show_insights:
                                st.markdown(f"<div style='font-size:0.78rem; color:#ccc; border-left:2px solid #3498db; padding-left:8px; margin-bottom:5px;'>{insight}</div>", unsafe_allow_html=True)

                # ── NEWS FEED (Auto-load, FinBERT Sentiment) ─────────────────
                st.markdown("<div style='color:#f39c12; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-top:16px; margin-bottom:8px; border-bottom:1px solid rgba(243,156,18,0.3); padding-bottom:6px;'>📰 Market Sentiment (FinBERT)</div>", unsafe_allow_html=True)
                try:
                    import feedparser
                    _rss_url = f"https://news.google.com/rss/search?q={deep_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
                    _feed = feedparser.parse(_rss_url)
                    _news_items = _feed.entries[:10]
                    if _news_items:
                        _pipe = get_finbert_pipeline()
                        _titles = [item.get("title", "").split(" - ")[0] for item in _news_items]
                        _sent_scores = []
                        
                        # Pre-calculate sentiment to show mood OUTSIDE popover
                        if _pipe:
                            _results = _pipe(_titles)
                            for _res in _results:
                                _lbl = _res['label'].upper()
                                _sc = _res['score']
                                _sent_scores.append(_sc if _lbl == 'POSITIVE' else (-_sc if _lbl == 'NEGATIVE' else 0))
                            
                            if _sent_scores:
                                _avg_sent = sum(_sent_scores) / len(_sent_scores)
                                _mood_lbl = "🚀 BULLISH" if _avg_sent > 0.1 else ("📉 BEARISH" if _avg_sent < -0.1 else "😴 NEUTRAL")
                                _mood_color = "#2ecc71" if _avg_sent > 0.1 else ("#e74c3c" if _avg_sent < -0.1 else "#f1c40f")
                                st.markdown(f"<div style='margin-bottom:12px;padding:8px 12px;background:rgba(255,255,255,0.04);border-radius:6px;border-left:3px solid {_mood_color};font-size:0.85rem;'><b style='color:{_mood_color};'>{_mood_lbl}</b> &nbsp;·&nbsp; FinBERT: {abs(_avg_sent):.2f}</div>", unsafe_allow_html=True)
                        
                        # Popover for details
                        with st.popover(f"View {len(_news_items)} Detailed Headlines", use_container_width=True):
                            st.markdown("### 📰 Recent Headlines")
                            if _pipe:
                                for _i, _res in enumerate(_results):
                                    _lbl = _res['label'].upper()
                                    _sc = _res['score']
                                    _icon = "🟢" if _lbl == 'POSITIVE' else ("🔴" if _lbl == 'NEGATIVE' else "⚪")
                                    _entry = _news_items[_i]
                                    with st.expander(f"{_icon} {_titles[_i][:70]}..."):
                                        st.caption(f"**Source:** {_entry.get('source', {}).get('title', 'Google News')} | **Date:** {_entry.get('published', 'N/A')}")
                                        st.markdown(f"[Read Article ↗]({_entry.get('link')})")
                            else:
                                for _entry in _news_items[:5]:
                                    _title = _entry.get("title", "").split(" - ")[0]
                                    with st.expander(f"📰 {_title[:70]}..."):
                                        st.caption(f"**Date:** {_entry.get('published', 'N/A')}")
                                        st.markdown(f"[Read Article ↗]({_entry.get('link')})")
                    else:
                        st.info("No recent news found for this ticker.")
                except Exception as _e:
                    st.caption(f"⚠️ News feed unavailable: {str(_e)[:60]}")


            with quant_col:
                tab_q, tab_m = st.tabs(["Quality Breakdown", "Momentum (FMI)"])
                
                # Build radar from score_details
                _radar_sd = compute_score_details(meta_enriched)
                _radar_breakdown = _radar_sd.get("breakdown", {})
                _sector_lc = meta.get("sector", "").lower() if meta.get("sector") else ""
                _is_tech = any(s in _sector_lc for s in ["tech", "semi", "software", "cloud", "comm", "ai"])
                _max_pts = {
                    "Valuation":       20,
                    "Profitability":   30 if _is_tech else 25,
                    "Fin. Health":     15,
                    "Net Yield":       5  if _is_tech else 10,
                    "Momentum":        25,   # Fixed: engine caps Context & Momentum at 25 (was 20)
                    "Analyst Est.":    5     # Fixed: engine caps Analyst Estimates at 5 (was 10)
                }
                _pillar_keys = {
                    "Valuation":       "Valuation",
                    "Profitability":   "Profitability",
                    "Fin. Health":     "Financial Health",
                    "Net Yield":       "Net Payout Yield",
                    "Momentum":        "Context & Momentum",
                    "Analyst Est.":    "Analyst Estimates"
                }
                _radar_labels = list(_max_pts.keys())
                _radar_vals   = [
                    round((_radar_breakdown.get(_pillar_keys[k], 0) / _max_pts[k]) * 100, 1)
                    for k in _radar_labels
                ]
                # Close the polygon
                _radar_labels_closed = _radar_labels + [_radar_labels[0]]
                _radar_vals_closed   = _radar_vals   + [_radar_vals[0]]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=_radar_vals_closed,
                    theta=_radar_labels_closed,
                    fill="toself",
                    fillcolor="rgba(0,255,204,0.08)",
                    line=dict(color="#00ffcc", width=2),
                    name=deep_ticker
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(
                            visible=True, range=[0, 100],
                            tickfont=dict(size=9, color="#666"),
                            gridcolor="rgba(255,255,255,0.06)",
                            linecolor="rgba(255,255,255,0.08)"
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=11, color="#bbb"),
                            gridcolor="rgba(255,255,255,0.06)"
                        )
                    ),
                    showlegend=False,
                    template="plotly_dark",
                    height=290,
                    margin=dict(t=20, b=10, l=40, r=40),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                with tab_q:
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Score summary under radar
                    _q_score = ai_score
                    _q_pct   = f"{_q_score}/100"
                    _q_color = ai_color
                    st.markdown(f"""
                    
                    """, unsafe_allow_html=True)

                # ── FMI Panel (below radar) ────────────────────────────────────
                with tab_m:
                    # Compute FMI live from quarterly data (no ETL re-run needed)
                    _qtrs = quarterly_fin[quarterly_fin["ticker"] == deep_ticker] if not quarterly_fin.empty else pd.DataFrame()
                    _anns = annual_fin[annual_fin["ticker"] == deep_ticker] if not annual_fin.empty else pd.DataFrame()
                    _fmi_data = compute_fmi_live(_qtrs, _anns)
                    _fmi_total = _fmi_data["total"]
                    _fmi_label = _fmi_data["label"]
                    _fmi_components = _fmi_data["components"]
                    _fmi_color = (
                        "#2ecc71" if _fmi_total >= 75 else
                        "#00ffcc" if _fmi_total >= 55 else
                        "#f39c12" if _fmi_total >= 40 else
                        "#e74c3c"
                    )
                    _fmi_bars = ""
                    _fmi_maxes = {"Revenue Acceleration": 30, "EPS Acceleration": 30, "Margin Expansion": 25, "Earnings Consistency": 15}
                    for comp_name, comp_val in _fmi_components.items():
                        max_pts = _fmi_maxes.get(comp_name, 30)
                        pct = min(100, int((comp_val / max_pts) * 100))
                        _fmi_bars += (
                            f"<div style='margin-bottom:5px;'>"
                            f"<div style='display:flex;justify-content:space-between;font-size:0.65rem;color:#aaa;margin-bottom:2px;'>"
                            f"<span>{comp_name}</span><span style='color:#fff;'>{comp_val}/{max_pts}</span>"
                            f"</div>"
                            f"<div style='background:rgba(255,255,255,0.07);border-radius:3px;height:5px;'>"
                            f"<div style='background:{_fmi_color};width:{pct}%;height:5px;border-radius:3px;'></div>"
                            f"</div></div>"
                        )
                    _fmi_html = (
                        "<div style='margin-top:10px;padding:10px;background:rgba(255,255,255,0.03);"
                        "border:1px solid rgba(255,255,255,0.07);border-radius:8px;'>"
                        "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
                        "<span style='font-size:0.72rem;font-weight:700;color:#8899aa;text-transform:uppercase;letter-spacing:1px;'>Fundamental Momentum Index</span>"
                        f"<span style='font-size:0.95rem;font-weight:900;color:{_fmi_color};'>{_fmi_total}/100"
                        f"&nbsp;<span style='font-size:0.7rem;font-weight:500;color:{_fmi_color};'>{_fmi_label}</span>"
                        "</span></div>"
                        + _fmi_bars +
                        "</div>"
                    )
                    st.markdown(_fmi_html, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<div style='margin-top:35px; margin-bottom:15px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #e74c3c; color:#e74c3c; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 4: DEEP DIAGNOSTICS & RAW DATA</div>", unsafe_allow_html=True)
            render_header("activity", "Diagnostic Metrics Portfolio")

            _card_style = "background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 4px 4px 4px;margin-bottom:4px;"
            _header_style = "color:#aabbcc;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;padding:0 8px 6px 8px;"
            
            with st.container():
                kcol1, kcol2, kcol3, kcol4, kcol5, kcol6 = st.columns(6)

                with kcol1:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Valuation & Size</div>", unsafe_allow_html=True)
                    m_cap = meta.get('market_cap', 0)
                    if m_cap >= 1e12: m_cap_txt = f"€{m_cap/1e12:.2f}T"
                    elif m_cap >= 1e9: m_cap_txt = f"€{m_cap/1e9:.1f}B"
                    else: m_cap_txt = f"€{m_cap/1e6:.0f}M"
                    
                    render_metric_row("Market Cap", m_cap_txt)
                    fwd_pe_txt = f"Fwd: {meta.get('forward_pe', 0):.1f}" if pd.notnull(meta.get('forward_pe')) and meta.get('forward_pe', 0) > 0 else ""
                    pe_val = f"{meta['pe_ratio']:.1f}" if pd.notnull(meta['pe_ratio']) else "N/A"
                    render_metric_row("P/E", pe_val, delta=fwd_pe_txt)
                    
                    peg_raw = meta.get('peg_ratio', 0)
                    peg_col = "#2ecc71" if pd.notnull(peg_raw) and 0 < peg_raw <= 1.0 else ("#e74c3c" if pd.notnull(peg_raw) and peg_raw > 2.0 else None)
                    render_metric_row("PEG", f"{peg_raw:.2f}" if pd.notnull(peg_raw) else "N/A", value_color=peg_col)
                    
                    ev_raw = meta.get('ev_to_ebitda', 0)
                    ev_col = "#2ecc71" if 0 < ev_raw <= 10 else ("#e74c3c" if ev_raw > 20 else None)
                    render_metric_row("EV/EBITDA",  f"{ev_raw:.2f}", value_color=ev_col, help_text="🟢 <10x (Value) | 🔴 >20x (Expensive)")
                    
                    ps_val = meta.get('price_to_sales', 0)
                    ps_col = "#2ecc71" if 0 < ps_val <= 2 else ("#e74c3c" if ps_val > 10 else None)
                    render_metric_row("Price/Sales",f"{ps_val:.2f}", value_color=ps_col, help_text="🟢 < 2x (Cheap) | 🔴 > 10x (Expensive)")
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol2:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Profit & Returns</div>", unsafe_allow_html=True)
                    
                    div_val = meta.get('dividend_yield_pct', 0)
                    div_col = "#2ecc71" if pd.notnull(div_val) and div_val > 4 else None
                    render_metric_row("Div Yield", f"{div_val:.2f}%" if pd.notnull(div_val) else "0.00%", value_color=div_col, help_text="🟢 > 4% (High Yielding)")
                    
                    # Net Payout = Div + Buybacks
                    net_payout = meta.get('net_payout_yield_pct', 0)
                    bb_yield   = meta.get('buyback_yield_pct', 0)
                    render_metric_row("Net Payout",   f"{net_payout:.2f}%", delta=f"BB: {bb_yield:.1f}%")
                    
                    roe_raw = meta.get('roe', 0) * 100
                    roe_col = "#2ecc71" if roe_raw >= 15 else ("#e74c3c" if roe_raw < 5 else None)
                    render_metric_row("ROE", f"{roe_raw:.1f}%", value_color=roe_col, help_text="🟢 > 15% (Strong Profitability) | 🔴 < 5% (Poor)")
                    
                    gm_val = meta.get('gross_margin', 0) * 100
                    gm_col = "#2ecc71" if gm_val >= 40 else ("#e74c3c" if gm_val < 10 else None)
                    render_metric_row("Gross Margin", f"{gm_val:.1f}%", value_color=gm_col, help_text="🟢 > 40% (Wide Moat) | 🔴 < 10% (Thin Margin)")
                    
                    op_margin = meta.get('operating_margin', 0)
                    op_val = op_margin*100 if pd.notnull(op_margin) else 0
                    op_col = "#2ecc71" if op_val >= 15 else ("#e74c3c" if op_val < 5 else None)
                    render_metric_row("Op Margin", f"{op_val:.1f}%", value_color=op_col)
                    
                    fcf_m = meta.get('fcf_margin', 0)
                    render_metric_row("FCF Margin", f"{fcf_m:.1f}%", value_color="#2ecc71" if fcf_m > 15 else None)
                    
                    rev_growth = meta.get('revenue_growth', 0) * 100
                    rev_col = "#2ecc71" if rev_growth > 20 else ("#e74c3c" if rev_growth < 0 else None)
                    render_metric_row("Rev Growth", f"{rev_growth:.1f}%", value_color=rev_col)
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol3:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Solvency</div>", unsafe_allow_html=True)
                    debt_eq_raw = meta.get('debt_to_equity', 0)
                    if pd.notnull(debt_eq_raw) and debt_eq_raw != 0:
                        debt_eq_txt = f"{(debt_eq_raw / 100.0):.2f}x"
                    else:
                        debt_eq_txt = "N/A (Neg Equity)" if meta.get('total_debt', 0) > 0 else "0.00x"
                    
                    curr_rat  = meta.get('current_ratio', 0)
                    quick_rat = meta.get('quick_ratio', 0)
                    
                    # Liquidity Status Colors
                    c_col = "#2ecc71" if curr_rat > 1.5 else ("#e74c3c" if curr_rat < 1.0 else "#f39c12")
                    q_col = "#2ecc71" if quick_rat > 1.0 else ("#e74c3c" if quick_rat < 0.7 else "#f39c12")
                    
                    _ebitda_val = 0.0
                    try: _ebitda_val = float(meta.get('ebitda', 0) or 0)
                    except (TypeError, ValueError): pass
                    _debt_val = 0.0
                    try: _debt_val = float(meta.get('total_debt', 0) or 0)
                    except (TypeError, ValueError): pass
                    debt_ebitda = (_debt_val / _ebitda_val) if _ebitda_val > 0 else 0
                    
                    de_col = "#2ecc71" if 0 < debt_ebitda < 3.0 else ("#e74c3c" if debt_ebitda >= 5.0 else "#f39c12")

                    render_metric_row("Debt/Eq", debt_eq_txt)
                    render_metric_row("Debt/EBITDA", f"{debt_ebitda:.2f}x" if debt_ebitda > 0 else "N/A", value_color=de_col)
                    render_metric_row("Current Ratio", f"{curr_rat:.2f}", value_color=c_col)
                    render_metric_row("Quick Ratio",   f"{quick_rat:.2f}", value_color=q_col)
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol4:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Risk & Volume</div>", unsafe_allow_html=True)
                    beta_val = meta.get('beta', 1.0)
                    if pd.notnull(beta_val) and beta_val != 0:
                        beta_col = "#e74c3c" if beta_val > 1.5 else ("#3498db" if beta_val < 0.8 else None)
                        render_metric_row("Beta", f"{beta_val:.2f}", value_color=beta_col, help_text="🔴 > 1.5 (High Volatility) | 🔵 < 0.8 (Defensive)")
                    else:
                        render_metric_row("Beta", "N/A")
                    
                    inst = meta.get('inst_ownership', 0) * 100
                    inst_col = "#2ecc71" if inst > 60 else ("#e74c3c" if inst < 10 else None)
                    render_metric_row("Inst Own", f"{inst:.0f}%", value_color=inst_col, help_text="🟢 > 60% (Strong Institutional Backing)")
                    
                    short_val = meta.get('short_percent_of_float', 0) * 100
                    short_col = "#e74c3c" if short_val > 10 else ("#2ecc71" if 0 <= short_val <= 2 else None)
                    render_metric_row("Short Float", f"{short_val:.1f}%", value_color=short_col, help_text="🔴 > 10% (Squeeze Risk) | 🟢 < 2% (Safe)")
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol5:
                    st.markdown(f"<div style='{_card_style}'><div style='{_header_style}'>Price & Context</div>", unsafe_allow_html=True)
                    render_metric_row("Target",       f"€{target_p:.2f}", delta=upside, is_pct=True)
                    
                    pe_5y_avg    = meta.get('pe_5y_avg', 0)
                    pe_cur       = meta.get('pe_ratio', 0)
                    pe_delta     = ((pe_cur / pe_5y_avg) - 1) * 100 if pe_5y_avg > 0 and pe_cur > 0 else 0
                    
                    render_metric_row("5Y Avg P/E",    f"{pe_5y_avg:.1f}" if pe_5y_avg > 0 else "N/A", delta=pe_delta, is_pct=True, color_invert=True)
                    
                    zs_col = "#2ecc71" if z_score < -1 else ("#e74c3c" if z_score > 1.5 else None)
                    render_metric_row("Z-Score (5Y)",  f"{z_score:.2f}", value_color=zs_col)
                    st.markdown("</div>", unsafe_allow_html=True)

                with kcol6:
                    # ── EARNINGS CALENDAR (v13.0) ──
                    e_row = earnings_cal[earnings_cal['ticker'] == deep_ticker]
                    e_header = _header_style
                    if not e_row.empty:
                        e_date = e_row.iloc[0]['earnings_date']
                        if pd.notnull(e_date):
                            # Handle both Timestamp and date objects safely
                            e_date_obj = e_date.date() if hasattr(e_date, 'date') else e_date
                            days_to_e = (e_date_obj - date.today()).days
                            if 0 <= days_to_e <= 7:
                                e_header = e_header.replace("#aabbcc", "#f39c12") # Highlight upcoming
                                e_date_str = f"⚠️ {e_date_obj.strftime('%b %d')}"
                            else:
                                e_date_str = e_date_obj.strftime('%b %d, %y')
                        else:
                            e_date_str = "TBD"
                        
                        eps_est = e_row.iloc[0]['eps_avg']
                        rev_est = e_row.iloc[0]['rev_avg']
                    else:
                        e_date_str = "N/A"
                        eps_est = None
                        rev_est = None

                    st.markdown(f"<div style='{_card_style}'><div style='{e_header}'>Earnings & Events</div>", unsafe_allow_html=True)
                    render_metric_row("Report Date", e_date_str)
                    render_metric_row("EPS Est",     f"{eps_est:.2f}" if pd.notnull(eps_est) else "N/A")
                    
                    if pd.notnull(rev_est) and rev_est > 0:
                        if rev_est >= 1e9: rev_txt = f"€{rev_est/1e9:.1f}B"
                        else: rev_txt = f"€{rev_est/1e6:.0f}M"
                    else:
                        rev_txt = "N/A"
                    render_metric_row("Revenue Est", rev_txt)
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")

            # Main Technical Chart (Full Width)

            fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                     vertical_spacing=0.05, 
                                     row_heights=[0.7, 0.3])
            
            fig_tech.add_trace(go.Candlestick(
                x=df_deep['date'],
                open=df_deep['price_open'], high=df_deep['price_high'],
                low=df_deep['price_low'], close=df_deep['price_close'],
                name="Price",
                increasing=dict(line=dict(color='#00e676', width=1), fillcolor='rgba(0,230,118,0.85)'),
                decreasing=dict(line=dict(color='#ff5252', width=1), fillcolor='rgba(255,82,82,0.85)')
            ), row=1, col=1)
            
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_20'], name='MA20', line=dict(color='#FFB300', width=1.5)), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_50'], name='MA50', line=dict(color='#40C4FF', width=1.5)), row=1, col=1)
            # 🏆 EXPERT: MA200 (Long-term trend anchor)
            if 'ma_200' in df_deep.columns:
                fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['ma_200'], name='MA200', line=dict(color='#E040FB', width=2.5)), row=1, col=1)
            
            # Support/Resistance → Scatter traces (appear in legend, not as annotations)
            _s3  = df_deep["price_low"].tail(252).min()
            _r3  = df_deep["price_high"].tail(252).max()

            dates_range = df_deep['date'].tolist()
            df_deep['rsi'] = df_deep['rsi'] if 'rsi' in df_deep.columns else _rsi_val  # RSI from get_tactical_metrics
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[_s1, _s1],
                name=f'S1 Support  €{_s1:.2f}', mode='lines',
                line=dict(color='#2ecc71', width=1, dash='dot'), opacity=0.8
            ), row=1, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[_r1, _r1],
                name=f'R1 Resistance  €{_r1:.2f}', mode='lines',
                line=dict(color='#e74c3c', width=1, dash='dot'), opacity=0.8
            ), row=1, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[_s2, _s2],
                name=f'S2 Support (50d)  €{_s2:.2f}', mode='lines',
                line=dict(color='#27ae60', width=1.5, dash='dash'), opacity=0.7
            ), row=1, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[_r2, _r2],
                name=f'R2 Resistance (50d)  €{_r2:.2f}', mode='lines',
                line=dict(color='#c0392b', width=1.5, dash='dash'), opacity=0.7
            ), row=1, col=1)
            # 🛡️ MAJOR INSTITUTIONAL LEVELS (1-Year)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[_s3, _s3],
                name=f'S3 Major Support (1Y)  €{_s3:.2f}', mode='lines',
                line=dict(color='#1b5e20', width=2.5, dash='solid'), opacity=0.5
            ), row=1, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[_r3, _r3],
                name=f'R3 Major Resistance (1Y)  €{_r3:.2f}', mode='lines',
                line=dict(color='#b71c1c', width=2.5, dash='solid'), opacity=0.5
            ), row=1, col=1)
            # 📈 AUTOMATED TRENDLINE (Linear Regression)
            # Calculate best-fit line for the current price window
            y_data = df_deep['price_close'].values
            x_data = np.arange(len(y_data))
            # Clean NaNs if any
            mask = ~np.isnan(y_data)
            if mask.any():
                slope, intercept = np.polyfit(x_data[mask], y_data[mask], 1)
                trendline_y = slope * x_data + intercept
                fig_tech.add_trace(go.Scatter(
                    x=df_deep['date'], y=trendline_y,
                    name='Regression Trendline',
                    line=dict(color='rgba(255, 215, 0, 0.4)', width=2, dash='dash'),
                    hoverinfo='skip'
                ), row=1, col=1)

            # RSI with overbought/oversold level traces in legend
            fig_tech.add_trace(go.Scatter(x=df_deep['date'], y=df_deep['rsi'], name='RSI (14)', line=dict(color='#9b59b6', width=2)), row=2, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[70, 70],
                name='RSI Overbought (70)', mode='lines',
                line=dict(color='rgba(231,76,60,0.5)', width=1, dash='dash'), showlegend=True
            ), row=2, col=1)
            fig_tech.add_trace(go.Scatter(
                x=[dates_range[0], dates_range[-1]], y=[30, 30],
                name='RSI Oversold (30)', mode='lines',
                line=dict(color='rgba(46,204,113,0.5)', width=1, dash='dash'), showlegend=True
            ), row=2, col=1)

            fig_tech.update_layout(
                title=dict(text=f"📈 {deep_ticker} — Technical Master Analysis", font=dict(size=20, color='#e8eaf6')),
                height=740,
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
                # Custom premium dark background
                paper_bgcolor='#0d0e14',
                plot_bgcolor='#11121a',
                font=dict(family="Inter, sans-serif", color="#b0bec5"),
                # Grid styling (subtle)
                xaxis=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                    zeroline=False, linecolor='rgba(255,255,255,0.1)'
                ),
                xaxis2=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                    zeroline=False, linecolor='rgba(255,255,255,0.1)',
                    tickprefix='€'
                ),
                yaxis2=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.04)',
                    zeroline=False
                ),
                # Legend → outside right side
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.01,
                    bgcolor="rgba(13,14,20,0.92)",
                    bordercolor="rgba(255,255,255,0.12)",
                    borderwidth=1,
                    font=dict(size=11, color='#cfd8dc'),
                    itemsizing="constant",
                    traceorder="normal"
                ),
                margin=dict(r=180, t=60, l=60, b=40)
            )
            fig_tech.update_yaxes(title_text="Price (€)", row=1, col=1)
            fig_tech.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            st.plotly_chart(fig_tech, use_container_width=True)

            # --- HISTORICAL FUNDAMENTAL TRENDS (Dual Axis) ---
            st.markdown("---")
            
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
                    
                    fig_fin.update_layout(
                        template="plotly_dark", height=500,
                        margin=dict(l=20, r=20, t=60, b=20),
                        hovermode="x unified",
                        barmode="group",
                        title_text=f"📊 {deep_ticker} Annual Financial Performance (Revenue, FCF & EPS)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    fig_fin.update_yaxes(title_text=f"Amount (€{unit})", secondary_y=False, range=[0, (max_val/scale)*1.3])
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
                        else:
                            df_fin_q_plot['free_cash_flow'] = None

                        # Calculate QoQ Growth locally to avoid NaN issues in DB
                        # Calculate Growth manually to handle negative values properly: (New - Old) / abs(Old)
                        df_fin_q_plot['rev_growth'] = (df_fin_q_plot['revenue'] - df_fin_q_plot['revenue'].shift(1)) / df_fin_q_plot['revenue'].shift(1).abs() * 100
                        df_fin_q_plot['eps_growth'] = (df_fin_q_plot['eps'] - df_fin_q_plot['eps'].shift(1)) / df_fin_q_plot['eps'].shift(1).abs() * 100
                        if 'free_cash_flow' in df_fin_q_plot.columns and df_fin_q_plot['free_cash_flow'].notna().any():
                            df_fin_q_plot['fcf_growth'] = (df_fin_q_plot['free_cash_flow'] - df_fin_q_plot['free_cash_flow'].shift(1)) / df_fin_q_plot['free_cash_flow'].shift(1).abs() * 100
                        else:
                            df_fin_q_plot['fcf_growth'] = None
                        
                        # Auto-scale based on max of Revenue and FCF
                        max_val_q = max(
                            df_fin_q_plot['revenue'].max() if not df_fin_q_plot['revenue'].empty else 0,
                            df_fin_q_plot['free_cash_flow'].max() if 'free_cash_flow' in df_fin_q_plot.columns and df_fin_q_plot['free_cash_flow'].notna().any() else 0
                        )
                        scale_q = 1e9 if max_val_q >= 1e9 else 1e6
                        unit_q = "B" if scale_q == 1e9 else "M"
                        
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
                                hovertemplate="<b>Quarter: %{x}</b><br>Revenue: €%{y:.2f}" + unit_q + "<br>QoQ Growth: %{text}<extra></extra>"
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
                                    hovertemplate="<b>Period: %{x}</b><br>FCF: €%{y:.2f}" + unit_q + "<br>QoQ Growth: %{text}<extra></extra>"
                                ),
                                secondary_y=False
                            )
                        
                        fig_fin_q.add_trace(
                            go.Scatter(
                                x=x_labels, 
                                y=df_fin_q_plot['eps'], 
                                name="EPS (€)", 
                                line=dict(color="orange", width=3), 
                                mode="lines+markers+text",
                                text=eps_text_q,
                                textposition="top center",
                                hovertemplate="<b>Quarter: %{x}</b><br>EPS: €%{y:.2f}<br>QoQ Growth: %{text}<extra></extra>"
                            ),
                            secondary_y=True
                        )
                        
                        fig_fin_q.update_layout(
                            template="plotly_dark", height=500,
                            margin=dict(l=20, r=20, t=60, b=20),
                            hovermode="x unified",
                            barmode="group",
                            title_text=f"📊 {deep_ticker} Quarterly Financial Performance (Revenue, FCF & EPS)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        y_range_q = [0, (max_val_q/scale_q)*1.3] if pd.notnull(max_val_q) else None
                        fig_fin_q.update_yaxes(title_text=f"Amount (€{unit_q})", secondary_y=False, range=y_range_q)
                        fig_fin_q.update_yaxes(title_text="Earnings Per Share (€)", secondary_y=True)
                        
                        st.plotly_chart(fig_fin_q, use_container_width=True)

                    else:
                        st.info("No historical quarterly financial data available for this ticker.")
                else:
                    st.info("Quarterly financials warehouse table is empty. Please run the ETL pipeline.")
            
            # ── DCF INTRINSIC VALUATION MODEL ───────────────────────────────
            st.markdown("---")
            render_header("gem", "Discounted Cash Flow (DCF) Intrinsic Valuation")
            st.write("Calculates the absolute mathematical fair value of the asset based on projected Future Free Cash Flows.")
            
            fcf = meta.get("free_cashflow")
            fcf = fcf if pd.notnull(fcf) else 0
            mcap = meta.get("market_cap")
            mcap = mcap if pd.notnull(mcap) else 0
            total_debt = meta.get("total_debt")
            total_debt = total_debt if pd.notnull(total_debt) else 0
            
            if fcf > 0 and mcap > 0:
                shares_out = mcap / cur_p
                
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    proj_growth = st.number_input("Projected FCF Growth Y1-Y5 (%)", value=15.0, step=1.0) / 100
                with col_d2:
                    term_growth = st.number_input("Terminal Growth Y6+ (%)", value=2.5, step=0.5) / 100
                with col_d3:
                    discount_rate = st.number_input("Discount Rate (WACC) (%)", value=9.0, step=0.5) / 100
                    
                if discount_rate > term_growth:
                    # 5-Year Projection
                    cash_flows = []
                    current_fcf = fcf
                    for year in range(1, 6):
                        current_fcf *= (1 + proj_growth)
                        pv_fcf = current_fcf / ((1 + discount_rate) ** year)
                        cash_flows.append(pv_fcf)
                    
                    # Terminal Value Calculation
                    tv = (current_fcf * (1 + term_growth)) / (discount_rate - term_growth)
                    pv_tv = tv / ((1 + discount_rate) ** 5)
                    
                    # Enterprise Value -> Equity Value
                    enterprise_value = sum(cash_flows) + pv_tv
                    intrinsic_equity = enterprise_value - total_debt
                    
                    intrinsic_per_share = intrinsic_equity / shares_out
                    margin_of_safety = (intrinsic_per_share - cur_p) / cur_p * 100
                    
                    dcf_color = "#2ecc71" if margin_of_safety > 0 else "#e74c3c"
                    verdict = "Undervalued / Discounted" if margin_of_safety > 0 else "Overvalued / Premium"
                    
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03); border-left:4px solid {dcf_color}; padding:15px; border-radius:4px; margin-top: 10px;'>
                        <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;'>
                            <div>
                                <span style='color:#bbb; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;'>Intrinsic Value per Share</span><br>
                                <span style='font-size:2.5rem; font-weight:800; color:#fff;'>€{intrinsic_per_share:,.2f}</span>
                            </div>
                            <div style='text-align:right;'>
                                <span style='color:#bbb; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;'>Margin of Safety</span><br>
                                <span style='font-size:1.8rem; font-weight:800; color:{dcf_color};'>{margin_of_safety:+.1f}%</span><br>
                                <span style='font-size:0.9rem; color:{dcf_color}; font-weight:600;'>[{verdict}]</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Discount Rate (WACC) must be strictly greater than Terminal Growth Rate to converge.")
            else:
                st.info("⚠️ Insufficient Positive Free Cash Flow data to perform a reliable DCF Valuation.")

            # ── OWNERSHIP & SHORT SQUEEZE RISK ──────────────────────────────
            st.markdown("---")
            render_header("search", "Smart Money Flow & Short Squeeze Risk")
            
            inst_own = meta.get("inst_ownership", 0)
            insider_own = meta.get("insider_ownership", 0)
            
            inst_own = float(inst_own) if pd.notnull(inst_own) else 0.0
            insider_own = float(insider_own) if pd.notnull(insider_own) else 0.0
            public_own = max(0, 1.0 - inst_own - insider_own)
            
            short_pct = meta.get("short_percent_of_float", 0)
            short_pct = float(short_pct) if pd.notnull(short_pct) else 0.0
            short_ratio = meta.get("short_ratio", 0)
            short_ratio = float(short_ratio) if pd.notnull(short_ratio) else 0.0
            
            col_own1, col_own2 = st.columns([1, 1])
            with col_own1:
                labels = ['Institutions (Smart Money)', 'Insiders', 'Public/Retail Float']
                values = [inst_own, insider_own, public_own]
                colors = ['#00d2ff', '#3a7bd5', 'rgba(255,255,255,0.05)']
                
                fig_own = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.65)])
                fig_own.update_traces(hoverinfo='label+percent', textinfo='none', marker=dict(colors=colors, line=dict(color='#0d0e14', width=2)))
                fig_own.update_layout(
                    title=dict(text="Corporate Ownership Structure", font=dict(size=18)),
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                
                fig_own.add_annotation(text=f"{(inst_own+insider_own)*100:.1f}%<br><b>Locked</b>", x=0.5, y=0.5, font_size=20, showarrow=False)
                st.plotly_chart(fig_own, use_container_width=True)
                
            with col_own2:
                squeeze_color = "#e74c3c" if short_pct > 0.15 else "#f39c12" if short_pct > 0.05 else "#2ecc71"
                
                fig_short = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = short_pct * 100,
                    number = {'suffix': "%", 'font': {'size': 45, 'color': squeeze_color}},
                    title = {'text': "Short % of Float (Squeeze Risk)", 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [None, max(30, (short_pct*100)+5)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': squeeze_color},
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 5], 'color': "rgba(46, 204, 113, 0.15)"},
                            {'range': [5, 15], 'color': "rgba(243, 156, 18, 0.15)"},
                            {'range': [15, 100], 'color': "rgba(231, 76, 60, 0.15)"}],
                    }
                ))
                fig_short.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_short, use_container_width=True)
                
                st.markdown(f"<p style='text-align:center; color:#bbb; font-size:1rem;'>Short Ratio (Days to Cover): <b>{short_ratio:.1f} days</b></p>", unsafe_allow_html=True)


            # ── PEER COMPARISON ────────────────────────────────────────────
            st.markdown("---")
            render_header("package", f"Peer Comparison — {meta['sector']} Sector")

            # Get all peers in same sector (excluding indices + the stock itself)
            peer_companies = companies_full[
                (companies_full['sector'] == meta['sector']) &
                (~companies_full['ticker'].isin(indices_list)) &
                (companies_full['ticker'] != deep_ticker)
            ].copy()

            if not peer_companies.empty:
                # Merge with latest price to get RSI/signal for peers
                peer_prices = prices.sort_values('date').groupby('ticker').tail(1)[['ticker', 'price_close', 'rsi', 'ma_signal']]
                peer_df = peer_companies.merge(peer_prices, on='ticker', how='left')
                peer_df["upside_pct"] = (peer_df["target_mean_price"] / peer_df["price_close"] - 1) * 100

                # 6 comparison metrics
                metrics_cfg = [
                    ("P/E Ratio",        "pe_ratio",            False),  # lower is better
                    ("P/B Ratio",        "price_to_book",       False),
                    ("ROE (%)",          "roe",                 True,  100),   # higher is better
                    ("FCF Margin (%)",   "fcf_margin",          True),
                    ("Analyst Upside %", "upside_pct",          True),
                    ("Quality Score",    None,                  True),   # computed
                ]

                # Build a comparison dataframe
                rows = []
                # Add the selected stock first
                sel_row = meta.copy()
                sel_row_prices = prices[prices['ticker'] == deep_ticker].tail(1)
                if not sel_row_prices.empty:
                    sel_row['rsi'] = sel_row_prices.iloc[0]['rsi']
                    sel_row['ma_signal'] = sel_row_prices.iloc[0]['ma_signal']
                sel_row['upside_pct'] = upside
                sel_row['quality_score'] = compute_score(sel_row)

                rows.append({
                    'ticker': deep_ticker,
                    'company': meta['company'],
                    'pe_ratio': meta.get('pe_ratio'),
                    'price_to_book': meta.get('price_to_book'),
                    'roe_pct': (meta.get('roe') or 0) * 100,
                    'fcf_margin': meta.get('fcf_margin') or 0,
                    'upside_pct': upside,
                    'quality_score': compute_score(sel_row),
                    'is_selected': True
                })

                for _, pr in peer_df.iterrows():
                    pr_score_input = pr.copy()
                    pr_score_input['upside_pct'] = pr.get('upside_pct', 0) or 0
                    rows.append({
                        'ticker': pr['ticker'],
                        'company': pr.get('company', pr['ticker']),
                        'pe_ratio': pr.get('pe_ratio'),
                        'price_to_book': pr.get('price_to_book'),
                        'roe_pct': (pr.get('roe') or 0) * 100,
                        'fcf_margin': pr.get('fcf_margin') or 0,
                        'upside_pct': pr.get('upside_pct', 0) or 0,
                        'quality_score': compute_score(pr_score_input),
                        'is_selected': False
                    })

                comp_df = pd.DataFrame(rows).set_index('ticker')

                # Sector averages for reference line
                sector_avg = comp_df.mean(numeric_only=True)

                # Display as styled dataframe
                peer_table = comp_df[['company', 'pe_ratio', 'price_to_book', 'roe_pct', 'fcf_margin', 'upside_pct', 'quality_score']].copy()
                peer_table.columns = ['Company', 'P/E', 'P/B', 'ROE %', 'FCF%', 'Upside %', 'Quality']
                peer_table['P/E']  = peer_table['P/E'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
                peer_table['P/B']  = peer_table['P/B'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
                peer_table['ROE %'] = peer_table['ROE %'].apply(lambda x: f"{x:.1f}%")
                peer_table['FCF%'] = peer_table['FCF%'].apply(lambda x: f"{x:.1f}%")
                peer_table['Upside %'] = peer_table['Upside %'].apply(lambda x: f"{x:+.1f}%")


                st.dataframe(peer_table, use_container_width=True,
                             column_config={"Quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%d")})
            else:
                st.info(f"No peers found in the **{meta['sector']}** sector to compare with.")

            st.markdown("---")
            render_header("activity", f"Performance Alpha (Cumulative % vs SPY)")
            
            df_ticker_ret = df_deep.set_index('date')['price_close']
            df_spy_ret = spy_prices.set_index('date')['price_close']
            common_dates = df_ticker_ret.index.intersection(df_spy_ret.index)
            if not common_dates.empty:
                ticker_cum = (df_ticker_ret.loc[common_dates] / df_ticker_ret.loc[common_dates].iloc[0] - 1) * 100
                spy_cum = (df_spy_ret.loc[common_dates] / df_spy_ret.loc[common_dates].iloc[0] - 1) * 100
            else:
                ticker_cum = pd.Series()
                spy_cum = pd.Series()

            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=common_dates, y=ticker_cum, name=f"{deep_ticker} (%)", line=dict(color="#3498db", width=3)))
            fig_rel.add_trace(go.Scatter(x=common_dates, y=spy_cum, name="SPY (%)", line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dot")))
            fig_rel.update_layout(template="plotly_dark", height=450, yaxis_title="Return (%)", hovermode="x unified", margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig_rel, use_container_width=True)

            st.markdown("<div style='margin-top:35px; padding:6px 12px; background:rgba(255,255,255,0.03); border-left:4px solid #2ecc71; color:#2ecc71; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:1.5px;'>LAYER 5: PORTFOLIO IDEA MANAGEMENT</div>", unsafe_allow_html=True)
            # --- WATCHLIST QUICK SAVE WORKFLOW ---
            with st.expander("📥 📝 Save Idea to Watchlist Pipeline", expanded=False):
                with st.form(f"quick_save_form_{deep_ticker}"):
                    st.write("**Idea Management & Catalyst Tracking**")
                    _wl_col1, _wl_col2 = st.columns(2)
                    with _wl_col1:
                        # Auto-suggest status based on Logic
                        _s_index = 1 if act_str.startswith("🔥") or "ACCUMULATE" in act_str else 0
                        opt_status = st.selectbox("Status", ["🔵 PENDING", "🟢 ACTIVE", "🟡 REVIEW", "🔴 INVALIDATED", "⚫ CLOSED"], index=_s_index)
                        opt_thesis = st.text_area("Investment Thesis (Why buy/hold?)", value=act_desc, height=110)
                    with _wl_col2:
                        opt_catalyst = st.text_input("Upcoming Catalyst (Earnings, FDA, Macro, etc.)", placeholder="e.g. Q4 Earnings expected positive...")
                        
                        _kcol1, _kcol2, _kcol3 = st.columns(3)
                        with _kcol1: opt_entry = st.number_input("Entry (€)", value=float(_s1), step=1.0)
                        with _kcol2: opt_inval = st.number_input("Inval / Stop (€)", value=float(_stop_loss), step=1.0)
                        with _kcol3: opt_tp = st.number_input("Take Profit (€)", value=float(_tp1), step=1.0)

                        opt_erd = meta.get("next_earnings_date", "TBD")
                        if pd.isna(opt_erd): opt_erd = "TBD"
                        st.caption(f"Next Earnings: **{opt_erd}**")
                        
                    if st.form_submit_button("💾 Save Candidate to Watchlist", type="primary"):
                        try:
                            wl_df = load_watchlist()
                            # Delete existing to overwrite
                            wl_df = wl_df[wl_df["Ticker"] != deep_ticker]
                            
                            new_row = pd.DataFrame([{
                                "Ticker": deep_ticker,
                                "Status": opt_status,
                                "Thesis": opt_thesis,
                                "Catalyst": opt_catalyst,
                                "Entry Target": round(opt_entry, 2),
                                "Invalidation Level": round(opt_inval, 2),
                                "Take Profit": round(opt_tp, 2),
                                "Next Earnings": str(opt_erd),
                                "Added Date": pd.Timestamp.now().strftime("%Y-%m-%d")
                            }])
                            wl_df = pd.concat([wl_df, new_row], ignore_index=True)
                            save_watchlist(wl_df)
                            st.success(f"✅ Successfully added **{deep_ticker}** to Watchlist Pipeline!")
                        except Exception as e:
                            st.error(f"Error saving to watchlist: {e}")

            st.markdown("---")


# ── FEATURE 1.5: Correlation Matrix ──────────────────────────────────────────

# ── TAB 6: WATCHLIST PIPELINE ────────────────────────────────────────────────
if active_tab == "6. Watchlist":

    render_header("calendar", "Watchlist & Idea Pipeline")
    st.write("Track and prune your high-conviction ideas. A thesis without an invalidation level is just a gamble.")
    
    wl_df = load_watchlist()
    if wl_df.empty:
        st.info("Your watchlist is empty. Go to the **Decision Engine** to add your first candidate.")
    else:
        # Display summary metrics
        st.markdown("### Active Candidates Pipeline")
        _w1, _w2, _w3, _w4 = st.columns(4)
        _w1.metric("Total Ideas", len(wl_df))
        _w2.metric("Active (Triggered)", len(wl_df[wl_df["Status"].str.contains("ACTIVE")]))
        _w3.metric("Pending", len(wl_df[wl_df["Status"].str.contains("PENDING")]))
        _w4.metric("Invalidated", len(wl_df[wl_df["Status"].str.contains("INVALIDATED")]))
        st.markdown("---")
        
        # Interactive Editor
        config = {
            "Status": st.column_config.SelectboxColumn("Status", options=["🔵 PENDING", "🟢 ACTIVE", "🟡 REVIEW", "🔴 INVALIDATED", "⚫ CLOSED"], width="medium"),
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True, width="small"),
            "Entry Target": st.column_config.NumberColumn("Entry Target €", format="€%.2f"),
            "Invalidation Level": st.column_config.NumberColumn("Inval / Stop €", format="€%.2f"),
            "Take Profit": st.column_config.NumberColumn("TP Target €", format="€%.2f"),
            "Thesis": st.column_config.TextColumn("Thesis", width="large"),
            "Catalyst": st.column_config.TextColumn("Catalyst", width="medium"),
            "Next Earnings": st.column_config.TextColumn("Next Earnings", width="small")
        }
        
        with st.form("watchlist_editor_form"):
            st.caption("Double-click any cell to edit your notes, update Stop Loss levels or change Workflow Status. Click the trash icon to remove an idea.")
            
            # Type safety: Ensure text columns are explicitly strings before rendering in editor
            _wl_df_safe = wl_df.copy()
            for col in ["Thesis", "Catalyst", "Next Earnings"]:
                if col in _wl_df_safe.columns:
                    _wl_df_safe[col] = _wl_df_safe[col].fillna("").astype(str)

            edited_df = st.data_editor(
                _wl_df_safe,
                column_config=config,
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
                height=400
            )
            
            if st.form_submit_button("💾 Synchronize Watchlist Changes", type="primary"):
                try:
                    save_watchlist(edited_df)
                    st.success("✅ Watchlist synced successfully! Database updated.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to sync memory: {e}")

# ── TAB 7: PORTFOLIO MANAGEMENT ──────────────────────────────────────────────
if active_tab == "7. Portfolio Builder":
    render_header("package", "Professional Bulk Portfolio Suite", level="###")
    st.write("Craft your portfolio by selecting tickers and entering your holdings below. High-density quantitative analysis will follow.")

    # 1. LOCAL TICKER SELECTION
    all_available_tickers = sorted(prices["ticker"].unique().tolist())
    # Exclude indices for portfolio building
    indices = ["^VIX", "SPY", "^GSPC", "^DJI", "^IXIC"]
    stock_tickers = [t for t in all_available_tickers if t not in indices]
    
    # ── 1. PORTFOLIO PERSISTENCE (SUPABASE SYNC) ─────────────────────────────
    # Initial Load from Cloud DB
    if 'portfolio_db_synced' not in st.session_state:
        db_portfolio = load_portfolio_from_db()
        if db_portfolio:
            st.session_state.portfolio_tickers = sorted(list(db_portfolio.keys()))
            st.session_state.portfolio_shares = {t: db_portfolio[t]["shares"] for t in db_portfolio}
            st.session_state.portfolio_cost = {t: db_portfolio[t]["cost"] for t in db_portfolio}
        else:
            defaults = ["AAPL", "NVDA", "META"] # Safe defaults for new users
            st.session_state.portfolio_tickers = defaults
            st.session_state.portfolio_shares = {t: 10.0 for t in defaults}
            st.session_state.portfolio_cost = {t: 150.0 for t in defaults} # Placeholder defaults
            
        st.session_state.portfolio_db_synced = True

    p_tickers = st.multiselect(
        "Select Tickers for Portfolio Construction", 
        stock_tickers, 
        default=st.session_state.portfolio_tickers, 
        key="p_ticker_select"
    )

    # Detect UI Selection Changes
    if p_tickers != st.session_state.portfolio_tickers:
        st.session_state.portfolio_tickers = p_tickers
        # Retain old weights if ticker already existed, else initialize default 10 shares
        new_shares = {}
        new_cost = {}
        for t in p_tickers:
            new_shares[t] = st.session_state.portfolio_shares.get(t, 10.0)
            new_cost[t] = st.session_state.portfolio_cost.get(t, 0.0)
        st.session_state.portfolio_shares = new_shares
        st.session_state.portfolio_cost = new_cost

    if p_tickers:
        latest_prices = prices[prices["ticker"].isin(p_tickers)].groupby("ticker")["price_close"].last().to_dict()
        
        # Build Initial DataFrame for Editor (ONLY if tickers list actually changed or structure is missing/stale)
        if 'last_portfolio_tickers' not in st.session_state or \
           st.session_state.last_portfolio_tickers != p_tickers or \
           'portfolio_df' not in st.session_state or \
           "Cost Basis (€)" not in st.session_state.portfolio_df.columns or \
           "Region" not in st.session_state.portfolio_df.columns:
            st.session_state.last_portfolio_tickers = p_tickers
            init_data = []
            for t in p_tickers:
                # Enrich with m_df data for professional look
                meta = m_df[m_df["Ticker"] == t].iloc[0] if not m_df[m_df["Ticker"] == t].empty else {}
                init_data.append({
                    "Ticker": t,
                    "Company": meta.get("Company", t),
                    "Sector": meta.get("Sector", "N/A"),
                    "Region": meta.get("Region", "US"),
                    "Market Cap (B)": meta.get("MCap (B)", 0),
                    "Price (€)": latest_prices.get(t, 0),
                    "Shares": st.session_state.portfolio_shares.get(t, 10.0),
                    "Cost Basis (€)": st.session_state.portfolio_cost.get(t, latest_prices.get(t, 0))
                })
            st.session_state.portfolio_df = pd.DataFrame(init_data)
        
        # 2. BULK DATA EDITOR
        render_header("layers", "Capital Allocation Grid", level="#####")
        
        with st.form("portfolio_editor_form"):
            # KEY FIX: The data_editor should be the ONLY way to change weights for the current tickers
            edited_df = st.data_editor(
                st.session_state.portfolio_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                    "Company": st.column_config.TextColumn("Company", disabled=True),
                    "Region": st.column_config.TextColumn("Region", disabled=True),
                    "Price (€)": st.column_config.NumberColumn("Market Price", format="€%.2f", disabled=True),
                    "Shares": st.column_config.NumberColumn("Shares owned", min_value=0.0, step=0.01, format="%.2f"),
                    "Cost Basis (€)": st.column_config.NumberColumn("Avg Cost Basis", min_value=0.0, step=0.01, format="€%.2f")
                },
                hide_index=True,
                width="stretch",
                key="p_portfolio_editor_final"
            )
            
            # PASSIVE SYNC: Use a button to lock in changes and update Database
            recompute = st.form_submit_button("Save & Calculate", use_container_width=True, type="primary")

        if recompute:
            st.session_state.portfolio_df = edited_df.copy()
            shares_dict = edited_df.set_index("Ticker")["Shares"].to_dict()
            cost_dict = edited_df.set_index("Ticker")["Cost Basis (€)"].to_dict()
            st.session_state.portfolio_shares = shares_dict
            st.session_state.portfolio_cost = cost_dict
            # Upload to Supabase 
            save_portfolio_to_db(shares_dict, cost_dict)
            st.toast("☁️ Portfolio sync to Supabase Database successful!", icon="🚀")
            st.rerun()
        else:
            edited_df = st.session_state.portfolio_df.copy()
        
        # 3. WEIGHT & VALUE CALCULATION
        edited_df["Market Value"] = edited_df["Price (€)"] * edited_df["Shares"]
        total_p_val = edited_df["Market Value"].sum()
        


    
    # 3. WEIGHT & VALUE CALCULATION
    edited_df["Market Value"] = edited_df["Price (€)"] * edited_df["Shares"]
    total_p_val = edited_df["Market Value"].sum()
    
    if total_p_val > 0:
        edited_df["Weight (%)"] = (edited_df["Market Value"] / total_p_val) * 100
        weights = (edited_df["Market Value"] / total_p_val).values
        current_tickers = edited_df["Ticker"].tolist()
        weights = (edited_df["Market Value"] / total_p_val).values
        n_assets = len(current_tickers)

        # ── 4. PERFORMANCE ENGINE (Weighted) ──
        # Use filtered 'prices' to follow the global date filter
        p_prices = prices[prices["ticker"].isin(current_tickers)]
        ret_matrix = p_prices.pivot(index="date", columns="ticker", values="daily_return_pct").fillna(0) / 100
        # Ensure column order matches current_tickers for correct weighting
        ret_matrix = ret_matrix[current_tickers]
        
        # ── Pre-compute matrices for Optimizer & Analytics ──
        cov_matrix = ret_matrix.cov() * 252
        hist_rets  = ret_matrix.mean() * 252

        # Show Total Summary
        total_cost_basis = (edited_df["Cost Basis (€)"] * edited_df["Shares"]).sum()
        total_pnl = total_p_val - total_cost_basis
        pnl_pct = (total_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0

        port_daily = (ret_matrix * weights).sum(axis=1)
        cum_returns = (1 + port_daily).cumprod()

        # Risk Metrics
        risk_free = 0.04 / 252
        excess_returns = port_daily - risk_free
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_dd = drawdown.min() * 100
        vol = port_daily.std() * np.sqrt(252) * 100
        confidence_level = 0.05
        var_95 = np.percentile(port_daily, confidence_level * 100) * 100
        cvar_95 = port_daily[port_daily <= np.percentile(port_daily, confidence_level * 100)].mean() * 100

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 1 · PORTFOLIO HEALTH DASHBOARD
        # ═══════════════════════════════════════════════════════════════════
        render_header("activity", "Portfolio Health Dashboard")
        l1_left, l1_right = st.columns([1, 2])
        with l1_left:
            st.metric("Total Market Value", f"€{total_p_val:,.2f}")
            st.metric("Total Cost Basis",   f"€{total_cost_basis:,.2f}")
            st.metric("Overall PnL",         f"€{total_pnl:,.2f}", delta=f"{pnl_pct:.2f}%")
            st.markdown("<br>", unsafe_allow_html=True)

            # Risk tiles stacked vertically
            render_metric_tile("Weighted Return", f"{(cum_returns.iloc[-1]-1)*100:.1f}%", delta=(cum_returns.iloc[-1]-1)*100)
            st.caption(f"Timeframe: {selected_horizon}")
            if sharpe > 2.0: s_label, s_color = "💎 ELITE", "#00ffcc"
            elif sharpe > 1.5: s_label, s_color = "🟢 STRONG", "#2ecc71"
            elif sharpe > 1.0: s_label, s_color = "🟡 OK", "#f1c40f"
            else: s_label, s_color = "🔴 POOR", "#e74c3c"
            render_metric_tile("Sharpe Ratio", f"{sharpe:.2f} · {s_label}", help_text="< 1.0 Poor | 1.0–1.5 Acceptable | 1.5–2.0 Strong | > 2.0 Elite")
            render_metric_tile("Max Drawdown",  f"{max_dd:.1f}%")
            render_metric_tile("Annual Vol",    f"{vol:.1f}%")
            render_metric_tile("VaR (95%)",     f"{var_95:.2f}%")
            render_metric_tile("CVaR (95%)",    f"{cvar_95:.2f}%")

        with l1_right:
            # ── Benchmark Growth Simulation ──────────────────────────────
            render_header("chart", "Growth Simulation vs Benchmark", level="#####")
            bench_options = {
                "S&P 500 (SPY)": "SPY",
                "Nasdaq 100 (QQQ)": "QQQ",
                "DAX 40 (^GDAXI)": "^GDAXI",
                "MSCI World (IWDA.AS)": "IWDA.AS"
            }
            sel_bench_label = st.selectbox("Select Performance Benchmark", options=list(bench_options.keys()), index=0, key="bench_l1")
            sel_bench_ticker = bench_options[sel_bench_label]

            initial_investment = 10000
            backtest_df = pd.DataFrame({'date': cum_returns.index, 'cum_return': cum_returns.values})
            backtest_df["portfolio_value"] = backtest_df["cum_return"] * initial_investment

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=backtest_df["date"], y=backtest_df["portfolio_value"],
                name="Your Portfolio", line=dict(color="#00ffcc", width=3)
            ))

            bench_prices = prices_full[
                (prices_full["ticker"] == sel_bench_ticker) &
                (prices_full["date"] >= t_start) &
                (prices_full["date"] <= t_end)
            ].sort_values("date")
            if not bench_prices.empty:
                bench_prices = bench_prices[bench_prices["date"].isin(cum_returns.index)]
                if not bench_prices.empty and bench_prices["daily_return_pct"].notna().any():
                    bench_daily = bench_prices["daily_return_pct"].fillna(0) / 100
                    bench_cum   = (1 + bench_daily).cumprod()
                    bench_cum   = bench_cum / bench_cum.iloc[0]
                    bench_prices = bench_prices.copy()
                    bench_prices["bench_value"] = bench_cum.values * initial_investment
                    fig_bt.add_trace(go.Scatter(
                        x=bench_prices["date"], y=bench_prices["bench_value"],
                        name=sel_bench_label, line=dict(color="#f1c40f", width=2, dash="dot")
                    ))
            fig_bt.update_layout(
                template="plotly_dark", height=500,
                yaxis_title="Value (€)", margin=dict(t=10, l=10, r=10, b=10)
            )
            st.plotly_chart(fig_bt, use_container_width=True)

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 2 · STRUCTURAL DIAGNOSIS
        # ═══════════════════════════════════════════════════════════════════
        render_header("globe", "Structural Diagnosis — Exposure & Correlation")
        l2c1, l2c2, l2c3 = st.columns(3)
        with l2c1:
            render_header("globe", "Geographic Exposure", level="#####")
            reg_dist = edited_df.groupby("Region")["Market Value"].sum().reset_index()
            fig_reg = px.pie(reg_dist, values='Market Value', names='Region', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_reg.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=10,b=10),
                                  legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
            st.plotly_chart(fig_reg, use_container_width=True)

        with l2c2:
            render_header("layers", "Thematic Exposure (Sector)", level="#####")
            sec_dist = edited_df.groupby("Sector")["Market Value"].sum().reset_index()
            fig_sec = px.pie(sec_dist, values='Market Value', names='Sector', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Safe)
            fig_sec.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=10,b=10),
                                  legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
            st.plotly_chart(fig_sec, use_container_width=True)

        with l2c3:
            render_header("activity", "Asset Correlation Matrix", level="#####")
            corr_matrix = ret_matrix.corr()
            mean_corr = (corr_matrix.values.sum() - n_assets) / (n_assets**2 - n_assets) if n_assets > 1 else 0
            if mean_corr > 0.45:
                st.warning(f"⚠️ High Correlation ({mean_corr:.2f}) — risk concentration!")
            fig_corr = px.imshow(
                corr_matrix, text_auto=".2f",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                template="plotly_dark", aspect="auto"
            )
            fig_corr.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # LAYER 3 · STRATEGIC REBALANCING & OPTIMIZATION
        # ═══════════════════════════════════════════════════════════════════
        # ── 4.6. AI REBALANCING COMMAND CENTER (PREMIUM CARD GRID) ──────────────
        render_header("ai", "Institutional Rebalancing Command Center")
        
        # Use chunks for grid layout (ULTRA-DENSE: 6 per row)
        n_cols = 6
        tickers_list = edited_df.to_dict('records')
        
        for i in range(0, len(tickers_list), n_cols):
            cols = st.columns(n_cols)
            chunk = tickers_list[i : i + n_cols]
            
            for idx, row in enumerate(chunk):
                t = row["Ticker"]
                w = row["Weight (%)"]
                
                # Fetch AI target from reco_df
                ai_meta = reco_df[reco_df["ticker"] == t].iloc[0] if not reco_df[reco_df["ticker"] == t].empty else None
                
                if ai_meta is not None:
                    ai_score = ai_meta["score"]
                    upside = ai_meta["upside_pct"]

                    # ── Read action from the shared m_df source ──────────────
                    status = _action_map.get(t, "HOLD / NEUTRAL")

                    # Derive display color from canonical action label
                    if status == "STRONG BUY":          color = "#00ffcc"; border = "2px solid #00ffcc"
                    elif "BUY" in status:               color = "#2ecc71"; border = "1px solid #2ecc71"
                    elif "SELL" in status:              color = "#ff4b4b"; border = "2px solid #ff4b4b"
                    elif "REDUCE" in status:            color = "#e67e22"; border = "1px solid #e67e22"
                    else:                               color = "#3498db"; border = "1px solid rgba(255,255,255,0.1)"

                    reason = f"Quality score {ai_score}"
                    if upside > 10: reason = f"Upside potential (+{upside:.1f}%)"
                    if w > 20: reason = "Risk concentration limit exceeded"

                    with cols[idx]:
                        st.markdown(f"""
                        <div style='background:rgba(255,255,255,0.02); border:{border}; border-radius:5px; padding:6px; margin-bottom:4px;'>
                            <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                                <span style='font-size:0.8rem; font-weight:800; color:{color};'>{t}</span>
                                <span style='background:{color}22; color:{color}; padding:1px 4px; border-radius:2px; font-size:0.45rem; font-weight:700;'>{status}</span>
                            </div>
                            <div style='display:grid; grid-template-columns: 1fr 1fr; gap:4px; margin-bottom:4px;'>
                                <div><div style='color:#777; font-size:0.45rem; text-transform:uppercase;'>WGT</div><div style='font-size:0.75rem; font-weight:700;'>{w:.1f}%</div></div>
                                <div><div style='color:#777; font-size:0.45rem; text-transform:uppercase;'>SCORE</div><div style='font-size:0.75rem; font-weight:700;'>{ai_score}</div></div>
                            </div>
                            <div style='color:#666; font-size:0.55rem; border-top:1px solid rgba(255,255,255,0.05); padding-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>
                                {reason}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with cols[idx]:
                        st.markdown(f"""
                        <div style='background:rgba(255,255,255,0.01); border:1px dashed rgba(255,255,255,0.08); border-radius:5px; padding:6px; text-align:center;'>
                            <div style='color:#555; font-size:0.5rem;'>{t}</div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── 4.7. REBALANCING OPTIMIZER ───────────────────────────────────────────
        st.markdown("### 📊 Portfolio Rebalancing Hub")
        
        with st.expander("Institutional Rebalancing Protocol & Rulebook", expanded=False):
            st.markdown("""
            **1. Security Assessment Construct (5-Pillar Matrix)**  
            The analytical engine issues tactical recommendations based on a composite score derived from 5 independent pillars: Technical Trend, AI Quality, Sector-weighted Valuation, Volatility Risk, and Support/Resistance R/R.
            * **STRONG BUY:** The security achieves optimal alignment across all quantitative pillars. It exhibits elite fundamental quality coupled with highly favorable Risk/Reward metrics. Represents an ideal entry zone.
            * **BUY / ACCUMULATE:** Strong underlying fundamentals and robust long-term signals, though potentially undergoing short-term consolidation. Suitable for progressive accumulation.
            * **HOLD / NEUTRAL:** Mixed signals or lack of clear directional advantage. This also applies to elite assets currently trading at premium multiples (overbought). Capital allocation should be deferred pending a structural pullback.
            * **REDUCE / UNDERPERFORM:** Asset is technically overextended (RSI > 70) yielding elevated tactical risk. Recommends partial profit-taking to mitigate impending mean reversion.
            * **SELL / AVOID:** Significant deterioration in technical trends and poor profitability metrics. High probability of capital depreciation. Focus shifts to capital preservation.

            **2. Portfolio Strategy Optimization (Modern Portfolio Theory)**  
            * **Minimum Volatility:** Prioritizes capital preservation by overwriting cap-weights with a mathematical minimization of portfolio variance. It actively strips out high-beta components. **Application:** Systemic risk spikes, macroeconomic distress, or defensive posturing.
            * **Risk Parity:** Discards market capitalization entirely. Allocates capital such that the *marginal risk contribution* of each asset forms an equal slice of the total portfolio risk. **Application:** Core long-term portfolio structuring (e.g., All-Weather framework), ensuring no single asset dictates volatility.
            * **Equal Weight:** A disciplined 1/N allocation scaling. Functionally enforces buying low and selling high during rebalancing cycles. **Application:** Mitigating concentration risk in cap-weighted indices (e.g., extreme mega-cap tech dominance) and maximizing broad diversification.
            * **Max Sharpe (Optimal MPT):** Implements Markowitz Mean-Variance Optimization. Locates the exact tangency portfolio on the Efficient Frontier, mathematically yielding the maximum return per unit of volatility. **Application:** Standard bullish to neutral market environments demanding optimal risk-adjusted growth.
            * **Maximum Return:** Agnostic to portfolio variance. Hyper-concentrates capital into the assets demonstrating the highest historical momentum and largest expected returns. **Application:** Aggressive short-term tactical plays during high-conviction momentum rallies.
            """)

        if 'pending_optimization' not in st.session_state:
            st.session_state.pending_optimization = None
        if 'pending_opt_strategy' not in st.session_state:
            st.session_state.pending_opt_strategy = None

        # ── Strategy Controls ──────────────────────────────────────────────
        strat_col, min_w_col, _ = st.columns([2, 1, 1])
        with strat_col:
            strategy_options = {
                "🛡️ Minimum Volatility (Lowest Risk)":   "min_vol",
                "⚖️ Risk Parity (Strategic Balance)":   "risk_parity",
                "🌐 Equal Weight (Max Diversification)": "equal_weight",
                "🚀 Max Sharpe (Risk-Adjusted Growth)":  "max_sharpe",
                "🎯 Maximum Return (Highest Growth)":    "max_return",
            }
            sel_strategy_label = st.selectbox(
                "Optimization Strategy",
                options=list(strategy_options.keys()), index=2,
                help="Choose how to distribute capital: from lowest risk (Min Vol) to highest growth (Max Return)."
            )
            sel_strategy = strategy_options[sel_strategy_label]

        with min_w_col:
            min_weight_pct = st.slider(
                "Min Weight / Ticker (%)",
                min_value=0, max_value=10, value=2, step=1,
                help="Floor constraint: no ticker will be weighted below this level. Prevents the optimizer from fully selling out a position."
            )
            min_w = min_weight_pct / 100.0

        # ── Strategy Descriptions ──────────────────────────────────────────
        _strategy_descriptions = {
            "min_vol": (
                "**🛡️ Minimum Volatility** — Finds the allocation with the **lowest possible portfolio variance**, "
                "regardless of expected returns. Ideal for capital preservation and bear market defense. "
                "Widely used by pension funds and the MSCI Minimum Volatility Index family."
            ),
            "risk_parity": (
                "**⚖️ Risk Parity** — Allocates capital so each asset contributes **equally** to total portfolio risk. "
                "Pioneer strategy used by Ray Dalio (Bridgewater) for the 'All Weather' portfolio. "
                "High-volatility assets receive less capital; stable assets receive more."
            ),
            "equal_weight": (
                "**🌐 Equal Weight (1/N)** — Splits capital evenly across all holdings. Simple yet powerful "
                "diversification popularized by the S&P 500 Equal Weight Index (RSP). "
                "Avoids the estimation errors often found in complex mathematical models."
            ),
            "max_sharpe": (
                "**🚀 Max Sharpe (Markowitz MVO)** — Finds the allocation that maximizes return per unit of risk "
                "(Sharpe Ratio). Based on Harry Markowitz's Nobel Prize-winning theory. Best for risk-adjusted "
                "growth but results tend to be concentrated in top-performing assets."
            ),
            "max_return": (
                "**🎯 Maximum Return** — Maximizes expected annual return with no regard for volatility. "
                "A high-conviction, aggressive strategy favored by George Soros and Stanley Druckenmiller: "
                "'To make superior returns, concentrate on what you are most right about.' **Use with caution.**"
            ),
        }
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.03); padding:10px 14px; border-radius:8px; "
            f"border-left:3px solid #00d4ff; margin-bottom:12px; font-size:0.83rem;'>"
            f"{_strategy_descriptions[sel_strategy]}</div>",
            unsafe_allow_html=True
        )

        # ── Core Optimization Functions ────────────────────────────────────
        def _run_max_sharpe(hist_rets, cov_matrix, n_assets, min_w):
            """Maximize Sharpe Ratio (Markowitz MVO) with per-asset floor constraint."""
            def portfolio_stats(w):
                p_ret = np.sum(hist_rets.values * w)
                p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                p_sharpe = (p_ret - 0.04) / p_vol if p_vol > 0 else 0
                return p_ret, p_vol, p_sharpe

            # Ensure floor doesn't exceed 1/n (prevent infeasibility)
            floor = min(min_w, 0.9 / n_assets)
            cap = min(0.40, 1.0 - floor * (n_assets - 1))
            bounds = tuple((floor, cap) for _ in range(n_assets))
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            ]
            init_w = np.array([1.0 / n_assets] * n_assets)
            result = minimize(lambda w: -portfolio_stats(w)[2], init_w,
                              method='SLSQP', bounds=bounds, constraints=constraints)
            return result.x if result.success else init_w

        def _run_risk_parity(cov_matrix, n_assets, min_w):
            """Risk Parity: equalize marginal risk contribution of each asset."""
            def risk_contributions(w, cov):
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                marginal  = np.dot(cov, w) / port_vol
                contrib   = w * marginal
                return contrib

            def rp_objective(w):
                rc = risk_contributions(w, cov_matrix.values)
                target = 1.0 / n_assets
                return np.sum((rc / rc.sum() - target) ** 2)

            floor = min(min_w, 0.9 / n_assets)
            bounds = tuple((floor, 1.0) for _ in range(n_assets))
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            init_w = np.array([1.0 / n_assets] * n_assets)
            result = minimize(rp_objective, init_w, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'ftol': 1e-10, 'maxiter': 1000})
            return result.x if result.success else init_w

        def _run_equal_weight(n_assets):
            """Equal Weight (1/N): simple uniform allocation."""
            return np.array([1.0 / n_assets] * n_assets)

        def _run_min_vol(cov_matrix, n_assets, min_w):
            """Minimum Volatility: minimize portfolio standard deviation."""
            floor = min(min_w, 0.9 / n_assets)
            cap = min(0.40, 1.0 - floor * (n_assets - 1))
            bounds = tuple((floor, cap) for _ in range(n_assets))
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            init_w = np.array([1.0 / n_assets] * n_assets)

            def port_vol(w):
                return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

            result = minimize(port_vol, init_w, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'ftol': 1e-12, 'maxiter': 1000})
            return result.x if result.success else init_w

        def _run_max_return(hist_rets, n_assets, min_w):
            """Maximum Return: maximize expected annual return (ignores volatility)."""
            # Simple analytical solution: concentrate on highest-return assets
            floor = min(min_w, 0.9 / n_assets)
            cap   = min(0.40, 1.0 - floor * (n_assets - 1))
            # Sort assets by expected return descending
            sorted_idx = np.argsort(hist_rets.values)[::-1]
            w = np.full(n_assets, floor)
            remaining = 1.0 - floor * n_assets
            for i in sorted_idx:
                alloc = min(cap - floor, remaining)
                w[i] += alloc
                remaining -= alloc
                if remaining <= 1e-9:
                    break
            return w

        # ── Action Buttons ─────────────────────────────────────────────────
        act_col1, act_col2 = st.columns([1, 1])
        with act_col1:
            if st.button("🚀 GENERATE OPTIMAL REBALANCE", use_container_width=True, type="primary"):
                try:
                    if sel_strategy == "min_vol":
                        opt_weights = _run_min_vol(cov_matrix, n_assets, min_w)
                    elif sel_strategy == "risk_parity":
                        opt_weights = _run_risk_parity(cov_matrix, n_assets, min_w)
                    elif sel_strategy == "equal_weight":
                        opt_weights = _run_equal_weight(n_assets)
                    elif sel_strategy == "max_sharpe":
                        opt_weights = _run_max_sharpe(hist_rets, cov_matrix, n_assets, min_w)
                    else:  # max_return
                        opt_weights = _run_max_return(hist_rets, n_assets, min_w)

                    # Build comparison table
                    comparison_data = []
                    for idx, ticker in enumerate(current_tickers):
                        price   = latest_prices.get(ticker, 1)
                        curr_w  = weights[idx] * 100
                        rec_w   = opt_weights[idx] * 100
                        curr_s  = st.session_state.portfolio_shares.get(ticker, 0)
                        rec_s   = (opt_weights[idx] * total_p_val) / price
                        delta_s = rec_s - curr_s
                        action  = "HOLD"
                        if delta_s >  0.1: action = "BUY"
                        elif delta_s < -0.1: action = "SELL"
                        comparison_data.append({
                            "Ticker":            ticker,
                            "Current Weight %":  curr_w,
                            "Optimal Weight %":  rec_w,
                            "Current Shares":    curr_s,
                            "Optimal Shares":    rec_s,
                            "Action":            action,
                            "Delta Shares":      delta_s,
                            "Est. Value (€)":    delta_s * price,
                        })

                    st.session_state.pending_optimization = pd.DataFrame(comparison_data)
                    st.session_state.pending_opt_strategy = sel_strategy_label
                    st.rerun()

                except Exception as e:
                    st.error(f"Optimization failed: {e}")

        with act_col2:
            pass
            # csv = edited_df.to_csv(index=False).encode('utf-8')
            # st.download_button(
            #     label="📥 DOWNLOAD PORTFOLIO (CSV)",
            #     data=csv,
            #     file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            #     mime="text/csv",
            #     use_container_width=True
            # )

        # ── Display Suggestions ────────────────────────────────────────────
        if st.session_state.pending_optimization is not None:
            st.markdown("---")
            _used_strategy = st.session_state.pending_opt_strategy or "Unknown Strategy"
            st.info(f"🎯 **Suggested Rebalancing · Strategy: {_used_strategy}**")

            # ── Expected metrics after rebalancing ──────────────────────
            _opt_w_arr = np.array(st.session_state.pending_optimization["Optimal Weight %"].values) / 100
            try:
                _exp_ret  = np.sum(hist_rets.values * _opt_w_arr) * 100
                _exp_vol  = np.sqrt(np.dot(_opt_w_arr.T, np.dot(cov_matrix, _opt_w_arr))) * 100
                _exp_srp  = (_exp_ret/100 - 0.04) / (_exp_vol/100) if _exp_vol > 0 else 0
                _curr_ret = np.sum(hist_rets.values * weights) * 100
                _curr_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * 100
                _curr_srp = (_curr_ret/100 - 0.04) / (_curr_vol/100) if _curr_vol > 0 else 0

                _m1, _m2, _m3, _m4 = st.columns(4)
                with _m1: render_metric_tile("Expected Annual Return", f"{_exp_ret:.1f}%", delta=_exp_ret - _curr_ret)
                with _m2: render_metric_tile("Expected Annual Vol",    f"{_exp_vol:.1f}%")
                with _m3: render_metric_tile("Expected Sharpe",        f"{_exp_srp:.2f}", delta=_exp_srp - _curr_srp)
                with _m4: render_metric_tile("Min Weight Floor",       f"{min_weight_pct}%")
                st.markdown("<br>", unsafe_allow_html=True)
            except Exception:
                pass  # metrics are optional - skip on error

            # ── Action summary bar ──────────────────────────────────────
            _sell_count = (st.session_state.pending_optimization["Action"] == "SELL").sum()
            _buy_count  = (st.session_state.pending_optimization["Action"] == "BUY").sum()
            _hold_count = (st.session_state.pending_optimization["Action"] == "HOLD").sum()
            st.markdown(
                f"<div style='font-size:0.82rem; margin-bottom:8px;'>Summary: "
                f"<span style='color:#2ecc71; font-weight:700;'>▲ {_buy_count} BUY</span> &nbsp;|&nbsp; "
                f"<span style='color:#3498db; font-weight:700;'>— {_hold_count} HOLD</span> &nbsp;|&nbsp; "
                f"<span style='color:#e74c3c; font-weight:700;'>▼ {_sell_count} SELL</span></div>",
                unsafe_allow_html=True
            )

            # ── Rebalancing table ───────────────────────────────────────
            st.dataframe(
                st.session_state.pending_optimization,
                column_config={
                    "Current Weight %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Optimal Weight %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Current Shares":   st.column_config.NumberColumn(format="%.2f"),
                    "Optimal Shares":   st.column_config.NumberColumn(format="%.2f"),
                    "Delta Shares":     st.column_config.NumberColumn(format="%+.2f"),
                    "Est. Value (€)":   st.column_config.NumberColumn(format="€%+.2f"),
                    "Action":           st.column_config.TextColumn("Action"),
                },
                hide_index=True, use_container_width=True
            )

            # ── DISCARD / APPLY buttons ────────────────────────────────
            sc1, sc2, sc3 = st.columns([2, 1, 1])
            with sc2:
                if st.button("❌ DISCARD", use_container_width=True):
                    st.session_state.pending_optimization = None
                    st.session_state.pending_opt_strategy = None
                    st.rerun()
            with sc3:
                if st.button("✅ APPLY REBALANCE", use_container_width=True, type="primary"):
                    new_shares_dict = st.session_state.pending_optimization.set_index("Ticker")["Optimal Shares"].to_dict()
                    st.session_state.portfolio_shares = new_shares_dict
                    for idx, row in st.session_state.portfolio_df.iterrows():
                        st.session_state.portfolio_df.at[idx, "Shares"] = new_shares_dict.get(row["Ticker"], 0)
                    st.session_state.pending_optimization = None
                    st.session_state.pending_opt_strategy = None
                    st.toast("✅ Portfolio updated to suggested optimal weights!", icon="🎯")
                    st.rerun()

        st.markdown("---")

        # ── 5. ADVANCED ANALYTICS (Efficient Frontier & Risk) ──────────
        if len(current_tickers) > 1:
            render_header("activity", "Markowitz Efficient Frontier — Strategy Tactical Map")

            curr_r = np.sum(hist_rets.values * weights)
            curr_v = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            curr_sharpe = (curr_r - 0.04) / curr_v if curr_v > 0 else 0

            # ── Efficient Frontier (Monte Carlo simulation background) ────
            st.info(
                "ℹ️ Each dot represents a randomly weighted portfolio. "
                "The **5 labeled markers** show where each optimization strategy lands on the risk/return map. "
                "Your current portfolio ★ reveals which strategy style you are closest to."
            )
            n_sims  = 2000
            sim_res = np.zeros((3, n_sims))
            for i in range(n_sims):
                w_rnd  = np.random.dirichlet(np.ones(n_assets))
                r_rnd  = np.sum(hist_rets.values * w_rnd)
                v_rnd  = np.sqrt(np.dot(w_rnd.T, np.dot(cov_matrix, w_rnd)))
                sim_res[0, i] = v_rnd
                sim_res[1, i] = r_rnd
                sim_res[2, i] = (r_rnd - 0.04) / v_rnd if v_rnd > 0 else 0

            fig_mpt = go.Figure()

            # ── Background: Monte Carlo cloud ─────────────────────────────
            fig_mpt.add_trace(go.Scatter(
                x=sim_res[0, :], y=sim_res[1, :], mode="markers",
                marker=dict(
                    color=sim_res[2, :], colorscale="Viridis",
                    showscale=True, size=4, opacity=0.25,
                    colorbar=dict(title="Sharpe", x=1.02)
                ),
                name="Simulated Portfolios",
                hovertemplate="Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
            ))

            # ── Compute & plot The Big 5 strategies ───────────────────────
            _default_min_w = max(0.02, 1.0 / (n_assets * 5))  # sensible floor for EF display
            _big5 = []
            try:
                _big5 = [
                    {
                        "name":   "🛡️ Min Volatility",
                        "color":  "#3498db",
                        "symbol": "diamond",
                        "w":      _run_min_vol(cov_matrix, n_assets, _default_min_w),
                        "desc":   "Markowitz / MSCI Min Vol Index",
                    },
                    {
                        "name":   "🚀 Max Sharpe",
                        "color":  "#2ecc71",
                        "symbol": "square",
                        "w":      _run_max_sharpe(hist_rets, cov_matrix, n_assets, _default_min_w),
                        "desc":   "Harry Markowitz — Nobel Prize MVO",
                    },
                    {
                        "name":   "🎯 Max Return",
                        "color":  "#e67e22",
                        "symbol": "triangle-up",
                        "w":      _run_max_return(hist_rets, n_assets, _default_min_w),
                        "desc":   "Soros / Druckenmiller — Aggressive Growth",
                    },
                    {
                        "name":   "⚖️ Risk Parity",
                        "color":  "#9b59b6",
                        "symbol": "cross",
                        "w":      _run_risk_parity(cov_matrix, n_assets, _default_min_w),
                        "desc":   "Ray Dalio — Bridgewater All Weather",
                    },
                    {
                        "name":   "🌐 Equal Weight",
                        "color":  "#bdc3c7",
                        "symbol": "circle",
                        "w":      _run_equal_weight(n_assets),
                        "desc":   "S&P 500 Equal Weight (RSP) — 1/N Rule",
                    },
                ]
            except Exception:
                pass  # skip if optimizer fails (e.g. single asset)

            for strat in _big5:
                w_s = strat["w"]
                r_s = np.sum(hist_rets.values * w_s)
                v_s = np.sqrt(np.dot(w_s.T, np.dot(cov_matrix, w_s)))
                sh_s = (r_s - 0.04) / v_s if v_s > 0 else 0
                fig_mpt.add_trace(go.Scatter(
                    x=[v_s], y=[r_s],
                    mode="markers+text",
                    marker=dict(
                        color=strat["color"], size=16,
                        symbol=strat["symbol"],
                        line=dict(color="white", width=1.5)
                    ),
                    text=[strat["name"]],
                    textposition="top center",
                    textfont=dict(size=10, color=strat["color"]),
                    name=strat["name"],
                    hovertemplate=(
                        f"<b>{strat['name']}</b><br>"
                        f"{strat['desc']}<br>"
                        "Vol:    %{x:.1%}<br>"
                        "Return: %{y:.1%}<br>"
                        f"Sharpe: {sh_s:.2f}"
                        "<extra></extra>"
                    ),
                ))

            # ── Current portfolio star ─────────────────────────────────────
            fig_mpt.add_trace(go.Scatter(
                x=[curr_v], y=[curr_r], mode="markers+text",
                marker=dict(color="#e74c3c", size=20, symbol="star",
                            line=dict(color="white", width=2)),
                text=[f"YOUR PORTFOLIO<br>Sharpe {curr_sharpe:.2f}"],
                textposition="top center",
                textfont=dict(size=10, color="#e74c3c"),
                name="★ Current Portfolio"
            ))
            fig_mpt.update_layout(
                template="plotly_dark", height=580,
                xaxis_title="Annual Volatility (Risk)",
                yaxis_title="Annual Historical Return",
                xaxis=dict(tickformat=".0%"),
                yaxis=dict(tickformat=".0%"),
                margin=dict(t=40, b=120, l=10, r=60),
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=-0.28,
                    xanchor="center", x=0.45,
                    font=dict(size=11),
                    itemsizing="constant",
                    bgcolor="rgba(0,0,0,0.3)",
                    bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1,
                ),
                annotations=[dict(
                    text="← Lower Risk          Higher Return →",
                    xref="paper", yref="paper",
                    x=0.0, y=1.03, showarrow=False,
                    font=dict(size=10, color="#666"),
                    align="left"
                )],
            )
            st.plotly_chart(fig_mpt, use_container_width=True)

            st.markdown("---")
            # ── Risk Contribution ─────────────────────────────────────────
            render_header("risk", "Global Risk Contribution", level="#####")
            mctr         = np.dot(cov_matrix, weights) / (curr_v if curr_v > 0 else 1)
            risk_contrib = weights * mctr
            risk_pct     = risk_contrib / np.sum(np.abs(risk_contrib)) * 100

            fig_risk_b = px.bar(
                x=current_tickers, y=risk_pct,
                labels={"x": "Ticker", "y": "Risk Contribution (%)"},
                template="plotly_dark",
                color=risk_pct, color_continuous_scale="Reds"
            )
            fig_risk_b.update_layout(height=380)
            st.plotly_chart(fig_risk_b, use_container_width=True)


        else:
            st.warning("⚠️ Total portfolio value is 0. Please enter the number of shares owned to activate the analysis.")
    else:
        st.info("🎯 Start by selecting tickers at the top to build your institutional-grade portfolio.")


        st.markdown("---")

        # 6. ── FEATURE 7: Alert Configurator (Moved Here) ──
        render_header("activity", "Dynamic Alert Center", level="###")
        with st.form("alert_form"):
            colX, colY, colZ = st.columns(3)
            with colX: a_ticker = st.selectbox("Ticker", all_tickers, format_func=format_ticker)
            with colY: a_metric = st.selectbox("Metric", ["Price", "Volume", "Daily Return %", "RSI"])
            with colZ: a_condition = st.selectbox("Condition", ["above", "below"])
            a_value = st.number_input("Threshold Value", value=100.0)
            # ── Standardized Common Ratings ──
            if ai_score > 80: 
                action_label = "💎 STRONG BUY"
                action_color = "#2ecc71"
            elif ai_score > 65: 
                action_label = "🟢 BUY / ACCUMULATE"
                action_color = "#27ae60"
            elif ai_score > 45: 
                action_label = "🟡 HOLD / NEUTRAL"
                action_color = "#f1c40f"
            elif ai_score > 35: 
                action_label = "🟠 REDUCE / UNDERPERFORM"
                action_color = "#e67e22"
            else: 
                action_label = "🔴 SELL / AVOID"
                action_color = "#e74c3c"
            a_email = st.text_input("Notify Email", value="dgl.rocketmail94@gmail.com")
            submitted = st.form_submit_button("Deploy Alert Rule")
            if submitted:
                st.toast(f"Alert rule created for {a_ticker}!")
                st.success(f"✅ Rule saved: If **{a_ticker} {a_metric}** is **{a_condition} {a_value}**, notify **{a_email}**.")

# ── FEATURE 3: AI Price & Monte Carlo Forecasting ────────────────────────────



            





# ── TAB: MARKET SCANNER & OPPORTUNITY RADAR ──────────────────────────────────
if active_tab == "2. Opportunity Radar":
    render_header("search", "Market Scanner & Opportunity Radar", level="###")
    st.info("📊 **Quantitative Engine Note:** All scoring systems, filters, and opportunity signals in this tab are generated entirely through **Hard-coded Quantitative Mathematics** (evaluating Quality, Momentum, and Valuation metrics). They **do not** factor in Qualitative AI Sentiment Reading or NLP Analysis.")
    st.write("Scan the entire ticker universe for institutional-grade opportunities based on Valuation, Momentum, and Quality Scores.")
    m_df = get_master_screener_data(companies_full, prices_full, quarterly_fin, annual_fin)

    
    # ── Quick Filter Modes (High-Fidelity Redesign) ────────────────────────────
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: rgba(255, 255, 255, 0.05);
            color: #ccc;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 10px 20px;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        div.stButton > button:hover {
            border-color: #0668E1;
            color: white;
            background-color: rgba(6, 104, 225, 0.1);
            transform: translateY(-2px);
        }
        div.stButton > button:active {
            background-color: #0668E1;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    # Final Compact Dropdown Layout (Removed Redundant Reset Button)
    scan_presets = [
        "🔍 All Stock Universe",
        "🏆 Institutional Pulse (Quality > 75 & Bullish)",
        "📈 Trend Following (MA20 > MA50)",
        "📉 RSI Mean Reversion (Oversold < 30)",
        "🚀 Buy on Dip (Bullish + Oversold)",
        "⚡ Multi-Indicator Breakout (Bullish + RSI > 50)",
        "──────────────────────────────",
        "⚠️ Structural Caution (Quality < 40 & Bearish)",
        "📉 Negative Momentum (MA20 < MA50)",
        "🔥 Overbought Alert (> 70)",
        "🎈 Valuation Exhaustion (Z-Score > +2.0)",
        "⚔️ Exit on Strength (Bearish + Overbought)",
        "💔 Multi-Indicator Breakdown (Bearish + RSI < 50)"
    ]
    
    # Initialize session state for scan mode if not exists
    if 'scan_mode' not in st.session_state:
        st.session_state.scan_mode = scan_presets[0]
        
    # ── Applied Logic (Synced with Backtest Engine) ───────────────────────────
    st.markdown("#### Geographic, Sector & Strategy Filters")
    r_col1, r_col2, r_col3 = st.columns([1, 1, 1.5])
    
    with r_col1:
        # Region Filter (Dropdown style)
        all_regions = ["🌎 All Regions"] + sorted(m_df["Region"].unique().tolist())
        selected_region = st.selectbox(
            "Filter by Region", 
            options=all_regions, 
            index=0,
            key="p_region_filter"
        )
        
    with r_col2:
        # Sector Filter (Dropdown style)
        all_sectors = ["🌍 All Sectors"] + sorted(m_df["Sector"].unique().tolist())
        selected_sector = st.selectbox(
            "Filter by Sector",
            options=all_sectors,
            index=0,
            key="p_sector_filter"
        )
        
    with r_col3:
        scan_mode = st.selectbox(
            "Intelligence Strategy Preset", 
            options=scan_presets, 
            key="scan_mode",
            label_visibility="visible"
        )

    # Apply both filters (Supporting "All" options)
    f_df = m_df.copy()
    if selected_region != "🌎 All Regions":
        f_df = f_df[f_df["Region"] == selected_region]
    if selected_sector != "🌍 All Sectors":
        f_df = f_df[f_df["Sector"] == selected_sector]
    if "Institutional Pulse" in scan_mode:
        f_df = f_df[(f_df["Quality"] >= 75) & (f_df["Trend"] == "BULLISH")]
        st.success("🏆 Institutional Pulse: Quality Score > 75 and Bullish Trend (Institutional Conviction)")
    elif "Trend Following" in scan_mode:
        f_df = f_df[f_df["Trend"] == "BULLISH"]
        st.info("📈 Trend Following: Stocks in confirmed MA20 > MA50 bullish alignment")
    elif "RSI Mean Reversion" in scan_mode:
        f_df = f_df[f_df["RSI (14)"] < 30]
        st.warning("📉 RSI Mean Reversion: Extremely Oversold (RSI < 30) candidates")
    elif "Deep Value" in scan_mode:
        f_df = f_df[f_df["Z-Score"] < -2.0]
        st.success("💎 Deep Value: Prices at -2.0 Std Dev relative to 5Y mean (Historical Bargains)")
    elif "Buy on Dip" in scan_mode:
        f_df = f_df[(f_df["Trend"] == "BULLISH") & (f_df["RSI (14)"] < 40)]
        st.info("🚀 Buy on Dip: Bullish Trend with short-term RSI cooling (< 40)")
    elif "Multi-Indicator Breakout" in scan_mode:
        f_df = f_df[(f_df["Trend"] == "BULLISH") & (f_df["RSI (14)"] > 50)]
        st.success("⚡ Breakout: Strong Momentum (Trend Bullish + RSI > 50)")
    elif "Structural Caution" in scan_mode:
        f_df = f_df[(f_df["Quality"] < 40) & (f_df["Trend"] == "BEARISH")]
        st.error("⚠️ Structural Caution: High Risk! Low Quality (Score < 40) + Confirmed Downtrend (MA20 < MA50)")
    elif "Negative Momentum" in scan_mode:
        f_df = f_df[f_df["Trend"] == "BEARISH"]
        st.error("📉 Negative Momentum: Stocks in confirmed MA20 < MA50 bearish alignment. Avoid jumping in too early.")
    elif "Overbought Alert" in scan_mode:
        f_df = f_df[f_df["RSI (14)"] > 70]
        st.warning("🔥 Overbought Alert: Extremely Overbought (RSI > 70). High risk of price correction.")
    elif "Valuation Exhaustion" in scan_mode:
        f_df = f_df[f_df["Z-Score"] > 2.0]
        st.error("🎈 Valuation Exhaustion: Prices at +2.0 Std Dev relative to 5Y mean. Likely overvalued.")
    elif "Exit on Strength" in scan_mode:
        f_df = f_df[(f_df["Trend"] == "BEARISH") & (f_df["RSI (14)"] > 60)]
        st.warning("⚔️ Exit on Strength: Bearish general trend but experiencing a short-term rally (RSI > 60). Prime short setup.")
    elif "Multi-Indicator Breakdown" in scan_mode:
        f_df = f_df[(f_df["Trend"] == "BEARISH") & (f_df["RSI (14)"] < 50)]
        st.error("💔 Breakdown: Extreme downside momentum (Trend Bearish + RSI < 50). Falling knife.")
    elif "──" in scan_mode:
        # Just to catch the separator line if selected
        st.warning("Please select a valid screening preset.")

    # ── Custom Refinement ─────────────────────────────────────────────────────
    with st.expander("Custom Refinement Sliders"):
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            min_score = st.slider("Min Quality Score", 0, 100, 0)
            rsi_range = st.slider("RSI Range", 0, 100, (0, 100))
        with rcol2:
            max_pe = st.slider("Max Forward P/E", 0, 100, 100)
            min_upside = st.slider("Min Analyst Upside (%)", -50, 100, -50)
        with rcol3:
            min_fmi = st.slider("Min FMI Score", 0, 100, 0,
                                help="Fundamental Momentum Index (0-100). Higher = stronger earnings/revenue acceleration.")

    f_df = f_df[
        (f_df["Quality"] >= min_score) &
        (f_df["RSI (14)"].between(rsi_range[0], rsi_range[1])) &
        (f_df["P/E (Fwd)"] <= max_pe) &
        (f_df["Upside (%)"] >= min_upside) &
        (f_df["FMI"] >= min_fmi)
    ]

    # ── Display Results ───────────────────────────────────────────────────────
    display_cols = ["Ticker", "Company", "Sector", "Action", "Quality", "FMI",
                    "Upside (%)", "MCap (B)", "RSI (14)", "Z-Score",
                    "vs MA200 (%)", "P/E (Fwd)", "EV/EBITDA", "PEG", "FCF Margin (%)",
                    "ROE (%)", "Yield (%)", "Net Payout (%)", "Debt/EBITDA"]
    display_df = f_df.sort_values(["Quality", "FMI"], ascending=False)[display_cols]

    st.markdown(f"**Found {len(display_df)} active opportunities** — Sorted by Quality + FMI")
    
    # ── PAGINATION / LIMIT LOGIC ──────────────────────────────────────────────
    if 'radar_limit' not in st.session_state:
        st.session_state.radar_limit = 50
        
    paged_df = display_df.iloc[:st.session_state.radar_limit]
    st.dataframe(
        paged_df,
        use_container_width=True, 
        height=520,
        column_config={
            "Quality":         st.column_config.ProgressColumn("Quality Score", min_value=0, max_value=100, format="%d"),
            "FMI":             st.column_config.ProgressColumn("FMI Score", min_value=0, max_value=100, format="%d"),
            "Upside (%)": st.column_config.NumberColumn("Upside %", format="%+.1f%%"),
            "Debt/EBITDA":     st.column_config.NumberColumn("Debt/EBITDA", format="%.2f"),
        }
    )

    # Load All Button
    if len(display_df) > st.session_state.radar_limit:
        if st.button(f"📥 Load All (Showing {st.session_state.radar_limit} of {len(display_df)})", use_container_width=True):
            st.session_state.radar_limit = len(display_df)
            st.rerun()
    elif len(display_df) > 50:
        if st.button("🔄 Reset to Top 50", use_container_width=True):
            st.session_state.radar_limit = 50
            st.rerun()

    # ── Quality Score Methodology Note (v3.0 — synced with etl/utils.py) ────────
    st.markdown("""
    <div style='margin-top:16px; padding:14px 18px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:10px;'>
        <div style='font-size:0.78rem; font-weight:700; color:#8899aa; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>
            Quality Score Methodology v3.0 — 6 Pillars, Max 100 Points
        </div>
        <div style='display:grid; grid-template-columns: repeat(6, 1fr); gap:10px;'>
            <div style='background:rgba(52,152,219,0.08); border-left:3px solid #3498db; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#3498db; font-weight:700;'>VALUATION</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>PEG · P/E · P/B · ROE bonus</div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 20 pts</div>
            </div>
            <div style='background:rgba(46,204,113,0.08); border-left:3px solid #2ecc71; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#2ecc71; font-weight:700;'>PROFITABILITY</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>FCF Margin · ROE<br><span style='color:#f1c40f;'>Tech: ≤ 30 pts</span></div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 25 pts</div>
            </div>
            <div style='background:rgba(241,196,15,0.08); border-left:3px solid #f1c40f; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#f1c40f; font-weight:700;'>FINANCIAL HEALTH</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>Debt / EBITDA ratio<br>Sector-aware bands</div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 15 pts</div>
            </div>
            <div style='background:rgba(155,89,182,0.08); border-left:3px solid #9b59b6; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#9b59b6; font-weight:700;'>NET PAYOUT YIELD</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>Dividend + Buyback<br><span style='color:#f1c40f;'>Tech capped: ≤ 5 pts</span></div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 10 pts</div>
            </div>
            <div style='background:rgba(0,210,255,0.08); border-left:3px solid #00d2ff; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#00d2ff; font-weight:700;'>TECHNICAL TREND</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>MA Signal · RSI · Z-Score</div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 25 pts</div>
            </div>
            <div style='background:rgba(231,76,60,0.08); border-left:3px solid #e74c3c; padding:8px 10px; border-radius:5px;'>
                <div style='font-size:0.7rem; color:#e74c3c; font-weight:700;'>ANALYST ESTIMATES</div>
                <div style='font-size:0.65rem; color:#aaa; margin-top:3px;'>Upside % + Consensus</div>
                <div style='font-size:1rem; font-weight:800; color:#fff;'>≤ 5 pts</div>
            </div>
        </div>
        <div style='margin-top:10px; display:grid; grid-template-columns: 1fr 1fr; gap:8px; font-size:0.68rem;'>
            <div style='background:rgba(231,76,60,0.07); border-left:2px solid #e74c3c; padding:6px 10px; border-radius:4px; color:#ccc;'>
                🚨 <b style='color:#e74c3c;'>Red Flag Penalties:</b>
                Negative P/E &amp; no-growth (−10) · Debt/EBITDA &gt; 10 (−10) · Value Trap: Z &lt; −1.5 + Sell consensus (−5)
            </div>
            <div style='background:rgba(241,196,15,0.07); border-left:2px solid #f1c40f; padding:6px 10px; border-radius:4px; color:#ccc;'>
                ⚡ <b style='color:#f1c40f;'>Beta Risk Adjustment (v3.0 NEW):</b>
                High Beta &gt; 1.8 → penalty up to −5 · Low Beta &lt; 0.8 (non-tech) → bonus up to +5
            </div>
        </div>
        <div style='margin-top:6px; font-size:0.65rem; color:#556677;'>
            Score is fully sector-aware — Tech growth stocks use different P/E bands &amp; profitability weights vs. Utilities/Financials. All thresholds use linear interpolation (np.interp) to eliminate cliff effects.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💡 Tactical Interpretation Guide"):
        st.write("""
        - **If Strong Buy + High Upside**: Consider Scaling In.
        - **If High Upside but Neutral/Bearish Trend**: Potential Value Trap. Wait for MA20 breakout.
        - **If High Quality + RSI < 30**: Extreme Oversold opportunity for mean reversion.
        """)
if active_tab == "4. Quantitative Forecast (ML)":
    import torch
    import optuna
    import numpy as np
    from arch import arch_model
    render_header("zap", "Context-Aware Direct Multi-Step Forecasting (v11.0)", "Institutional-Grade Adaptive ML Ensemble")
    
    # ── ML Model Architectures (Support for 13th Feature: Market Regime) ──────────────
    
    # ── LSTM Architecture (v7.0: Direct Multi-step Mapping) ───────────
    class StockLSTM(torch.nn.Module):
        def __init__(self, input_size=13, hidden_size=64, num_layers=2, output_size=30):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers  = num_layers
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
            self.attention = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)
            self.fc = torch.nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            # Temporal Attention
            attn_output, _ = self.attention(out, out, out)
            return self.fc(attn_output.mean(dim=1))

    # ── Transformer Architecture (v8.0: Pure Attention — Parallel Multi-step) ────────
    class StockTransformer(torch.nn.Module):
        def __init__(self, input_size=13, d_model=64, nhead=4, num_layers=2, output_size=30, dropout=0.1):
            super().__init__()
            self.input_proj = torch.nn.Linear(input_size, d_model)
            self.pos_enc = torch.nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation="gelu"
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = torch.nn.LayerNorm(d_model)
            self.fc   = torch.nn.Linear(d_model, output_size)

        def forward(self, x):
            B, T, _ = x.shape
            x = self.input_proj(x)
            x = x + self.pos_enc[:, :T, :]
            x = self.encoder(x)
            x = self.norm(x[:, -1, :])
            return self.fc(x)

    # ── PatchTST Architecture (v10.0: Channel-Independent Transformer) ────────
    class StockPatchTST(torch.nn.Module):
        def __init__(self, c_in=13, context_window=120, target_window=30,
                     patch_len=16, stride=8, d_model=64, nhead=4,
                     num_layers=2, dropout=0.1):
            super().__init__()
            self.patch_len = patch_len
            self.stride    = stride
            self.c_in      = c_in
            self.num_patches = (context_window - patch_len) // stride + 1
            self.patch_embed = torch.nn.Linear(patch_len, d_model)
            self.pos_enc     = torch.nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
            encoder_layer    = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation="gelu"
            )
            self.encoder     = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm        = torch.nn.LayerNorm(d_model)
            self.head        = torch.nn.Linear(self.num_patches * d_model, target_window)

        def forward(self, x):
            B, T, C = x.shape
            x = x.permute(0, 2, 1)                      
            x = x.unfold(-1, self.patch_len, self.stride) 
            P = x.shape[2]
            x = x.reshape(B * C, P, self.patch_len)
            x = self.patch_embed(x)                  
            x = x + self.pos_enc[:, :P, :]            
            x = self.encoder(x)                       
            x = self.norm(x)                          
            x = x.reshape(B, C, P * x.shape[-1])     
            x = x.mean(dim=1)                         
            return self.head(x)

    def _get_regime_history():
        """
        Computes historical 0-100 Market Regime Score for the last 500 days.
        Used as the 13th feature for context-aware forecasting.
        """
        try:
            # 1. Price Trend (50 pts)
            spy = df_spy_global.tail(500).copy()
            spy['trend_score'] = (spy['price_close'] > spy['ma_50']).astype(int) * 25
            spy['trend_score'] += (spy['price_close'] > spy['ma_200']).astype(int) * 25
            
            # 2. Breadth (30 pts)
            br = breadth_ts_global.copy()
            
            # 3. Volatility (20 pts)
            vix_h = prices_full[prices_full['ticker'] == '^VIX'].sort_values('date').tail(500).copy()
            vix_h['vix_score'] = vix_h['price_close'].apply(lambda v: 20 if v < 20 else 10 if v < 30 else 0)
            
            # Sync all on date
            reg_df = spy[['date', 'trend_score']].merge(br, on='date', how='left').merge(vix_h[['date', 'vix_score']], on='date', how='left')
            reg_df['breadth_score'] = (reg_df['breadth_pct'] / 100 * 30).fillna(15)
            reg_df['regime_score'] = reg_df['trend_score'] + reg_df['breadth_score'] + reg_df['vix_score']
            return reg_df[['date', 'regime_score']].fillna(50)
        except Exception:
            return pd.DataFrame()

    def _precompute_features(df_ticker):
        """
        Shared 13-factor feature engineering (Context-Aware v11.0).
        Injected 'Market Regime Score' as the 13th strategic input.

        Returns a dict with:
            data_scaled  : np.ndarray [N, 12]
            price_scaler : MinMaxScaler fitted on raw price_close column
            features     : list[str] of 12 feature names
            data         : np.ndarray [N, 12] (raw, unscaled)
            df           : pd.DataFrame with all features
            n_feat       : int (12)
        Returns None on failure.
        """
        import warnings; warnings.filterwarnings('ignore')
        try:
            ticker_id = df_ticker['ticker'].iloc[0] if not df_ticker.empty else None
            if ticker_id is None:
                return None

            # Cache key includes date to invalidate when data refreshes
            max_date  = str(df_ticker['date'].max()) if 'date' in df_ticker.columns else ''
            cache_key = f"feat_cache_{ticker_id}_{max_date}"
            if cache_key in st.session_state:
                return st.session_state[cache_key]

            df = df_ticker.copy().sort_values("date").reset_index(drop=True).tail(500).reset_index(drop=True)

            # ── Macro & Volatility ──
            df['vol_surge'] = df['volume'] / (df['volume'].rolling(20).mean().fillna(df['volume']))
            spy_df = prices_full[prices_full['ticker']=='SPY'][['date','daily_return_pct']].rename(columns={'daily_return_pct':'spy_ret'})
            vix_df = prices_full[prices_full['ticker']=='^VIX'][['date','daily_return_pct']].rename(columns={'daily_return_pct':'vix_ret'})
            df = df.merge(spy_df, on='date', how='left').merge(vix_df, on='date', how='left')
            df['spy_ret'] = df['spy_ret'].fillna(0)
            df['vix_ret'] = df['vix_ret'].fillna(0)

            # ── Technical ──
            if 'rsi' in df.columns:           df['rsi'] = df['rsi'].fillna(50.0)
            else:                              df['rsi'] = 50.0
            if 'price_z_score' in df.columns: df['price_z_score'] = df['price_z_score'].fillna(0.0)
            else:                              df['price_z_score'] = 0.0

            # ── Fundamentals ──
            co_row = companies_full[companies_full['ticker'] == ticker_id].iloc[0].to_dict() \
                if not companies_full[companies_full['ticker'] == ticker_id].empty else {}
            _ebitda = float(co_row.get('ebitda', 1) or 1)
            _debt   = float(co_row.get('total_debt', 0) or 0)
            df['pe_ratio']    = float(np.clip(float(co_row.get('pe_ratio', 20) or 20), 0, 150))
            df['roe']         = float(np.clip(float(co_row.get('roe', 0) or 0) * 100, -50, 100))
            df['fcf_margin']  = float(np.clip(float(co_row.get('fcf_margin', 0) or 0), -50, 80))
            df['debt_ebitda'] = float(np.clip(_debt / max(_ebitda, 1), 0, 12))
            df['rev_growth']  = float(np.clip(float(co_row.get('revenue_growth', 0) or 0) * 100, -50, 100))

            # ── Market Regime Overlay (13th Feature) ──
            rh = _get_regime_history()
            if not rh.empty:
                df = df.merge(rh, on='date', how='left')
                df['regime_score'] = df['regime_score'].fillna(method='ffill').fillna(50)
            else:
                df['regime_score'] = 50.0

            features = [
                'price_close', 'daily_return_pct',
                'spy_ret', 'vix_ret',
                'vol_surge', 'rsi', 'price_z_score',
                'pe_ratio', 'roe', 'fcf_margin',
                'debt_ebitda', 'rev_growth', 'regime_score'
            ]
            data = df[features].ffill().fillna(0).values.astype(np.float32)

            from sklearn.preprocessing import MinMaxScaler
            scaler       = MinMaxScaler(feature_range=(-1, 1))
            data_scaled  = scaler.fit_transform(data)
            price_scaler = MinMaxScaler(feature_range=(-1, 1))
            price_scaler.fit(data[:, 0:1])

            result = {
                'data_scaled' : data_scaled,
                'price_scaler': price_scaler,
                'features'    : features,
                'data'        : data,
                'df'          : df,
                'n_feat'      : len(features),
            }
            st.session_state[cache_key] = result
            return result
        except Exception:
            return None

    def _run_lstm_core(df_ticker, lookback=60, forecast_days=30, sector_name=None, quality_score=50):
        import warnings
        warnings.filterwarnings('ignore')
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # ── Shared Feature Engineering (cached per ticker) ──
        feat = _precompute_features(df_ticker)
        if feat is None:
            return None, None, None
        data_scaled  = feat['data_scaled']
        price_scaler = feat['price_scaler']
        features     = feat['features']
        data         = feat['data']
        df           = feat['df']

        # Adaptive Lookback tuning (v7.0)
        ticker_vol = df['daily_return_pct'].tail(60).std()
        spy_vol_s  = prices_full[prices_full['ticker']=='SPY']['daily_return_pct'].tail(60)
        spy_vol    = spy_vol_s.std() if not spy_vol_s.empty else 1.0

        # 🛡️ ADAPTIVE CLIPPING & WEIGHTS (v7.0)
        vol_ratio = ticker_vol / (spy_vol + 1e-6)
        dynamic_clamp = 0.05 + min(0.05, 0.02 * vol_ratio)

        if ticker_vol > 2.0 * spy_vol:   lstm_w, arima_w, lookback = 0.70, 0.30, max(60, min(lookback, 60))
        elif ticker_vol < 0.8 * spy_vol: lstm_w, arima_w, lookback = 0.40, 0.60, 120
        else:                             lstm_w, arima_w, lookback = 0.60, 0.40, max(90, lookback)

        if len(data_scaled) < lookback + forecast_days + 30:
            return None, None, None

        X, y = [], []
        for i in range(len(data_scaled) - lookback - forecast_days):
            X.append(data_scaled[i:(i+lookback), :])
            y.append(data_scaled[i+lookback : i+lookback+forecast_days, 0])
        
        X_t = torch.FloatTensor(np.array(X)).to(device)
        y_t = torch.FloatTensor(np.array(y)).to(device)
        
        # 🛡️ TEMPORAL FEATURE DECAY
        decay_weights = torch.exp(torch.linspace(-0.5, 0, lookback)).to(device).view(1, lookback, 1)
        X_t = X_t * decay_weights
        
        ticker_id    = df_ticker['ticker'].iloc[0] if not df_ticker.empty else "unknown"
        MODEL_VERSION = f"v7_direct_{forecast_days}"
        if "optuna_cache" not in st.session_state or st.session_state.get("optuna_version") != MODEL_VERSION:
            st.session_state.optuna_cache = {}; st.session_state.optuna_version = MODEL_VERSION
            
        if ticker_id in st.session_state.optuna_cache:
            best = st.session_state.optuna_cache[ticker_id]
        else:
            hpo_split = int(len(X_t)*0.8)
            X_hpo, y_hpo = X_t[:hpo_split], y_t[:hpo_split]
            def objective(trial):
                h  = trial.suggest_categorical("hidden_size",[32,64,128])
                nl = trial.suggest_int("num_layers",1,2)
                lr = trial.suggest_float("lr",5e-4,2e-3,log=True)
                m  = StockLSTM(input_size=len(features),hidden_size=h,num_layers=nl,output_size=forecast_days).to(device)
                cr = torch.nn.HuberLoss(delta=1.0)
                op = torch.optim.Adam(m.parameters(),lr=lr)
                m.train()
                import time
                for _ in range(20): # Optimized: 20 epochs for HPO
                    indices = torch.randperm(X_hpo.size(0), device=device)
                    for start_idx in range(0, X_hpo.size(0), 128):
                        idx = indices[start_idx:start_idx+128]
                        X_b, y_b = X_hpo[idx], y_hpo[idx]
                        
                        y_baseline = X_b[:, -1, 0].unsqueeze(1) # Anchor
                        op.zero_grad()
                        o = m(X_b) + y_baseline
                        l = cr(o, y_b)
                        l.backward()
                        op.step()
                    time.sleep(0.005) # Micro-yield
                return l.item()
            with st.spinner(f"Tuning Direct Intelligence for {ticker_id}..."):
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=5, timeout=10) # Optimized: 5 trials, 10s
                best = study.best_params; best['epochs']=80
                st.session_state.optuna_cache[ticker_id] = best
        
        # ── Final Training (v7.2: Direct Multi-step Architecture) ──
        model = StockLSTM(input_size=len(features), hidden_size=best['hidden_size'], num_layers=best['num_layers'], output_size=forecast_days).to(device)
        cr = torch.nn.HuberLoss(delta=1.0); op = torch.optim.Adam(model.parameters(), lr=best['lr'])
        model.train(); prev_loss = 1e9
        
        for epoch in range(best['epochs']):
            indices = torch.randperm(X_t.size(0), device=device)
            for start_idx in range(0, X_t.size(0), 128):
                idx = indices[start_idx:start_idx+128]
                X_b, y_b = X_t[idx], y_t[idx]
                
                y_baseline = X_b[:, -1, 0].unsqueeze(1)
                op.zero_grad()
                o = model(X_b) + y_baseline
                l_core = cr(o, y_b)
                
                # Multi-step Directional Penalty
                pred_diff = o - y_baseline
                true_diff = y_b - y_baseline
                penalty = torch.mean(torch.clamp(-pred_diff * true_diff, min=0)) * 0.5
                
                l = l_core + penalty
                l.backward()
                op.step()
            
            if torch.isnan(l): break
            l_val = l.item()
            if abs(prev_loss - l_val) < (prev_loss * 5e-5) and epoch > 30: break
            prev_loss = l_val
            import time; time.sleep(0.005) # Micro-yield
            
        # ── INFERENCE (v7.2: Single Shot Direct) ──
        model.eval()
        last_seq = data_scaled[-lookback:].copy()
        last_seq_t = torch.FloatTensor(last_seq).unsqueeze(0).to(device)
        last_seq_t = last_seq_t * decay_weights # Apply temporal decay
        with torch.no_grad():
            y_base_inf   = last_seq_t[:, -1, 0].unsqueeze(1)
            preds_scaled = (model(last_seq_t) + y_base_inf).cpu().numpy().flatten()
        
        lstm_predicted_prices = price_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
        
        # ── Raw ARIMA ──
        ts_raw = df['price_close'].values
        try:
            from pmdarima import auto_arima
            arima_predicted_prices = auto_arima(ts_raw, seasonal=False, stepwise=True, suppress_warnings=True).predict(n_periods=forecast_days)
        except Exception:
            try:
                from statsmodels.tsa.arima.model import ARIMA
                arima_predicted_prices = ARIMA(ts_raw,order=(1,1,1)).fit().forecast(steps=forecast_days)
            except Exception:
                arima_predicted_prices = np.full(forecast_days,ts_raw[-1])
        
        ensemble_prices = (lstm_w * lstm_predicted_prices) + (arima_w * arima_predicted_prices)
        
        # 🛡️ ADAPTIVE VOLATILITY CLIPPING (v7.2)
        clamped_prices = [df_ticker['price_close'].iloc[-1]]
        for t in range(len(ensemble_prices)):
            p_raw = ensemble_prices[t]
            p_prev = clamped_prices[-1]
            p_clamped = np.clip(p_raw, p_prev * (1 - dynamic_clamp), p_prev * (1 + dynamic_clamp))
            clamped_prices.append(p_clamped)
            
        current_price = data[-1,0]
        if np.isnan(ensemble_prices[-1]) or current_price==0: return None,None,None
        
        model.eval()
        X_explain = last_seq_t.clone().requires_grad_(True)
        out_explain = model(X_explain)
        torch.sum(out_explain).backward() # Backprop through entire multi-step output
        importances = torch.abs(X_explain.grad[0]).mean(dim=0).cpu().numpy()
        importances = importances / (np.sum(importances) + 1e-9) * 100
        feat_imp_dict = dict(zip(features, importances))

        return clamped_prices[1:], (clamped_prices[-1]-clamped_prices[0])/clamped_prices[0], feat_imp_dict

    def calculate_backtest_accuracy(df_full, sector_name=None, quality_score=50, test_size=21):
        """Phase 10: Honest Backtest - Strict Train/Test Separation"""
        if len(df_full) < 150: return None, None
        # We slice raw data to ensure NO LEAKAGE from the future
        train_df = df_full.iloc[:-test_size].copy()
        actual_prices = df_full["price_close"].iloc[-test_size:].values
        
        # Run forecast strictly on training data
        # No re-training or HPO on the test window allowed
        predicted,_,_ = _run_lstm_core(train_df, lookback=120, forecast_days=test_size, sector_name=sector_name, quality_score=quality_score)
        
        if predicted is None or len(predicted) < test_size: return None, None
        mape = np.mean(np.abs((actual_prices - predicted) / actual_prices))
        return max(0.0, min(100.0, 100*(1-mape))), float(mape)

    @st.cache_data(show_spinner="Training Adaptive AI Ensemble (LSTM + ARIMA)...")
    def train_predict_lstm(df_ticker, lookback=60, forecast_days=30, sector_name=None, quality_score=50):
        return _run_lstm_core(df_ticker, lookback=lookback, forecast_days=forecast_days, sector_name=sector_name, quality_score=quality_score)

    @st.cache_data(show_spinner="🤖 Training Temporal Transformer (Attention Engine v8.0)...")
    def train_predict_transformer(df_ticker, lookback=90, forecast_days=30, sector_name=None, quality_score=50):
        """
        Drop-in replacement for train_predict_lstm using the pure Transformer architecture.
        Returns the same (path_array, return_pct, feature_importance) tuple.
        """
        import warnings
        warnings.filterwarnings('ignore')
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

            # ── Shared Feature Engineering (cached per ticker) ──
            feat = _precompute_features(df_ticker)
            if feat is None:
                return None, 0.0, {}
            data_scaled  = feat['data_scaled']
            price_scaler = feat['price_scaler']
            features     = feat['features']
            data         = feat['data']
            df           = feat['df']

            if len(data) < lookback + forecast_days:
                return None, 0.0, {}

            X, y = [], []
            for i in range(lookback, len(data_scaled) - forecast_days):
                X.append(data_scaled[i-lookback:i])
                y.append(data_scaled[i:i+forecast_days, 0])

            X = torch.FloatTensor(np.array(X)).to(device)
            y = torch.FloatTensor(np.array(y)).to(device)

            model = StockTransformer(input_size=len(features), d_model=64, nhead=4,
                                     num_layers=2, output_size=forecast_days).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            criterion = torch.nn.HuberLoss(delta=0.5)

            model.train()
            for epoch in range(60):
                indices = torch.randperm(X.size(0), device=device)
                for start_idx in range(0, X.size(0), 128):
                    idx = indices[start_idx:start_idx+128]
                    X_b, y_b = X[idx], y[idx]
                    
                    optimizer.zero_grad()
                    out = model(X_b)
                    
                    y_baseline = X_b[:, -1, 0].unsqueeze(1)
                    out = out + y_baseline
                    
                    loss = criterion(out, y_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                import time; time.sleep(0.005)

            # Inference
            model.eval()
            with torch.no_grad():
                last_seq = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device)
                y_baseline_inf = last_seq[:, -1, 0].unsqueeze(1)
                pred_scaled = (model(last_seq) + y_baseline_inf).cpu().numpy()[0]

            # Inverse-transform only price column
            price_scaler = MinMaxScaler(feature_range=(-1, 1))
            price_scaler.fit(data[:, 0:1])
            full_pred = np.zeros((forecast_days, len(features)))
            full_pred[:, 0] = pred_scaled
            forecast_raw = price_scaler.inverse_transform(full_pred[:, 0:1]).flatten()

            last_price = data[-1, 0]
            total_return = (forecast_raw[-1] / last_price - 1) if last_price > 0 else 0.0

            # ── v9.1: Real gradient attribution (not random) ──
            feat_imp = {}
            try:
                model.eval()
                last_seq_t = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device).requires_grad_(True)
                out_t = model(last_seq_t)
                torch.sum(out_t).backward()
                imp_t = torch.abs(last_seq_t.grad[0]).mean(dim=0).cpu().numpy()
                imp_t = imp_t / (imp_t.sum() + 1e-9) * 100
                feat_imp = {f: round(float(v), 1) for f, v in zip(features, imp_t)}
            except Exception:
                feat_imp = {f: round(100/len(features), 1) for f in features}

            return forecast_raw, total_return, feat_imp
        except Exception as e:
            return None, 0.0, {}

    @st.cache_data(show_spinner="🧬 Training PatchTST (SOTA Channel-Independent Engine v10.0)...")
    def train_predict_patchtst(df_ticker, lookback=120, forecast_days=30, sector_name=None, quality_score=50):
        """
        PatchTST Channel-Independent engine. Each of the 12 factors is processed
        by the SAME Transformer independently (no cross-channel noise), then averaged.
        Best for long-horizon, fundamental-driven forecasts.
        """
        import warnings
        warnings.filterwarnings('ignore')
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

            # ── Shared Feature Engineering (cached per ticker) ──
            feat = _precompute_features(df_ticker)
            if feat is None:
                return None, 0.0, {}
            data_scaled  = feat['data_scaled']
            price_scaler = feat['price_scaler']
            features     = feat['features']
            data         = feat['data']

            if len(data) < lookback + forecast_days:
                return None, 0.0, {}

            # ── Patch parameters ──
            patch_len = 16; stride = 8
            num_patches = (lookback - patch_len) // stride + 1

            X, y = [], []
            for i in range(lookback, len(data_scaled) - forecast_days):
                X.append(data_scaled[i-lookback:i])
                y.append(data_scaled[i:i+forecast_days, 0])
            X = torch.FloatTensor(np.array(X)).to(device)  # [N, T, C]
            y = torch.FloatTensor(np.array(y)).to(device)  # [N, forecast_days]

            model = StockPatchTST(
                c_in=len(features), context_window=lookback,
                target_window=forecast_days, patch_len=patch_len,
                stride=stride, d_model=64, nhead=4, num_layers=2
            ).to(device)
            optimizer  = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)
            criterion  = torch.nn.HuberLoss(delta=0.5)
            scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)

            model.train()
            for epoch in range(80):
                indices = torch.randperm(X.size(0), device=device)
                for start_idx in range(0, X.size(0), 128):
                    idx = indices[start_idx:start_idx+128]
                    X_b, y_b = X[idx], y[idx]
                    
                    optimizer.zero_grad()
                    out = model(X_b)
                    
                    y_baseline = X_b[:, -1, 0].unsqueeze(1)
                    out = out + y_baseline
                    
                    loss = criterion(out, y_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                import time; time.sleep(0.005)

            # Inference
            model.eval()
            with torch.no_grad():
                last_seq_p   = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device)
                y_baseline_p = last_seq_p[:, -1, 0].unsqueeze(1)
                pred_scaled  = (model(last_seq_p) + y_baseline_p).cpu().numpy()[0]

            # Inverse-transform price
            full_pred_p    = np.zeros((forecast_days, len(features)))
            full_pred_p[:, 0] = pred_scaled
            forecast_raw_p = price_scaler.inverse_transform(full_pred_p[:, 0:1]).flatten()

            last_price_p = data[-1, 0]
            total_return_p = (forecast_raw_p[-1] / last_price_p - 1) if last_price_p > 0 else 0.0

            # Gradient-based feature importance
            feat_imp_p = {}
            try:
                last_seq_grad = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device).requires_grad_(True)
                out_grad = model(last_seq_grad)
                torch.sum(out_grad).backward()
                imp_p = torch.abs(last_seq_grad.grad[0]).mean(dim=0).cpu().numpy()
                imp_p = imp_p / (imp_p.sum() + 1e-9) * 100
            except Exception:
                feat_imp_p = {f: round(100/len(features), 1) for f in features}

            return forecast_raw_p, total_return_p, feat_imp_p
        except Exception:
            return None, 0.0, {}

    @st.cache_data(show_spinner="Smart Blend: Training all 3 AI Engines (LSTM + Transformer + PatchTST)...")
    def train_predict_ensemble(df_ticker, lookback=90, forecast_days=30, sector_name=None, quality_score=50):
        """
        Performance-Weighted Ensemble: trains all 3 engines, evaluates each on the
        most recent holdout period (last forecast_days of known data), and blends
        their forecasts using weights proportional to 1/RMSE.
        """
        import warnings; warnings.filterwarnings('ignore')
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            feat = _precompute_features(df_ticker)
            if feat is None: return None, 0.0, {}, {}
            data_scaled, price_scaler, features, data, n_feat = feat['data_scaled'], feat['price_scaler'], feat['features'], feat['data'], feat['n_feat']
            if len(data) < lookback + 2 * forecast_days: return None, 0.0, {}, {}

            X_arr, y_arr = [], []
            for i in range(lookback, len(data_scaled) - forecast_days):
                X_arr.append(data_scaled[i-lookback:i]); y_arr.append(data_scaled[i:i+forecast_days, 0])
            X_t = torch.FloatTensor(np.array(X_arr)).to(device)
            y_t = torch.FloatTensor(np.array(y_arr)).to(device)
            if len(X_t) < 5: return None, 0.0, {}, {}

            def _eval_holdout(mdl):
                n_d = len(data_scaled)
                eval_x = data_scaled[n_d - lookback - forecast_days : n_d - forecast_days]
                actual_s = data_scaled[n_d - forecast_days : n_d, 0]
                mdl.eval()
                with torch.no_grad():
                    inp = torch.FloatTensor(eval_x).unsqueeze(0).to(device)
                    pred_s = mdl(inp).cpu().numpy().flatten()[:forecast_days]
                fp = np.zeros((forecast_days, n_feat)); fp[:, 0] = pred_s
                fa = np.zeros((forecast_days, n_feat)); fa[:, 0] = actual_s
                pp = price_scaler.inverse_transform(fp[:, 0:1]).flatten()
                ap = price_scaler.inverse_transform(fa[:, 0:1]).flatten()
                return float(np.sqrt(np.mean((pp - ap) ** 2))), float(np.mean(np.abs((ap - pp) / (np.abs(ap) + 1e-8))) * 100), float(1.0 if (pp[-1] > pp[0]) == (ap[-1] > ap[0]) else 0.0)

            def _infer(mdl):
                mdl.eval()
                with torch.no_grad():
                    last_x = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device)
                    y_base = last_x[:, -1, 0].unsqueeze(1)
                    pr = (mdl(last_x) + y_base).cpu().numpy().flatten()[:forecast_days]
                fp = np.zeros((forecast_days, n_feat)); fp[:, 0] = pr
                return price_scaler.inverse_transform(fp[:, 0:1]).flatten()

            results = {}
            import time as _time

            # (A) LSTM
            try:
                ml = StockLSTM(input_size=n_feat, hidden_size=64, num_layers=2, output_size=forecast_days).to(device)
                ol = torch.optim.Adam(ml.parameters(), lr=1e-3, weight_decay=1e-5); cl = torch.nn.HuberLoss(delta=1.0)
                ml.train()
                for _ in range(30):
                    idxs = torch.randperm(X_t.size(0), device=device)
                    for si in range(0, X_t.size(0), 128):
                        idx = idxs[si:si+128]; Xb, yb = X_t[idx], y_t[idx]; ybl = Xb[:, -1, 0].unsqueeze(1)
                        ol.zero_grad(); ls = cl(ml(Xb) + ybl, yb); ls.backward(); torch.nn.utils.clip_grad_norm_(ml.parameters(), 1.0); ol.step()
                    _time.sleep(0.005)
                path_l = _infer(ml); rm, mp, dr = _eval_holdout(ml)
                results['LSTM'] = {'path': path_l, 'rmse': rm, 'mape': mp, 'dir': dr, 'model': ml}
            except Exception: pass

            # (B) Transformer
            try:
                mt = StockTransformer(input_size=n_feat, d_model=64, nhead=4, num_layers=2, output_size=forecast_days).to(device)
                ot = torch.optim.Adam(mt.parameters(), lr=1e-3, weight_decay=1e-5); ct = torch.nn.HuberLoss(delta=0.5)
                mt.train()
                for _ in range(30):
                    idxs = torch.randperm(X_t.size(0), device=device)
                    for si in range(0, X_t.size(0), 128):
                        idx = idxs[si:si+128]; Xb, yb = X_t[idx], y_t[idx]; ybl = Xb[:, -1, 0].unsqueeze(1)
                        ot.zero_grad(); ls = ct(mt(Xb) + ybl, yb); ls.backward(); torch.nn.utils.clip_grad_norm_(mt.parameters(), 1.0); ot.step()
                    _time.sleep(0.005)
                path_t = _infer(mt); rm, mp, dr = _eval_holdout(mt)
                results['Transformer'] = {'path': path_t, 'rmse': rm, 'mape': mp, 'dir': dr, 'model': mt}
            except Exception: pass

            # (C) PatchTST
            try:
                patch_len, stride = 16, 8
                mp_m = StockPatchTST(c_in=n_feat, context_window=lookback, target_window=forecast_days, patch_len=patch_len, stride=stride, d_model=64, nhead=4, num_layers=2).to(device)
                op = torch.optim.Adam(mp_m.parameters(), lr=8e-4, weight_decay=1e-5); cp = torch.nn.HuberLoss(delta=0.5); sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=30, eta_min=1e-5)
                mp_m.train()
                for _ in range(30):
                    idxs = torch.randperm(X_t.size(0), device=device)
                    for si in range(0, X_t.size(0), 128):
                        idx = idxs[si:si+128]; Xb, yb = X_t[idx], y_t[idx]; ybl = Xb[:, -1, 0].unsqueeze(1)
                        op.zero_grad(); ls = cp(mp_m(Xb) + ybl, yb); ls.backward(); torch.nn.utils.clip_grad_norm_(mp_m.parameters(), 1.0); op.step()
                    sp.step(); _time.sleep(0.005)
                path_p = _infer(mp_m); rm, mp_v, dr = _eval_holdout(mp_m)
                results['PatchTST'] = {'path': path_p, 'rmse': rm, 'mape': mp_v, 'dir': dr, 'model': mp_m}
            except Exception: pass

            if not results: return None, 0.0, {}, {}
            inv_rmse = {k: 1.0 / max(v['rmse'], 0.01) for k, v in results.items()}; total_inv = sum(inv_rmse.values()); weights = {k: round(v / total_inv, 4) for k, v in inv_rmse.items()}
            blended = np.zeros(forecast_days)
            for k, v in results.items(): blended += weights[k] * v['path'][:forecast_days]
            last_price_e = data[-1, 0]; total_return_e = (blended[-1] / last_price_e - 1) if last_price_e > 0 else 0.0
            feat_imp_e = {f: 0.0 for f in features}
            try:
                last_seq_e = torch.FloatTensor(data_scaled[-lookback:]).unsqueeze(0).to(device)
                for k, v in results.items():
                    inp_g = last_seq_e.detach().clone().requires_grad_(True); torch.sum(v['model'](inp_g)).backward()
                    imp_g = torch.abs(inp_g.grad[0]).mean(dim=0).cpu().numpy(); imp_g = imp_g / (imp_g.sum() + 1e-9)
                    for i, f in enumerate(features): feat_imp_e[f] += weights[k] * float(imp_g[i]) * 100
                feat_imp_e = {f: round(v, 1) for f, v in feat_imp_e.items()}
            except Exception: feat_imp_e = {f: round(100/n_feat, 1) for f in features}
            metrics_dict = {k: {'RMSE': round(v['rmse'], 2), 'MAPE (%)': round(v['mape'], 2), 'Dir. Acc': f"{v['dir']*100:.0f}%", 'Weight': f"{weights[k]*100:.1f}%", 'Target': round(v['path'][-1], 2)} for k, v in results.items()}
            return blended, total_return_e, feat_imp_e, metrics_dict
        except Exception: return None, 0.0, {}, {}

    # ── PERSISTENT PERFORMANCE LOGGER ──
    import json as _json; from pathlib import Path as _Path; from datetime import datetime as _dt
    _LOG_PATH = _Path("model_performance_log.json")
    def _load_perf_log():
        if _LOG_PATH.exists():
            try: return _json.loads(_LOG_PATH.read_text())
            except: pass
        return {"logs": []}
    def _save_perf_log(d):
        try: _LOG_PATH.write_text(_json.dumps(d, indent=2))
        except: pass
    def _log_ensemble_run(ticker, horizon, vix_level, em):
        if not em: return
        anchor = max(em.keys(), key=lambda k: float(em[k]['Weight'].replace('%','')))
        entry = {"ts": _dt.now().isoformat()[:19], "ticker": ticker, "horizon": horizon, "vix": round(vix_level, 2), "regime": regime, "anchor": anchor,
                 "models": {k: {"rmse": v["RMSE"], "mape": v["MAPE (%)"], "weight": float(v["Weight"].replace("%",""))/100} for k, v in em.items()}}
        d = _load_perf_log(); d["logs"].append(entry); d["logs"] = d["logs"][-500:]; _save_perf_log(d)

    render_header("trending-up", "Price & Monte Carlo Forecasting", level="###")
    
    # ── AI STRATEGIST GUIDE (Synchronized with Master Tactical Regime) ──
    if regime == "STRONG BULLISH" or regime == "BULLISH":
        _rec_model, _rec_reason = "Neural v9.1 · Transformer (Direct 12F)", f"In a <b>{regime}</b> environment, prioritize <b>Pattern Recognition</b> & momentum capture via the Transformer ensemble."
    elif regime == "NEUTRAL / SIDEWAYS":
        _rec_model, _rec_reason = "Hybrid LSTM + Multi-Head Attention", f"Current <b>{regime}</b> regime favors <b>Temporal Stability</b>. LSTM is best for mean-reversion and sequential price discovery."
    else: # BEARISH / CAUTION
        _rec_model, _rec_reason = "PatchTST (Channel-Independent) + Monte Carlo", f"Tactical <b>{regime}</b> detected. Shift to <b>Structural Robustness</b>. PatchTST is less prone to noise during trend breakdowns."

    st.markdown(f"""
    <div style='background:rgba(52,152,219,0.08); border:1px solid #3498db; padding:16px; border-radius:8px; margin-bottom:25px;'>
        <div style='display:flex; align-items:center; margin-bottom:8px;'>{SVG_ICONS["brain"]} <b style='color:#3498db; font-size:1rem; margin-left:4px;'>ML Model Strategist</b></div>
        <div style='font-size:0.92rem; color:#e0e0e0; line-height:1.5;'>Market Context: <b style='color:{regime_ui_color};'>{regime}</b><br>Recommended Model: <b style='color:#00ffcc;'>{_rec_model}</b><br>Rationale: <i>{_rec_reason}</i></div>
        <div style='margin-top:12px; font-size:0.82rem; color:#8899aa; border-top:1px solid rgba(255,255,255,0.1); padding-top:8px;'><b>Pro Tip:</b> LSTM+ARIMA anchors mean-reversion • Transformer excels in patterns • <b>PatchTST (v10.0)</b> gives high fundamental resolution — ideal for stable markets.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── ROW 1: Forecast Configuration (Horizontal Form) ──────────────────────
    with st.form("forecast_config_form"):
        fcol1, fcol2, fcol3, fcol4 = st.columns([2, 1, 1, 1])
        with fcol1:
            # Sync active_ticker into fc_selector if available and not already set
            if 'fc_selector_form' not in st.session_state:
                _at = st.session_state.get('active_ticker', None)
                if _at and _at in current_universe:
                    st.session_state['fc_selector_form'] = _at
                
            fc_ticker = st.selectbox("Select Ticker to Forecast", current_universe, 
                                     format_func=format_ticker,
                                     index=None,
                                     placeholder="Choose a Ticker...",
                                     key="fc_selector_form")
        with fcol2:
            forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 7, key="fc_days_form")
        with fcol3:
            n_sims = st.selectbox("Monte Carlo Simulations", [500, 1000, 1500, 2000, 5000], index=3, key="n_sims_form")
        with fcol4:
            engine_mode = st.radio(
                "Core Engine",
                options=["LSTM Core", "Transformer", "PatchTST (SOTA)", "Smart Blend (Best of 3)"],
                index=3,
                key="engine_mode_form",
                help="LSTM Core: stable mean-reversion • Transformer: high-vol pattern recognition • PatchTST: channel-independent fundamentals • Smart Blend: trains all 3 engines and auto-weights them by accuracy (RMSE)."
            )
            
        run_forecast = st.form_submit_button("🎯 EXECUTE ML ENSEMBLE FORECAST", use_container_width=True, type="primary")

    # Initialize before the forecast block so the metrics panel never hits NameError
    ensemble_metrics = {}

    if run_forecast and fc_ticker:
        fc_ticker = st.session_state.fc_selector_form
        forecast_days = st.session_state.fc_days_form
        n_sims = st.session_state.n_sims_form
        engine_mode = st.session_state.engine_mode_form
        
        df_fc = prices_full[prices_full["ticker"] == fc_ticker].sort_values("date")
        ts = df_fc["price_close"].values
        
        # Pre-fetch Company data for Sector context
        co_data = companies_full[companies_full["ticker"] == fc_ticker].iloc[0] if not companies_full[companies_full["ticker"] == fc_ticker].empty else None
        sector_val = co_data['sector'] if co_data is not None else None
        
        # 1. ML Prediction — branch on engine_mode (Standardized lookback by horizon)
        if forecast_days <= 14:   std_lookback = 90
        elif forecast_days <= 45: std_lookback = 180
        else:                     std_lookback = 252

        drift_score    = compute_score(co_data) if co_data is not None else 50
        use_ensemble    = (engine_mode == "Smart Blend (Best of 3)")
        use_patchtst    = (engine_mode == "PatchTST (SOTA)")
        use_transformer = (engine_mode == "Transformer")
        if len(df_fc) < 30:
            st.warning(f"⚠️ Insufficient historical data ({len(df_fc)} days) to train the ML neural network. At least 30 days are required.")
            lstm_path, lstm_return, feat_imp = None, 0.0, {}
            st.session_state['ensemble_metrics'] = {}
        elif use_ensemble:
            with st.spinner(f"Smart Blend: Training all 3 ML engines ({std_lookback}D Lookback)..."):
                _ens = train_predict_ensemble(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            if _ens[0] is not None:
                lstm_path, lstm_return, feat_imp, _em = _ens
                st.session_state['ensemble_metrics'] = _em
                # Log this run for meta-evaluation
                _vix_now = float(prices_full[prices_full['ticker']=='^VIX']['price_close'].iloc[-1]) \
                    if not prices_full[prices_full['ticker']=='^VIX'].empty else 20.0
                _log_ensemble_run(fc_ticker, forecast_days, _vix_now, _em)
            else:
                st.warning("⚠️ Ensemble failed. Falling back to LSTM...")
                lstm_path, lstm_return, feat_imp = train_predict_lstm(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
                st.session_state['ensemble_metrics'] = {}
        elif use_patchtst:
            with st.spinner(f"🧬 Running PatchTST ({std_lookback}D Lookback)..."):
                lstm_path, lstm_return, feat_imp = train_predict_patchtst(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            if lstm_path is None:
                st.warning(f"⚠️ PatchTST needs {std_lookback}+ days. Falling back to LSTM...")
                lstm_path, lstm_return, feat_imp = train_predict_lstm(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            st.session_state['ensemble_metrics'] = {}
        elif use_transformer:
            with st.spinner(f"🤖 Running Transformer ({std_lookback}D Lookback)..."):
                lstm_path, lstm_return, feat_imp = train_predict_transformer(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            if lstm_path is None:
                st.warning(f"⚠️ Transformer needs {std_lookback}+ days. Falling back to LSTM...")
                lstm_path, lstm_return, feat_imp = train_predict_lstm(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            st.session_state['ensemble_metrics'] = {}
        else:  # LSTM Core
            with st.spinner(f"Running LSTM Core ({std_lookback}D Lookback)..."):
                lstm_path, lstm_return, feat_imp = train_predict_lstm(
                    df_fc, lookback=std_lookback, forecast_days=forecast_days,
                    sector_name=sector_val, quality_score=drift_score)
            st.session_state['ensemble_metrics'] = {}
        
        # 2. News Sentiment (High-Accuracy FinBERT) using Google News
        import feedparser
        rss_url = f"https://news.google.com/rss/search?q={fc_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        titles = [entry.get("title", "").split(" - ")[0] for entry in feed.entries[:10]]
        avg_sent = analyze_sentiment_finbert(titles) if titles else 0
        
        # 4. Monte Carlo Simulation (AI-Enhanced & Dynamic Volatility)
        returns = df_fc["daily_return_pct"].dropna() / 100
        mu = returns.mean()
        sigma_long_term = returns.std()
        
        # Calculate current 'heat' (14-day rolling volatility)
        sigma_current = returns.tail(14).std() if len(returns) >= 14 else sigma_long_term
        
        last_price = ts[-1]
        
        drift_bias = 0
        if drift_score >= 75: drift_bias += 0.0005 
        elif drift_score <= 40: drift_bias -= 0.0005 
        drift_bias += (avg_sent * 0.001) 
        if lstm_return is not None and lstm_return > 0.05: drift_bias += 0.0005
        
        # ── Phase 7: Monte Carlo GARCH(1,1) (Volatility Clustering) ───────────
        try:
            # Fit GARCH(1,1) to captured historical returns (using 500-day window)
            # Scaling by 100 for numerical stability in the solver
            garch_data = returns.tail(500) * 100
            am = arch_model(garch_data, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
            res = am.fit(disp='off')
            
            # Forecast volatility term structure for the horizon
            forecasts = res.forecast(horizon=forecast_days)
            # Variance -> Std Dev, and rescale back from percent
            sigma_forecast = np.sqrt(forecasts.variance.values[-1, :]) / 100.0
            
            # Ensure no zero/nan vol (fallback to long-term avg)
            sigma_forecast = np.nan_to_num(sigma_forecast, nan=sigma_long_term)
            sigma_forecast[sigma_forecast == 0] = sigma_long_term
            
        except Exception:
            # Robust Fallback to Mean Reversion (OU Process) if GARCH fails to converge
            kappa = 0.1 
            sigma_forecast = []
            s_t = sigma_current
            for _ in range(forecast_days):
                s_t = s_t + kappa * (sigma_long_term - s_t)
                sigma_forecast.append(s_t)
            
        # ── Phase 7.5: Monte Carlo — AI-Anchored GBM ────────────────────────────
        # Best Practice: use AI ensemble's implied drift + residual-calibrated vol
        # instead of raw historical mean return (which ignores the AI's forward view).

        # (A) AI-IMPLIED DRIFT: annualized daily drift from the AI forecast path
        if lstm_path is not None and len(lstm_path) >= 2 and last_price > 0:
            # Log-return implied by AI path from today to horizon end
            ai_total_log_return = np.log(lstm_path[-1] / last_price)
            mu_ai = ai_total_log_return / forecast_days   # per-day log drift
        else:
            mu_ai = mu + drift_bias  # fallback to historical if AI path unavailable

        # (B) RESIDUAL-CALIBRATED VOLATILITY:
        # Measure how much actual recent prices deviated from the AI's in-sample fit.
        # We approximate this by: residual_vol = std of (actual_return - AI implied step)
        # If unavailable, blend GARCH vol with rolling 21-day realized vol.
        try:
            actual_recent = df_fc['price_close'].values[-forecast_days-1:]
            if lstm_path is not None and len(actual_recent) >= 2:
                ai_step_returns = np.diff(np.log(lstm_path + 1e-9))[:len(actual_recent)-1]
                actual_step_returns = np.diff(np.log(actual_recent + 1e-9))
                min_len = min(len(ai_step_returns), len(actual_step_returns))
                residuals = actual_step_returns[:min_len] - ai_step_returns[:min_len]
                residual_vol = float(np.std(residuals)) if min_len > 2 else sigma_current
            else:
                residual_vol = sigma_current
            # Blend: 60% GARCH structure + 40% AI residual (retains clustering + calibration)
            sigma_blended = np.array([
                0.6 * float(s) + 0.4 * residual_vol for s in sigma_forecast
            ])
            sigma_blended = np.clip(sigma_blended, sigma_long_term * 0.3, sigma_long_term * 4.0)
        except Exception:
            sigma_blended = np.array(sigma_forecast)

        # (C) SIMULATE PATHS anchored on AI-implied drift, noise from residual vol
        # (C) SIMULATE PATHS (Vectorized NumPy implementation for M3 Speed)
        Z = np.random.normal(size=(forecast_days, n_sims))
        s_v = sigma_blended.reshape(-1, 1) # (days, 1) for broadcasting
        
        # Calculate all log-returns in one shot (GBM formula: r = (mu - 0.5*sigma^2) + sigma*Z)
        daily_log_rets = (mu_ai - 0.5 * s_v**2) + (s_v * Z)
        
        # Prepend zeros row for starting point (Price at T=0)
        cum_log_rets = np.vstack([np.zeros(n_sims), np.cumsum(daily_log_rets, axis=0)])
        
        # Final price paths: P_t = P_0 * exp(sum of daily log rets)
        simulated_paths = last_price * np.exp(cum_log_rets)

        
        # 1.5 Backtest Accuracy (Diagnostic) — Dynamic Horizon Sync (Phase 8)
        with st.spinner(f"Validating {forecast_days}-Day Accuracy..."):
            precision_score, mape_raw = calculate_backtest_accuracy(df_fc, sector_name=sector_val, quality_score=drift_score, test_size=forecast_days)

        # ── ROW 2: AI Metrics (Horizontal Cards) ─────────────────────────────
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("ML Ensemble Target", f"€{lstm_path[-1]:.2f}" if lstm_path is not None else "N/A", delta=f"{lstm_return*100:.2f}%" if lstm_return else "N/A")
            if lstm_path is not None:
                st.session_state[f"ai_target_for_de_{fc_ticker}"] = float(lstm_path[-1])
                st.caption(f"→ Synced to Decision Engine TP1")
        
        with mcol2:
            sent_label = "Bullish" if avg_sent > 0.1 else "Bearish" if avg_sent < -0.1 else "Neutral"
            st.metric("News Sentiment Mood", sent_label, delta=f"{avg_sent:.2f}")
        with mcol3:
            # Smart Money Momentum Logic
            # Compute OBV ROC directly from df_fc (raw data always available)
            _raw_obv = (np.sign(df_fc['price_close'].diff().fillna(0)) * df_fc['volume']).cumsum()
            _obv_roc = _raw_obv.pct_change(5).replace([np.inf, -np.inf], 0).fillna(0)
            obv_short = _obv_roc.tail(5).mean()
            obv_long  = _obv_roc.tail(20).mean()
            sm_spirit = "Accumulation" if obv_short > obv_long else "Distribution"
            st.metric("Smart Money Spirit", sm_spirit, delta="Positive Flow" if sm_spirit == "Accumulation" else "Heavy Selling")
        with mcol4:
            if precision_score is not None:
                p_val = f"{precision_score:.1f}%"
                p_label = f"Model Precision ({forecast_days}d Holdout)"
                p_delta = f"±{mape_raw*100:.1f}% uncertainty" if mape_raw else None
            else:
                p_val, p_label, p_delta = "N/A", f"Model Precision ({forecast_days}d Holdout)", None
            st.metric(p_label, p_val, delta=p_delta)
            
        # Highlight divergence
        if (sent_label == "Bearish" and sm_spirit == "Accumulation") or (sent_label == "Bullish" and sm_spirit == "Distribution"):
            div_type = "BULLISH DIVERGENCE (Smart Money Accumulating despite Retail Fear)" if sent_label == "Bearish" else "BEARISH DIVERGENCE (Smart Money Distributing despite Retail Greed)"
            div_color = "#2ecc71" if sent_label == "Bearish" else "#e74c3c"
            div_icon = "📈" if sent_label == "Bearish" else "📉"
            st.markdown(f"<div style='margin-top:10px; padding:12px 18px; background:linear-gradient(90deg, {div_color}22, rgba(0,0,0,0)); border-left:4px solid {div_color}; border-radius:6px;'><b style='color:{div_color}; font-size:1.0rem;'>{div_icon} HIGH PROBABILITY SET-UP: {div_type}</b><br><span style='font-size:0.85rem; color:#ccc;'>Institutions and Smart Money are actively positioning in direct opposition to retail sentiment. This severe dislocation heavily tilts risk/reward for a contrarian entry. <b>Actionable edge: Wait for break of structure in direction of Smart Money.</b></span></div>", unsafe_allow_html=True)

        # ── AI TRADING SIGNATURE ─────────────────────────────────────────────
        # Pre-compute all levels for the card
        p5_final   = np.percentile(simulated_paths[-1, :], 5)
        p10_final  = np.percentile(simulated_paths[-1, :], 10)
        p90_final  = np.percentile(simulated_paths[-1, :], 90)
        p95_final  = np.percentile(simulated_paths[-1, :], 95)
        _ai_target = float(lstm_path[-1]) if lstm_path is not None else last_price
        _ai_stop   = float(p10_final)
        _ai_tp2    = float(p90_final)
        _ai_upside = (lstm_return * 100) if lstm_return is not None else 0

        # ── Conviction Score (3-Pillar: 0-3) ────────────────────────────────
        _conv_pts  = 0
        _conv_pts += 1 if _ai_upside >= 3 else 0
        _conv_pts += 1 if sm_spirit == "Accumulation" else 0
        _conv_pts += 1 if avg_sent > 0.05 else 0

        # R/R based on Monte Carlo bands
        _sig_risk   = last_price - _ai_stop
        _sig_reward = _ai_target - last_price
        _sig_rr     = (_sig_reward / _sig_risk) if _sig_risk > 0 else 0

        # ── Executive Verdict ────────────────────────────────────────────────
        if _conv_pts == 3 and _sig_rr >= 1.5:
            _sig_verdict, _sig_color, _sig_badge = "STRONG LONG", "#00ffcc", "HIGH CONVICTION"
            _sig_desc = (f"All 3 pillars are aligned: Projects +{_ai_upside:.1f}% upside, "
                         f"institutions are in Accumulation mode, and news sentiment is "
                         f"{'Bullish' if avg_sent > 0.1 else 'leaning constructive'}. "
                         f"A {_sig_rr:.1f}x R/R setup with Monte Carlo support — ideal for a full position.")
        elif _conv_pts >= 2 and _sig_rr >= 1.0:
            _sig_verdict, _sig_color, _sig_badge = "BUY / ACCUMULATE", "#2ecc71", "MODERATE CONVICTION"
            _sig_desc = (f"2 of 3 pillars are constructive. Targets €{_ai_target:.2f} "
                         f"({_ai_upside:+.1f}%), Smart Money shows {sm_spirit}. "
                         f"R/R of {_sig_rr:.1f}x supports a partial position entry. "
                         f"Reserve allocation for a dip toward €{_ai_stop:.2f}.")
        elif _ai_upside <= -3:
            _sig_verdict, _sig_color, _sig_badge = "REDUCE / HEDGE", "#e74c3c", "BEARISH SIGNAL"
            _sig_desc = (f"Model projects {_ai_upside:.1f}% downside to €{_ai_target:.2f}. "
                         f"Smart Money shows {sm_spirit} and sentiment is {sent_label}. "
                         f"Consider reducing exposure or hedging until price stabilizes above €{_ai_stop:.2f}.")
        elif _conv_pts == 0:
            _sig_verdict, _sig_color, _sig_badge = "AVOID / WAIT", "#e74c3c", "NO CONVICTION"
            _sig_desc = (f"All 3 pillars are negative: Upside is weak ({_ai_upside:+.1f}%), "
                         f"Smart Money shows {sm_spirit}, and sentiment is {sent_label}. "
                         f"Best to stay flat or look for a better setup.")
        else:
            _sig_verdict, _sig_color, _sig_badge = "NEUTRAL / MONITOR", "#f1c40f", "MIXED SIGNALS"
            _sig_desc = (f"Conflicting signals: Projects {_ai_upside:+.1f}% to €{_ai_target:.2f}, "
                         f"but Smart Money ({sm_spirit}) and sentiment ({sent_label}) "
                         f"are not fully aligned. Monitor for a confluence trigger before entry.")

        # ── Reasoning pills ─────────────────────────────────────────────────
        def _pill(label, value, ok):
            c = "#2ecc71" if ok else "#e74c3c"
            return (f"<span style='display:inline-flex; align-items:center; gap:5px; background:rgba(255,255,255,0.05); "
                    f"border:1px solid {c}55; border-radius:20px; padding:4px 10px; font-size:0.78rem; margin:3px;'>"
                    f"<span style='color:{c}; font-weight:700;'>{'✓' if ok else '✗'}</span> "
                    f"<span style='color:#ccc;'>{label}:</span> "
                    f"<span style='color:#fff; font-weight:700;'>{value}</span></span>")

        _pill_ai   = _pill("Upside",    f"{_ai_upside:+.1f}%",  _ai_upside >= 3)
        _pill_sm   = _pill("Smart Money",  sm_spirit,              sm_spirit == "Accumulation")
        _pill_sent = _pill("Sentiment",    sent_label,             avg_sent > 0.05)
        _pill_rr   = _pill("R/R",          f"{_sig_rr:.1f}x",      _sig_rr >= 1.5)
        _pill_prec = _pill("ML Precision", f"{precision_score:.1f}%" if precision_score else "N/A", (precision_score or 0) >= 75)

        _unc_str = f"±{mape_raw*100:.1f}% CI" if mape_raw else ""
        _vix_now_sig = float(prices_full[prices_full['ticker']=='^VIX']['price_close'].iloc[-1]) \
            if not prices_full[prices_full['ticker']=='^VIX'].empty else 20.0
        _playbook = ("Mean Reversion / Range Trading" if _vix_now_sig > 25
                     else "Trend Following / Breakout" if _vix_now_sig < 15
                     else "Selective / Stock Picker's Market")

        def _hex_rgb(h): h=h.lstrip('#'); return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
        _bg_rgb = _hex_rgb(_sig_color)

        html_content = f"""
<div style='background:rgba(10,15,25,0.7); border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:22px 26px; margin:18px 0;'>
<!-- Header Row -->
<div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px; margin-bottom:18px;'>
<div>
<div style='font-size:0.65rem; color:#8899aa; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px;'>AI Trading Signature</div>
<div style='font-size:1.8rem; font-weight:900; color:{_sig_color}; text-shadow:0 0 20px rgba({_bg_rgb},0.5); line-height:1;'>{_sig_verdict}</div>
<div style='font-size:0.75rem; color:{_sig_color}; background:rgba({_bg_rgb},0.12); border:1px solid rgba({_bg_rgb},0.35); border-radius:20px; display:inline-block; padding:2px 10px; margin-top:6px;'>{_sig_badge}</div>
</div>
<div style='text-align:right;'>
<div style='font-size:0.65rem; color:#8899aa; text-transform:uppercase; margin-bottom:4px;'>VIX Context</div>
<div style='font-size:1.1rem; font-weight:700; color:#f1c40f;'>VIX {_vix_now_sig:.1f}</div>
<div style='font-size:0.78rem; color:#aaa;'>{_playbook}</div>
</div>
</div>
<!-- Signal Pills -->
<div style='margin-bottom:16px;'>{_pill_ai}{_pill_sm}{_pill_sent}{_pill_rr}{_pill_prec}</div>
<!-- Rationale -->
<div style='font-size:0.88rem; color:#dde; line-height:1.6; margin-bottom:18px; border-left:3px solid rgba({_bg_rgb},0.6); padding-left:14px;'>{_sig_desc}</div>
<!-- Trade Setup Snapshot -->
<div style='border-top:1px solid rgba(255,255,255,0.08); padding-top:16px;'>
<div style='font-size:0.65rem; color:#8899aa; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;'>Trade Setup Snapshot</div>
<div style='display:grid; grid-template-columns:repeat(5,1fr); gap:8px; font-size:0.82rem;'>
<div style='background:rgba(255,255,255,0.04); border-radius:8px; padding:10px 12px; border-top:2px solid #3498db;'>
<div style='color:#8899aa; font-size:0.68rem; margin-bottom:4px;'>CURRENT PRICE</div>
<div style='color:#fff; font-weight:800; font-size:1.05rem;'>€{last_price:.2f}</div>
</div>
<div style='background:rgba(46,204,113,0.08); border-radius:8px; padding:10px 12px; border-top:2px solid #2ecc71;'>
<div style='color:#8899aa; font-size:0.68rem; margin-bottom:4px;'>ENTRY (NOW)</div>
<div style='color:#2ecc71; font-weight:800; font-size:1.05rem;'>€{last_price:.2f}</div>
<div style='color:#8899aa; font-size:0.65rem;'>{forecast_days}d forecast</div>
</div>
<div style='background:rgba(231,76,60,0.08); border-radius:8px; padding:10px 12px; border-top:2px solid #e74c3c;'>
<div style='color:#8899aa; font-size:0.68rem; margin-bottom:4px;'>STOP (MC P10)</div>
<div style='color:#e74c3c; font-weight:800; font-size:1.05rem;'>€{_ai_stop:.2f}</div>
<div style='color:#8899aa; font-size:0.65rem;'>Risk: {((last_price-_ai_stop)/last_price*100):.1f}%</div>
</div>
<div style='background:rgba(0,255,204,0.06); border-radius:8px; padding:10px 12px; border-top:2px solid #00ffcc;'>
<div style='color:#8899aa; font-size:0.68rem; margin-bottom:4px;'>TARGET 1 (ML)</div>
<div style='color:#00ffcc; font-weight:800; font-size:1.05rem;'>€{_ai_target:.2f}</div>
<div style='color:#8899aa; font-size:0.65rem;'>{_unc_str} · {_ai_upside:+.1f}%</div>
</div>
<div style='background:rgba(52,152,219,0.06); border-radius:8px; padding:10px 12px; border-top:2px solid #3498db;'>
<div style='color:#8899aa; font-size:0.68rem; margin-bottom:4px;'>TARGET 2 (MC P90)</div>
<div style='color:#3498db; font-weight:800; font-size:1.05rem;'>€{_ai_tp2:.2f}</div>
<div style='color:#8899aa; font-size:0.65rem;'>Extended scenario</div>
</div>
</div>
<!-- R/R Progress Bar -->
<div style='margin-top:14px; display:flex; align-items:center; gap:12px;'>
<div style='color:#8899aa; font-size:0.75rem; white-space:nowrap;'>R/R Ratio</div>
<div style='flex:1; background:rgba(255,255,255,0.08); border-radius:4px; height:8px; position:relative; overflow:hidden;'>
<div style='width:{min(100, _sig_rr/3.0*100):.0f}%; height:100%; background:linear-gradient(90deg,#e74c3c,#f1c40f,#2ecc71,#00ffcc); border-radius:4px;'></div>
</div>
<div style='color:{_sig_color}; font-weight:800; font-size:0.9rem; white-space:nowrap;'>{_sig_rr:.2f}x</div>
<div style='color:#8899aa; font-size:0.75rem; white-space:nowrap;'>{'FAVORABLE' if _sig_rr>=1.5 else 'MARGINAL' if _sig_rr>=1.0 else 'POOR'}</div>
</div>
<!-- 90% Confidence Interval note -->
<div style='margin-top:10px; font-size:0.78rem; color:#8899aa; text-align:center;'>
Monte Carlo 90% CI: <b style='color:#fff;'>€{p5_final:.2f}</b> ↔ <b style='color:#fff;'>€{p95_final:.2f}</b> &nbsp;·&nbsp; {forecast_days}-Day Horizon &nbsp;·&nbsp; {n_sims:,} simulations
</div>
</div>
</div>
"""
        st.markdown(html_content, unsafe_allow_html=True)

        # ── ROW 3: Main Chart (Full Width) ── (Moved to Top) ───────────────────
        render_header("ai", f"AI Ensemble vs Stochastic Monte Carlo: {fc_ticker}")
        fig_fc = go.Figure()
        # Include today's date so all lines start from the last known price point
        future_dates = pd.date_range(start=df_fc["date"].max(), periods=forecast_days+1, freq='B')
        
        for i in range(min(n_sims, 50)): 
            fig_fc.add_trace(go.Scatter(x=future_dates, y=simulated_paths[:, i], mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
        
        mean_path = simulated_paths.mean(axis=1)
        fig_fc.add_trace(go.Scatter(x=future_dates, y=mean_path, name="Monte Carlo Mean Path", line=dict(color="rgba(241, 196, 15, 0.5)", width=2, dash="dash")))
        
        if lstm_path is not None:
            # Prepend today's price to visually close the gap on the chart
            lstm_plot_y = np.insert(lstm_path, 0, last_price)
            
            # ── Ensemble Uncertainty Bands (Calibrated from Backtest MAPE) ─────────
            if mape_raw is not None:
                # Temporal Confidence Decay: band widens with sqrt(t)
                time_decay = np.zeros(len(lstm_plot_y))
                time_decay[1:] = np.sqrt(np.arange(1, len(lstm_path) + 1) / len(lstm_path))
                lstm_upper = lstm_plot_y * (1 + mape_raw * time_decay)
                lstm_lower = lstm_plot_y * (1 - mape_raw * time_decay)
                # Shaded confidence region
                fig_fc.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates[::-1]),
                    y=list(lstm_upper) + list(lstm_lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(0,229,255,0.08)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'Ensemble ±{mape_raw*100:.1f}% Confidence',
                    showlegend=True
                ))
            # Central Ensemble path (on top)
            fig_fc.add_trace(go.Scatter(
                x=future_dates, y=lstm_plot_y,
                name="AI Ensemble Most Likely Path",
                line=dict(color="#00E5FF", width=4)
            ))
        
        p10 = np.percentile(simulated_paths, 10, axis=1)
        p90 = np.percentile(simulated_paths, 90, axis=1)
        fig_fc.add_trace(go.Scatter(x=future_dates, y=p10, name="Lower Risk Bound (90%)", line=dict(color="rgba(255,0,0,0.5)", width=2, dash="dot")))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=p90, name="Upper Reward Bound (90%)", line=dict(color="rgba(0,255,0,0.5)", width=2, dash="dot")))
 
        fig_fc.update_layout(template="plotly_dark", height=600, yaxis_title="Price (€)", margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown("---")

        # ── ROW 2.5: Intelligence Diagnostic (Breakdown) ──────────────────────
        _fc_meta = companies_full[companies_full['ticker'] == fc_ticker]
        render_header("activity", "AI Reasoning & Diagnostic Insight")
        dcol1, dcol2 = st.columns([1, 1])
        
        with dcol1:
            # Model Input Reasoning (SHAP)
            render_header("activity", "Model Input Reasoning (SHAP)")
            if feat_imp:
                pretty_feat_map = {
                    'price_close': 'Price Level',
                    'daily_return_pct': 'Volatility/Return',
                    'spy_ret': 'Market (SPY)', 
                    'vix_ret': 'Fear Index (VIX)',
                    'vol_surge': 'Volume Spike', 
                    'quality_score_norm': 'Quality Score'
                }
                imp_df = pd.DataFrame([
                    {'Feature': pretty_feat_map.get(k, k), 'Weight (%)': v}
                    for k, v in feat_imp.items()
                ]).sort_values('Weight (%)', ascending=True)

                fig_imp = px.bar(
                    imp_df, x='Weight (%)', y='Feature', orientation='h',
                    template="plotly_dark", height=300,
                    color='Weight (%)', color_continuous_scale="Viridis"
                )
                fig_imp.update_layout(xaxis_title="Influence (%)", showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Insufficient data for SHAP analysis.")
            
        with dcol2:
            # ── Weighted Ensemble Metrics Panel (persisted via session_state) ──
            _em_display = st.session_state.get('ensemble_metrics', {})
            if _em_display:
                st.markdown("")
                render_header("activity", "Ensemble Performance Breakdown")
                rows_html = ""
                for model_name, m in _em_display.items():
                    icon_key = "brain" if model_name == "LSTM" else "bot" if model_name == "Transformer" else "dna"
                    icon_svg = SVG_ICONS[icon_key].replace('width="18"','width="14"').replace('height="18"','height="14"')
                    w_pct = float(m["Weight"].replace("%", ""))
                    bar_color = "#00ffcc" if w_pct == max(float(v["Weight"].replace("%", "")) for v in _em_display.values()) else "#3498db"
                    conf_score = max(0.0, 100.0 - m["MAPE (%)"])
                    conf_color = "#2ecc71" if conf_score >= 90 else "#f1c40f" if conf_score >= 80 else "#e74c3c"
                    rows_html += f"""
                    <tr>
                        <td style='padding:8px 12px; font-weight:600;'>{icon_svg} {model_name}</td>
                        <td style='padding:8px 12px; text-align:center; color:#e74c3c;'>${m["RMSE"]}</td>
                        <td style='padding:8px 12px; text-align:center; color:#e67e22;'>{m["MAPE (%)"]:.1f}%</td>
                        <td style='padding:8px 12px; text-align:center; font-weight:700; color:{conf_color};'>{conf_score:.1f}%</td>
                        <td style='padding:8px 12px; text-align:center;' title='0% happens when mean-reverting models predict flatlines during a trending test set.'>{m["Dir. Acc"]}</td>
                        <td style='padding:8px 12px; text-align:center; font-weight:700; color:#f1c40f;'>€{m.get('Target', 0):.2f}</td>
                        <td style='padding:8px 12px; min-width:120px;'>
                            <div style='display:flex; align-items:center; gap:6px;'>
                                <div style='background:{bar_color}; height:8px; border-radius:4px; width:{w_pct:.0f}%; max-width:80px;'></div>
                                <span style='color:{bar_color}; font-weight:700; font-size:0.9rem;'>{m["Weight"]}</span>
                            </div>
                        </td>
                    </tr>"""
                st.markdown(f"""
                <table style='width:100%; border-collapse:collapse; font-size:0.88rem; color:#e0e0e0;'>
                    <thead>
                        <tr style='border-bottom:1px solid rgba(255,255,255,0.15); color:#8899aa; font-size:0.78rem; text-transform:uppercase;'>
                            <th style='padding:6px 12px; text-align:left;'>Model</th>
                            <th style='padding:6px 12px; text-align:center;'>RMSE ($)</th>
                            <th style='padding:6px 12px; text-align:center;'>MAPE</th>
                            <th style='padding:6px 12px; text-align:center;' title='Confidence Score (100 - MAPE)'>Confidence <span style='cursor:help;'>ⓘ</span></th>
                            <th style='padding:6px 12px; text-align:center;' title='Directional Accuracy evaluated on the holdout window'>Dir. Acc <span style='cursor:help;'>ⓘ</span></th>
                            <th style='padding:6px 12px; text-align:center;'>Target Vote</th>
                            <th style='padding:6px 12px; text-align:left;'>Weight</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)
                st.caption("💡 Weight ∝ 1/RMSE — the model with the lowest error has the highest influence on the final forecast.")
                st.markdown("<div style='font-size:0.85rem; color:#8899aa; margin-top:4px;'><b>Note on Dir. ACC 0%:</b> LSTM & Transformer are mathematically prone to 0% Directional Accuracy because they tend to output mean-reverting flatlines. If the real price trends slightly, the strict binary direction check fails. <b>PatchTST</b>, functioning as a structural forecaster, is more likely to yield 100% on trajectory direction.</div>", unsafe_allow_html=True)

            # ── Meta Intelligence Panel ──────────────────────────────────────
            with st.expander("🧪 Meta Intelligence — Anchor History & VIX Regime Analysis", expanded=False):
                _perf_data = _load_perf_log()["logs"]
                if len(_perf_data) < 2:
                    st.info("📊 Insufficient history. Please run Smart Blend at least twice to build Regime analysis.")
                else:
                    # Section A: Recent Anchor History
                    st.markdown("​**Anchor Model by Run (Latest 20):**")
                    anchor_rows = [{
                        "Time": e["ts"], "Ticker": e["ticker"],
                        "Horizon": f"{e['horizon']}D",
                        "VIX": e["vix"],
                        "Regime": "⬆️ High" if e["regime"]=="high_vix" else "⬇️ Low",
                        "Anchor Model": e["anchor"]
                    } for e in _perf_data[-20:][::-1]]
                    st.dataframe(pd.DataFrame(anchor_rows), use_container_width=True, hide_index=True)

                    # Section B: VIX-Regime Performance Chart
                    if len(_perf_data) >= 3:
                        regime_rows = []
                        for e in _perf_data:
                            for mn, mv in e["models"].items():
                                regime_rows.append({
                                    "Model": mn,
                                    "Regime": "High VIX (>25)" if e["regime"]=="high_vix" else "Low VIX (≤25)",
                                    "RMSE": mv["rmse"]
                                })
                        regime_df = pd.DataFrame(regime_rows)
                        avg_r = regime_df.groupby(["Model","Regime"])["RMSE"].mean().reset_index()
                        if not avg_r.empty:
                            st.markdown("​**📈 Model RMSE by VIX Regime:**")
                            fig_meta = px.bar(
                                avg_r, x="Model", y="RMSE", color="Regime",
                                barmode="group", template="plotly_dark", height=260,
                                color_discrete_map={"High VIX (>25)": "#e74c3c", "Low VIX (≤25)": "#2ecc71"},
                                labels={"RMSE": "Avg RMSE ($)"}
                            )
                            fig_meta.update_layout(margin=dict(t=10,b=0,l=0,r=0),
                                                   legend=dict(orientation="h", y=1.12))
                            st.plotly_chart(fig_meta, use_container_width=True)
                            st.caption("💡 Lower bar = more effective model in that market regime. Key question: Does LSTM or Transformer perform better during VIX spikes?")

            st.markdown("<br>", unsafe_allow_html=True)


# ── STRATEGY ENGINE ──────────────────────────────────────────────────────────
def run_backtest_simulation(bt_ticker, bt_prices, strategy_type, sl_pct, tp_pct, tx_cost_pct, initial_capital, reco_df):
    """
    Core simulator: Runs a single-ticker backtest for a specific strategy.
    Returns a dict with processed metrics and curves.
    """
    if len(bt_prices) < 60:
        return None

    ticker_score_row = reco_df[reco_df["ticker"] == bt_ticker]
    static_score = int(ticker_score_row["score"].iloc[0]) if not ticker_score_row.empty else 50

    prices_arr  = bt_prices["price_close"].values
    returns_arr = bt_prices["daily_return_pct"].values / 100
    dates_arr   = bt_prices["date"].values
    
    # Fetch indicators
    ma20_arr = bt_prices.get("ma_20", np.zeros_like(prices_arr)).values
    ma50_arr = bt_prices.get("ma_50", np.zeros_like(prices_arr)).values
    rsi_arr  = bt_prices.get("rsi", np.full_like(prices_arr, 50)).values

    # Z-Score Calculation
    price_series = pd.Series(prices_arr)
    ma60 = price_series.rolling(60).mean().values
    std60 = price_series.rolling(60).std().values
    z_scores = np.zeros_like(prices_arr)
    for j in range(len(prices_arr)):
        if std60[j] > 0 and not np.isnan(std60[j]):
            z_scores[j] = (prices_arr[j] - ma60[j]) / std60[j]

    position   = np.zeros(len(bt_prices))
    in_position = False
    entry_price_val = 0.0
    trade_log = []
    
    for i in range(1, len(prices_arr)):
        current_price = prices_arr[i]
        p_date = str(dates_arr[i])[:10]
        position[i] = position[i-1]
        
        # 1. Exit Conditions
        if in_position:
            pnl_pct = (current_price - entry_price_val) / entry_price_val
            exit_signal = False
            exit_reason = ""
            
            if pnl_pct <= -sl_pct:
                exit_signal = True; exit_reason = "Stop Loss"
            elif tp_pct > 0 and pnl_pct >= tp_pct:
                exit_signal = True; exit_reason = "Take Profit"
            
            if not exit_signal:
                if "Trend Following" in strategy_type:
                    if ma20_arr[i] < ma50_arr[i]: exit_signal = True; exit_reason = "MA Death Cross"
                elif "RSI Mean Reversion" in strategy_type:
                    if rsi_arr[i] > 70: exit_signal = True; exit_reason = "RSI Overbought"
                elif "Z-Score" in strategy_type:
                    if z_scores[i] > 0.5: exit_signal = True; exit_reason = "Z-Score > 0.5"
                elif "Institutional Quality" in strategy_type:
                    if static_score < 60 or ma20_arr[i] < ma50_arr[i]: exit_signal = True; exit_reason = "Quality Degraded/Trend Change"
                elif "Buy on Dip" in strategy_type or "Multi-Indicator Breakout" in strategy_type:
                    if current_price < ma50_arr[i]: exit_signal = True; exit_reason = "Price < MA50"
                    
            if exit_signal:
                position[i] = 0; in_position = False
                trade_log.append({"Date": p_date, "Action": "🔴 SELL", "Reason": exit_reason, "Price": f"€{current_price:.2f}", "PnL": f"{pnl_pct*100:+.1f}%"})
        
        # 2. Entry Conditions
        if not in_position:
            entry_signal = False; entry_reason = ""
            if "Trend Following" in strategy_type:
                if ma20_arr[i] > ma50_arr[i] and ma20_arr[i-1] <= ma50_arr[i-1]: entry_signal = True; entry_reason = "MA Golden Cross"
            elif "RSI Mean Reversion" in strategy_type:
                if rsi_arr[i] < 30 and rsi_arr[i-1] >= 30: entry_signal = True; entry_reason = "RSI Oversold"
            elif "Z-Score" in strategy_type:
                if z_scores[i] < -2.0 and z_scores[i-1] >= -2.0: entry_signal = True; entry_reason = "Z-Score < -2.0"
            elif "Institutional Quality" in strategy_type:
                if static_score >= 75 and ma20_arr[i] > ma50_arr[i]: entry_signal = True; entry_reason = "High Quality + Trend"
            elif "Buy on Dip" in strategy_type:
                if ma20_arr[i] > ma50_arr[i] and rsi_arr[i] < 40 and rsi_arr[i-1] >= 40: entry_signal = True; entry_reason = "Uptrend + RSI Dip"
            elif "Multi-Indicator Breakout" in strategy_type:
                if current_price > ma50_arr[i] and rsi_arr[i] > 50 and rsi_arr[i-1] <= 50: entry_signal = True; entry_reason = "Price>MA50 + RSI>50"
                
            if entry_signal:
                position[i] = 1; in_position = True; entry_price_val = current_price
                trade_log.append({"Date": p_date, "Action": "🟢 BUY", "Reason": entry_reason, "Price": f"€{current_price:.2f}", "PnL": "-"})

    pos_shifted = np.roll(position, 1); pos_shifted[0] = 0
    signal_changes = np.abs(np.diff(pos_shifted, prepend=pos_shifted[0]))
    strategy_returns = returns_arr * pos_shifted - signal_changes * tx_cost_pct
    cum_strategy = (1 + strategy_returns).cumprod()
    equity_curve = cum_strategy * initial_capital
    
    cum_bnh = (1 + returns_arr).cumprod()
    bnh_curve = cum_bnh * initial_capital
    
    total_return = (equity_curve[-1] / initial_capital - 1) * 100
    bnh_return = (bnh_curve[-1] / initial_capital - 1) * 100
    sharpe = ( (strategy_returns - 0.04/252).mean() / (strategy_returns.std() + 1e-9) ) * np.sqrt(252)
    max_dd = ((equity_curve - np.maximum.accumulate(equity_curve)) / np.maximum.accumulate(equity_curve)).min() * 100
    
    trade_returns = strategy_returns[signal_changes == 1]
    win_rate = (trade_returns > 0).sum() / max(len(trade_returns), 1) * 100
    n_trades = int(signal_changes.sum() / 2)

    return {
        "ticker": bt_ticker, "strategy": strategy_type,
        "total_return": total_return, "bnh_return": bnh_return, "sharpe": sharpe, "max_dd": max_dd,
        "win_rate": win_rate, "n_trades": n_trades, "dates_arr": dates_arr, "equity_curve": equity_curve,
        "bnh_curve": bnh_curve, "trade_log": trade_log
    }

# ── TAB: STRATEGY BACKTEST ───────────────────────────────────────────────────
if active_tab == "5. Backtest Lab":

    render_header("activity", "Strategy Backtesting Engine — Signal Simulator")
    st.markdown("""
    <div style='background:rgba(0,255,204,0.05); border:1px solid rgba(0,255,204,0.2);
                border-radius:8px; padding:12px 16px; margin-bottom:16px; font-size:0.85rem; color:#aaa;'>
    <span style='color:#00ffcc; font-weight:900;'>[INFO]</span> <b>How it works:</b> Select a trading rule or run a tournament to find the best logic for a specific ticker.
    The engine simulates every signal on <b>5 years of historical data</b>.
    </div>
    """, unsafe_allow_html=True)

    bt_col1, bt_col2 = st.columns([1, 2])

    with bt_col1:
        st.markdown("#### Trading Rule Configuration")
        _bt_options = [t for t in all_tickers if t not in ["^VIX","SPY","^GSPC","^DJI","^IXIC"]]
        
        # Move Mode Toggle OUTSIDE the form to trigger immediate UI rerun for 'disabled' logic
        bt_mode = st.radio("Simulation Mode", ["Single Strategy", "Find Best Strategy (Auto-Run All)"], index=0, horizontal=True)
        
        # --- Market Regime Integration ---
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03); border:1px solid {regime_ui_color}; 
                    border-radius:6px; padding:8px 12px; margin-bottom:12px;'>
            <span style='font-size:0.75rem; color:#aaa; font-weight:700; text-transform:uppercase;'>Global Market Regime</span><br>
            <span style='color:{regime_ui_color}; font-weight:900;'>{regime}</span>
        </div>
        """, unsafe_allow_html=True)
        
        _strat_options = [
            "Institutional Quality Pulse (AI Score > 75)",
            "Trend Following (MA20/50 Cross)", 
            "RSI Mean Reversion (30/70)",
            "Z-Score Mean Reversion (Deep Value)",
            "Buy on Dip (Uptrend + Oversold)",
            "Multi-Indicator Breakout (Price>MA50 + RSI>50)"
        ]
        
        _rec_idx = 0
        _rec_msg = ""
        if "BEAR" in regime.upper() or "CAUTION" in regime.upper():
            _rec_idx = 3 # Z-Score Mean Reversion
            _rec_msg = "💡 **Regime Filter:** Z-Score or RSI Mean Reversion typically outperforms in sideways/volatile markets."
        elif "NEUTRAL" in regime.upper() or "SIDEWAYS" in regime.upper():
            _rec_idx = 2 # RSI Mean Reversion
            _rec_msg = "💡 **Regime Filter:** Mean Reversion strategies are preferred during range-bound regimes."
        elif "BULL" in regime.upper():
            _rec_idx = 1 # Trend Following
            _rec_msg = "💡 **Regime Filter:** Trend Following and Breakout strategies capture maximum upside in risk-on markets."
            
        if _rec_msg:
            st.caption(_rec_msg)
        
        with st.form("backtest_form"):
            bt_ticker = st.selectbox("Select Ticker to Backtest", options=_bt_options, format_func=format_ticker, key="bt_ticker_form")
            
            strategy_type = st.selectbox(
                "Select Strategy (if Single) 🎯 Regime Aligned",
                options=_strat_options,
                index=_rec_idx,
                disabled=(bt_mode != "Single Strategy")
            )
            
            st.markdown("###### Risk Management")
            sl_col, tp_col = st.columns(2)
            with sl_col: stop_loss = st.slider("Stop Loss (%)", 0, 30, 8)
            with tp_col: take_profit = st.slider("Take Profit (%)", 0, 100, 25)
                
            st.markdown("###### Capital Constraints")
            initial_capital = st.number_input("Initial Capital (€)", 1000, 1_000_000, 10000, step=1000)
            tx_cost_v = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, step=0.05)
            
            run_backtest = st.form_submit_button("▶ Run Backtest", use_container_width=True, type="primary")

    with bt_col2:
        if run_backtest and bt_ticker:
            tx_cost_pct = tx_cost_v / 100.0
            sl_pct = stop_loss / 100.0
            tp_pct = take_profit / 100.0
            
            # Filter bt_prices using the globally filtered 'prices' (respects Horizon sidebar)
            bt_prices = prices[prices["ticker"] == bt_ticker].sort_values("date").copy()
            
            if bt_mode == "Single Strategy":
                res = run_backtest_simulation(bt_ticker, bt_prices, strategy_type, sl_pct, tp_pct, tx_cost_pct, initial_capital, reco_df)
                if res:
                    st.session_state["bt_results"] = res
                    st.session_state["bt_leaderboard"] = None
            else:
                # AUTO-RUN ALL STRATEGIES
                all_strats = [
                    "Institutional Quality Pulse (AI Score > 75)",
                    "Trend Following (MA20/50 Cross)", 
                    "RSI Mean Reversion (30/70)",
                    "Z-Score Mean Reversion (Deep Value)",
                    "Buy on Dip (Uptrend + Oversold)",
                    "Multi-Indicator Breakout (Price>MA50 + RSI>50)"
                ]
                results = []
                progress_bar = st.progress(0)
                for idx, s in enumerate(all_strats):
                    progress_bar.progress((idx + 1) / len(all_strats), text=f"Simulating: {s}")
                    r = run_backtest_simulation(bt_ticker, bt_prices, s, sl_pct, tp_pct, tx_cost_pct, initial_capital, reco_df)
                    if r: results.append(r)
                progress_bar.empty()
                
                if results:
                    st.session_state["bt_leaderboard"] = results
                    # Set the best one as current results for main metrics display
                    best_res = max(results, key=lambda x: x["sharpe"])
                    st.session_state["bt_results"] = best_res

        # ── RENDER RESULTS ────────────────────────────────────────────────────
        if "bt_results" in st.session_state and st.session_state["bt_results"]:
            r = st.session_state["bt_results"]
            l_board = st.session_state.get("bt_leaderboard")
            
            if l_board:
                render_header("trophy", f"Strategy Tournament Leaderboard — {r['ticker']}")
                
                # Build Comparison Table
                comp_data = []
                for s_res in l_board:
                    comp_data.append({
                        "Strategy": s_res["strategy"],
                        "Return %": s_res["total_return"],
                        "Sharpe": s_res["sharpe"],
                        "Max DD %": s_res["max_dd"],
                        "Win Rate %": s_res["win_rate"],
                        "Trades": s_res["n_trades"]
                    })
                
                comp_df = pd.DataFrame(comp_data).sort_values("Sharpe", ascending=False)
                
                # Highlight Winner
                best_strat_name = comp_df.iloc[0]["Strategy"]
                st.markdown(f"""
                <div style='background:rgba(46, 204, 113, 0.1); border-left:4px solid #2ecc71; padding:15px; border-radius:4px; margin-bottom:20px;'>
                    <span style='color:#2ecc71; font-weight:800; font-size:1.1rem;'>WINNER: {best_strat_name}</span><br>
                    <span style='color:#bbb; font-size:0.9rem;'>For {r['ticker']}, this strategy offers the superior risk-adjusted performance (Sharpe: {comp_df.iloc[0]['Sharpe']:.2f}).</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(comp_df, use_container_width=True, hide_index=True,
                             column_config={
                                 "Return %": st.column_config.NumberColumn("Return", format="%.1f%%"),
                                 "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                                 "Max DD %": st.column_config.NumberColumn("Max DD", format="%.1f%%"),
                                 "Win Rate %": st.column_config.NumberColumn("Win Rate", format="%.0f%%")
                             })
                st.markdown("---")

            # Main Metrics (of best/selected)
            st.caption(f"Showing detailed analytics for: **{r['strategy']}**")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: render_metric_tile("Total Return",  f"{r['total_return']:+.1f}%", delta=r["total_return"])
            with m2: render_metric_tile("vs Buy&Hold",   f"{r['total_return']-r['bnh_return']:+.1f}%", delta=r["total_return"]-r["bnh_return"])
            with m3: render_metric_tile("Sharpe Ratio",  f"{r['sharpe']:.2f}")
            with m4: render_metric_tile("Max Drawdown",  f"{r['max_dd']:.1f}%")
            with m5: render_metric_tile("Win Rate",      f"{r['win_rate']:.0f}% ({r['n_trades']} trades)")

            # Chart
            fig_bt = go.Figure()
            # Overlay B&H
            fig_bt.add_trace(go.Scatter(x=r["dates_arr"], y=r["bnh_curve"], name="Buy & Hold", line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dot")))
            
            if l_board:
                # Add Top 3 Curves
                colors = ["#00ffcc", "#3498db", "#9b59b6"]
                for i, row in enumerate(comp_df.head(3).itertuples()):
                    # Find matches in results
                    s_dat = next(x for x in l_board if x["strategy"] == row.Strategy)
                    fig_bt.add_trace(go.Scatter(x=s_dat["dates_arr"], y=s_dat["equity_curve"], name=f"Rank {i+1}: {row.Strategy}", line=dict(color=colors[i], width=2 if i>0 else 3.5)))
            else:
                fig_bt.add_trace(go.Scatter(x=r["dates_arr"], y=r["equity_curve"], name=r["strategy"], line=dict(color="#00ffcc", width=3)))
            
            fig_bt.update_layout(template="plotly_dark", height=450, yaxis_title="Equity (€)", hovermode="x unified", legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig_bt, use_container_width=True)

            with st.expander("📋 View Trade Log"):
                st.dataframe(pd.DataFrame(r["trade_log"]), use_container_width=True, hide_index=True)
        else:
            st.info("👈 Configure your trading rule on the left and click **Run Simulation** to start.")


# ── TAB 8: SYSTEM METHODOLOGY ────────────────────────────────────────────────
if active_tab == "8. System Methodology":
    render_header("book", "DSS Framework & System Methodology")
    st.write("Transparency is the foundation of institutional-grade decision making. This document outlines the technical assumptions and boundaries of this Decision Support System.")
    
    st.markdown("---")
    
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("### 📡 1. Data Architecture & Sources")
        st.markdown("""
        The system operates on a hybrid ELT (Extract-Load-Transform) pipeline designed for low-latency financial analysis:
        - **Market Data**: Ingested via Yahoo Finance API. Includes adjusted close prices, historical volume, and corporate actions.
        - **Fundamentals**: Sourced from normalized Income Statements, Balance Sheets, and Cash Flow statements.
        - **Warehouse**: All processed intelligence is stored in a **DuckDB OLAP** database for sub-second query performance during deep-dives.
        """)
        
        st.markdown("### ⏳ 2. Lag & Latency Assumptions")
        st.info("""
        **Crucial**: Investors must account for the following inherent data latency:
        1. **Price Data**: T-1 (End of Day). This system is NOT designed for HFT or intra-day scalping.
        2. **Fundamental Metrics**: Subject to 'Reporting Lag'. Quarterly data is typically available 45-90 days after period end. The DSS always uses the *Latest Truly Available* data point to avoid look-ahead bias.
        """)

        st.markdown("### 🧪 3. Backtest Framework & Constraints")
        st.warning("""
        Backtest results are simulations and carry the following constraints:
        - **Transaction Costs**: Friction is modeled as a fixed % fee (default 0.1%).
        - **Slippage**: Assumes perfect liquidity (execution at Close price). Real-world slippage in low-cap stocks may degrade performance.
        - **Survivorship Bias**: The engine currently scans a fixed universe of active tickers.
        """)

    with m_col2:
        st.markdown("### 🛡️ 4. Data Integrity & Leakage Control")
        st.success("""
        To ensure "Professional Reliability", the system enforces **Point-in-Time** logic:
        - **No Look-Ahead**: When backtesting or training AI, the system strictly isolates information. A signal for Jan 1st 2024 is strictly prevented from 'seeing' any data point from Jan 2nd onwards.
        - **Ensemble Validation**: Predictions are balanced between Mean-Reversion (ARIMA) and Pattern-Recognition (LSTM) to avoid single-model overfitting.
        """)

        st.markdown("### 🔮 5. AI Forecast Boundaries")
        st.markdown("""
        Predictive models (Monte Carlo / LSTM / ARIMA) are **Probabilistic**, not Deterministic:
        - **Stochastic Nature**: Monte Carlo paths represent a distribution of possibilities based on historical volatility clustering (GARCH).
        - **Exogenous Risk**: The model does *not* account for geopolitical shocks, sudden regulatory changes, or 'Black Swan' events that have no historical numerical precedent.
        - **Confidence Intervals**: The 90% shadow bands represent statistical likelihood, leaving a 10% 'tail risk' for extreme movements.
        """)

        st.markdown("""
        <div style='background:rgba(255,255,255,0.05); padding:15px; border-radius:10px; border:1px solid rgba(255,255,255,0.1); margin-top:20px;'>
            <span style='color:#8899aa; font-weight:700; font-size:0.8rem; text-transform:uppercase;'>System Integrity Signature</span><br>
            <span style='font-family:monospace; color:#556677; font-size:0.75rem;'>SHA-256: DSS_VERSION_9.1_STABLE_KERNEL</span>
        </div>
        """, unsafe_allow_html=True)

st.sidebar.markdown("---")

# ── FEATURE 4: Sidebar Export Hub (HIDDEN) ──────────────────────────────────
# st.sidebar.markdown("---")
# st.sidebar.subheader("📥 Export Data (CSV)")
# csv_reco = reco_df.to_csv(index=False).encode('utf-8')
# st.sidebar.download_button("🔽 Download Recommendations", data=csv_reco, file_name="ai_reco.csv", mime="text/csv")
# 
# csv_prices = prices.to_csv(index=False).encode('utf-8')
# st.sidebar.download_button("🔽 Download Price History", data=csv_prices, file_name="price_history.csv", mime="text/csv")

auth.render_user_profile()
