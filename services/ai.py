"""LLM (Cohere) narratives and FinBERT sentiment."""
import numpy as np
import pandas as pd
import streamlit as st


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
- **Current Price**: €{_fmt(price)}
- **AI Quality Score**: {ai_score}/100
- **Analyst Recommendation**: {action} | **Consensus**: {consensus}
- **Analyst Price Target**: €{_fmt(target_p)} (Implied Upside: {_fmt(upside, 1)}%)
- **52-Week Position**: {_fmt(w52_pos, 0)}% of range

### Technical Indicators
- **RSI (14)**: {_fmt(rsi, 1)} {rsi_note}
- **MA Trend Signal**: {ma_signal}
- **Price Z-Score**: {_fmt(z_score, 2)}σ (deviation from 5-year historical mean; >+2=Historically expensive, <-2=Historically cheap)

### Valuation & Profitability
- **Forward P/E**: {_fmt(metrics.get('forward_pe', pe), 1)}x | **Trailing P/E**: {_fmt(pe, 1)}x | **PEG**: {_fmt(peg, 2)} | **P/B**: {_fmt(pb, 2)}x
- **ROE**: {_fmt(roe, 1, '%')} | **FCF Margin**: {_fmt(fcf, 1, '%')} | **Dividend Yield**: {_fmt(div_yield, 2, '%')}
- **Beta**: {_fmt(beta, 2)} | **Market Regime**: {regime}

---
Write a structured analysis with exactly these THREE sections:

### 1. 🎯 Investment Verdict
One clear paragraph (3-5 sentences). State the overall investment thesis, Buy/Hold/Sell, and primary driver.

### 2. 📊 Technical & Fundamental Analysis
One paragraph (3-5 sentences). Analyze the interplay between technical signals and the fundamental picture. Highlight any divergence or confirmation.

### 3. ⚠️ Key Risks & Catalysts
- Risk 1: (specific, quantitative)
- Risk 2: (specific, quantitative)
- Catalyst 1: (specific, quantitative)
- Catalyst 2: (specific, quantitative)

Rules: English only. Be direct and decisive. Reference specific data points. Under 350 words."""

        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
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


def get_unified_verdict(api_key: str, metrics: dict, nlp_result: dict, language: str = "English") -> str:
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
        rsi       = metrics.get("rsi", None)  # None = no data; avoid fake neutral RSI=50 in prompt
        ma_signal = metrics.get("ma_signal", "N/A")
        smart_money = metrics.get("smart_money", "N/A")
        pe        = metrics.get("pe_ratio", "N/A")
        peg       = metrics.get("peg_ratio", "N/A")
        fcf       = metrics.get("fcf_margin", "N/A")
        roe_raw   = metrics.get("roe", None)
        roe_pct   = round(float(roe_raw) * 100, 1) if not pd.isna(roe_raw) and roe_raw is not None else None
        op_margin = metrics.get("operating_margin", None)
        op_margin_pct = round(float(op_margin) * 100, 1) if not pd.isna(op_margin) and op_margin is not None else None
        gross_margin = metrics.get("gross_margin", None)
        gross_margin_pct = round(float(gross_margin) * 100, 1) if not pd.isna(gross_margin) and gross_margin is not None else None
        ev_ebitda  = metrics.get("ev_to_ebitda", None)
        # FCF Yield = Free Cash Flow / Market Cap
        _fcf_abs   = metrics.get("free_cashflow", None)
        _mktcap    = metrics.get("market_cap", None)
        def _notna(v): return v is not None and not pd.isna(v)
        fcf_yield  = round(float(_fcf_abs) / float(_mktcap) * 100, 1) if _notna(_fcf_abs) and _notna(_mktcap) and float(_mktcap) > 0 else None
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

        # Macro data
        vix_current = metrics.get("vix_current", "N/A")
        spy_trend   = metrics.get("spy_trend", 0)
        spy_str     = f"{float(spy_trend):+.2f}%" if isinstance(spy_trend, (int, float)) else "N/A"

        # NLP data
        nlp_score     = nlp_result.get("red_flag_score", 0)
        nlp_sentiment = nlp_result.get("sentiment", "Neutral")
        nlp_category  = nlp_result.get("risk_category", "None")
        nlp_reco      = nlp_result.get("recommendation", "N/A")
        nlp_insights  = nlp_result.get("key_insights", [])
        nlp_headlines = nlp_result.get("headlines_analyzed", 0)

        # Signal alignment check
        _ai_sc = int(ai_score) if str(ai_score).isdigit() else 50
        quant_bullish = _ai_sc >= 65
        quant_bearish = _ai_sc <= 38  # v4.0: aligned with WEAK tier threshold
        
        news_bullish  = nlp_score <= 35
        news_bearish  = nlp_score >= 50 or nlp_sentiment in ["Negative", "Critical"]
        
        if quant_bullish and news_bullish:
            alignment = "CONVERGENCE (BULLISH) — Both quantitative and qualitative signals are strong."
        elif quant_bullish and news_bearish:
            alignment = "DIVERGENCE — Strong fundamentals but negative news/high risk. High probability of surprise downside."
        elif quant_bearish and news_bullish:
            alignment = "DIVERGENCE — Positive news but weak fundamentals. Rally may be an unsustainable trap."
        elif quant_bearish and news_bearish:
            alignment = "CONVERGENCE (BEARISH) — Both quantitative and qualitative signals are weak."
        else:
            alignment = "MIXED/NEUTRAL — Signals are mixed. Weigh both fundamental value and immediate news risks carefully."

        # All prices in DB are already normalized to EUR
        prompt = f"""You are a Chief Investment Officer (CIO) at a top-tier hedge fund.
You have received both QUANTITATIVE data and QUALITATIVE news intelligence for a stock.
Your task: synthesize both and issue ONE definitive, actionable investment verdict.

## Stock: {ticker} ({company}) | Sector: {sector}

### DECISION GUIDANCE (apply judgment, not mechanical rules)
- Very weak fundamentals (AI Score < 35) + high news risk (Red Flag ≥ 60) → lean REDUCE/AVOID
- Elevated sentiment risk (Red Flag ≥ 70) → exercise caution regardless of fundamentals
- Statistically stretched price (Z-Score > +2.5) with mediocre fundamentals → avoid a BUY call
- If News Red Flag ≥ 40 or Sentiment is Neutral/Negative, explicitly mention negative catalysts in the verdict

---

### QUANTITATIVE (Fundamental & Technical)
- AI Quality Score: {ai_score}/100
- Price Z-Score: {z_score:+.2f}σ (deviation from 5-year historical mean; >+2=Historically expensive, <-2=Historically cheap)
- Price: €{_f(price)} | Analyst Target: €{_f(target_p)} | Implied Upside: {_f(upside, 1)}%
- 52W Range: €{_f(low_52w)} – €{_f(high_52w)} | % from MA200: {pct_from_ma200} | 52W Position: {_f(metrics.get('w52_pos','N/A'), 0)}%
- Technical Levels: S1=€{_f(support_s1)} | S2=€{_f(support_s2)} | R1=€{_f(resistance_r1)} | R2=€{_f(resistance_r2)} | Stop=€{_f(stop_loss_tech)}
- Moving Averages: MA20=€{_f(ma20_cur)} | MA50=€{_f(ma50_cur)}
- RSI: {_f(rsi, 1) if rsi is not None else 'N/A'} | MA Signal: {ma_signal} | Smart Money: {smart_money} | Forward P/E: {_f(metrics.get('forward_pe', pe), 1)}x | PEG: {_f(peg, 2)} | FCF Margin: {_f(fcf, 1, '%')}
- Profitability: ROE: {f"{roe_pct:+.1f}%" if roe_pct is not None else 'N/A'} | Operating Margin: {f"{op_margin_pct:.1f}%" if op_margin_pct is not None else 'N/A'} | Gross Margin: {f"{gross_margin_pct:.1f}%" if gross_margin_pct is not None else 'N/A'}
- Enterprise Valuation: EV/EBITDA: {f"{float(ev_ebitda):.1f}x" if ev_ebitda is not None else 'N/A'} | FCF Yield: {f"{fcf_yield:.1f}%" if fcf_yield is not None else 'N/A'}
- Debt/EBITDA: {_f(metrics.get('debt_ebitda', 'N/A'), 2)}x | Net Payout Yield: {_f(metrics.get('net_payout_yield_pct', 'N/A'), 2)}%
- Revenue Growth (YoY): {f"{float(metrics.get('revenue_growth') or 0) * 100:+.1f}%" if metrics.get('revenue_growth') is not None else 'N/A'} | EPS Growth (YoY): {f"{float(metrics.get('earnings_growth') or 0) * 100:+.1f}%" if metrics.get('earnings_growth') is not None else 'N/A'}
- Earnings Surprise (last 2Q): {metrics.get('earnings_surprise_summary', 'N/A')}
- Market Regime: {regime}

### MACRO ENVIRONMENT
- VIX (Fear Index): {vix_current} (>25 = high panic, <15 = complacency)
- SPY (S&P 500) Trend: {spy_str}

### QUALITATIVE (News Intelligence — {nlp_headlines} sources analyzed)
- News Red Flag Score: {nlp_score}/100
- Sentiment: {nlp_sentiment} | Risk Category: {nlp_category}
- NLP Recommendation: "{nlp_reco}"
- Key News Signals: {'; '.join(nlp_insights[:3]) if nlp_insights else 'None'}

### SIGNAL ALIGNMENT
{alignment}

---
Write a unified analysis with FIVE sections. Total length: under 550 words.

### 💼 CIO Verdict
One decisive paragraph (3-4 sentences). Final call referencing BOTH quantitative and qualitative data.

### 💡 Signal Convergence Analysis
One paragraph on the interplay between news sentiment and fundamentals. If diverging, state which side you trust more and why.

### 🎯 Actionable Recommendation & Execution
State: **STRONG BUY / BUY / ACCUMULATE / HOLD / WATCH / REDUCE / AVOID**
CRITICAL LOGIC RULES:
1. If the Base Case target is lower than the Current Price (€{_f(price)}), the verdict MUST be HOLD, WATCH, or REDUCE. Never BUY or ACCUMULATE.
2. Only recommend ACCUMULATE/BUY if the Base Case Target is greater than or equal to the Current Price (€{_f(price)}).
3. If AI Quality Score is < 50, DO NOT recommend a straight BUY or STRONG BUY. The maximum bullish verdict allowed is WATCH or ACCUMULATE. Institutional discipline requires high structural quality, not just cheap valuation or buybacks.
4. Institutional Skepticism: Do not ignore cyclical or sector macro headwinds (e.g., China demand, EV transition, margin pressure) even if the News Red Flag Score is 0. A "clean" news feed does not negate underlying industry risks.
5. Execution framework: Do not suggest buying at Technical Support (S1/S2) if they are unrealistically far (e.g. >15% below spot). If Support is too far, advise waiting for a meaningful pullback.

### 📊 3-Scenario Valuation
**ANCHOR: Current Price = €{_f(price)} | Analyst Consensus Target = €{_f(target_p)}**
Scenarios are expressed as absolute price targets AND % change from CURRENT PRICE. The 3 scenarios MUST be logically consistent:
- **🐻 Bear Case (20% probability):** [Downside catalyst (e.g. multiple compression)] → Target: €[price] ([negative %] vs current). MUST be BELOW current price.
- **📈 Base Case (60% probability):** [Expected trajectory (e.g. earnings confirm uplift but market cools)] → Target: €[price] ([+/- %] vs current). If this is negative, the tone MUST be cautious.
- **🚀 Bull Case (20% probability):** [Upside catalyst (e.g. demand validation)] → Target: €[price] ([positive %] vs current). MUST be higher than Base Case.

### ⚠️ Key Risks to Monitor
3 concise bullet points on the most material risks.


Rules: Your final output MUST be written in {language} (translate section headers too, but keep emojis). Be decisive. Reference specific numbers. Start each section header on its own line."""

        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        return response.message.content[0].text, prompt

    except Exception as e:
        err = str(e)
        if "invalid api key" in err.lower() or "unauthorized" in err.lower():
            return "❌ **Invalid API Key.** Please check your Cohere API Key.", ""
        elif "rate limit" in err.lower():
            return "⏳ **Rate limit reached.** Please wait a moment and try again.", ""
        else:
            return f"❌ **AI Engine Error:** {err}", ""


@st.cache_resource(show_spinner="📥 Loading Institutional NLP Engine (FinBERT ~440MB)...")
def get_finbert_pipeline():
    """Loads the ProsusAI/finbert model for financial-specific sentiment analysis."""
    import os
    import warnings
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from transformers import pipeline
        try:
            return pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None


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
