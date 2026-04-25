"""
LLM Risk Audit Engine — Phase 5 of Hedge-Fund Engine v8.0
Uses Google Gemini (Free Tier) to analyze news headlines and detect
hidden risks that quantitative data cannot surface.

Usage:
    from etl.llm_parser import analyze_risk_with_llm
    result = analyze_risk_with_llm("AAPL", "Apple Inc.")
    # Returns: {"red_flag_score": 15, "sentiment": "Positive", "key_insights": [...]}
"""
import os
import json
import feedparser
import urllib.parse
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Cohere Client Initialization ──────────────────────────────────────────────
_COHERE_KEY = os.getenv("COHERE_API_KEY", "")
_client = None

def _get_client():
    """Lazy-initialize the Cohere client (avoids import-time errors if key is missing)."""
    global _client
    if _client is None:
        if not _COHERE_KEY:
            raise ValueError("COHERE_API_KEY not found in .env file.")
        import cohere
        _client = cohere.ClientV2(api_key=_COHERE_KEY)
    return _client


# ── News Fetcher (Free — Google News RSS) ────────────────────────────────────
def _fetch_recent_headlines(ticker: str, company_name: str, max_items: int = 12) -> list[str]:
    """Fetch recent news headlines from Google News RSS for a given ticker."""
    
    # 1. Use URL encoding for company names with spaces
    # 2. Append when:7d to ensure news is fresh and relevant
    q_company = urllib.parse.quote(f"{company_name} stock when:7d")
    q_ticker = urllib.parse.quote(f"{ticker} stock when:7d")
    
    queries = [
        f"https://news.google.com/rss/search?q={q_company}&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q={q_ticker}&hl=en-US&gl=US&ceid=US:en",
    ]
    headlines = []
    for url in queries:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                title = entry.get("title", "").split(" - ")[0].strip()
                if title and len(title) > 10:
                    headlines.append(title)
        except Exception:
            pass
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for h in headlines:
        if h.lower() not in seen:
            seen.add(h.lower())
            unique.append(h)
    return unique[:15]


# ── LLM Risk Analysis (Core Engine) ─────────────────────────────────────────
RISK_AUDIT_PROMPT = """You are a **Chief Risk Officer (CRO)** at a top-tier investment bank.

You will receive a list of recent news headlines about a specific public company,
PLUS a Quantitative Snapshot from our internal scoring system.
Your task is to assess hidden risks that standard financial ratios (P/E, ROE, etc.) CANNOT detect.

**Current Macro Environment: {macro_context}**
Use this macro context to calibrate your risk assessment. In a RISK-OFF or BEARISH environment,
weight downside risks more heavily. In a BULLISH / RISK-ON environment, consider whether
the company is positioned to benefit from the current trend.

**Quantitative Snapshot (Internal System):**
- AI Quality Score: {quant_score}/100 (composite fundamental + momentum score)
- RSI (14-day): {rsi} — momentum indicator; >70 = overbought, <30 = oversold
- Z-Score (60-day): {z_score} — valuation vs history; >2 = stretched, <-2 = deep value
- 52-Week Position: {w52_pos}% (0% = at 52W Low, 100% = at 52W High)
- Smart Money Signal: {smart_money} (Accumulation = institutional buying, Distribution = selling)
- Debt/EBITDA: {debt_ebitda}
- Earnings Surprise (last 2Q): {earnings_surprise}

Use this quantitative data to cross-validate against the news headlines.
If the headlines are negative but the quant signals are strong (high score, accumulation),
note this divergence explicitly. If both quant AND news are bearish, escalate the red_flag_score.

Return your assessment as a **valid JSON object** with the following structure:
{{
    "red_flag_score": <int 0-100, measures SEVERITY of tail risk — not just sentiment>,
    "headline_sentiment_score": <int 0-100, pure tone measure: 0=very positive, 100=very negative>,
    "confidence": <int 0-100, how confident you are given the quality and quantity of evidence>,
    "sentiment": "<Positive | Neutral | Negative | Critical>",
    "risk_categories": ["<category1>", "<category2>"],
    "key_insights": [
        "<Insight 1: specific risk or signal, cite the headline that supports it>",
        "<Insight 2: cite the headline>",
        "<Insight 3: cite the headline>"
    ],
    "evidence_map": {{
        "<Insight 1 short label>": "<exact or paraphrased headline that supports it>",
        "<Insight 2 short label>": "<exact or paraphrased headline that supports it>",
        "<Insight 3 short label>": "<exact or paraphrased headline that supports it>"
    }},
    "recommendation": "<A one-sentence actionable recommendation for a portfolio manager>"
}}

**Field definitions:**
- `red_flag_score`: Measures the SEVERITY and MATERIALITY of risks to shareholder value. A short-term negative headline with no tail risk should score low (< 30), even if sentiment is negative. Reserve high scores (> 60) for structural, legal, or financial risks with real capital implications.
- `headline_sentiment_score`: Pure emotional tone of the headlines. 0 = uniformly positive, 100 = uniformly negative. This is independent of whether the news creates actual investment risk.
- `confidence`: Lower this if headlines are few (< 5), duplicated, vague, or contradictory. If information is insufficient or conflicting, say so in key_insights and set confidence < 50.
- `risk_categories`: An array. A company can face multiple simultaneous risk types. Allowed values: "None", "Legal", "Operational", "Financial", "Reputational", "Geopolitical", "Macro".
- `evidence_map`: Each insight MUST be grounded in at least one specific headline. Do not generate insights without evidence.

**Scoring Guide for red_flag_score (risk severity, not sentiment):**
- 0-20: No material risks. Positive or neutral news with no structural threat.
- 21-40: Minor concerns. Monitor but no capital action needed.
- 41-60: Moderate risk. M&A uncertainty, margin pressure, regulatory inquiry — investigate further.
- 61-80: Significant risk. Credible legal, operational, or financial issues with capital implications.
- 81-100: Critical. Imminent threat: fraud, major lawsuit, insolvency signal, CEO departure under fire.

**CRITICAL INSTRUCTIONS:**
1. Return ONLY the JSON object. No markdown, no explanation, no code fences.
2. DO NOT hallucinate. Every insight in `key_insights` must appear in `evidence_map` with a supporting headline.
3. Distinguish between SENTIMENT (temporary noise) and RISK (structural threat). A stock falling 5% on earnings miss is a sentiment event, not a red_flag risk, unless guidance was cut dramatically.
4. M&A rumors: classify as Moderate Risk (score 40-55) by default, but adjust UP if the deal looks dilutive or financially strained, or DOWN if the deal looks strategically sound.
5. Debt rule: if Debt/EBITDA > 4.0 AND news mentions rising rates or credit concerns, escalate risk_categories to include "Financial" and add +10 to red_flag_score.
6. Macro regime: in RISK_OFF/BEARISH regimes, apply +5 to +10 to red_flag_score if the company is cyclically exposed.
7. Insufficient data rule: if fewer than 5 headlines are available, or headlines are repetitive/vague, set confidence < 50 and note this explicitly in key_insights.

IMPORTANT: Return ONLY the JSON object. No markdown, no explanation, no code fences.

Company: {company} ({ticker})
Recent Headlines:
{headlines}
"""

def analyze_risk_with_llm(
    ticker: str,
    company_name: str,
    macro_context: str = "NEUTRAL | VIX=N/A",
    quant_context: Optional[dict] = None,
) -> dict:
    """
    Main entry point for LLM Risk Audit.
    Returns a dict with red_flag_score, sentiment, key_insights, risk_category, recommendation.
    On failure, returns a safe default dict.

    Args:
        ticker: Stock ticker symbol.
        company_name: Full company name.
        macro_context: Regime string, e.g. 'BEARISH / CAUTION | VIX=28.5 | DXY+0.4%'
        quant_context: Optional dict with keys: quant_score, rsi, z_score, w52_pos,
                       smart_money, debt_ebitda, earnings_surprise.
    """
    _q = quant_context or {}
    prompt_quant = {
        "quant_score":       _q.get("quant_score", "N/A"),
        "rsi":               _q.get("rsi", "N/A"),
        "z_score":           _q.get("z_score", "N/A"),
        "w52_pos":           _q.get("w52_pos", "N/A"),
        "smart_money":       _q.get("smart_money", "N/A"),
        "debt_ebitda":       _q.get("debt_ebitda", "N/A"),
        "earnings_surprise": _q.get("earnings_surprise", "N/A"),
    }

    default_result = {
        "red_flag_score": 0,
        "headline_sentiment_score": 50,
        "confidence": 0,
        "sentiment": "N/A",
        "risk_categories": ["None"],
        "risk_category": "None",  # backward-compat for UI
        "key_insights": ["LLM analysis unavailable."],
        "evidence_map": {},
        "recommendation": "Rely on quantitative scoring only.",
        "error": None,
    }

    try:
        # 1. Fetch headlines
        headlines = _fetch_recent_headlines(ticker, company_name)
        if not headlines:
            default_result["key_insights"] = ["No recent news found for this ticker."]
            return default_result

        # 2. Build prompt with macro + quant context injected
        headlines_text = "\n".join([f"- {h}" for h in headlines])
        prompt = RISK_AUDIT_PROMPT.format(
            company=company_name, ticker=ticker, headlines=headlines_text,
            macro_context=macro_context,
            **prompt_quant
        )

        # 3. Call Cohere
        client = _get_client()
        response = client.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": prompt}]
        )

        raw_text = response.message.content[0].text.strip()

        # 4. Parse JSON (with fallback for markdown code fences)
        if raw_text.startswith("```"):
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[-1].split("```")[0].strip()
            else:
                raw_text = raw_text.split("```")[-1].split("```")[0].strip()


        result = json.loads(raw_text)

        # Validate required keys with new schema
        result.setdefault("red_flag_score", 0)
        result.setdefault("headline_sentiment_score", result.get("red_flag_score", 50))
        result.setdefault("confidence", 70)
        result.setdefault("sentiment", "N/A")
        result.setdefault("risk_categories", ["None"])
        result.setdefault("key_insights", [])
        result.setdefault("evidence_map", {})
        result.setdefault("recommendation", "No recommendation.")
        # Backward-compat: derive risk_category string from array for existing UI
        cats = result["risk_categories"]
        result["risk_category"] = " | ".join(cats) if cats else "None"
        result["error"] = None
        result["headlines_analyzed"] = len(headlines)
        result["_prompt_debug"] = prompt   # for in-UI debug expander

        return result

    except json.JSONDecodeError as e:
        default_result["error"] = f"LLM returned invalid JSON: {str(e)[:80]}"
        default_result["key_insights"] = [f"Raw LLM output could not be parsed. Error: {str(e)[:100]}"]
        return default_result
    except Exception as e:
        default_result["error"] = str(e)[:120]
        default_result["key_insights"] = [f"API Error: {str(e)[:100]}"]
        return default_result
