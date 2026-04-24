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

You will receive a list of recent news headlines about a specific public company.
Your task is to assess hidden risks that standard financial ratios (P/E, ROE, etc.) CANNOT detect.

**Current Macro Environment: {macro_context}**
Use this macro context to calibrate your risk assessment. In a RISK-OFF or BEARISH environment,
weight downside risks more heavily. In a BULLISH / RISK-ON environment, consider whether
the company is positioned to benefit from the current trend.

Analyze the headlines and return your assessment as a **valid JSON object** with the following structure:
{{
    "red_flag_score": <int 0-100>,
    "sentiment": "<Positive | Neutral | Negative | Critical>",
    "key_insights": [
        "<Insight 1: A specific risk or positive signal you identified>",
        "<Insight 2: Another finding>",
        "<Insight 3: Another finding>"
    ],
    "risk_category": "<None | Legal | Operational | Financial | Reputational | Geopolitical>",
    "recommendation": "<A one-sentence actionable recommendation for a portfolio manager>"
}}

Scoring Guide for red_flag_score:
- 0-20: No material risks detected. Predominantly positive news.
- 21-40: Minor concerns. Monitor but no action needed.
- 41-60: Moderate risk. Headline-level concerns worth investigating (e.g., M&A rumors causing uncertainty or subsidiary stock drops).
- 61-80: Significant risk. Potential legal, operational, or financial issues.
- 81-100: Critical. Imminent threat to shareholder value (lawsuits, fraud, CEO departure).

CRITICAL INSTRUCTIONS:
1. Return ONLY the JSON object. No markdown, no explanation, no code fences.
2. DO NOT hallucinate or assume positive business strategies (e.g., 'fiber infrastructure', 'clean energy') unless explicitly mentioned in the provided headlines.
3. If headlines show the stock or its subsidiary dropping due to M&A rumors, you MUST classify it as at least Moderate Risk (score > 40) and state the uncertainty.
4. Factor in the macro environment: in RISK_OFF/BEARISH regimes, apply an additional 5-10 point premium to your red_flag_score if the company has exposure to cyclical headwinds.

IMPORTANT: Return ONLY the JSON object. No markdown, no explanation, no code fences.

Company: {company} ({ticker})
Recent Headlines:
{headlines}
"""

def analyze_risk_with_llm(ticker: str, company_name: str, macro_context: str = "NEUTRAL | VIX=N/A") -> dict:
    """
    Main entry point for LLM Risk Audit.
    Returns a dict with red_flag_score, sentiment, key_insights, risk_category, recommendation.
    On failure, returns a safe default dict.

    Args:
        ticker: Stock ticker symbol.
        company_name: Full company name.
        macro_context: Regime string to inject into prompt, e.g. 'BEARISH / CAUTION | VIX=28.5 | DXY+0.4%'
    """
    default_result = {
        "red_flag_score": 0,
        "sentiment": "N/A",
        "key_insights": ["LLM analysis unavailable."],
        "risk_category": "None",
        "recommendation": "Rely on quantitative scoring only.",
        "error": None,
    }

    try:
        # 1. Fetch headlines
        headlines = _fetch_recent_headlines(ticker, company_name)
        if not headlines:
            default_result["key_insights"] = ["No recent news found for this ticker."]
            return default_result

        # 2. Build prompt with macro context injected
        headlines_text = "\n".join([f"- {h}" for h in headlines])
        prompt = RISK_AUDIT_PROMPT.format(
            company=company_name, ticker=ticker, headlines=headlines_text,
            macro_context=macro_context
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

        # Validate required keys
        result.setdefault("red_flag_score", 0)
        result.setdefault("sentiment", "N/A")
        result.setdefault("key_insights", [])
        result.setdefault("risk_category", "None")
        result.setdefault("recommendation", "No recommendation.")
        result["error"] = None
        result["headlines_analyzed"] = len(headlines)

        return result

    except json.JSONDecodeError as e:
        default_result["error"] = f"LLM returned invalid JSON: {str(e)[:80]}"
        default_result["key_insights"] = [f"Raw LLM output could not be parsed. Error: {str(e)[:100]}"]
        return default_result
    except Exception as e:
        default_result["error"] = str(e)[:120]
        default_result["key_insights"] = [f"API Error: {str(e)[:100]}"]
        return default_result
