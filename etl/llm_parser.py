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
8. LANGUAGE: All text values in your JSON response (key_insights, evidence_map keys/values, recommendation) MUST be written in {language}. Do not change the JSON keys, only the values.

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
    language: str = "English"
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
            language=language,
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


# ── LLM Portfolio Review (CIO Board Meeting) ─────────────────────────────────
PORTFOLIO_REVIEW_PROMPT = """You are the **Chief Risk Officer (CRO)** of a top-tier hedge fund chairing a monthly portfolio review session.

You have received a complete snapshot of a client's equity portfolio. Your role is to identify structural vulnerabilities, macro misalignments, and provide decisive rebalancing guidance.

## Portfolio Snapshot

**Macro Environment:** {macro_context}

**Portfolio-Level Risk Metrics:**
- Total Market Value: €{total_value:,.0f}
- Total Unrealized PnL: {pnl_pct:+.1f}%
- Portfolio Beta (vs S&P 500): {port_beta:.2f}
- Sharpe Ratio: {sharpe:.2f}
- Max Drawdown: {max_dd:.1f}%
- Annual Volatility: {vol:.1f}%
- VaR (95%): {var_95:.2f}% (daily)
- Largest Sector Exposure: {top_sector} ({top_sector_weight:.1f}% of portfolio)

**Full Position-Level Data:**
{positions_table}

## Your Task
Write a CIO Portfolio Review with FOUR sections. Total length: under 750 words.

### 📡 Macro-Alignment Assessment
Is this portfolio positioned correctly for the current macro regime? Assess Beta, sector mix, and volatility relative to the VIX and market regime. Flag any obvious misalignment (e.g., high-beta tech-heavy portfolio in a RISK-OFF environment).

### 🔬 Structural Vulnerability Scan
Identify the top 2-3 structural weaknesses:
- Concentration risk (overweight positions or sectors)
- Low-quality anchors (positions with poor Quality Score and negative PnL — deadweight)
- Correlation traps (positions that appear diversified but likely move together)

### 🗂️ Position Buckets
Classify EVERY position from the data into exactly one of the four buckets below. Use a table-style format:

**🏛️ Core Compounders** — High Quality Score (≥65), positive or stable PnL, structural competitive moat. These are long-term holds.
| Ticker | Quality | PnL% | Weight% | Rationale |
|--------|---------|------|---------|-----------|

**🔄 Cyclical Beta** — Moderate Quality (40-65), performance tied to macro cycle (rates, commodities, GDP). Hold when regime is bullish, reduce in risk-off.
| Ticker | Quality | PnL% | Weight% | Rationale |
|--------|---------|------|---------|-----------|

**🎲 Speculative Positions** — Low Quality (<40) OR early-stage/high-volatility growth plays. High upside but asymmetric tail risk.
| Ticker | Quality | PnL% | Weight% | Rationale |
|--------|---------|------|---------|-----------|

**✂️ Trim Candidates** — Positions that should be reduced or exited: poor quality + negative PnL, excessive concentration, or thesis broken. Prioritized for capital redeployment.
| Ticker | Quality | PnL% | Weight% | Exit Rationale |
|--------|---------|------|---------|----------------|

### ⚡ Rebalancing Directives (3 Actions)
State exactly 3 concrete actions the portfolio manager should take. Be specific: name the ticker, the direction (reduce/increase/exit), and the rationale. Format as:
1. **[ACTION] [TICKER]:** [Rationale with specific numbers]
2. **[ACTION] [TICKER]:** [Rationale with specific numbers]
3. **[ACTION] [TICKER]:** [Rationale with specific numbers]

Rules: Your output MUST be written in {language}. Be decisive and institutional. Reference specific tickers and numbers from the data. Every position MUST appear in exactly one bucket — no omissions.
**ETF/Index Fund Rule:** Positions marked as Type=ETF/Index Fund are inherently diversified vehicles (e.g., Nasdaq 100 ETF). They MUST NOT be flagged for concentration risk. In Position Buckets, classify them as Cyclical Beta or Core Compounders based on their macro role. Start each section header on its own line."""


def analyze_portfolio_with_llm(
    api_key: str,
    portfolio_data: dict,
    macro_context: str = "NEUTRAL | VIX=N/A",
    language: str = "English",
) -> tuple[str, str]:
    """
    CIO Board Meeting — AI Portfolio Review.
    Analyzes the full portfolio snapshot and returns a structured risk report.

    Args:
        api_key: Cohere API key.
        portfolio_data: Dict with keys: positions, total_value, pnl_pct, port_beta,
                        sharpe, max_dd, vol, var_95, top_sector, top_sector_weight.
        macro_context: Regime string, e.g. 'BEARISH | VIX=22.5'.
        language: Output language for the report ('English' or 'Vietnamese').
    Returns:
        Tuple of (report_text, prompt_debug).
    """
    try:
        import cohere
        co = cohere.ClientV2(api_key=api_key)

        # Build positions table string
        _ETF_KEYWORDS = {"etf", "fund", "trust", "index", "ishares", "vanguard",
                          "amundi", "lyxor", "xtrackers", "invesco", "spdr", "ucits"}
        positions = []
        for _pr in portfolio_data.get("positions", []):
            _company_lc = str(_pr.get("company", "")).lower()
            _sector_raw = str(_pr.get("sector", "N/A"))
            _is_etf = (
                _sector_raw in ("N/A", "nan", "None", "") and
                any(kw in _company_lc for kw in _ETF_KEYWORDS)
            )
            positions.append({
                "ticker": _pr.get("ticker", "?"),
                "company": _pr.get("company", "N/A"),
                "asset_type": "ETF/Index Fund" if _is_etf else "Stock",
                "sector": "Diversified (ETF)" if _is_etf else _sector_raw,
                "weight_pct": _pr.get("weight_pct", 0),
                "quality_score": _pr.get("quality_score", "N/A"),
                "pnl_pct": _pr.get("pnl_pct", 0),
                "price": _pr.get("price", 0),
            })

        lines = ["Type           | Ticker | Company              | Sector             | Weight% | Quality | PnL%  | Price€"]
        lines.append("-" * 110)
        for p in positions:
            lines.append(
                f"{p['asset_type']:14} | {p['ticker']:6} | {p['company'][:20]:20} | "
                f"{p['sector'][:18]:18} | {p['weight_pct']:5.1f}% | "
                f"{str(p['quality_score']):7} | {p['pnl_pct']:+5.1f}% | "
                f"€{p['price']:8.2f}"
            )
        positions_table = "\n".join(lines)

        prompt = PORTFOLIO_REVIEW_PROMPT.format(
            macro_context=macro_context,
            total_value=portfolio_data.get("total_value", 0),
            pnl_pct=portfolio_data.get("pnl_pct", 0),
            port_beta=portfolio_data.get("port_beta", 1.0),
            sharpe=portfolio_data.get("sharpe", 0),
            max_dd=portfolio_data.get("max_dd", 0),
            vol=portfolio_data.get("vol", 0),
            var_95=portfolio_data.get("var_95", 0),
            top_sector=portfolio_data.get("top_sector", "N/A"),
            top_sector_weight=portfolio_data.get("top_sector_weight", 0),
            positions_table=positions_table,
            language=language,
        )

        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
        )
        return response.message.content[0].text, prompt

    except Exception as e:
        err = str(e)
        if "invalid api key" in err.lower() or "unauthorized" in err.lower():
            return "❌ **Invalid API Key.**", ""
        elif "rate limit" in err.lower():
            return "⏳ **Rate limit reached.** Please wait a moment.", ""
        else:
            return f"❌ **Portfolio Review Error:** {err[:120]}", ""
