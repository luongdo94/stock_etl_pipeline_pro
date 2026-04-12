# 🧠 Unified Alpha-Risk Intelligence Hub (AI Logic)

This document explains the architecture and decision-making logic behind the integrated AI analysis system in the Dashboard. The system acts as a virtual **Chief Investment Officer (CIO)**, cross-referencing quantitative metrics with market news to deliver a definitive investment thesis.

## 1. System Overview
The **Unified Alpha-Risk Intelligence Hub** is more than a data summarizer. It performs "Signal Convergence" by analyzing two distinct worlds:
1.  **Quantitative:** Financial ratios, technical indicators, and algorithmic scores.
2.  **Qualitative:** News sentiment, regulatory risks, macroeconomic shifts, and public perception.

## 2. Input Pipeline

### Quantitative Metrics
The AI is provided with a comprehensive set of "hard" data points:
- **AI Score (0-100):** A synthesized fundamental quality score.
- **Financial Momentum (FMI):** Real-time measurement of earnings and revenue acceleration.
- **Technicals:** RSI (Overbought/Oversold), MA Signals (Moving Average trends).
- **Valuation:** P/E Ratio, PEG Ratio, FCF (Free Cash Flow) Margin.
- **Market Regime:** The global market context (Bullish/Bearish/Neutral).

### Qualitative Intelligence (NLP Results)
Via the `analyze_risk_with_llm` function, the system scans 15 recent headlines from Google News:
- **Red Flag Score (0-100):** Assessed risk based on news content.
- **Sentiment:** Overall tone (Positive, Negative, Critical).
- **Risk Category:** Focused risk area (Legal, Technical, Financial, Reputational).

---

## 3. Signal Alignment Engine

The AI analyzes the relationship between these two streams to detect potential conflicts:

- **CONVERGENCE:** Both fundamentals and news sentiment are aligned (Bullish). This triggers the highest conviction ratings (Strong Buy).
- **DIVERGENCE:** 
    - *Risk Conflict:* Strong fundamentals but negative technical/regulatory news. The AI will often downgrade the verdict to protect capital.
    - *Opportunity Conflict:* Weak internals but highly positive news (e.g., rumors of a buyout). The AI identifies this as high-risk speculation.
- **BEARISH ALIGNMENT:** Both quantitative and qualitative signals are negative. The AI issues an Avoid/Reduce verdict.

---

## 4. Investment Verdicts (Action Vocabulary)

The system is constrained to issue one of exactly six definitive actions:

| Action | Definition | Typical Conditions |
| :--- | :--- | :--- |
| **STRONG BUY** | High Conviction Buy | Perfect convergence, favorable valuation, strong news support. |
| **BUY** | Standard Buy | Good fundamentals, no significant news headwinds. |
| **WATCH & ACCUMULATE** | Tactical Overweight | Sideways price action or news-heavy environments with upside potential. |
| **HOLD** | Neutral Position | Fair valuation with no clear immediate catalyst. |
| **REDUCE** | Underweight | Initial breakdown in fundamentals or minor negative news. |
| **AVOID** | Sell / Do Not Buy | Significant risks detected (Red Flag > 70) or severe fundamental deterioration. |

---

## 5. Operational Notes
- **CIO Persona:** The AI is designed with a critical mindset. It may contradict mathematical formulas if it perceives qualitative risks that the math cannot see.
- **Refresh Frequency:** News is scanned in real-time when the button is pressed. The analysis is valid for the current trading context.
- **API Limits:** The system currently utilizes the Cohere Trial tier (limited to approximately 20 high-fidelity calls per month).

> [!IMPORTANT]
> AI recommendations are for informational purposes only. They are a decision-support tool. Investors are responsible for their own financial decisions.
