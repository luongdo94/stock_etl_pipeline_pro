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
Via the `analyze_risk_with_llm()` function in `etl/llm_parser.py`, the system scans **15 recent headlines** from **Google News**:
- **Red Flag Score (0-100):** Assessed risk based on news content (0 = no risk, 100 = critical risk).
- **Sentiment:** Overall tone (Positive, Negative, Neutral, Critical).
- **Risk Category:** Focused risk area (Legal, Technical, Financial, Reputational, Regulatory, Operational).
- **LLM Provider:** Cohere Command-R+ (Trial tier: ~20 high-fidelity calls/month limit).

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
- **API Limits:** The system currently utilizes the **Cohere Trial tier** (limited to approximately **20 high-fidelity calls per month**). For production use, upgrade to Cohere Production tier for unlimited calls.
- **News Source:** Headlines are fetched from **Google News RSS feeds** via the `feedparser` library, filtered for relevance and recency (last 7 days).
- **Function Location:** `etl/llm_parser.py` → `analyze_risk_with_llm(ticker: str, company_name: str) -> dict`

---

## 6. AI Market Scanner Strategy Presets

The scanner has **14 presets**, defined in `core/scan_presets.py` and tested in `tests/test_scan_presets.py`.
Thresholds use the Quality tiers (ELITE 75, SOLID 60, FAIR 45). Unknown values never match.

### 6.1 Opportunity (9)

| Preset | Rule |
|---|---|
| 🏆 Institutional Pulse | Quality ≥ 75 and uptrend (MA50 > MA200) |
| 💎 Quality at a Fair Price | Quality ≥ 75 and Value ≥ 50 |
| 🏷️ Deep Value | Value ≥ 70 and Quality ≥ 60 |
| 📈 Rising Estimates | Revisions ≥ 65 and Quality ≥ 60 |
| 🌱 GARP | 0 < PEG < 1.0 and Quality ≥ 60 |
| ⚙️ Both Accelerating | EPS and revenue both up > 10% QoQ for 2 quarters |
| 🚀 Buy on Dip | Uptrend and RSI < 40 |
| ⚡ Strong Breakout | Uptrend, > 5% above MA200, Momentum ≥ 70, RSI 50–70 |
| 💰 Quality Dividend | Yield > 2.5%, Quality ≥ 60, dividend covered by FCF, net debt/EBITDA < 3x (unknown, e.g. banks, kept) |

### 6.2 Risk & Warning (5)

| Preset | Rule |
|---|---|
| 🪤 Value Trap Risk | Value ≥ 65 and Quality < 45 |
| 🚩 Red Flags | Any Quality penalty |
| 📉 Downtrend | MA50 < MA200 and (RSI < 50 or Quality < 45) |
| ⚠️ Earnings Deterioration | EPS down > 10% QoQ for 2 quarters, revenue not accelerating |
| 🎈 Overextended | RSI > 70 and Z-Score > +2 (a price statistic, not a valuation) |

### 6.3 Why the list was cut from 27 (October 2026)

Measured on the 812-stock universe: Strong Breakout sat 99% inside Bullish Momentum (320 hits, 39% of the
universe); Negative Momentum and Multi-Indicator Breakdown overlapped 95%; Oversold Reversal Setup returned 0
stocks, Mean Reversion Elite 2, Short Squeeze Watch 4 (all already red-flagged), Distribution Warning 6 and
Contrarian Value 8 (a subset of GARP). Accumulation / Distribution rest only on a volume heuristic, and Exit on
Strength is a trading signal, not a risk. Those single-indicator screens remain possible with the Custom
Refinement sliders (RSI, Z-Score, PEG, Smart Money).

## 7. Smart Money Indicator v5.0

The **Smart Money** indicator tracks institutional buying and selling patterns using On-Balance Volume (OBV) divergence analysis to identify where professional investors are positioning.

### Calculation Methodology (Enhanced v5.0)

**Two-Layer Architecture:**

**Layer 1 - OBV Divergence (Priority):**
- Detects when OBV and price move in OPPOSITE directions
- **Hidden Accumulation**: Price falling but OBV rising → institutions buying dips
- **Hidden Distribution**: Price rising but OBV falling → institutions selling into rallies
- Uses adaptive window (15-25 days) based on ATR/volatility
- Stricter magnitude guard (0.12 × avg_volume × window) to filter noise

**Layer 2 - OBV Trend vs MA(21) (Fallback):**
- Classic institutional flow: OBV above/below its 21-day MA
- Requires 3 of last 5 days consistently above/below MA
- Applied only when no clear divergence detected

**Key Improvements over v4.0:**
1. **Adaptive Window**: High volatility stocks use wider windows (25 days), low volatility use narrower (15 days)
2. **Stricter Magnitude Guard**: Increased from 0.05 to 0.12 (240% avg volume threshold)
3. **Strength Scoring**: Returns confidence score 0-100 based on:
   - OBV magnitude (40 points)
   - Price magnitude (25 points)
   - Volume confirmation (20 points)
   - Consistency across window (15 points)
4. **Layer Detection**: Identifies whether signal came from DIVERGENCE or TREND layer

### Output Format

Returns a dictionary with three components:
```python
{
    "signal": "ACCUMULATION" | "DISTRIBUTION" | "NEUTRAL",
    "strength": 0-100,  # Confidence score
    "layer": "DIVERGENCE" | "TREND" | "NONE"
}
```

### Interpretation

| Signal | Strength | Meaning | Action |
|---|---|---|---|
| **ACCUMULATION** | 70-100 | Strong institutional buying | High conviction entry |
| **ACCUMULATION** | 40-69 | Moderate institutional buying | Cautious entry |
| **ACCUMULATION** | 0-39 | Weak institutional buying | Monitor, wait for confirmation |
| **DISTRIBUTION** | 70-100 | Strong institutional selling | High conviction exit |
| **DISTRIBUTION** | 40-69 | Moderate institutional selling | Reduce position |
| **DISTRIBUTION** | 0-39 | Weak institutional selling | Monitor, consider hedging |
| **NEUTRAL** | 0 | No clear institutional flow | Wait for clearer signal |

**Layer Priority:**
- **DIVERGENCE** signals are highest priority (catches hidden institutional activity)
- **TREND** signals are fallback (classic OBV vs MA confirmation)

### Integration with Strategies
- No scanner preset relies on this indicator alone; use the Smart Money filter under Custom Refinement

### Advantages Over Traditional Methods
- **Adaptive to volatility**: Window size adjusts automatically
- **Noise filtering**: Stricter magnitude guard reduces false signals
- **Confidence scoring**: Strength metric helps prioritize signals
- **Layer transparency**: Know whether signal is from divergence or trend
- **Volume confirmation**: Recent volume patterns validate signals

### Limitations
- Based on publicly available price/volume data (cannot see dark pools)
- OBV is cumulative and path-dependent (uses last 126 days to avoid bias)
- Should be combined with other indicators for confirmation
- Strength scoring is relative, not absolute probability

---

## 5. Operational Notes
- **CIO Persona:** The AI is designed with a critical mindset. It may contradict mathematical formulas if it perceives qualitative risks that the math cannot see.
- **Refresh Frequency:** News is scanned in real-time when the button is pressed. The analysis is valid for the current trading context.
- **API Limits:** The system currently utilizes the **Cohere Trial tier** (limited to approximately **20 high-fidelity calls per month**). For production use, upgrade to Cohere Production tier for unlimited calls.
- **News Source:** Headlines are fetched from **Google News RSS feeds** via the `feedparser` library, filtered for relevance and recency (last 7 days).
- **Function Location:** `etl/llm_parser.py` → `analyze_risk_with_llm(ticker: str, company_name: str) -> dict`

> [!IMPORTANT]
> AI recommendations are for informational purposes only. They are a decision-support tool. Investors are responsible for their own financial decisions.

---

## Quick Reference: Strategy Cheat Sheet

### Strategy Count Summary
- **Total Presets:** 14 (cut from 27 in October 2026, see section 6.3)
- **Opportunity:** 9 · **Risk / Warning:** 5
- Full list and rules: section 6 and `core/scan_presets.py`

### Key Metrics to Monitor
- **Quality Score** - Fundamental strength (0-100)
- **RSI** - Momentum and overbought/oversold (0-100)
- **Z-Score** - Valuation vs historical mean (±3)
- **Smart Money** - Institutional flow (Accumulation/Distribution/Neutral)
- **Trend** - Technical direction (Bullish/Bearish)


---

## 8. 6-Pillar Institutional Rating System v14.0

The **Institutional Rating Engine** synthesizes six independent pillars to generate actionable investment recommendations (STRONG BUY, BUY, HOLD, SELL, AVOID). This system is used consistently across both the Opportunity Radar screener and Deep Dive tab.

### 8.1. Rating Architecture

**Function:** `compute_institutional_rating()` in `app.py`

**Pillars:**
1. **Technical Trend** (0-1 points): MA signals, RSI confirmation
2. **Quality** (0-1 points): AI Score (fundamental quality)
3. **Valuation** (0-1 points): Sector-adjusted P/E, PEG, upside potential
4. **Risk** (0-1 points): 52-week position
5. **Conviction** (0-1 points): Risk/Reward ratio
6. **Smart Money** (-1.25 to +1.25 points): Institutional flow with strength-based scoring

**Total Range:** -1.25 to 6.25 points

### 8.2. Smart Money Soft Scoring (NEW in v14.0)

Instead of binary 0/1 points, Smart Money now uses **graduated scoring** based on signal strength:

#### ACCUMULATION Scoring

| Strength Range | Points | Label | Color |
|---|---|---|---|
| **≥ 80** | +1.25 | ACCUMULATION_STRONG | #00ffcc (Cyan) |
| **65-79** | +1.0 | ACCUMULATION_STRONG | #2ecc71 (Green) |
| **40-64** | +0.5 | ACCUMULATION_WEAK | #3498db (Blue) |
| **< 40** | 0.0 | ACCUMULATION_WEAK | #95a5a6 (Gray) |

#### DISTRIBUTION Scoring

| Strength Range | Points | Label | Color |
|---|---|---|---|
| **≥ 80** | -1.25 | DISTRIBUTION_STRONG | #c0392b (Dark Red) |
| **65-79** | -1.0 | DISTRIBUTION_STRONG | #e74c3c (Red) |
| **40-64** | -0.5 | DISTRIBUTION_WEAK | #e67e22 (Orange) |
| **< 40** | 0.0 | DISTRIBUTION_WEAK | #95a5a6 (Gray) |

**Rationale:**
- Weak signals (< 40 strength) are ignored to prevent noise
- Moderate signals (40-64) get half weight
- Strong signals (65-79) get full weight
- Very strong signals (≥ 80) get bonus/penalty weight

### 8.3. Action Label Thresholds

| Total Points | Conditions | Action Label |
|---|---|---|
| **≥ 5.0** | Quality not weak | **STRONG BUY** |
| **≥ 3.5** | Trend not bearish | **BUY / ACCUMULATE** |
| **≤ 2.0** | Trend + Valuation both weak | **SELL / AVOID** |
| **≤ 2.0** | Quality weak | **SELL / AVOID** |
| **≤ 2.0** | Strong distribution (SM ≤ -0.5) | **SELL / AVOID** |
| **≤ 2.5** | Quality strong | **HOLD / NEUTRAL** |
| **≤ 4.5** | RSI > 70 | **REDUCE / UNDERPERFORM** |
| **Other** | - | **HOLD / NEUTRAL** |

### 8.4. Examples

#### Example 1: Very Strong Accumulation Bonus
```
Trend: ✅ (1.0)
Quality: ✅ (1.0)
Valuation: ✅ (1.0)
Risk: ✅ (1.0)
R/R: ❌ (0.0)
Smart Money: ACCUMULATION (85, DIVERGENCE) → +1.25

Total: 4.0 + 1.25 = 5.25 → STRONG BUY
```

#### Example 2: Weak Signal Ignored
```
Trend: ✅ (1.0)
Quality: ✅ (1.0)
Valuation: ✅ (1.0)
Risk: ✅ (1.0)
R/R: ❌ (0.0)
Smart Money: ACCUMULATION (25, TREND) → +0.0

Total: 4.0 + 0.0 = 4.0 → BUY (not STRONG BUY)
```

#### Example 3: Distribution Penalty
```
Trend: ✅ (1.0)
Quality: ✅ (1.0)
Valuation: ✅ (1.0)
Risk: ❌ (0.0)
R/R: ❌ (0.0)
Smart Money: DISTRIBUTION (75, DIVERGENCE) → -1.0

Total: 3.0 - 1.0 = 2.0 → SELL / AVOID
```

### 8.5. Benefits of Soft Scoring

1. **Precision:** Weak OBV signals don't trigger STRONG BUY
2. **Reward Quality:** Very strong divergence (≥80) gets bonus weight
3. **Risk Management:** Strong distribution actively downgrades ratings
4. **Transparency:** Users see exact point contribution
5. **Flexibility:** Easy to adjust thresholds without code changes

### 8.6. Integration with Other Systems

- **Opportunity Radar:** Uses rating to filter and sort stocks
- **Deep Dive:** Displays 6-pillar matrix with color coding
- **AI Tab:** Incorporates rating into convergence analysis
- **Portfolio Builder:** Uses rating for position sizing recommendations

---
