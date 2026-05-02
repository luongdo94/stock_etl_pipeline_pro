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

The AI Market Scanner provides **23 curated strategy filters** (optimized from 26 to eliminate redundancy) that combine quantitative metrics with technical indicators to identify specific market opportunities and risks. Each preset is designed for a specific investment thesis or risk scenario.

**Recent Optimization (May 2026):**
- Removed 3 redundant presets: "Trend Following" (too simple), "Deep Value" (superseded by Mean Reversion Elite), "RSI Mean Reversion" (superseded by Oversold Reversal Setup)
- Renamed 2 presets for clarity: "Multi-Indicator Breakout" → "Bullish Momentum", "Momentum Breakout" → "Strong Breakout"
- Final count: **15 Opportunity + 8 Risk/Warning strategies**

### 6.1 Opportunity Strategies (15 presets)

#### 🏆 Institutional Pulse
**Thesis:** Elite quality stocks in confirmed uptrends - institutional conviction plays.

**Criteria:**
- Quality Score ≥ 70 (ELITE tier)
- Trend = BULLISH (MA20 > MA50)

**Use Case:** Highest conviction opportunities combining fundamental excellence with technical confirmation.

---

#### 🚀 Buy on Dip
**Thesis:** Bullish stocks experiencing temporary RSI cooling - tactical entry points.

**Criteria:**
- Trend = BULLISH (confirmed uptrend)
- RSI < 40 (short-term pullback)

**Use Case:** Enter quality uptrends during healthy pullbacks before momentum resumes.

---

#### 🚀 Bullish Momentum
**Thesis:** Strong uptrends with RSI confirmation - ride the momentum.

**Criteria:**
- Trend = BULLISH (MA20 > MA50)
- RSI > 50 (momentum strength confirmed)

**Use Case:** Classic trend-following setup for established bullish moves.

---

#### 📈 Both Accelerating
**Thesis:** Strongest fundamental momentum - both top-line and bottom-line growth accelerating.

**Criteria:**
- EPS Momentum = "Accelerating" (QoQ growth >+10% for 2 consecutive quarters)
- Revenue Momentum = "Accelerating" (QoQ growth >+10% for 2 consecutive quarters)

**Use Case:** Fundamental momentum leaders - companies firing on all cylinders.

---

#### 🌱 GARP (Growth at Reasonable Price)
**Thesis:** Peter Lynch-style growth investing - quality growth at fair valuations.

**Criteria:**
- PEG ratio 0-1.5 (growth at reasonable price)
- Quality Score > 55 (SOLID tier or better)

**Use Case:** Balanced growth investing with valuation discipline.

---

#### 💰 High Quality Dividend
**Thesis:** Income + quality + momentum - defensive income with upside potential.

**Criteria:**
- Dividend Yield > 2.5%
- Quality Score > 65 (STRONG tier)
- Trend = BULLISH (confirmed uptrend)

**Use Case:** Core income holdings with quality fundamentals and technical strength.

---

#### 🔥 Short Squeeze Watch
**Thesis:** High short interest + oversold + bullish reversal = potential squeeze setup.

**Criteria:**
- Short % of Float > 15%
- RSI < 45 (oversold)
- Trend = BULLISH (reversal confirmed)

**Use Case:** Event-driven, high-volatility setups. Requires tight risk management.

---

#### 🎯 Smart Money Accumulation
**Thesis:** Follow institutional buying activity in quality stocks before the broader market catches on.

**Criteria:**
- Smart Money indicator shows "ACCUMULATION"
- Quality Score ≥ 55 (SOLID tier or better)
- RSI < 50 (not overbought, room to run)

**Use Case:** Early-stage institutional accumulation before price breakout. Institutions typically accumulate quietly before major moves.

---

#### 🔄 Mean Reversion Elite
**Thesis:** High-quality assets temporarily oversold present low-risk entry points.

**Criteria:**
- Quality Score ≥ 65 (STRONG tier)
- RSI < 35 (extreme oversold)
- Z-Score < -1.0 (below historical mean)

**Use Case:** "Elite assets on sale" - temporary weakness in fundamentally strong companies. Classic mean reversion setup with quality buffer. **Supersedes old "Deep Value" preset with quality filter.**

---

#### ⚡ Strong Breakout
**Thesis:** Ride established uptrends with healthy momentum (not overextended).

**Criteria:**
- Price > MA200 by 5%+ (strong uptrend)
- RSI between 50-70 (healthy momentum, not overbought)
- Trend = BULLISH (MA20 > MA50 confirmation)

**Use Case:** Trend-following strategy for established moves. Avoids early entries and overextended rallies. **Renamed from "Momentum Breakout" for clarity.**

---

#### 💎 Contrarian Value
**Thesis:** Quality companies in temporary downtrends at attractive valuations - wait for reversal signal.

**Criteria:**
- Quality Score ≥ 60 (STRONG tier)
- Trend = BEARISH (currently in downtrend)
- Z-Score < -1.5 (significantly undervalued)
- PEG ratio 0-1.2 (growth at discount)

**Use Case:** Contrarian opportunity - DO NOT enter immediately. Wait for technical reversal signal (RSI upturn, MA crossover) before entry. For patient investors building watchlists.

---

#### 🏰 Defensive Moat
**Thesis:** Fortress balance sheets for all-weather portfolios - recession-resistant companies.

**Criteria:**
- Debt/EBITDA < 2.0 (low leverage)
- ROE > 15% (high profitability)
- Dividend Yield > 2.0% (income component)
- Quality Score ≥ 60 (STRONG tier)

**Use Case:** Core portfolio holdings for risk-averse investors. Companies with strong competitive moats and financial flexibility.

---

#### 🌊 Oversold Reversal Setup
**Thesis:** Extreme oversold conditions + institutional buying = high-probability bounce.

**Criteria:**
- RSI < 30 (extreme oversold)
- Smart Money shows "ACCUMULATION"
- Quality Score ≥ 50 (minimum quality threshold)

**Use Case:** Short-term tactical bounce plays. Institutions buying into panic creates asymmetric risk/reward. **Supersedes old "RSI Mean Reversion" preset with Smart Money confirmation.**

---

#### 📊 Balanced Growth
**Thesis:** Sustainable growth at fair prices - the "Goldilocks" setup.

**Criteria:**
- Quality Score 55-75 (SOLID to STRONG tier)
- Forward P/E 15-30x (reasonable valuation)
- ROE > 12% (profitable growth)
- Trend = BULLISH (confirmed uptrend)

**Use Case:** Core growth holdings with balanced risk/reward. Not the fastest growers, but sustainable with margin of safety.

---

### 6.2 Risk & Warning Strategies (8 presets)

#### 🚨 Distribution Warning
**Thesis:** Institutions exiting weak stocks during overbought conditions - strong exit signal.

**Criteria:**
- Smart Money shows "DISTRIBUTION" (institutional selling)
- RSI > 60 (overbought, rally losing steam)
- Quality Score < 55 (WEAK to FAIR tier)

**Use Case:** Exit signal for existing positions. Combination of weak fundamentals + institutional selling + technical exhaustion.

---

#### ⚠️ Earnings Deterioration
**Thesis:** Fundamental breakdown - both top-line and bottom-line declining.

**Criteria:**
- EPS Momentum = "Decelerating" (QoQ decline >-10% for 2 consecutive quarters)
- Revenue Momentum = "Decelerating" (QoQ decline >-10% for 2 consecutive quarters)

**Use Case:** Early warning system for fundamental deterioration. Avoid or exit before broader market recognition.

---

#### ⚠️ Structural Caution
**Thesis:** High risk - weak fundamentals + confirmed downtrend.

**Criteria:**
- Quality Score < 38 (WEAK tier)
- Trend = BEARISH (MA20 < MA50)

**Use Case:** Avoid list. Combination of poor fundamentals and negative technical momentum creates high-risk environment.

---

#### 📉 Negative Momentum
**Thesis:** Confirmed bearish trend - avoid catching falling knives.

**Criteria:**
- Trend = BEARISH (MA20 < MA50 confirmed)

**Use Case:** Simple bearish trend filter. Wait for trend reversal before considering entry.

---

#### 🔥 Overbought Alert
**Thesis:** Elevated RSI signals potential short-term pullback risk.

**Criteria:**
- RSI > 65 (overbought territory)

**Use Case:** Take-profit or reduce exposure signal. Elevated risk of mean reversion.

---

#### 🎈 Valuation Exhaustion
**Thesis:** Prices significantly above historical mean - likely overvalued.

**Criteria:**
- Z-Score > +2.0 (2 standard deviations above 5-year mean)

**Use Case:** Valuation warning. Price has extended far beyond historical norms.

---

#### ⚔️ Exit on Strength
**Thesis:** Short-term rally within bearish trend - opportunity to exit or short.

**Criteria:**
- Trend = BEARISH (overall downtrend)
- RSI > 60 (short-term overbought bounce)

**Use Case:** Exit existing positions on strength. For short sellers, prime entry point (rally into resistance).

---

#### 💔 Multi-Indicator Breakdown
**Thesis:** Extreme downside momentum - "falling knife" scenario.

**Criteria:**
- Trend = BEARISH (confirmed downtrend)
- RSI < 50 (momentum accelerating downward)

**Use Case:** Avoid catching falling knives. Wait for stabilization before considering entry.

---**Use Case:** Exit signal for existing positions. Combination of weak fundamentals + institutional selling + technical exhaustion.

---

#### ⚠️ Earnings Deterioration
**Thesis:** Fundamental breakdown - both top-line and bottom-line declining.

**Criteria:**
- EPS Momentum = "Decelerating"
- Revenue Momentum = "Decelerating"
- Both declining QoQ > -10% for 2 consecutive quarters

**Use Case:** Early warning system for fundamental deterioration. Avoid or exit before broader market recognition.

---

#### ⚠️ Structural Caution
**Thesis:** High risk - weak fundamentals + confirmed downtrend.

**Criteria:**
- Quality Score < 38 (WEAK tier)
- Trend = BEARISH (MA20 < MA50)

**Use Case:** Avoid list. Combination of poor fundamentals and negative technical momentum creates high-risk environment.

---

#### 💔 Multi-Indicator Breakdown
**Thesis:** Extreme downside momentum - "falling knife" scenario.

**Criteria:**
- Trend = BEARISH (confirmed downtrend)
- RSI < 50 (momentum accelerating downward)

**Use Case:** Avoid catching falling knives. Wait for stabilization before considering entry.

---

#### ⚔️ Exit on Strength
**Thesis:** Short-term rally within bearish trend - opportunity to exit or short.

**Criteria:**
- Trend = BEARISH (overall downtrend)
- RSI > 60 (short-term overbought bounce)

**Use Case:** Exit existing positions on strength. For short sellers, prime entry point (rally into resistance).

---

### 6.3 Strategy Selection Guide

| Investment Style | Recommended Strategies |
|---|---|
| **Growth Investor** | Strong Breakout, Balanced Growth, Smart Money Accumulation, Both Accelerating |
| **Value Investor** | Mean Reversion Elite, Contrarian Value, GARP |
| **Income Investor** | Defensive Moat, High Quality Dividend |
| **Momentum Trader** | Strong Breakout, Bullish Momentum, Buy on Dip |
| **Contrarian** | Contrarian Value, Oversold Reversal Setup, Mean Reversion Elite |
| **Risk Manager** | Distribution Warning, Structural Caution, Earnings Deterioration |

**Note:** Removed presets (superseded by better alternatives):
- "Trend Following" → Use "Bullish Momentum" instead (adds RSI confirmation)
- "Deep Value" → Use "Mean Reversion Elite" instead (adds quality filter)
- "RSI Mean Reversion" → Use "Oversold Reversal Setup" instead (adds Smart Money confirmation)

### 6.4 Combining Strategies

**Best Practices:**
1. **Start with opportunity strategy** to build watchlist
2. **Cross-reference with risk strategies** to eliminate dangerous setups
3. **Apply custom refinement sliders** for additional filtering
4. **Monitor Smart Money indicator** for institutional flow confirmation
5. **Check multiple timeframes** - scanner shows current snapshot, verify trend persistence

**Example Workflow:**
```
Step 1: Run "Smart Money Accumulation" → 45 results
Step 2: Apply custom filter: Min Quality 60, RSI 30-45 → 12 results
Step 3: Check "Distribution Warning" to eliminate → 10 results
Step 4: Manual review of top 10 for final selection
```

---

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
- **Smart Money Accumulation** strategy specifically targets ACCUMULATION signals with strength ≥40
- **Distribution Warning** strategy flags DISTRIBUTION signals with strength ≥40
- **Oversold Reversal Setup** requires ACCUMULATION confirmation with strength ≥50

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

### Top 5 Opportunity Strategies
1. **🎯 Smart Money Accumulation** - Follow institutional buying (Quality ≥55 + RSI <50)
2. **🔄 Mean Reversion Elite** - Quality assets on sale (Quality ≥65 + RSI <35 + Z<-1.0)
3. **⚡ Strong Breakout** - Ride established trends (Price >MA200 by 5%+ + RSI 50-70)
4. **🏰 Defensive Moat** - Fortress balance sheets (Low debt + High ROE + Dividend)
5. **📊 Balanced Growth** - Sustainable growth at fair price (Quality 55-75 + PE 15-30x)

### Top 5 Risk Warnings
1. **🚨 Distribution Warning** - Institutions exiting (Smart Money DISTRIBUTION + RSI >60)
2. **⚠️ Earnings Deterioration** - Fundamental breakdown (EPS + Revenue both declining)
3. **💔 Multi-Indicator Breakdown** - Falling knife (Bearish + RSI <50)
4. **⚠️ Structural Caution** - Weak + Bearish (Quality <38 + Downtrend)
5. **⚔️ Exit on Strength** - Rally in downtrend (Bearish + RSI >60)

### Strategy Count Summary
- **Total Presets:** 23 (optimized from 26)
- **Opportunity Strategies:** 15
- **Risk/Warning Strategies:** 8
- **Removed (redundant):** 3 (Trend Following, Deep Value, RSI Mean Reversion)
- **Renamed (clarity):** 2 (Multi-Indicator Breakout → Bullish Momentum, Momentum Breakout → Strong Breakout)

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
