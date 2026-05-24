# System Algorithms & Indicator Logic

This document explains the computational principles behind the intelligent indicators on the Stock ETL Pipeline Dashboard. All system data (Price, Revenue, Market Cap) is normalized to **Euro (EUR)** during the ETL Pipeline's Extract phase.

---

## 1. Trend Confidence Score & Market Regime
This indicator (0-100) measures the strength of the overall market trend, focusing on the US market (S&P 500) but normalized to **Euro (EUR)** for system-wide synchronization.

### 1.1. Trend Confidence Score Calculation (Max 100 points)
The score is calculated by aggregating technical weights of the SPY index (S&P 500) and market breadth:

| Factor | Condition | Weight |
| :--- | :--- | :--- |
| **SPY Medium-term** | SPY Closing Price > MA50 | +25 points |
| **SPY Long-term** | SPY Closing Price > MA200 | +25 points |
| **Market Breadth** | % of stocks in Universe > MA50 exceeds 50% | +30 points |
| **Macro Alignment** | Macro state (vix, dxy, tnx) is `RISK_ON` | +20 points |

*Note: If Macro is in `NEUTRAL` state, only +10 points are added.*

### 1.2. Market Regime Classification
Based on the total `conf_score_global`, the system classifies the market into 4 scenarios:

- **STRONG BULLISH ($\ge 75$):** Strong growth, high consensus.
- **BULLISH ($\ge 50$):** Established uptrend, low risk.
- **NEUTRAL / SIDEWAYS ($\ge 35$):** Sideways market, unclear trend.
- **BEARISH / CAUTION ($< 35$):** Weak market, high risk.

---

## 2. Quality Index & Individual Quality Score
This indicator represents the "intrinsic quality" of the market or a specific stock.

### 2.1. Market Quality Index
A Market Cap-weighted average of all stocks in the Universe:
`Market Quality Index = Σ(Quality Score * Market Cap) / Σ(Market Cap)`

### 2.2. Individual Quality Score v4.1 (Scale of 100)
Each stock is evaluated across **7 financial pillars** (config-driven from `config/scoring_rules.yaml`):

#### Pillar 1: Valuation (Max 20 pts)
- **PEG Ratio:** Preferred < 1.5 (growth at reasonable price). Scores 0-12 pts.
- **P/E Ratio:** Sector-adjusted bands (Tech: 15-35 ideal, Value: 10-22 ideal). Scores 0-12 pts.
- **P/B Ratio:** Financials have different norms (1.0-1.8 ideal) vs. Tech/Industrial (< 3.0). Scores 0-8 pts.
- **Early Stage Logic:** Pre-profit growth stocks (negative P/E + revenue growth > 15% + improving EPS) are exempt from harsh P/E penalties and scored on revenue acceleration instead.

#### Pillar 2: Profitability (Max 25-30 pts)
- **FCF Margin:** > 15% = excellent (15 pts), > 8% = good (12 pts), > 5% = fair (6 pts).
- **ROE:** > 15% = excellent (10 pts), > 10% = good (8 pts), > 5% = fair (4 pts).
- **Tech Bonus:** +5 pts if FCF > 20% (exceptional cash generation for tech/growth stocks).
- **Early Stage Credit:** Partial profitability credit (0-7 pts) when losses are shrinking (positive earnings growth).
- **Cap:** 30 pts for Tech/Growth sectors, 25 pts for others.

#### Pillar 3: Financial Health (Max 15 pts)
- **Debt/EBITDA Ratio:** < 2.0 = excellent (15 pts), < 4.0 = good (8 pts), > 8.0 = red flag territory.
- **Sector-adjusted:** Financials/Utilities have higher tolerance (< 6.0 acceptable due to business model).

#### Pillar 4: Net Payout Yield (Max 10 pts, Tech cap 5 pts)
- **Dividend + Buyback Yield:** 4-6% = ideal (9-10 pts), 2.5-4% = good (6 pts), 1-2.5% = fair (3 pts).
- **Tech Cap:** Growth stocks capped at 5 pts to avoid penalizing reinvestment strategies.

#### Pillar 5: Context & Momentum (Max 15 pts) — **Reduced from 25 in v3.0**
- **MA Signal:** Bullish = +8 pts, Neutral = +3 pts, Bearish = 0 pts.
- **RSI:** 40-60 (neutral zone) = +5 pts, < 30 (oversold) = contrarian bonus (0-3 pts), > 70 (overbought) = penalty (0 to -2 pts).
- **Z-Score:** < -1.5 (deep value) = +4 pts, > +2.0 (overheated) = -2 to -4 pts.

#### Pillar 6: Analyst Estimates (Max 10 pts) — **Increased from 5 in v3.0**
- **Upside Potential:** 30%+ = +5 pts, 15-30% = +4 pts, 5-15% = +2 pts, < 5% = +1 pt.
- **Consensus Quality:** Strong Buy = +5 pts, Buy = +3 pts, Hold = +1 pt, Sell/Underperform = -2 pts.
- **Rationale:** Collective analyst research reflects deep fundamental due diligence and is a high-signal indicator.

#### Pillar 7: Revenue Consistency (Max 5 pts) — **NEW in v4.0**
- **Accelerating:** Revenue growth > 15% + Earnings growth > 10% = 5 pts (strong double-digit growth on both).
- **Stable:** Revenue growth > 5% + Earnings not declining = 3 pts (moderate growth, losses not widening).
- **Positive:** Revenue growth > 0% = 2 pts (at least top-line is growing).
- **Declining:** Revenue < -5% = 0 pts (no credit for shrinking business).

#### Red Flags (Instant Penalties) — **Strengthened in v4.0**
- **Negative P/E:** -3 pts (early stage with high growth), -8 pts (high growth but unprofitable), -15 pts (stagnant unprofitable).
- **High Debt:** D/EBITDA > 8 = -5 pts, > 12 = -15 pts (critical distress signal). Threshold tightened from 10 in v3.0.
- **Value Trap:** Z-Score < -1.5 + Sell consensus = -5 pts (cheap for a reason).
- **Beta Risk:** > 1.8 = -1 to -5 pts (high volatility penalty), < 0.8 (non-tech) = +2 to +5 pts (defensive stability bonus).

**Config-Driven Architecture:** All thresholds and weights are loaded from `config/scoring_rules.yaml`, enabling easy tuning without code changes. Improved error handling with safe fallbacks for missing data.

---

## 3. Smart Money Detection v6.0 (Institutional Flow Analysis)

This advanced indicator detects institutional buying/selling activity through a three-layer architecture with multi-factor validation.

### 3.1. Architecture Overview

The system uses a **priority hierarchy** to detect institutional flow:

1. **Layer 1 (Highest Priority):** OBV Divergence + MFI Confirmation
2. **Layer 2 (Medium Priority):** Institutional Volume Pattern Detection
3. **Layer 3 (Fallback):** OBV Trend vs MA(21)

### 3.2. Layer 1: OBV Divergence + MFI Confirmation

**On-Balance Volume (OBV) Divergence** detects when price and volume move in opposite directions:

- **Hidden Accumulation:** Price ↓ + OBV ↑ → Institutions buying dips
- **Hidden Distribution:** Price ↑ + OBV ↓ → Institutions selling into rallies

**NEW in v6.0: Money Flow Index (MFI) Cross-Validation**

MFI is RSI applied to money flow (price × volume) instead of just price. When MFI confirms OBV divergence:
- **Accumulation:** Both OBV and MFI rising while price falls
- **Distribution:** Both OBV and MFI falling while price rises
- **Strength Bonus:** +15 points when MFI confirms the signal

**Adaptive Window Sizing:**
```
Volatility (ATR/Price) > 4.0% → 25-day window (high volatility)
Volatility 2.5-4.0%           → 20-day window (medium volatility)
Volatility < 2.5%             → 15-day window (low volatility)
```

**Strength Scoring (0-100 points):**
- **OBV Magnitude (0-35 pts):** Size of OBV change relative to average volume
- **Price Magnitude (0-20 pts):** Size of price move (larger divergence = stronger signal)
- **Volume Confirmation (0-15 pts):** Recent 5-day volume vs window average
- **Consistency (0-15 pts):** % of days in window supporting the divergence
- **MFI Confirmation (0-15 pts):** Bonus when MFI confirms OBV direction

### 3.3. Layer 2: Institutional Volume Pattern Detection

Detects **large block trades** (volume spikes > 2x average) that indicate institutional activity:

**Detection Logic:**
1. Identify days with volume > threshold (sector-adjusted)
2. Check if large volume aligns with price direction
3. **Institutional Buying:** Large volume on up days (≥60% consistency)
4. **Institutional Selling:** Large volume on down days (≥60% consistency)

**Sector-Specific Thresholds:**
- **Tech/Growth:** 2.5x average (higher retail participation, need stronger signal)
- **Banks/Financials:** 1.8x average (lower volume, easier to detect institutional)
- **Other Sectors:** 2.0x average (default)

**Strength Scoring:**
```
Strength = (Consistency × 50) + (Volume Spike Magnitude × 30) + (Volume Quality × 20)
```

Minimum threshold: 40 points to trigger signal (filters out weak patterns)

### 3.4. Layer 3: OBV Trend vs MA(21) — Fallback

Classic institutional flow indicator when no divergence or volume pattern is detected:

- **OBV > MA(21) for 3+ of last 5 days:** ACCUMULATION
- **OBV < MA(21) for 3+ of last 5 days:** DISTRIBUTION

**Enhanced Strength Scoring in v6.0:**
```
Strength = (Consistency × 40) + (Distance from MA × 30) + (Volume Quality × 30)
```

Volume Quality component (new in v6.0) rewards institutional-style volume patterns.

### 3.5. Volume Quality Score (0-100)

Distinguishes institutional volume patterns from retail activity:

**Factor 1: Volume Concentration (30 pts)**
- Measures frequency of large block trades
- Institutional: Concentrated large trades
- Retail: Distributed small trades

**Factor 2: Volume-Price Correlation (40 pts)**
- Measures how volume aligns with price moves
- Institutional: High correlation (conviction moves)
- Retail: Low correlation (erratic trading)

**Factor 3: Volume Consistency (30 pts)**
- Measures volume stability (coefficient of variation)
- Institutional: Steady, predictable volume
- Retail: Erratic, unpredictable volume

### 3.6. Signal Interpretation

| Signal | Strength | Meaning | Action |
|---|---|---|---|
| ACCUMULATION_STRONG | ≥80 | Very strong institutional buying | High conviction buy signal |
| ACCUMULATION_STRONG | 65-79 | Strong institutional buying | Buy signal |
| ACCUMULATION_WEAK | 40-64 | Moderate institutional buying | Cautious buy signal |
| ACCUMULATION_WEAK | <40 | Weak signal (ignored) | No action |
| DISTRIBUTION_STRONG | ≥80 | Very strong institutional selling | High conviction sell signal |
| DISTRIBUTION_STRONG | 65-79 | Strong institutional selling | Sell signal |
| DISTRIBUTION_WEAK | 40-64 | Moderate institutional selling | Cautious sell signal |
| DISTRIBUTION_WEAK | <40 | Weak signal (ignored) | No action |
| NEUTRAL | - | No clear institutional flow | No action |

### 3.7. Integration with Master Positioning Score

Smart Money contributes to the 6-Pillar Institutional Rating:

```python
if strength >= 80:
    points = +1.25 (ACCUMULATION) or -1.25 (DISTRIBUTION)
elif strength >= 65:
    points = +1.0 (ACCUMULATION) or -1.0 (DISTRIBUTION)
elif strength >= 40:
    points = +0.5 (ACCUMULATION) or -0.5 (DISTRIBUTION)
else:
    points = 0.0 (signal too weak, ignored)
```

**Example:** "DISTRIBUTION_WEAK (TREND) Strength: 60/100 Points: -0.50"
- **DISTRIBUTION:** Institutions selling
- **WEAK:** Moderate signal (not strong)
- **(TREND):** Detected via Layer 3 (OBV trend), not divergence
- **60/100:** Confidence score
- **-0.50:** Penalty applied to positioning score

### 3.8. Advantages Over Traditional OBV

| Aspect | Traditional OBV | Smart Money v6.0 |
|---|---|---|
| **Validation** | OBV only | OBV + MFI cross-validation |
| **Volume Analysis** | Binary (up/down) | Quality scoring (institutional vs retail) |
| **Sector Awareness** | None | Sector-specific thresholds |
| **False Positives** | High (sideways markets) | Low (multi-layer filtering) |
| **Strength Scoring** | None | 0-100 confidence score |
| **Block Trade Detection** | No | Yes (Layer 2) |

---

## 4. Fundamental Momentum Index (FMI) v4.0
---

## 4. Fundamental Momentum Index (FMI) v4.0
FMI measures the **acceleration** of fundamental factors. A stock might have a low Quality score (due to high valuation) but a high FMI (due to explosive growth).

### 4.1. FMI Score Structure (Max 100 points)
1.  **Revenue Acceleration (30 pts):** Most recent quarterly revenue growth vs. annual average.
2.  **EPS Acceleration (30 pts):** Speed of Earnings Per Share (EPS) growth.
3.  **Margin Expansion (25 pts):** Profit margin expansion (EPS growing faster than revenue).
4.  **Earnings Consistency (15 pts):** Number of positive growth quarters in the last 4.

---

## 5. AI Trading Signature (Executive Verdict)
Located in the **Predictive Suite** tab, this system makes the final trading decision based on the convergence of AI data, cash flow, and risk management.

### 5.1. Conviction Score (Scale of 3)
The verdict is based on the consensus of 3 "pillars":
1.  **AI Upside:** Model-predicted gain (LSTM/Transformer/PatchTST) $\ge 3\%$. (+1 pt)
2.  **Smart Money:** OBV ROC indicator shows Accumulation. (+1 pt)
3.  **News Sentiment:** Sentiment from FinBERT $> 0.05$. (+1 pt)

### 5.2. Risk Management R/R (Risk/Reward)
- **Reward:** Distance from current price to AI target (`_ai_target`).
- **Risk:** Distance from current price to statistical Stop-loss (`_ai_stop` - 10th percentile of Monte Carlo).
- **R/R Ratio:** `Reward / Risk`.

### 5.3. Recommendation Logic (Action Hierarchy)
The system combines **Conviction Score** AND **R/R** to issue a Verdict:

- **STRONG LONG:** 3/3 Conviction points AND R/R $\ge 1.5$.
- **BUY / ACCUMULATE:** $\ge 2/3$ Conviction points AND R/R $\ge 1.0$.
- **REDUCE / HEDGE:** When AI predicts a decline $\le -3\%$.
- **AVOID / WAIT:** 0/3 Conviction points (all signals contradict).
- **NEUTRAL / MONITOR:** All other cases (mixed signals).

---

## 6. Zone-Based Support & Resistance Detection v2.0

The system uses an advanced **zone-based approach** to identify support and resistance levels, recognizing that S/R in real markets are **price ranges** (zones) rather than single precise points.

### 6.1. Core Philosophy: Zones vs. Levels

**Traditional Approach (Deprecated):**
- Support/Resistance as single price points (e.g., S1 = $95.00)
- Fixed window sizes regardless of volatility
- Simple recency + volume weighting

**Modern Zone-Based Approach (v2.0):**
- Support/Resistance as **price ranges** (e.g., S1 zone = $94.50-$95.50)
- **Adaptive window** based on ATR/volatility
- **Clustering** of nearby swing points
- **Multi-factor strength scoring**

### 5.2. Adaptive Window Sizing

Window size automatically adjusts based on stock volatility (ATR):

```python
volatility_pct = (ATR_14 / current_price) × 100

if volatility_pct > 5.0:      # High volatility
    window = base_window + 4
elif volatility_pct > 3.0:    # Medium volatility  
    window = base_window + 2
else:                          # Low volatility
    window = base_window
```

**Rationale:** High-volatility stocks require wider windows to filter noise; stable stocks use narrower windows for precision.

### 5.3. Swing Point Detection

**Swing Low (Support Candidate):**
- A price low that is the **local minimum** within its window
- Example (window=5): Day 3 is swing low if `low[3] < min(low[1], low[2], low[4], low[5])`

**Swing High (Resistance Candidate):**
- A price high that is the **local maximum** within its window
- Example (window=5): Day 3 is swing high if `high[3] > max(high[1], high[2], high[4], high[5])`

### 5.4. Clustering Algorithm

Nearby swing points are merged into zones:

1. Sort all swing points by price
2. Group points within `±zone_width_pct` of each other
3. Calculate zone midpoint as average of all points in cluster

**Example:**
```
Swing lows: 94.80, 95.20, 95.10, 98.50
Zone width: ±1.0%

Result:
- Zone 1: [94.80, 95.20, 95.10] → midpoint = 95.03
- Zone 2: [98.50] → midpoint = 98.50
```

### 5.5. Zone Strength Scoring (Composite Formula)

Each zone receives a strength score (0-1) based on **4 factors**:

```
Strength = (Recency × 0.30) + 
           (Pivot Volume × 0.25) + 
           (Retest Count × 0.25) + 
           (Reaction Magnitude × 0.20)
```

#### Factor 1: Recency (30%)
- More recent swing points = stronger signal
- Normalized by position in lookback window: `avg_index / window_length`

#### Factor 2: Pivot Volume (25%)
- Higher volume at swing point = stronger zone
- Normalized: `avg_pivot_volume / avg_volume` (capped at 3.0)

#### Factor 3: Retest Count (25%)
- More tests of the zone = stronger validation
- Formula: `min(test_count / 5.0, 1.0)` (capped at 5 tests)

#### Factor 4: Reaction Magnitude (20%)
- Larger price bounce/rejection = stronger zone
- Support: measures % bounce from low
- Resistance: measures % drop from high
- Formula: `min(reaction_pct / 10.0, 1.0)` (capped at 10%)

**Example:**
```
Zone at $95.00:
- 3 swing lows (most recent 40 days ago)
- Average volume: 2.5× daily average
- 3 retests
- Average bounce: 8%

Strength = (0.67 × 0.30) + (0.83 × 0.25) + (0.60 × 0.25) + (0.80 × 0.20)
         = 0.201 + 0.208 + 0.150 + 0.160
         = 0.719 (Strong zone)
```

### 5.6. Multi-Timeframe Architecture

The system calculates zones across 3 timeframes:

| Level | Timeframe | Lookback | Base Window | Purpose |
|---|---|---|---|---|
| **S1/R1** | Short-term | 20 days | 3 days | Tactical trading (intraday to swing) |
| **S2/R2** | Medium-term | 60 days | 5 days | Position trading (weeks to months) |
| **S3/R3** | Long-term | 252 days | 7 days | Strategic investing (quarters to years) |

### 5.7. Smart Hierarchy Selection

**Old Method (Deprecated):**
- Force S2 = S1 × 0.98 if S2 ≥ S1 (artificial adjustment)
- Force R2 = R1 × 1.02 if R2 ≤ R1

**New Method (v2.0):**
- Only use S2 if it's **meaningfully different** (≥3% below S1)
- Only use S3 if it's **meaningfully different** (≥5% below S2)
- If zones overlap, **skip that level** rather than forcing artificial values

**Rationale:** Preserves integrity of detected zones; avoids "bending" real market structure.

### 5.8. Derived Trading Levels

From zone midpoints, the system calculates actionable levels:

```python
# Zone-aware stop loss (below S1 zone boundary)
stop_loss = S1 × (1 - zone_width × 1.5)

# Zone-aware target (above R1 zone boundary)
TP1 = R1 × (1 + zone_width × 1.5)

# Secondary targets use zone midpoints
TP2 = R2
TP3 = R3
```

### 5.9. Risk/Reward Calculation

```python
risk_distance = current_price - stop_loss
reward_distance = TP1 - current_price

R/R Ratio = reward_distance / risk_distance
```

**Interpretation:**
- **R/R ≥ 2.5:** Asymmetric opportunity (High conviction)
- **R/R 1.2-2.5:** Acceptable setup (Medium conviction)
- **R/R < 1.2:** Unfavorable setup (Low conviction)

### 5.10. Practical Example

**Stock: AAPL, Current Price: $175.00**

**Step 1: Detect Swing Points (20-day lookback)**
- ATR = $3.50 → Volatility = 2.0% → Window = 3 (low vol)
- Found 4 swing lows: $172.50, $173.00, $172.80, $168.00

**Step 2: Cluster into Zones**
- Zone width = 1.0% (based on ATR)
- Zone 1: [$172.50, $173.00, $172.80] → midpoint = $172.77
- Zone 2: [$168.00] → midpoint = $168.00

**Step 3: Score Zones**
- Zone 1 strength: 0.82 (3 tests, high volume, 4% bounce)
- Zone 2 strength: 0.45 (1 test, medium volume, 2% bounce)

**Step 4: Select Best Zone**
- S1 = $172.77 (Zone 1 - highest strength, nearest to price)

**Step 5: Calculate Trading Levels**
- Stop Loss = $172.77 × (1 - 0.01 × 1.5) = $170.18
- TP1 = $178.50 (R1 zone)
- R/R = ($178.50 - $175.00) / ($175.00 - $170.18) = 0.73 (Low - wait for better entry)

### 5.11. Advantages Over Traditional Methods

| Aspect | Traditional | Zone-Based v2.0 |
|---|---|---|
| **Precision** | Single point (unrealistic) | Price range (realistic) |
| **Adaptability** | Fixed window | ATR-adaptive window |
| **Validation** | Simple volume weight | 4-factor strength score |
| **Clustering** | None (noise) | Merges nearby points |
| **Hierarchy** | Forced adjustment | Smart selection |
| **Stop Loss** | Arbitrary % | Zone-boundary aware |

### 5.12. Implementation Notes

- **Function:** `detect_swing_zones()` in `app.py`
- **Called by:** `get_tactical_metrics()` for all tabs (Screener, Deep Dive, Portfolio)
- **Caching:** Results cached per ticker to avoid redundant calculations
- **Fallback:** If insufficient data (<15 days), falls back to simple min/max

---

## 7. Portfolio Optimization Strategies

The system provides 3 optimization strategies in the **Portfolio Builder** tab, allowing investors to customize based on risk appetite.

### 7.1. Max Sharpe (Markowitz MVO)
- **Principle:** Based on **Modern Portfolio Theory (MPT)**, it finds weights ($w$) to maximize the **Sharpe Ratio**:
  $$\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}$$
- **Characteristics:** Concentrates capital on assets with the best risk-adjusted efficiency.
- **Suitability:** Investors seeking optimal returns who accept higher concentration.

### 7.2. Risk Parity
- **Principle:** Allocates capital such that each asset contributes **an equal amount of risk** to the total portfolio risk.
  $$\min \sum_{i=1}^{n} (RC_i - \frac{1}{n})^2$$
  Where Risk Contribution ($RC_i$) is: $RC_i = \frac{w_i (\Sigma w)_i}{\sqrt{w^T \Sigma w}}$
- **Characteristics:** High Volatility assets receive less capital, stable assets receive more.
- **Suitability:** Defensive portfolios, prioritizing stability and absolute safety.

### 7.3. Equal Weight (1/N)
- **Principle:** Equal capital allocation across all assets: $w_i = \frac{1}{n}$.
- **Characteristics:** Maximum diversification, not dependent on historical return/volatility estimates.
- **Suitability:** Long-term diversification believers who want to avoid "putting all eggs in one basket."

### 7.4. System Constraints
To ensure practical safety, all optimization models adhere to:
1.  **Full Investment:** $\sum w_i = 100\%$.
2.  **Concentration Cap:** Max **40%** per asset (for MVO/RP).
3.  **Min Weight Floor:** Users can set a minimum threshold (e.g., 2%).

---

## 8. Reference Forecasting Models
The system uses an Ensemble of deep learning architectures:
- **LSTM (v7.2):** Optimized for cyclicality and temporal stability.
- **Transformer (v8.0):** Optimized for high-volatility pattern identification.
- **PatchTST (v10.0):** Channel-independent processing, optimized for long-term fundamental-based forecasting.

---

## 9. Portfolio Performance & Risk Metrics

Key health metrics found in the **Portfolio Builder** tab.

### 9.1. Weighted Return
The actual return of the entire portfolio based on capital allocation:
$$R_p = \sum_{i=1}^{n} w_i R_i$$

### 9.2. Annual Vol (Annualized Volatility)
Measures systematic risk through the standard deviation of returns:
$$\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$$

### 9.3. Value at Risk (VaR 95%)
Maximum expected 1-day loss with 95% confidence. If VaR is -2%, there is a 95% probability that the portfolio will not lose more than 2% in a day.

### 9.4. Conditional Value at Risk (CVaR / Expected Shortfall)
The average loss in "worst-case" scenarios (the 5% tail beyond VaR).

---

## 10. Unified Alpha-Risk Intelligence Hub
Advanced AI system integrated into the **Deep Dive** tab for cross-checking and data unification (Convergence Analysis).

- **Objective:** Consolidate quantitative data (Metrics) and qualitative data (News NLP) to provide decisive action recommendations.
- **Mechanism Details:** See [AI_INTELLIGENCE.md](./AI_INTELLIGENCE.md).
