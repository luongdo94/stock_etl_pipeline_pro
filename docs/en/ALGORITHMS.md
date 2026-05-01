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

## 3. Fundamental Momentum Index (FMI) v4.0
FMI measures the **acceleration** of fundamental factors. A stock might have a low Quality score (due to high valuation) but a high FMI (due to explosive growth).

### 3.1. FMI Score Structure (Max 100 points)
1.  **Revenue Acceleration (30 pts):** Most recent quarterly revenue growth vs. annual average.
2.  **EPS Acceleration (30 pts):** Speed of Earnings Per Share (EPS) growth.
3.  **Margin Expansion (25 pts):** Profit margin expansion (EPS growing faster than revenue).
4.  **Earnings Consistency (15 pts):** Number of positive growth quarters in the last 4.

---

## 4. AI Trading Signature (Executive Verdict)
Located in the **Predictive Suite** tab, this system makes the final trading decision based on the convergence of AI data, cash flow, and risk management.

### 4.1. Conviction Score (Scale of 3)
The verdict is based on the consensus of 3 "pillars":
1.  **AI Upside:** Model-predicted gain (LSTM/Transformer/PatchTST) $\ge 3\%$. (+1 pt)
2.  **Smart Money:** OBV ROC indicator shows Accumulation. (+1 pt)
3.  **News Sentiment:** Sentiment from FinBERT $> 0.05$. (+1 pt)

### 4.2. Risk Management R/R (Risk/Reward)
- **Reward:** Distance from current price to AI target (`_ai_target`).
- **Risk:** Distance from current price to statistical Stop-loss (`_ai_stop` - 10th percentile of Monte Carlo).
- **R/R Ratio:** `Reward / Risk`.

### 4.3. Recommendation Logic (Action Hierarchy)
The system combines **Conviction Score** AND **R/R** to issue a Verdict:

- **STRONG LONG:** 3/3 Conviction points AND R/R $\ge 1.5$.
- **BUY / ACCUMULATE:** $\ge 2/3$ Conviction points AND R/R $\ge 1.0$.
- **REDUCE / HEDGE:** When AI predicts a decline $\le -3\%$.
- **AVOID / WAIT:** 0/3 Conviction points (all signals contradict).
- **NEUTRAL / MONITOR:** All other cases (mixed signals).

---

## 5. Portfolio Optimization Strategies

The system provides 3 optimization strategies in the **Portfolio Builder** tab, allowing investors to customize based on risk appetite.

### 5.1. Max Sharpe (Markowitz MVO)
- **Principle:** Based on **Modern Portfolio Theory (MPT)**, it finds weights ($w$) to maximize the **Sharpe Ratio**:
  $$\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}$$
- **Characteristics:** Concentrates capital on assets with the best risk-adjusted efficiency.
- **Suitability:** Investors seeking optimal returns who accept higher concentration.

### 5.2. Risk Parity
- **Principle:** Allocates capital such that each asset contributes **an equal amount of risk** to the total portfolio risk.
  $$\min \sum_{i=1}^{n} (RC_i - \frac{1}{n})^2$$
  Where Risk Contribution ($RC_i$) is: $RC_i = \frac{w_i (\Sigma w)_i}{\sqrt{w^T \Sigma w}}$
- **Characteristics:** High Volatility assets receive less capital, stable assets receive more.
- **Suitability:** Defensive portfolios, prioritizing stability and absolute safety.

### 5.3. Equal Weight (1/N)
- **Principle:** Equal capital allocation across all assets: $w_i = \frac{1}{n}$.
- **Characteristics:** Maximum diversification, not dependent on historical return/volatility estimates.
- **Suitability:** Long-term diversification believers who want to avoid "putting all eggs in one basket."

### 5.4. System Constraints
To ensure practical safety, all optimization models adhere to:
1.  **Full Investment:** $\sum w_i = 100\%$.
2.  **Concentration Cap:** Max **40%** per asset (for MVO/RP).
3.  **Min Weight Floor:** Users can set a minimum threshold (e.g., 2%).

---

## 6. Reference Forecasting Models
The system uses an Ensemble of deep learning architectures:
- **LSTM (v7.2):** Optimized for cyclicality and temporal stability.
- **Transformer (v8.0):** Optimized for high-volatility pattern identification.
- **PatchTST (v10.0):** Channel-independent processing, optimized for long-term fundamental-based forecasting.

---

## 7. Portfolio Performance & Risk Metrics

Key health metrics found in the **Portfolio Builder** tab.

### 7.1. Weighted Return
The actual return of the entire portfolio based on capital allocation:
$$R_p = \sum_{i=1}^{n} w_i R_i$$

### 7.2. Annual Vol (Annualized Volatility)
Measures systematic risk through the standard deviation of returns:
$$\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$$

### 7.3. Value at Risk (VaR 95%)
Maximum expected 1-day loss with 95% confidence. If VaR is -2%, there is a 95% probability that the portfolio will not lose more than 2% in a day.

### 7.4. Conditional Value at Risk (CVaR / Expected Shortfall)
The average loss in "worst-case" scenarios (the 5% tail beyond VaR).

---

## 8. Unified Alpha-Risk Intelligence Hub
Advanced AI system integrated into the **Deep Dive** tab for cross-checking and data unification (Convergence Analysis).

- **Objective:** Consolidate quantitative data (Metrics) and qualitative data (News NLP) to provide decisive action recommendations.
- **Mechanism Details:** See [AI_INTELLIGENCE.md](./AI_INTELLIGENCE.md).
