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

### 2.2. Individual Quality Score v3.0 (Scale of 100)
Each stock is evaluated across 6 financial pillars:
1.  **Valuation (20 pts):** Evaluates P/E, P/B, and PEG (Lower PEG preferred).
2.  **Profitability (25-30 pts):** Focuses on FCF Margin and ROE.
3.  **Financial Health (15 pts):** Debt/EBITDA ratio tailored to industry standards.
4.  **Net Payout Yield (10 pts):** Total capital returned to shareholders (Dividends + Buybacks).
5.  **Context & Momentum (25 pts):** Technical signals (MA), relative strength (RSI), and price deviation (Z-Score).
6.  **Analyst Estimates (5 pts):** Implied upside and expert consensus.

**Red Flags (Penalties):** Heavy points deduction if P/E is negative, Debt/EBITDA > 10, or Beta is too high (>1.8).

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
