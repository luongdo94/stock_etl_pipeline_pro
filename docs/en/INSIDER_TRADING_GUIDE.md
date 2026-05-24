# 📊 Insider Trading Intelligence Guide

## Overview

Insider trading data provides powerful signals about company health and future prospects. When executives, directors, and major shareholders buy or sell their own company's stock, it often reflects their confidence (or lack thereof) in the business.

## 🎯 Why Insider Trading Matters

### Key Principles:

1. **Insiders have superior information** - They know the business better than anyone
2. **Buying is more significant than selling** - Insiders sell for many reasons (diversification, taxes, personal needs), but they buy for only one: they think the stock will go up
3. **Cluster buying is powerful** - Multiple insiders buying simultaneously is a strong bullish signal
4. **Size matters** - Large purchases relative to insider's wealth are more meaningful

## 📈 How to Interpret Insider Signals

### 🟢 BULLISH SIGNALS (Strong Buy Indicators)

#### 1. **Net Insider Buying**
- `insider_signal = 'NET BUY'`
- `insider_net_shares_6m > 0`
- Multiple insiders buying over 6 months

**Example:**
```
Ticker: NKE (Nike)
Net Shares (6M): +763,000
Signal: NET BUY 🟢
Interpretation: Strong insider confidence
```

#### 2. **CEO/CFO Purchases**
- C-suite executives buying with personal money
- Especially significant if:
  - Purchase > $100K
  - Multiple purchases in short period
  - First purchase in years

#### 3. **Cluster Buying**
- 3+ insiders buying within same month
- Suggests shared positive outlook
- Often precedes good news

#### 4. **Open Market Purchases**
- `transaction_type = 'Buy'`
- More significant than stock awards/options
- Real money at risk

### 🔴 BEARISH SIGNALS (Caution Indicators)

#### 1. **Heavy Insider Selling**
- `insider_signal = 'NET SELL'`
- `insider_net_shares_6m < -100,000`
- Multiple insiders selling simultaneously

**Example:**
```
Ticker: CSCO (Cisco)
Net Shares (6M): -348,000
Signal: NET SELL 🔴
Interpretation: Insiders reducing exposure
```

#### 2. **CEO Selling Before Lockup Expiry**
- Selling immediately when allowed
- Suggests urgency to exit

#### 3. **Selling at 52-Week Highs**
- Insiders selling near peak prices
- May indicate overvaluation

### ⚪ NEUTRAL SIGNALS

#### 1. **Routine Sales**
- Regular 10b5-1 planned sales
- Small amounts relative to holdings
- Diversification/tax planning

#### 2. **Stock Awards/Options Exercise**
- `transaction_type = 'Award'` or `'Exercise'`
- Part of compensation, not discretionary
- Less meaningful than open market buys

## 🔍 Database Schema

### Tables

#### 1. `raw.insider_summary`
6-month aggregated insider activity per ticker.

```sql
SELECT 
    ticker,
    insider_purchases_6m,    -- Total shares purchased
    insider_sales_6m,        -- Total shares sold
    net_shares,              -- Net position (buys - sells)
    pct_buy,                 -- % of shares bought
    pct_sell                 -- % of shares sold
FROM raw.insider_summary
WHERE net_shares > 0         -- Filter for net buyers
ORDER BY net_shares DESC
```

#### 2. `raw.insider_transactions`
Detailed transaction-level data.

```sql
SELECT 
    ticker,
    insider_name,
    position,                -- CEO, CFO, Director, etc.
    transaction_type,        -- Buy, Sale, Award, Exercise, Gift
    shares,
    value,                   -- Transaction value in USD
    transaction_date,
    ownership_type           -- D=Direct, I=Indirect
FROM raw.insider_transactions
WHERE transaction_type = 'Buy'
  AND transaction_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY value DESC
```

#### 3. `marts.dim_companies` (Enhanced)
Insider metrics integrated into company dimension.

```sql
SELECT 
    ticker,
    company,
    insider_ownership,       -- % of shares held by insiders
    insider_net_shares_6m,   -- Net insider activity (6M)
    insider_purchases_6m,
    insider_sales_6m,
    insider_pct_buy,
    insider_pct_sell,
    insider_signal           -- 'NET BUY', 'NET SELL', 'NEUTRAL'
FROM marts.dim_companies
WHERE insider_signal = 'NET BUY'
  AND insider_net_shares_6m > 10000
```

## 📊 Practical Queries

### 1. Find Stocks with Strong Insider Buying

```sql
SELECT 
    c.ticker,
    c.company,
    c.sector,
    c.insider_net_shares_6m,
    c.insider_signal,
    c.pe_ratio,
    c.peg_ratio,
    p.close AS current_price
FROM marts.dim_companies c
LEFT JOIN marts.fct_daily_returns p 
    ON c.ticker = p.ticker 
    AND p.date = (SELECT MAX(date) FROM marts.fct_daily_returns)
WHERE c.insider_signal = 'NET BUY'
  AND c.insider_net_shares_6m > 50000  -- Significant buying
  AND c.peg_ratio < 1.5                -- Reasonable valuation
  AND c.market_cap > 1e9               -- Mid-cap+
ORDER BY c.insider_net_shares_6m DESC
LIMIT 20
```

### 2. Recent CEO Purchases

```sql
SELECT 
    ticker,
    insider_name,
    shares,
    value,
    transaction_date,
    ROUND(value / 1000000, 2) AS value_millions
FROM raw.insider_transactions
WHERE position LIKE '%Chief Executive Officer%'
  AND transaction_type = 'Buy'
  AND transaction_date >= CURRENT_DATE - INTERVAL '90 days'
  AND value > 100000  -- $100K+ purchases
ORDER BY transaction_date DESC, value DESC
```

### 3. Cluster Buying Detection

```sql
WITH insider_counts AS (
    SELECT 
        ticker,
        COUNT(DISTINCT insider_name) AS num_buyers,
        SUM(shares) AS total_shares,
        SUM(value) AS total_value,
        MIN(transaction_date) AS first_buy,
        MAX(transaction_date) AS last_buy
    FROM raw.insider_transactions
    WHERE transaction_type = 'Buy'
      AND transaction_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY ticker
)
SELECT 
    ic.*,
    c.company,
    c.sector,
    c.pe_ratio,
    c.peg_ratio
FROM insider_counts ic
JOIN marts.dim_companies c USING (ticker)
WHERE ic.num_buyers >= 3  -- 3+ insiders buying
  AND ic.total_value > 500000  -- $500K+ total
ORDER BY ic.num_buyers DESC, ic.total_value DESC
```

### 4. Insider Buying vs Stock Performance

```sql
WITH insider_buys AS (
    SELECT 
        ticker,
        SUM(CASE WHEN transaction_type = 'Buy' THEN shares ELSE 0 END) AS buy_shares,
        MIN(transaction_date) AS first_buy_date
    FROM raw.insider_transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY ticker
    HAVING SUM(CASE WHEN transaction_type = 'Buy' THEN shares ELSE 0 END) > 0
),
price_change AS (
    SELECT 
        ticker,
        (MAX(close) - MIN(close)) / MIN(close) * 100 AS pct_change
    FROM marts.fct_daily_returns
    WHERE date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY ticker
)
SELECT 
    ib.ticker,
    c.company,
    ib.buy_shares,
    ib.first_buy_date,
    pc.pct_change AS stock_return_pct,
    CASE 
        WHEN pc.pct_change > 10 THEN '🚀 Strong Gain'
        WHEN pc.pct_change > 0 THEN '📈 Positive'
        WHEN pc.pct_change > -10 THEN '📉 Slight Loss'
        ELSE '💥 Large Loss'
    END AS performance
FROM insider_buys ib
JOIN price_change pc USING (ticker)
JOIN marts.dim_companies c USING (ticker)
ORDER BY pc.pct_change DESC
```

## 🎯 Trading Strategies

### Strategy 1: Insider Buy + Value Play

**Criteria:**
- `insider_signal = 'NET BUY'`
- `insider_net_shares_6m > 25,000`
- `peg_ratio < 1.0`
- `pe_ratio < 20`
- `price_z_score < -0.5` (below average)

**Logic:** Insiders buying undervalued stocks = high conviction value play

### Strategy 2: Cluster Buying Momentum

**Criteria:**
- 3+ insiders buying in last 30 days
- Total purchase value > $500K
- Stock near 52-week low (`pct_from_52w_high < -20%`)
- Positive earnings growth

**Logic:** Multiple insiders buying weakness = turnaround opportunity

### Strategy 3: CEO Conviction Play

**Criteria:**
- CEO purchase > $250K
- First CEO purchase in 12+ months
- Company profitable (net income > 0)
- Reasonable debt (debt/equity < 1.5)

**Logic:** CEO putting significant personal capital at risk = strong signal

## ⚠️ Important Caveats

### 1. **Data Availability**
- Insider data is **primarily available for US stocks**
- International markets have different reporting requirements
- Data may be delayed (SEC Form 4 filings can be 2 days behind)

### 2. **False Signals**
- Not all insider buying leads to stock gains
- Insiders can be wrong about timing
- External factors (market crashes) can override insider signals

### 3. **Context Matters**
- Check **why** insiders are selling:
  - Diversification? (Neutral)
  - Margin calls? (Bearish)
  - Pre-planned 10b5-1? (Neutral)
- Consider **insider ownership %**:
  - High ownership (>20%) = aligned interests
  - Low ownership (<1%) = less meaningful

### 4. **Combine with Other Signals**
Never rely on insider data alone. Use in conjunction with:
- Fundamental analysis (PE, PEG, FCF)
- Technical indicators (RSI, MA signals)
- Earnings trends
- Sector health

## 🔄 Data Refresh Schedule

### Airflow DAG Integration

- **Daily (Fast Mode)**: Skip insider data (too slow)
- **Weekly (Full Mode - Mondays)**: Extract insider summary for US stocks
- **Manual**: Run `python3 load_insider_data.py` for immediate update

### Coverage

- **US Stocks**: Full coverage (transactions + summary)
- **International**: Limited (only `insider_ownership` %)

## 📚 Further Reading

### Regulatory Framework
- **SEC Form 4**: Insider transaction disclosure (US)
- **Section 16**: Defines "insiders" (officers, directors, 10%+ owners)
- **10b5-1 Plans**: Pre-arranged trading plans (less meaningful)

### Academic Research
- **Seyhun (1986)**: Insiders earn abnormal returns
- **Lakonishok & Lee (2001)**: Insider purchases predict returns
- **Jeng et al. (2003)**: Insider purchases > sales for prediction

### Tools & Resources
- **SEC EDGAR**: Official filing database
- **OpenInsider.com**: Aggregated insider data
- **GuruFocus**: Insider tracking with alerts

---

## 🚀 Quick Start

```bash
# 1. Load insider data
python3 load_insider_data.py

# 2. Run transform to update dim_companies
python3 run.py --fast

# 3. Query insider signals
python3 -c "
import duckdb
conn = duckdb.connect('warehouse/stock_dw.duckdb', read_only=True)
df = conn.execute('''
    SELECT ticker, company, insider_signal, insider_net_shares_6m
    FROM marts.dim_companies
    WHERE insider_signal = 'NET BUY'
    ORDER BY insider_net_shares_6m DESC
    LIMIT 10
''').df()
print(df)
"
```

---

**Last Updated:** 2026-05-15  
**Version:** 1.0  
**Author:** Honest Quant Intelligence Platform
