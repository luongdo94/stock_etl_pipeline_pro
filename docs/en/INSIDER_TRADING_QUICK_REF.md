# 🎯 Insider Trading - Quick Reference Card

## 📊 Database Tables

| Table | Records | Purpose |
|-------|---------|---------|
| `raw.insider_transactions` | 2,486 | Detailed transaction history |
| `raw.insider_summary` | 36 | 6-month aggregated data |
| `marts.dim_companies` | All | Enhanced with insider signals |

## 🔍 Key Columns in `dim_companies`

```sql
insider_ownership          -- % shares held by insiders
insider_net_shares_6m      -- Net buying/selling (6M)
insider_purchases_6m       -- Total purchases
insider_sales_6m           -- Total sales
insider_signal             -- 'NET BUY' | 'NET SELL' | 'NEUTRAL'
```

## 🚀 Quick Queries

### 1. Find Net Buyers
```sql
SELECT ticker, company, insider_net_shares_6m
FROM marts.dim_companies
WHERE insider_signal = 'NET BUY'
ORDER BY insider_net_shares_6m DESC
LIMIT 10
```

### 2. Recent CEO Buys
```sql
SELECT ticker, insider_name, shares, value, transaction_date
FROM raw.insider_transactions
WHERE position LIKE '%CEO%'
  AND transaction_type = 'Buy'
  AND transaction_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY value DESC
```

### 3. Cluster Buying
```sql
SELECT ticker, COUNT(DISTINCT insider_name) AS buyers, SUM(value) AS total_value
FROM raw.insider_transactions
WHERE transaction_type = 'Buy'
  AND transaction_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY ticker
HAVING COUNT(DISTINCT insider_name) >= 3
ORDER BY total_value DESC
```

## 🎯 Signal Interpretation

| Signal | Meaning | Action |
|--------|---------|--------|
| 🟢 NET BUY (>50K shares) | Strong insider confidence | **Bullish** - Consider buying |
| 🟢 NET BUY (<50K shares) | Moderate confidence | **Neutral-Bullish** - Monitor |
| ⚪ NEUTRAL | No significant activity | **Neutral** - Use other signals |
| 🔴 NET SELL (<-50K shares) | Insiders reducing exposure | **Bearish** - Caution |
| 🔴 NET SELL (<-100K shares) | Heavy selling | **Very Bearish** - Avoid |

## 📈 Top Current Signals (as of 2026-05-15)

### 🟢 Strong Buyers:
- **NKE**: +763K shares
- **NOW**: +355K shares  
- **QCOM**: +209K shares
- **TMO**: +206K shares

### 🔴 Sellers:
- **CSCO**: -348K shares
- **INTU**: -69K shares

## 🔄 Update Commands

```bash
# Manual update
python3 load_insider_data.py

# Check status
python3 quick_check_insider.py

# Run transform
python3 run.py --fast
```

## ⚠️ Important Rules

1. **Buying > Selling**: Insiders buy for one reason (stock will rise), sell for many
2. **Size Matters**: Large purchases (>$100K) more meaningful
3. **Cluster = Strong**: 3+ insiders buying = powerful signal
4. **CEO Buys = Gold**: C-suite purchases with personal money = highest conviction
5. **Context Required**: Always combine with fundamentals + technicals

## 📚 Full Documentation

- **Comprehensive Guide**: `docs/en/INSIDER_TRADING_GUIDE.md`
- **Implementation Summary**: `docs/en/INSIDER_TRADING_SUMMARY.md`

---

**Quick Tip**: Filter scanner for `insider_signal = 'NET BUY'` + `peg_ratio < 1.0` = High-conviction value plays! 🎯
