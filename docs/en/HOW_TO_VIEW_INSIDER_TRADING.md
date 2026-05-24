# 📖 How to View Insider Trading Data

## 🎯 Quick Start Guide

### Step 1: Open Dashboard
```bash
streamlit run app.py
```

### Step 2: Navigate to Stock Scanner
Click on the **🔭 Stock Scanner** tab at the top of the dashboard.

### Step 3: Scroll to Insider Section
Scroll down to the bottom of the page to find the **👥 Insider Trading Intelligence** section.

---

## 🔍 Using the Filters

### Filter Options:

#### 1. **🎯 Ticker**
- Select **"All"** to see all tickers
- Or choose a specific ticker (e.g., NVDA, AAPL, MSFT)
- Only shows tickers that have insider data

#### 2. **📝 Transaction Type**
- **All**: Show all transaction types
- **Buy**: Only insider purchases (most important!)
- **Sale**: Only insider sales
- **Award**: Stock awards/grants (compensation)
- **Exercise**: Option exercises
- **Gift**: Stock gifts (often to family/charity)
- **Unknown**: Unclassified transactions

#### 3. **📅 Period**
- **30 days**: Last month
- **60 days**: Last 2 months
- **90 days**: Last quarter (default)
- **180 days**: Last 6 months
- **365 days**: Last year

#### 4. **💰 Min Value ($K)**
- Filter out small transactions
- Enter minimum value in thousands of dollars
- Example: Enter "100" to show only transactions ≥ $100,000

---

## 📊 Understanding the Display

### Metrics Row:
- **📊 Transactions**: Total number of transactions matching filters
- **💰 Total Value**: Combined value of all transactions
- **👥 Insiders**: Number of unique insiders involved
- **🟢 Buy Ratio**: Percentage of buys vs sales (higher = more bullish)

### Charts:
- **Left Chart**: Transaction type breakdown (pie chart)
- **Right Chart**: Top 10 insiders by transaction value (bar chart)

### Transaction Table:
- **Color Coding**:
  - 🟢 Green background = Buy (bullish signal)
  - 🔴 Red background = Sale (bearish signal)
  - 🔵 Blue background = Award (neutral)
  - 🟠 Orange background = Exercise (neutral)
  - 🟣 Purple background = Gift (neutral)

### Columns Explained:
- **Ticker**: Stock symbol
- **Insider Name**: Name of the insider
- **Position**: Role (CEO, CFO, Director, Officer, etc.)
- **Type**: Transaction type
- **Shares**: Number of shares traded
- **Value**: Transaction value in thousands of USD
- **Date**: Transaction date
- **Own**: Ownership type (D=Direct, I=Indirect)
- **Description**: Full transaction details

---

## 💡 Key Insights Section

Click **"💡 Key Insights"** expander to see:

### 🟢 Recent Insider Buys
- Shows last 5 insider purchases
- **Green boxes** with transaction details
- Most important signal to watch!

### 🔴 Largest Insider Sales
- Shows top 5 sales by value
- **Red boxes** with transaction details
- Context matters: selling for diversification vs. lack of confidence

---

## 📥 Exporting Data

Click the **"📥 Download CSV"** button to export:
- All filtered transactions
- Includes all columns
- Filename format: `insider_transactions_{ticker}_{days}d.csv`

---

## 🎯 Example Use Cases

### Use Case 1: Find Recent CEO Purchases
1. Set **Transaction Type** = "Buy"
2. Set **Period** = "90 days"
3. Set **Min Value** = "100" ($100K+)
4. Look for **Position** = "Chief Executive Officer"
5. **Why**: CEO buying with personal money = strong conviction

### Use Case 2: Check Specific Stock
1. Select **Ticker** = "NVDA" (or any stock)
2. Set **Period** = "180 days"
3. Review all transaction types
4. Check **Buy Ratio** metric
5. **Why**: Understand insider sentiment for specific stock

### Use Case 3: Find Cluster Buying
1. Set **Transaction Type** = "Buy"
2. Set **Period** = "30 days"
3. Look for multiple insiders from same company
4. Check if 3+ insiders buying
5. **Why**: Multiple insiders buying = powerful signal

### Use Case 4: Monitor Large Sales
1. Set **Transaction Type** = "Sale"
2. Set **Min Value** = "500" ($500K+)
3. Set **Period** = "60 days"
4. Check if C-suite executives selling
5. **Why**: Heavy insider selling = caution signal

---

## 🚨 Red Flags to Watch

### ⚠️ Warning Signs:
1. **Heavy C-Suite Selling**
   - CEO/CFO selling large amounts
   - Multiple executives selling simultaneously
   - Selling near 52-week highs

2. **No Insider Buying**
   - Stock down 20%+ but no insider buys
   - Suggests insiders don't see value

3. **Selling Before Earnings**
   - Insiders selling weeks before earnings
   - May indicate weak results coming

### ✅ Bullish Signs:
1. **CEO Purchases**
   - CEO buying with personal money
   - Especially if first purchase in years
   - Large amounts (>$100K)

2. **Cluster Buying**
   - 3+ insiders buying within 30 days
   - Suggests shared positive outlook

3. **Buying Near Lows**
   - Insiders buying when stock down 20%+
   - Indicates they see value

---

## 🔧 Troubleshooting

### "No transactions found"
- **Cause**: Filters too restrictive or no data for selected ticker
- **Solution**: 
  - Try "All" ticker
  - Increase period to 180 or 365 days
  - Remove min value filter
  - Check if ticker is US-based (international stocks have limited data)

### "No insider data available"
- **Cause**: Database not populated yet
- **Solution**: Run `python3 load_insider_data.py`

### Table not loading
- **Cause**: Database locked or query timeout
- **Solution**: 
  - Close other connections to database
  - Reduce period or add filters
  - Restart dashboard

---

## 📚 Additional Resources

### Documentation:
- **Full Guide**: `docs/en/INSIDER_TRADING_GUIDE.md`
- **Quick Reference**: `INSIDER_TRADING_QUICK_REF.md`
- **Complete Status**: `INSIDER_TRADING_COMPLETE.md`

### CLI Tools:
```bash
# View transactions via command line
python3 view_insider_transactions.py --ticker NVDA --days 90

# Check database status
python3 quick_check_insider.py

# Update data manually
python3 load_insider_data.py
```

---

## 💡 Pro Tips

1. **Combine with Fundamentals**: Don't rely on insider data alone. Check PE, PEG, FCF, etc.

2. **Context Matters**: Understand WHY insiders are selling:
   - Diversification? (Neutral)
   - Pre-planned 10b5-1? (Neutral)
   - Margin calls? (Bearish)

3. **Size Matters**: A $1M CEO purchase is more meaningful than a $10K director sale

4. **Watch the Ratio**: Buy Ratio > 60% = bullish, < 40% = bearish

5. **Export for Analysis**: Download CSV and analyze in Excel/Python for deeper insights

---

**Happy Insider Tracking! 🎯**

Remember: Insiders buy for one reason (they think stock will rise), but sell for many reasons (diversification, taxes, personal needs). Focus on the BUYS! 🟢
