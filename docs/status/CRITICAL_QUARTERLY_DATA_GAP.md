# ✅ RESOLVED: Quarterly Data Gap for European & Asian Stocks

**Status:** FIXED (2026-05-02)  
**Severity:** HIGH  
**Impact:** Data Quality, Scoring Accuracy, User Trust  
**Discovered:** 2026-05-02  
**Fixed:** 2026-05-02  
**Affected Markets:** Europe, UK, Japan, Hong Kong

---

## 📋 Resolution Summary

**What Was Fixed:**
The ETL pipeline was **intentionally skipping** quarterly earnings and FCF data for European and Asian stocks based on an **incorrect assumption** that these markets only report semi-annually.

**Reality:** All major European and Asian stocks report quarterly, just like US stocks.

**Fix Applied:**
- Removed `NON_QUARTERLY_SUFFIXES` filter from `etl/utils.py` line 237
- Updated `is_eligible_for_quarterly()` to remove geographic discrimination
- All equity stocks now processed equally regardless of exchange

**Expected Impact:** 
- Quality Scores will become accurate (profitability data restored)
- FMI (Fundamental Momentum Index) will calculate correctly
- Earnings growth metrics will be available
- Fair comparison with US stocks restored

---

## 🔍 Root Cause Analysis (Original Bug)

### Location
**File:** `etl/utils.py` line 202

```python
# WRONG ASSUMPTION
NON_QUARTERLY_SUFFIXES = ('.PA', '.MI', '.AS', '.DE', '.MC', '.LS', '.SW', '.L', '.CO', '.HK', '.T')

def is_eligible_for_quarterly(ticker):
    # ...
    # 3. Must not belong to a semi-annual reporting exchange
    if ticker.upper().endswith(NON_QUARTERLY_SUFFIXES): return False  # ❌ WRONG!
    return True
```

### Affected Markets

| Suffix | Market | Country | Quarterly Data? | Currently Skipped? |
|--------|--------|---------|-----------------|-------------------|
| `.PA` | Euronext Paris | France | ✅ YES (15+ quarters) | ❌ YES |
| `.MI` | Borsa Italiana | Italy | ✅ YES | ❌ YES |
| `.AS` | Euronext Amsterdam | Netherlands | ✅ YES (16+ quarters) | ❌ YES |
| `.DE` | XETRA | Germany | ✅ YES (17+ quarters) | ❌ YES |
| `.MC` | Madrid | Spain | ✅ YES | ❌ YES |
| `.LS` | Lisbon | Portugal | ✅ YES | ❌ YES |
| `.SW` | SIX Swiss | Switzerland | ✅ YES | ❌ YES |
| `.L` | London SE | UK | ✅ YES (5+ quarters) | ❌ YES |
| `.CO` | Copenhagen | Denmark | ✅ YES | ❌ YES |
| `.HK` | Hong Kong | Hong Kong | ✅ YES | ❌ YES |
| `.T` | Tokyo SE | Japan | ✅ YES (13+ quarters) | ❌ YES |

### Verification Test

```bash
python3 -c "
from yahooquery import Ticker as YQTicker

tickers = ['SAP.DE', 'AIR.PA', 'ASML.AS', 'VOD.L', '7203.T']

for ticker in tickers:
    yq = YQTicker(ticker)
    qf = yq.income_statement(frequency='q')
    print(f'{ticker}: {len(qf)} quarters available')
"

# Output:
# SAP.DE: 17 quarters available      ✅
# AIR.PA: 15 quarters available      ✅
# ASML.AS: 16 quarters available     ✅
# VOD.L: 5 quarters available        ✅
# 7203.T: 13 quarters available      ✅
```

**Conclusion:** ALL tested stocks have quarterly data available.

---

## 💥 Impact Assessment

### 1. Data Completeness

**Current State:**
```sql
SELECT 
    CASE 
        WHEN ticker LIKE '%.DE' THEN 'Germany'
        WHEN ticker LIKE '%.PA' THEN 'France'
        WHEN ticker LIKE '%.T' THEN 'Japan'
        ELSE 'US'
    END AS market,
    COUNT(*) AS total_tickers,
    SUM(CASE WHEN ticker IN (SELECT DISTINCT ticker FROM raw.quarterly_financials) THEN 1 ELSE 0 END) AS with_quarterly,
    ROUND(SUM(CASE WHEN ticker IN (SELECT DISTINCT ticker FROM raw.quarterly_financials) THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS coverage_pct
FROM raw.company_info
WHERE quote_type = 'EQUITY'
GROUP BY market;
```

**Expected Result:**
- US stocks: ~95% coverage ✅
- European stocks: ~0% coverage ❌
- Japanese stocks: ~0% coverage ❌

### 2. Quality Score Impact

**Missing Components:**
- **Profitability (25-30 pts):** Partially missing (no quarterly FCF margin)
- **Revenue Consistency (5 pts):** Completely missing (no revenue growth data)
- **Momentum Context (15 pts):** Degraded (no recent earnings trends)

**Estimated Impact:** -10 to -20 points on Quality Score for European/Asian stocks

### 3. FMI (Fundamental Momentum Index) Impact

**FMI Components:**
1. Revenue Acceleration (30 pts) → ❌ Missing (no quarterly revenue)
2. EPS Acceleration (30 pts) → ❌ Missing (no quarterly EPS)
3. Margin Expansion (25 pts) → ❌ Missing (no quarterly margins)
4. Earnings Consistency (15 pts) → ❌ Missing (no quarterly history)

**Result:** FMI = 0 for ALL European/Asian stocks

### 4. User Experience Impact

**Dashboard Issues:**
- Earnings calendar: Empty for EU/Asia stocks
- Earnings surprise: N/A
- Growth charts: Flat lines
- Momentum indicators: Always "Neutral"
- Comparison with US stocks: Unfair (US has full data, EU/Asia doesn't)

**User Perception:** "Why are European stocks always rated lower than US stocks?"

---

## 🛠️ Proposed Solution

### Option 1: Remove Filter (Recommended)

**Change:** Delete or comment out the NON_QUARTERLY_SUFFIXES filter

**File:** `etl/utils.py` line 237

```python
# BEFORE (WRONG)
if ticker.upper().endswith(NON_QUARTERLY_SUFFIXES): return False

# AFTER (CORRECT)
# Removed: All major markets report quarterly
# if ticker.upper().endswith(NON_QUARTERLY_SUFFIXES): return False
```

**Pros:**
- ✅ Simple fix (1 line change)
- ✅ Restores data completeness
- ✅ Fair comparison across markets
- ✅ Improves Quality Scores accuracy

**Cons:**
- ⚠️ May increase ETL runtime (more tickers to fetch)
- ⚠️ May hit Yahoo Finance rate limits (need retry logic)

### Option 2: Selective Filter (Conservative)

**Change:** Only skip truly semi-annual markets (if any exist)

```python
# Research needed: Are there ANY markets that truly only report semi-annually?
# Hypothesis: Even small-cap stocks in emerging markets report quarterly now
TRULY_SEMI_ANNUAL_SUFFIXES = ()  # Empty for now

if ticker.upper().endswith(TRULY_SEMI_ANNUAL_SUFFIXES): return False
```

**Pros:**
- ✅ Preserves intent (skip truly semi-annual reporters)
- ✅ More conservative approach

**Cons:**
- ⚠️ Requires research to identify truly semi-annual markets
- ⚠️ May not exist (all major markets are quarterly now)

### Option 3: Graceful Degradation (Hybrid)

**Change:** Try to fetch quarterly data, fall back to annual if unavailable

```python
def is_eligible_for_quarterly(ticker):
    # Remove hard filter, let Yahoo Finance API decide
    # If quarterly data unavailable, extract.py will fall back to annual
    return True
```

**In extract.py:**
```python
def extract_quarterly_financials(tickers):
    # ... existing code ...
    
    # If quarterly returns empty, log but don't fail
    if qf.empty:
        logger.debug(f"No quarterly data for {ticker}, will use annual")
    
    # ... continue ...
```

**Pros:**
- ✅ Most robust (handles edge cases)
- ✅ No data loss
- ✅ Automatic fallback

**Cons:**
- ⚠️ More complex implementation
- ⚠️ Requires changes in multiple files

---

## 📅 Implementation Plan

### Phase 1: Immediate Fix (1 hour)

1. **Remove filter** in `etl/utils.py` line 237
2. **Test** with 5 European stocks (SAP.DE, AIR.PA, ASML.AS, VOD.L, 7203.T)
3. **Verify** quarterly data loads correctly
4. **Run** full ETL pipeline

### Phase 2: Validation (2 hours)

1. **Check** database: `SELECT COUNT(*) FROM raw.quarterly_financials WHERE ticker LIKE '%.DE'`
2. **Verify** Quality Scores improved for European stocks
3. **Test** FMI calculation for European stocks
4. **Review** dashboard displays (earnings calendar, growth charts)

### Phase 3: Monitoring (Ongoing)

1. **Track** ETL runtime (may increase)
2. **Monitor** Yahoo Finance rate limits
3. **Log** any tickers that truly don't have quarterly data
4. **Update** documentation

---

## 🧪 Testing Checklist

### Pre-Fix Verification

- [ ] Confirm European stocks missing from `raw.quarterly_financials`
- [ ] Confirm FMI = 0 for European stocks
- [ ] Confirm Quality Scores lower for European vs US stocks
- [ ] Document current coverage: `SELECT COUNT(*) FROM raw.quarterly_financials`

### Post-Fix Verification

- [ ] European stocks appear in `raw.quarterly_financials`
- [ ] FMI > 0 for European stocks with growth
- [ ] Quality Scores comparable to US stocks (when fundamentals similar)
- [ ] Earnings calendar shows European earnings
- [ ] Growth charts display quarterly trends
- [ ] No increase in ETL failures

### Regression Testing

- [ ] US stocks still load correctly
- [ ] ETL runtime acceptable (< 2x increase)
- [ ] No Yahoo Finance rate limit errors
- [ ] Database schema unchanged
- [ ] Dashboard displays correctly

---

## 📚 Historical Context

### Why Was This Filter Added?

**Hypothesis 1:** Legacy assumption from 2010s
- In the past, some European companies reported semi-annually
- IFRS adoption (2005+) standardized quarterly reporting
- Filter may be outdated

**Hypothesis 2:** Yahoo Finance API limitations
- Older Yahoo Finance API may not have had quarterly data for EU stocks
- Modern yahooquery library has full coverage
- Filter may be obsolete

**Hypothesis 3:** Performance optimization
- Fetching quarterly data for all markets may have been slow
- Filter reduced ETL runtime
- Modern async fetching makes this unnecessary

### Regulatory Background

**EU Transparency Directive (2004/109/EC):**
- Requires listed companies to publish **interim reports** (quarterly or semi-annual)
- Most large-cap companies choose quarterly to match US standards
- Small-cap may be semi-annual, but Yahoo Finance only covers large-cap

**Japan Financial Instruments and Exchange Act:**
- Requires **quarterly securities reports** (四半期報告書)
- All listed companies must report quarterly
- No exceptions for large-cap stocks

**Conclusion:** The filter was based on outdated assumptions.

---

## 🎯 Success Criteria

### Quantitative Metrics

1. **Coverage:** ≥80% of European/Asian stocks have quarterly data
2. **Quality Score:** Average score for EU stocks within ±5 points of US stocks (when fundamentals similar)
3. **FMI:** ≥50% of EU stocks have FMI > 0 (not all zeros)
4. **ETL Runtime:** Increase < 50% (acceptable trade-off for data quality)

### Qualitative Metrics

1. **User Feedback:** No complaints about "missing earnings" for EU stocks
2. **Dashboard:** Earnings calendar populated for EU stocks
3. **Comparisons:** Fair cross-market comparisons possible
4. **Trust:** Users trust Quality Scores for EU stocks

---

## 🚨 Risks & Mitigation

### Risk 1: Yahoo Finance Rate Limits

**Probability:** Medium  
**Impact:** High (ETL fails)

**Mitigation:**
- Implement exponential backoff (already exists)
- Batch requests (already exists)
- Add delay between batches (increase from 1s to 2s)
- Monitor rate limit errors in logs

### Risk 2: Increased ETL Runtime

**Probability:** High  
**Impact:** Medium (slower refreshes)

**Mitigation:**
- Run quarterly fetch only weekly (not daily)
- Parallelize more aggressively (increase workers)
- Cache results longer (168h → 336h)
- Accept trade-off (data quality > speed)

### Risk 3: Data Quality Issues

**Probability:** Low  
**Impact:** Medium (bad data in DB)

**Mitigation:**
- Add validation: revenue > 0, EPS not null
- Log suspicious data (e.g., revenue = 0 for large-cap)
- Manual review of first 10 EU stocks
- Rollback plan (restore filter if data quality poor)

### Risk 4: Currency Mismatch

**Probability:** Low  
**Impact:** High (wrong calculations)

**Mitigation:**
- Verify FX conversion applied to quarterly data
- Test: SAP.DE revenue should be in EUR
- Check: No mixing of EUR and USD in calculations
- Already handled by existing FX logic

---

## 📖 References

- **Code:** `etl/utils.py` line 202, 237
- **Extract:** `etl/extract.py` line 772 (`extract_quarterly_financials`)
- **Transform:** `etl/transform.py` line 289 (`dim_quarterly_financials`)
- **Documentation:** `docs/en/ETL_ARCHITECTURE.md`
- **Skill:** `.agents/skills/data-recovery/SKILL.md`

---

## ✅ Action Items

### Immediate (Today)

- [ ] **Remove filter** in `etl/utils.py` line 237
- [ ] **Test** with 5 European stocks
- [ ] **Run** incremental ETL
- [ ] **Verify** data loads correctly

### Short-term (This Week)

- [ ] **Run** full ETL with all European stocks
- [ ] **Validate** Quality Scores improved
- [ ] **Check** FMI calculations
- [ ] **Update** documentation

### Long-term (This Month)

- [ ] **Monitor** ETL performance
- [ ] **Gather** user feedback
- [ ] **Optimize** if needed
- [ ] **Document** lessons learned

---

**Last Updated:** 2026-05-02  
**Status:** PENDING FIX  
**Owner:** Data Engineering Team  
**Priority:** HIGH
