# Documentation Update — May 3, 2026

## Summary

Updated all ETL Architecture documentation files (English, Vietnamese, German) to reflect the new **TradingView Auto-Discovery System** and **Garbage Collection** features implemented in May 2026.

---

## Changes Since Commit `74414bd`

### 1. TradingView Auto-Discovery System (NEW)

**Commits:**
- `4eeb491`: Initial TradingView integration with institutional-grade filters
- `029ee55`: Added TradingView widgets (ticker tape, heatmap, economic calendar)
- `b0dfa9c`: Added garbage collection and symbol mapping utility
- `442f253`: Enhanced UI with quantitative filters discovery

**Key Features:**

#### 5 Institutional-Grade Filters
1. **Value Stocks**: P/E < 15, P/B < 1.5, Dividend Yield > 2%
2. **GARP (Growth at Reasonable Price)**: EPS Growth > 15%, Revenue Growth > 10%, P/E < 25
3. **Breakout Momentum**: Price > MA50 > MA200, RSI 60-75, Volume > 1M
4. **Quality Compounders**: ROIC > 15%, ROE > 20%, Operating Margin > 15%, D/E < 0.5
5. **High-Yield Dividend**: Dividend Yield > 4%, Payout Ratio < 60%, Revenue Growth > 0%

#### Global Market Coverage
- Scans 14 markets: US, Vietnam, UK, Germany, France, Japan, Hong Kong, China, Australia, Canada, India, Brazil, Taiwan, Korea
- Top 20 stocks per filter = ~100 additional tickers dynamically discovered daily
- Total coverage expanded from ~600 to 700+ tickers

#### Smart Deduplication
- Normalized name matching to prevent duplicate companies
- Filters out cross-listings, preferred shares, depositary receipts
- Ticker validation (no spaces, slashes, invalid characters)

#### Exchange Mapping
Automatic conversion of TradingView symbols to Yahoo Finance tickers:
```
XETR:SIE    → SIE.DE     (Frankfurt)
LSE:BP      → BP.L       (London)
HOSE:VNM    → VNM.VN     (Vietnam)
TSE:7203    → 7203.T     (Tokyo)
NASDAQ:AAPL → AAPL       (US)
```

#### Integration Points
- **`etl/extract.py`**: `fetch_dynamic_tv_tickers()` function with 5 preset filters
- **`etl/extract.py`**: `get_combined_tickers()` merges base config with dynamic tickers
- **`etl/pipeline.py`**: Passes combined `TICKERS` to smart recovery and transform stages
- **`etl/transform.py`**: Staging views filter to only include active tickers
- **`app.py`**: Auto-Discovery section in Overview tab, TradingView Filters tab in Screener

---

### 2. Garbage Collection System (NEW)

**Purpose:** Maintain data warehouse hygiene by automatically removing stale auto-discovered tickers.

**Logic:**
- Auto-discovered TradingView tickers that haven't been updated in **7+ days** are automatically purged
- Base tickers from `config/tickers.yaml` are **strictly protected** and never removed
- Rationale: TradingView filters return "Top 20" rankings that change daily; stocks falling out of rankings become stale

**Implementation:**
- **Function:** `cleanup_stale_tv_tickers()` in `etl/load.py`
- **Trigger:** Runs in Step 4.8 (after Transform, before Atomic Swap)
- **Scope:** Deletes from 9 raw tables:
  - `stock_prices`
  - `company_info`
  - `historical_financials`
  - `quarterly_financials`
  - `cashflows`
  - `earnings_calendar`
  - `earnings_surprise`
  - `forward_estimates`
  - `hist_fcf`
  - `hist_fcf_quarterly`

**Safety Measures:**
- Only runs after successful transform
- Requires base ticker list to be loaded (fails safe if config unavailable)
- Uses `_extracted_at` timestamp to identify stale records
- Executes in <1 second (simple DELETE with date filter)

**Lifecycle:**
- **Day 1-7:** Ticker actively tracked, data extracted daily, visible in dashboard
- **Day 8+:** Ticker falls out of Top 20 → `_extracted_at` becomes stale → Garbage collection removes → Disappears from dashboard
- **Re-discovery:** If ticker re-enters Top 20, automatically re-added

---

### 3. Active Ticker Filtering in Transform Layer (NEW)

**Change:** Staging views now filter out "dead" or stale tickers that are no longer in the active ticker pool.

**Implementation:**
- **`etl/transform.py`**: `run_transforms()` now accepts `active_tickers` parameter
- **`etl/transform.py`**: `_create_staging()` creates temp table `active_tickers` and adds filter clause:
  ```sql
  WHERE ... AND ticker IN (SELECT ticker FROM active_tickers)
  ```
- **Purpose:** Prevents zombie data from polluting marts after garbage collection

**Affected Views:**
- `staging.stg_stock_prices`
- `staging.stg_company_info`
- `staging.stg_historical_financials`
- All downstream intermediate and marts tables

---

### 4. Dashboard UI Enhancements

**Auto-Discovery Section (Overview Tab):**
- Shows count of newly discovered stocks
- Lists tickers with discovery source tags (e.g., `TV_VALUE_STOCKS`)
- Distinguishes from base configuration stocks

**TradingView Filters Tab (Screener):**
- Real-time filter results grouped by source
- Stock cards with sector/region metadata
- Status badges (Existing DB vs New Discovery)
- Summary metrics (total signals, active filters, top performer)
- Expandable sections per filter with visual indicators

**TradingView Widgets:**
- Ticker tape (global markets)
- Market heatmap (S&P 500 sectors)
- Economic calendar
- Advanced charts with pre-loaded indicators

---

## Documentation Files Updated

### English (Primary)
- ✅ `docs/en/ETL_ARCHITECTURE.md`
  - Updated Step 1 (Extract) with TradingView Auto-Discovery details
  - Updated Step 4 (Transform) with Active Ticker Filtering
  - Added Step 4.8 (Garbage Collection)
  - Added Section 5: TradingView Auto-Discovery System (comprehensive)

### Vietnamese (Translation)
- ✅ `docs/vi/ETL_ARCHITECTURE.md`
  - Translated all English updates to Vietnamese
  - Maintained technical accuracy and terminology consistency

### German (Translation)
- ✅ `docs/de/ETL_ARCHITECTURE.md`
  - Translated all English updates to German
  - Maintained technical accuracy and terminology consistency

---

## Technical Details

### Performance Considerations
- **API Rate Limits:** 5 filters × 20 stocks = 100 API calls per run (well within limits)
- **Deduplication Overhead:** O(n) normalized name comparison (negligible for <1000 tickers)
- **Storage Impact:** ~100 additional tickers × 9 tables = minimal (DuckDB handles efficiently)
- **Garbage Collection:** Runs in <1 second (simple DELETE with date filter)

### Code Changes Summary
- **`etl/extract.py`**: +166 lines (TradingView integration)
- **`etl/load.py`**: +54 lines (garbage collection)
- **`etl/pipeline.py`**: +10 lines (integration)
- **`etl/transform.py`**: +30 lines (active ticker filtering)
- **`app.py`**: +291 lines (UI enhancements)

---

## Migration Notes

**No Breaking Changes:**
- Existing `config/tickers.yaml` continues to work as before
- Base tickers are protected and never removed
- Auto-discovery is additive (expands coverage, doesn't replace)
- Garbage collection only affects auto-discovered tickers

**Backward Compatibility:**
- If TradingView API fails, system falls back to base configuration
- All existing ETL logic remains unchanged
- Dashboard gracefully handles missing auto-discovery data

---

## Future Enhancements

**Potential Improvements:**
1. **Configurable GC Threshold:** Allow users to adjust 7-day stale threshold
2. **Filter Customization:** Enable users to define custom TradingView filters
3. **Discovery History:** Track which tickers were discovered and when
4. **Performance Metrics:** Monitor filter effectiveness (hit rate, false positives)
5. **Multi-Region Optimization:** Prioritize filters by user's primary market

---

## Verification

To verify the updates are working:

```bash
# 1. Check auto-discovered tickers
python show_tv_discovered_stocks.py

# 2. Run ETL pipeline
python run.py --fast --sync

# 3. Check garbage collection logs
grep "Garbage Collection" logs/etl_*.log

# 4. Verify dashboard
streamlit run app.py
# Navigate to: Overview > Auto-Discovery section
# Navigate to: Screener > TradingView Filters tab
```

---

**Documentation Version:** 2.1.0  
**Last Updated:** May 3, 2026  
**Author:** System Documentation Bot  
**Reviewed By:** ETL Pipeline Team
