# 🏗️ ETL Pipeline Architecture

This document describes the technical architecture of the Stock ETL Pipeline, from raw data collection to the creation of "Analytics Ready" datasets in the DuckDB data warehouse.

## 1. Design Philosophy
The system is built on three core pillars:
1.  **Euro-First (Normalization):** All financial metrics (Price, Revenue, Market Cap) are converted to Euro at the source to ensure absolute comparability across global tickers.
2.  **Zero Down-time (Continuous Availability):** Utilizes a Shadow DB mechanism and Atomic Swap, ensuring the Dashboard remains functional even during intensive data loads.
3.  **Data Layering:** Adopts a dbt-style modeling approach (Raw -> Staging -> Intermediate -> Marts) for maximum transparency and maintainability.

---

## 2. The 5-Step Pipeline Lifecycle

The system operates via the `run_pipeline()` function in `etl/pipeline.py` through five rigorous stages, plus an automated garbage collection step:

### Step 0: Shadow DB Prep
The system creates a "shadow" copy of the production database. All new write operations are performed on this copy to prevent any impact on end-users currently accessing the Dashboard.

### Step 1: Extract & Currency Normalization
- **Sources:** Utilizes a dual-source strategy for maximum resilience:
    - `yahooquery`: Primary source for sensitive Financials, Cashflows, and Earnings to bypass blocks.
    - `yfinance`: Primary source for high-velocity Price and FX data.
- **TradingView Auto-Discovery (NEW May 2026):** Dynamic ticker discovery system that automatically expands coverage beyond static configuration:
    - **5 Institutional-Grade Filters:** Value Stocks, GARP (Growth at Reasonable Price), Breakout Momentum, Quality Compounders, High-Yield Dividend.
    - **Global Market Scanning:** Scans 14 global markets (US, Europe, Asia-Pacific) via TradingView Scanner API.
    - **Smart Deduplication:** Prevents duplicate companies (cross-listings, preferred shares, depositary receipts) using normalized name matching.
    - **Exchange Mapping:** Automatically maps TradingView symbols to Yahoo Finance tickers (e.g., `XETR:SIE` → `SIE.DE`).
    - **Top 20 Per Filter:** Fetches top 20 stocks per filter, dynamically refreshed daily.
    - **Metadata Enrichment:** Captures sector, region, and discovery source for each auto-discovered ticker.
- **Multi-tier Smart Refresh Strategy:** To maximize speed and avoid API throttling, the system groups data into three update frequencies:
    - **Tier 1 (High Velocity - 24h):** Price data and technical indicators. Always updated daily.
    - **Tier 2 (Tactical - 7 Days):** Quarterly financials, Free Cash Flow (FCF), and Earnings calendars.
    - **Tier 3 (Strategic - 30 Days):** Company metadata (Sectors, Industries) and Historical Annual financials.
- **Global Market Coverage:** All equity stocks are processed equally regardless of geographic location. The system extracts quarterly data for US, European, and Asian markets without discrimination (fixed May 2026 — previously EU/Asia stocks were incorrectly filtered).
- **Normalize:** Automatically fetches live FX rates (e.g., `USDEUR=X`) to normalize all values to Euro at ingestion.

### Step 2: Validate
Performs initial integrity checks on the extracted data (no negative prices, no null critical columns). If validation fails, the process is aborted (Fail-fast).

### Step 3: Load
Data is loaded into the `raw` schema within DuckDB. An `UPSERT` strategy is used in incremental mode to prevent duplicate records.

### Step 4: Transform (Multi-layer Processing)
This is the data factory where SQL transformations happen within DuckDB:
- **Active Ticker Filtering (NEW May 2026):** Staging layer now filters out "dead" or stale tickers that are no longer in the active ticker pool, preventing zombie data from polluting marts.
- **Staging Layer:** Cleaning, rounding, and record flagging.
- **Intermediate Layer:** Calculation of technical indicators (RSI, Moving Averages, Z-Score).
- **Marts Layer:** Final business-facing Fact (Daily returns) and Dimension (Company info) tables.

### Step 4.8: Garbage Collection (NEW May 2026)
Automated cleanup system that maintains data warehouse hygiene:
- **Stale Ticker Removal:** Auto-discovered TradingView tickers that haven't been updated in 7+ days are automatically purged from all raw tables.
- **Base Ticker Protection:** Tickers defined in `config/tickers.yaml` are strictly protected and never removed.
- **Rationale:** TradingView filters return "Top 20" rankings that change daily. Stocks that fall out of rankings become stale and should be removed to prevent database bloat.
- **Scope:** Deletes from 9 raw tables: `stock_prices`, `company_info`, `historical_financials`, `quarterly_financials`, `cashflows`, `earnings_calendar`, `earnings_surprise`, `forward_estimates`, `hist_fcf`, `hist_fcf_quarterly`.
- **Safety:** Only runs after successful transform, never on base configuration tickers.

### Step 5: Atomic Swap
Once the data is ready in the Shadow DB, the system performs a physical file swap on the disk. This happens in milliseconds, ensuring the Dashboard always displays the latest data without connection errors.

---

## 3. Warehouse Schema Structure

| Schema | Role | Typical Tables |
| :--- | :--- | :--- |
| **raw** | Pre-transformation raw ingestion. | `stock_prices`, `company_info` |
| **staging** | Cleaned and filtered source data. | `stg_stock_prices`, `stg_cashflows` |
| **intermediate** | Metric calculation (Business Logic). | `int_stock_metrics` (RSI, MA200...) |
| **marts** | Final analytical tables (BI Ready). | `dim_companies`, `fct_daily_returns` |

---

## 4. Quality Control (Data Quality - DQ)

At the end of every ETL cycle, the system runs an automated suite of tests:
- **Critical Tests:** Unique key checks and Not Null constraints. Failure results in a pipeline abort.
- **Soft Tests:** Warnings about gaps in fundamental data. Results are logged to `marts.dq_warnings` for Dashboard visibility.

---
> [!TIP]
> You can monitor this lifecycle via the console logs. Each stage is timed for performance tuning and transparency.

---

## 5. TradingView Auto-Discovery System (NEW May 2026)

The pipeline now features an intelligent dynamic ticker discovery system that automatically expands coverage beyond static configuration.

### 5.1. Architecture Overview

**Traditional Approach (Pre-May 2026):**
- Static ticker list in `config/tickers.yaml`
- Manual updates required to add new stocks
- Limited to ~600 pre-configured tickers

**Auto-Discovery Approach (May 2026+):**
- Dynamic ticker discovery via TradingView Scanner API
- Automatic expansion to 700+ tickers globally
- Daily refresh of top-performing stocks per filter

### 5.2. Five Institutional-Grade Filters

| Filter | Criteria | Target Profile |
|---|---|---|
| **Value Stocks** | P/E < 15, P/B < 1.5, Div Yield > 2% | Undervalued dividend payers |
| **GARP** | EPS Growth > 15%, Revenue Growth > 10%, P/E < 25 | Growth at reasonable price |
| **Breakout Momentum** | Price > MA50 > MA200, RSI 60-75, Volume > 1M | Technical breakouts |
| **Quality Compounders** | ROIC > 15%, ROE > 20%, Op Margin > 15%, D/E < 0.5 | High-quality businesses |
| **High-Yield Dividend** | Div Yield > 4%, Payout < 60%, Revenue Growth > 0% | Sustainable income |

### 5.3. Global Market Coverage

Scans 14 markets: `america`, `vietnam`, `uk`, `germany`, `france`, `japan`, `hongkong`, `china`, `australia`, `canada`, `india`, `brazil`, `taiwan`, `korea`

### 5.4. Exchange Mapping Logic

Automatically converts TradingView symbols to Yahoo Finance tickers:

```python
XETR:SIE    → SIE.DE     (Frankfurt)
LSE:BP      → BP.L       (London)
HOSE:VNM    → VNM.VN     (Vietnam)
TSE:7203    → 7203.T     (Tokyo)
NASDAQ:AAPL → AAPL       (US)
```

### 5.5. Smart Deduplication

Prevents duplicate companies using normalized name matching:

1. **Normalize company names:** Remove suffixes (Inc, Corp, Ltd), special characters, extra spaces
2. **Cross-listing detection:** Skip if normalized name matches existing ticker
3. **Preferred share filtering:** Exclude tickers with "Preferred", "PFD", "Depositary Share", "Warrant" in name
4. **Ticker validation:** Skip tickers with spaces, slashes, or invalid characters

**Example:**
```
Base config: AAPL (Apple Inc.)
TradingView returns: AAPL (Apple Inc.), AAPL34 (Apple BDR Brazil)
Result: Only AAPL kept (AAPL34 filtered as duplicate)
```

### 5.6. Integration with ETL Pipeline

**Function Flow:**
```python
# etl/extract.py
base_tickers = load_tickers_config()           # Load config/tickers.yaml
dynamic_tickers = fetch_dynamic_tv_tickers()   # Fetch from TradingView
TICKERS = {**dynamic_tickers, **base_tickers} # Merge (base takes precedence)
```

**Pipeline Integration:**
- `etl/pipeline.py`: Passes combined `TICKERS` to smart recovery and transform stages
- `etl/transform.py`: Filters staging views to only include active tickers
- `etl/load.py`: Garbage collection removes stale auto-discovered tickers

### 5.7. Lifecycle Management

**Day 1-7:** Auto-discovered ticker actively tracked
- Appears in TradingView filter results
- Data extracted and loaded daily
- Visible in dashboard with "TV_" discovery source tag

**Day 8+:** Ticker falls out of Top 20 rankings
- No longer returned by TradingView API
- `_extracted_at` timestamp becomes stale (>7 days old)
- Garbage collection removes from all raw tables
- Disappears from dashboard

**Re-discovery:** If ticker re-enters Top 20, it's automatically re-added

### 5.8. Dashboard Integration

**Auto-Discovery Section (Overview Tab):**
- Shows count of newly discovered stocks
- Lists tickers with discovery source tags
- Distinguishes from base configuration stocks

**TradingView Filters Tab (Screener):**
- Real-time filter results grouped by source
- Stock cards with sector/region metadata
- Status badges (Existing DB vs New Discovery)
- Summary metrics (total signals, active filters, top performer)

### 5.9. Performance Considerations

- **API Rate Limits:** 5 filters × 20 stocks = 100 API calls per run (well within limits)
- **Deduplication Overhead:** O(n) normalized name comparison (negligible for <1000 tickers)
- **Storage Impact:** ~100 additional tickers × 9 tables = minimal (DuckDB handles efficiently)
- **Garbage Collection:** Runs in <1 second (simple DELETE with date filter)

---
