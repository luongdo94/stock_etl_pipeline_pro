# 🏗️ ETL Pipeline Architecture

This document describes the technical architecture of the Stock ETL Pipeline, from raw data collection to the creation of "Analytics Ready" datasets in the DuckDB data warehouse.

## 1. Design Philosophy
The system is built on three core pillars:
1.  **Euro-First (Normalization):** All financial metrics (Price, Revenue, Market Cap) are converted to Euro at the source to ensure absolute comparability across global tickers.
2.  **Zero Down-time (Continuous Availability):** Utilizes a Shadow DB mechanism and Atomic Swap, ensuring the Dashboard remains functional even during intensive data loads.
3.  **Data Layering:** Adopts a dbt-style modeling approach (Raw -> Staging -> Intermediate -> Marts) for maximum transparency and maintainability.

---

## 2. The 5-Step Pipeline Lifecycle

The system operates via the `run_pipeline()` function in `etl/pipeline.py` through five rigorous stages:

### Step 0: Shadow DB Prep
The system creates a "shadow" copy of the production database. All new write operations are performed on this copy to prevent any impact on end-users currently accessing the Dashboard.

### Step 1: Extract & Currency Normalization
- **Sources:** Yahoo Finance (yfinance) & Google News RSS.
- **Modes:** 
    - `INCREMENTAL`: Only downloads data since the last watermark (fast, ~3-5s).
    - `FULL REFRESH`: Re-downloads the entire historical window (default 5 years).
- **Normalize:** Automatically fetches live FX rates (e.g., `USDEUR=X`) to multiply into prices and fundamentals during ingestion.

### Step 2: Validate
Performs initial integrity checks on the extracted data (no negative prices, no null critical columns). If validation fails, the process is aborted (Fail-fast).

### Step 3: Load
Data is loaded into the `raw` schema within DuckDB. An `UPSERT` strategy is used in incremental mode to prevent duplicate records.

### Step 4: Transform (Multi-layer Processing)
This is the data factory where SQL transformations happen within DuckDB:
- **Staging Layer:** Cleaning, rounding, and record flagging.
- **Intermediate Layer:** Calculation of technical indicators (RSI, Moving Averages, Z-Score).
- **Marts Layer:** Final business-facing Fact (Daily returns) and Dimension (Company info) tables.

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
