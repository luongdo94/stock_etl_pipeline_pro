# 🚀 Honest Quant Intelligence Platform (v10.5)

**A professional-grade, institutional-level stock analytics, diagnostic, and backtesting engine.**

Honest Quant is not just a stock screener. It is a comprehensive **Hybrid Intelligence Platform** that merges traditional quantitative finance algorithms with modern Deep Learning (AI) forecasting. Designed for advanced traders and portfolio managers, the platform transforms raw market data into institutional-grade, actionable execution plans.

---

## 🏗️ 1. Architecture & ETL Pipeline

The backbone of Honest Quant is a robust, production-ready Data Engineering pipeline running on a modern data stack.

### Data Ingestion & Extraction (`etl/extract.py`)
- **Data Providers**: Utilizes a dual-source strategy combining `yahooquery` (for sensitive Financials/Earnings) and `yfinance` (for high-velocity Price/FX data).
- **Concurrency**: Integrates `ThreadPoolExecutor` and `yahooquery` batch modes to handle parallel data fetching across 600+ tickers.
- **Resilience (Multi-Pass)**: Implements a two-pass surgical strategy:
    - **Pass 1**: Batch concurrent extraction for speed.
    - **Pass 2 (Surgical)**: Sequential, throttled retry for failed tickers with randomized jitter to bypass API blocks and reach 100% coverage.
- **Fundamental Recovery Engine**: A proprietary logic designed to bypass 'Data Gaps' in free APIs. If a stock's summary metrics (ROE/FCF) are missing, the pipeline automatically extracts raw **Income Statements** and **Balance Sheets** to manually reconstruct accurate TTM (Trailing Twelve Months) metrics.

### Intelligent Loading Strategy
- **Incremental Load (Watermarking)**: The system automatically detects the last available data point for each ticker. In daily runs, it strictly fetches only the missing "gap" (incremental window), reducing bandwidth consumption and avoiding IP blocks.
- **New Ticker Bootstrapping**: When a new ticker is added to `config/tickers.yaml`, the ETL engine automatically identifies its absence in the warehouse and triggers a **Full 5-Year History Download** specifically for that ticker, while keeping all other tickers on an incremental path.
- **Multi-tier Smart Refresh**: To maximize speed and avoid API throttling, the system implements a tiered caching strategy:
    - **Tier 1 (Daily)**: Stock Prices & Technicals. Always updated.
    - **Tier 2 (Weekly - 168h)**: Quarterly Financials, Cashflow, and Earnings.
    - **Tier 3 (Monthly - 720h)**: Company Metadata (Sector, Industry), Historical Annual Financials.
- **Coverage Guard**: Regardless of the timers, a deep refresh is automatically triggered if total warehouse coverage drops below **95%** (Metadata) or **90%** (Quarterly data).

### Data Transformation & Warehousing (`etl/transform.py` & `etl/load.py`)
- **Storage Layer**: Uses **DuckDB** (`stock_dw.duckdb`) as an embedded, highly optimized columnar database. This allows the Streamlit dashboard to execute complex aggregations with millisecond latency.
- **Star-Schema Modeling (dbt-style)**:
  - **`raw` schema**: Stores untyped, historical JSON dumps.
  - **`marts.dim_companies`**: The master static dimension table containing aggregated fundamental markers (Market Cap, Sub-Sector, Beta, Short Interest).
  - **`marts.fct_daily_returns`**: The time-series fact table computing daily logarithmic returns, Technical Indicators ($MA_{20}, MA_{50}, MA_{200}$, RSI), and Rolling Volatility arrays.
  - **`marts.dim_quarterly_financials` & `dim_annual_financials`**: Financial statements optimized for longitudinal queries.
- **Automation**: Fully containerized using **Apache Airflow** (via Docker). DAGs are scheduled to run daily exactly 2 hours before market open, ensuring the data is strictly point-in-time.

---

## 🧠 2. AI Predictive Suite

Honest Quant doesn't just analyze the past; it attempts to project the future utilizing cutting-edge Machine Learning.

### Long Short-Term Memory (LSTM) Networks
- **Architecture**: A custom PyTorch-based neural network trained on multivariate historical sequences. It captures non-linear, long-term dependencies in stock volatility.
- **30-Day Forecasts**: Outputs a deterministic price trajectory for the next 30 trading days based on momentum curves and historical volatility clusters.

### Stochastic Risk Modeling (Monte Carlo)
- **Path Simulation**: Generates 500+ random-walk price paths using Geometric Brownian Motion (GBM).
- **Risk Assessment**: Outputs the 5th and 95th percentile confidence intervals (Value-at-Risk parameters) to answer: *"What is the absolute worst-case scenario for this stock over the next 3 months?"*

### Sentiment-Driven Drift
- Integrates Natural Language Processing (NLP) over recent financial news to adjust the drift parameter of the LSTM model. If news sentiment is heavily negative, the AI's standard output is structurally downgraded.

---

## 🧮 3. Canonical Scoring Engine (Quant Diagnostics)

Located in `etl/utils.py`, the core proprietary logic quantifies stocks based on complex mathematics. 

### 🟢 Quality Score (0-100)
A rigorous safety and fundamental moat check, designed to find "Wonderful companies at a fair price":
1. **Valuation**: Penalizes high Forward P/E multiples but rewards low PEG (Price/Earnings-to-Growth) ratios to adjust for intrinsic value.
2. **Growth**: Evaluates the 4-year Compound Annual Growth Rate (CAGR) of Revenue & EPS.
3. **Profitability**: Strictly requires positive Free Cash Flow, high Operating Margins, and exceptional Return on Equity (ROE).
4. **Safety & Risk Mitigation**: 
   - Punishes excessive Debt-to-Equity ratios.
   - Adjusts for Beta (Systematic Risk).
   - Computes **Net Payout Yield** (Dividends + Share Buybacks) to ensure companies returning capital directly to shareholders (like Apple or Meta) are highly ranked.
   - **Linear Interpolation**: Eliminates "Cliff Effects" (e.g., P/E 19.9 vs 20.1) by scaling points mathematically rather than using hard cutoffs.

### 🚀 Fundamental Momentum Index (FMI) (0-100)
A CANSLIM-style growth accelerator index. Because free APIs often suffer from delayed/sparse data, FMI utilizes a hyper-dynamic *Live-Computed Engine*:
- **Earnings & Revenue Acceleration**: Compares the latest available quarter against the baseline of the most recent FULL year's growth. 
- **QoQ Fallback Engine**: If Year-over-Year (YoY) data is missing due to API limits, it automatically falls back to extrapolating Quarter-over-Quarter (QoQ) metrics.
- **Margin Expansion**: Mathematically checks if $EPS\_Growth > Revenue\_Growth$, indicating rising operating leverage.
- Outputs actionable labels: `Accelerating`, `Slowing`, `Turning Around`, `Bottoming`.

### 📉 Z-Score Mean Reversion & Deep Value
- Continuously calculates $Z = (Price - MA_{60}) / STD_{60}$.
- Easily identifies massive dislocations from intrinsic value, spotting deep panics ($Z < -2$) and severe overbought euphoria ($Z > +2$).

---

## 💻 4. The Tactical Dashboard (`app.py`)

A high-density **Streamlit** control room, heavily styled with custom CSS to provide a dark-mode "God-Mode" terminal experience.

### Tab 1: Global Macro Overview
- **Macro Pulse**: Real-time trackers for Volatility ($VIX), USD Strength ($DXY), and the S&P500 ($SPY).
- **Market Breadth**: Displays the percentage of stocks successfully trading above their 200-day moving average to declare whether the market is structurally `RISK-ON` or `RISK-OFF`.
- **Top Movers & Heatmaps**: Visualizes Sector-wide Capital Rotation.

### Tab 2: Single Stock Deep Dive
- A meticulously designed full-page tear sheet.
- **Radar Charts**: Powered by Plotly, breaks down the exact anatomy of the Quality Score.
- **Progress Panels**: Neon-colored metric bars displaying the real-time FMI Acceleration breakdown.
- Evaluates Short Interest vulnerability and Institutional accumulation flow.

### Tab 3: AI Market Scanner
- A dynamic, multi-condition screener. Filter thousands of stocks in milliseconds using DuckDB's backend.
- **23 Curated Strategy Presets** covering momentum, value, quality, and risk scenarios (optimized to eliminate redundancy):

#### Opportunity Strategies (15 presets)
- `🏆 Institutional Pulse` - Quality ≥70 (ELITE) + Bullish trend
- `🚀 Buy on Dip` - Bullish trend + RSI cooling (<40)
- `🚀 Bullish Momentum` - Strong uptrend + RSI >50 confirmation
- `📈 Both Accelerating` - EPS + Revenue both growing QoQ >+10% for 2 quarters
- `🌱 GARP` - Growth at Reasonable Price (PEG <1.5 + Quality >55)
- `💰 High Quality Dividend` - Yield >2.5% + Quality >65 + Bullish
- `🔥 Short Squeeze Watch` - High short interest + oversold + bullish reversal
- `🎯 Smart Money Accumulation` - Institutional buying + Quality ≥55 + RSI <50
- `🔄 Mean Reversion Elite` - High Quality (≥65) + Oversold (RSI<35) + Below Mean (Z<-1.0)
- `⚡ Strong Breakout` - Price >MA200 by 5%+ with healthy RSI (50-70)
- `💎 Contrarian Value` - Quality ≥60 in downtrend + cheap valuation (Z<-1.5, PEG<1.2)
- `🏰 Defensive Moat` - Low debt (<2x) + High ROE (>15%) + Dividend (>2%) + Quality ≥60
- `🌊 Oversold Reversal Setup` - Extreme oversold (RSI<30) + Smart Money buying + Quality ≥50
- `📊 Balanced Growth` - Quality 55-75 + PE 15-30x + ROE >12% + Bullish

#### Risk & Warning Strategies (8 presets)
- `⚠️ Earnings Deterioration` - EPS + Revenue declining QoQ >-10% for 2 quarters
- `⚠️ Structural Caution` - Quality <38 + Bearish trend
- `📉 Negative Momentum` - MA20 < MA50 bearish alignment
- `🔥 Overbought Alert` - RSI >65, elevated pullback risk
- `🎈 Valuation Exhaustion` - Z-Score >+2.0, likely overvalued
- `⚔️ Exit on Strength` - Bearish trend + short-term rally (RSI >60)
- `💔 Multi-Indicator Breakdown` - Bearish + RSI <50, falling knife
- `🚨 Distribution Warning` - Institutions selling + Overbought (RSI>60) + Weak Quality (<55)

### Tab 4: Strategy Backtester V2
An institutional-grade simulation engine that allows you to directly trade your fundamental setups via Technical triggers.
- **Zero Lookahead Bias**: Matrix operations (`np.roll`) ensure that if a signal triggers on day $T$, the execution and P&L strictly calculates on day $T+1$.
- **4 Integrated Strategies**:
  1. *Trend Following (Golden Cross / Death Cross)*
  2. *RSI Mean Reversion (Overbought / Oversold)*
  3. *Buy on Dip (RSI Dip within an MA50 Uptrend)*
  4. *Z-Score Reversion (Deep Value Catching)*
- **Risk Management**: Incorporates Capital constraints, Trade Slippage (Tx costs), Hard Stop-Loss (%), and Take-Profit (%).
- **Interactive Outputs**: Visualizes Equity Curves vs traditional Buy&Hold, alongside a granular, row-by-row Trade Log explaining exactly *why* a trade was executed.

---

## 🛡️ 5. Observability & System Integrity

Honest Quant is built for production reliability, incorporating an enterprise-grade observability layer to ensure data fidelity.

### Persistent Audit Layer (`marts.etl_audit`)
Every ETL execution is cryptographically logged in the warehouse. The system tracks:
- **Run Status**: `SUCCESS` or `FAILED` indicators.
- **Performance**: Start/End timestamps and total processing duration.
- **Intake volume**: Exact count of rows processed in each run.

### Data Quality (DQ) Guardrails (`marts.dq_warnings`)
The pipeline executes automated integrity checks post-transformation to detect anomalies before they reach the dashboard:
- **Schema Validation**: Ensures all critical columns exist.
- **Null Checks**: Detects missing prices or financials.
- **Volatility Thresholds**: Flags suspicious price jumps (e.g., >100% in a single day).

### Infrastructure Engine (Sidebar Health)
The Streamlit dashboard features a high-fidelity **Infrastructure Engine** indicator in the sidebar, providing real-time visibility into the last sync status and data integrity without cluttering the analytical views.

### Automated Testing Suite (`tests/`)
A comprehensive test suite powered by `pytest` ensures the pipeline's logic remains sound during refactors:
- **`test_config.py`**: Validates ticker lists and environment variables.
- **`test_transform.py`**: Verifies complex math for RSI, Z-Score, and FMI logic using mocked data.
- **`test_load.py`**: Ensures DuckDB persistence and schema alignment.

---

## 🛠️ 6. Installation & Deployment

### Global Requirements
- Python 3.9+ 
- Docker & Docker Compose (for Airflow Orchestration)

### Step 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/luongdo94/stock_etl_pipeline.git
cd stock_etl_pipeline

# Create and isolate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install heavy scientific & ML dependencies
pip install -r requirements.txt
```

### Step 2. Run ETL Pipeline
You have two main ways to run the pipeline depending on your needs:

**Daily Update (Fast & Safe)**
Updates only stock prices and runs technical indicators. Recommended for weekdays.
```bash
python run.py --fast --sync
```

**Weekly/Full Update (Deep Dive)**
Refreshes everything including Financials, Cashflows, and Earnings (if last update > 7 days).
```bash
python run.py --sync
```

**Force Rebuild**
Ignore all caches and download 5 years of full history for everything.
```bash
python run.py --full
```

### Step 3. Spin Up The Control Room
Run the Streamlit frontend locally:
```bash
./start_dashboard.sh
# Alternatively: streamlit run app.py
```

### Step 4. Cloud Deployment (Optional)
To run the dashboard on the web (e.g., Streamlit Cloud) without pushing the database to Git:
1. Set up a **Supabase** project and enable **S3-compatible Storage**.
2. Create a private bucket named `warehouse`.
3. Set the following environment variables in your deployment platform:
    - `SUPABASE_REMOTE_MODE = "true"`
    - `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`
    - `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_ENDPOINT`
4. The dashboard will now stream data directly from the cloud via HTTP Parquet querying.

### Step 5. Continuous Integration (Airflow)
To set up completely automated daily ETL runs so your data is always fresh:
```bash
docker-compose up -d
# Access the Airflow UI at http://localhost:8080 to trigger the master DAG.
```

---

## 📂 7. Directory Structure
```text
stock_etl_pipeline/
│
├── etl/                   # Data extraction, normalization, and Quant Engine functions
│   ├── extract.py         # ThreadPool API scrapers (yfinance)
│   ├── transform.py       # DuckDB Star Schema generation 
│   ├── load.py            # Local warehouse persistor
│   └── utils.py           # Core mathematics, Z-Scores, FMI, Quality Score algorithms
│
├── warehouse/             # Local database location
│   └── stock_dw.duckdb    # Compiled Analytics Database
│
├── airflow/               # Orchestration logic
│   ├── dags/              # Master DAGs
│   └── docker-compose.yml # Container definitions
│
├── tests/                 # Automated pytest suite
├── app.py                 # Main Streamlit Tactical Dashboard UI
├── run.py                 # Pipeline trigger entry point
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

---
*Architected and Engineered by GIA LUONG DO.*


---

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### 1. **Dashboard Loading Slowly**

**Symptoms:**
- Dashboard takes >30 seconds to load
- Browser becomes unresponsive
- High memory usage

**Solutions:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Or use the in-app refresh button
# Click "🔄 Refresh Data" in the sidebar
```

**Prevention:**
- Memory optimization is automatically applied to large DataFrames
- Vectorized scoring reduces computation time by 10x
- Cache TTL is set to 10 minutes for optimal performance

---

#### 2. **API Rate Limiting / "Circuit Breaker OPEN"**

**Symptoms:**
- Error message: "Circuit breaker OPEN: Too many failures"
- Missing data for multiple tickers
- Extraction fails repeatedly

**Solutions:**
```bash
# Wait 2 minutes for automatic circuit breaker reset
# Or manually reset by restarting the ETL pipeline

python run.py  # Will automatically reset circuit breakers
```

**Prevention:**
- Circuit breakers protect against cascading failures
- Automatic backoff with exponential delay
- 3-pass retry logic (batch → surgical → evasion)

**Configuration:**
Edit `etl/retry_utils.py` to adjust thresholds:
```python
YAHOO_FINANCE_BREAKER = CircuitBreaker(
    failure_threshold=10,  # Increase if needed
    timeout=120            # Seconds before reset
)
```

---

#### 3. **Missing Translations / Language Issues**

**Symptoms:**
- Text appears as keys (e.g., "app.title" instead of "Honest Quant")
- Language selector not working
- Mixed languages in UI

**Solutions:**
```bash
# Check if translation files exist
ls locales/

# Should show: en.json, vi.json

# Verify JSON syntax
python -m json.tool locales/en.json
```

**Add Missing Translations:**
Edit `locales/en.json` or `locales/vi.json`:
```json
{
  "app": {
    "title": "Honest Quant Intelligence",
    "subtitle": "Institutional-Grade Analytics"
  },
  "messages": {
    "welcome": "Welcome, {name}!"
  }
}
```

---

#### 4. **Data Quality Warnings**

**Symptoms:**
- Red warnings in sidebar: "⚠️ Data Quality Issues"
- Missing fundamental data for some tickers
- Stale prices or outdated financials

**Solutions:**
```bash
# Force full refresh of all data
python run.py --full-refresh

# Or clear warehouse and rebuild
rm warehouse/stock_dw.duckdb
python run.py
```

**Check Coverage:**
```python
# In Python console
import duckdb
conn = duckdb.connect('warehouse/stock_dw.duckdb')

# Check metadata coverage
result = conn.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(market_cap) as has_market_cap,
        COUNT(pe_ratio) as has_pe
    FROM marts.dim_companies
""").fetchone()

print(f"Coverage: {result[1]/result[0]*100:.1f}%")
```

---

#### 5. **Vectorized Scoring Errors**

**Symptoms:**
- Error: "KeyError: 'sector'"
- Scores showing as NaN
- Fallback to slow row-by-row scoring

**Solutions:**
```python
# Check required columns
required_cols = [
    'pe_ratio', 'peg_ratio', 'roe', 'fcf_margin',
    'total_debt', 'ebitda', 'revenue_growth', 'earnings_growth',
    'rsi', 'price_z_score', 'sector'
]

# Verify DataFrame has all columns
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
```

**Fallback Behavior:**
- System automatically falls back to row-by-row scoring if vectorized fails
- Check logs for specific error message
- Ensure all required columns exist in DataFrame

---

#### 6. **Database Lock Errors**

**Symptoms:**
- Error: "database is locked"
- Cannot write to warehouse
- ETL pipeline hangs

**Solutions:**
```bash
# Close all connections to database
# Kill any running Python processes
pkill -f "python.*run.py"

# Remove lock file if exists
rm warehouse/stock_dw.duckdb.wal

# Restart ETL
python run.py
```

**Prevention:**
- Use `read_only=True` for dashboard queries
- Ensure ETL pipeline completes before starting dashboard
- Don't run multiple ETL processes simultaneously

---

#### 7. **Memory Errors / Out of Memory**

**Symptoms:**
- Error: "MemoryError"
- System becomes unresponsive
- Dashboard crashes

**Solutions:**
```python
# Enable memory optimization in app.py
from etl.performance_utils import optimize_dataframe_memory

# Apply to large DataFrames
prices = optimize_dataframe_memory(prices)
companies = optimize_dataframe_memory(companies)
```

**Reduce Memory Usage:**
```python
# Use batch processing for large operations
from etl.performance_utils import batch_process_dataframe

result = batch_process_dataframe(
    df,
    process_func=my_function,
    batch_size=1000  # Adjust based on available memory
)
```

---

#### 8. **Configuration Not Loading**

**Symptoms:**
- Scoring uses default values instead of config
- Changes to YAML files not reflected
- Error: "Config file not found"

**Solutions:**
```bash
# Verify config files exist
ls config/

# Should show:
# - scoring_rules.yaml
# - etl_config.yaml
# - tickers.yaml

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config/scoring_rules.yaml'))"
```

**Force Config Reload:**
```python
from etl.config_manager import load_config

# Force reload from disk
config = load_config("scoring_rules", reload=True)
```

---

#### 9. **Test Failures**

**Symptoms:**
- Tests fail with import errors
- Mock objects not working
- Assertion errors

**Solutions:**
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run tests with verbose output
pytest tests/ -v -s

# Run specific test file
pytest tests/test_scoring_engine.py -v

# Run with coverage report
pytest tests/ --cov=etl --cov=utils --cov-report=html

# View coverage report
open htmlcov/index.html
```

---

#### 10. **Docker / Deployment Issues**

**Symptoms:**
- Container fails to start
- Port conflicts
- Volume mount errors

**Solutions:**
```bash
# Check Docker logs
docker-compose logs -f

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check port availability
lsof -i :8501  # Streamlit default port

# Fix port conflicts in docker-compose.yml
ports:
  - "8502:8501"  # Use different external port
```

---

### Performance Optimization Checklist

✅ **Memory Optimization**
```python
# Apply to all large DataFrames
df = optimize_dataframe_memory(df)
```

✅ **Vectorized Operations**
```python
# Use vectorized scoring (10x faster)
df['score'] = vectorized_compute_scores(df)
```

✅ **Batch Processing**
```python
# Process large datasets in batches
result = batch_process_dataframe(df, func, batch_size=1000)
```

✅ **Caching**
```python
# Use Streamlit caching for expensive operations
@st.cache_data(ttl=600)
def load_data():
    # ...
```

✅ **Database Optimization**
```python
# Use read-only connections for queries
with get_db_connection(read_only=True) as conn:
    df = conn.execute("SELECT ...").df()
```

---

### Debugging Tips

#### Enable Debug Logging

```python
# In run.py or app.py
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### Check ETL Audit Log

```python
import duckdb

conn = duckdb.connect('warehouse/etl_audit.duckdb', read_only=True)
audit = conn.execute("""
    SELECT * FROM etl.audit_log 
    ORDER BY start_time DESC 
    LIMIT 10
""").df()

print(audit)
```

#### Verify Data Freshness

```python
import duckdb

conn = duckdb.connect('warehouse/stock_dw.duckdb', read_only=True)

# Check latest price date
latest = conn.execute("""
    SELECT MAX(date) as latest_date, COUNT(DISTINCT ticker) as tickers
    FROM raw.stock_prices
""").fetchone()

print(f"Latest data: {latest[0]}, Tickers: {latest[1]}")
```

---

### Getting Help

1. **Check Documentation**
   - API Reference: `docs/en/API.md`
   - Architecture: `docs/en/ETL_ARCHITECTURE.md`
   - Testing Guide: `docs/en/TESTING.md`

2. **Review Test Files**
   - Examples: `tests/test_*.py`
   - Mock patterns: `tests/test_extract.py`

3. **Check Configuration**
   - Scoring rules: `config/scoring_rules.yaml`
   - ETL config: `config/etl_config.yaml`
   - Tickers: `config/tickers.yaml`

4. **Verify Installation**
   ```bash
   pip list | grep -E "pandas|numpy|streamlit|duckdb|yfinance"
   ```

5. **System Requirements**
   - Python 3.9+
   - 8GB RAM minimum (16GB recommended)
   - 2GB disk space for warehouse
   - Internet connection for API calls

---

### Known Limitations

1. **Free API Constraints**
   - Yahoo Finance rate limits: ~2000 requests/hour
   - Some tickers may have incomplete data
   - Delayed data (15-20 minutes for real-time quotes)

2. **Memory Usage**
   - Large price history (5 years × 600 tickers) requires ~2GB RAM
   - Vectorized operations need contiguous memory
   - Consider batch processing for very large datasets

3. **Currency Normalization**
   - FX rates updated daily
   - Historical FX rates use forward-fill
   - Some exotic currencies may use default rates

4. **Scoring Limitations**
   - Requires minimum data: PE, market cap, sector
   - Early-stage companies (negative earnings) penalized
   - Sector adjustments may not fit all business models

---

### Emergency Recovery

If all else fails, perform a complete reset:

```bash
# 1. Backup current data (optional)
cp -r warehouse warehouse_backup_$(date +%Y%m%d)

# 2. Remove all cached data
rm -rf warehouse/*.duckdb
rm -rf warehouse/*.parquet

# 3. Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# 4. Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt

# 5. Rebuild warehouse from scratch
python run.py --full-refresh

# 6. Restart dashboard
streamlit run app.py
```

---

*For additional support, check the test files in `tests/` for working examples of all major functions.*
