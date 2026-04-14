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

### Intelligent Loading Strategy
- **Incremental Load (Watermarking)**: The system automatically detects the last available data point for each ticker. In daily runs, it strictly fetches only the missing "gap" (incremental window), reducing bandwidth consumption and avoiding IP blocks.
- **New Ticker Bootstrapping**: When a new ticker is added to `config/tickers.yaml`, the ETL engine automatically identifies its absence in the warehouse and triggers a **Full 5-Year History Download** specifically for that ticker, while keeping all other tickers on an incremental path.
- **Smart Fundamental Refresh**: Financial data is expensive and sensitive. The system implements a **7-day freshness cache** (168h) and a **95% coverage threshold**. It only triggers a deep fundamental crawl (using `yahooquery` batching) if the last update was > 7 days ago OR if total warehouse coverage drops below 95%. Users can also force-skip this via `--fast` mode.

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
- Embedded Presets: `🚀 High Momentum (FMI > 80)`, `💎 Deep Value (Z < -2, P/E < 15)`, `🛡️ Defensive Yields`.

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

## 🛠️ 5. Installation & Deployment

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
Updates only stock prices and runs technical indicators. Skips heavy fundamental API calls. Recommended for weekdays.
```bash
python etl/pipeline.py --fast
```

**Weekly/Full Update (Deep Dive)**
Refreshes everything including Financials, Cashflows, and Earnings (if last update > 7 days).
```bash
python etl/pipeline.py
```

**Force Rebuild**
Ignore all caches and download 5 years of full history for everything.
```bash
python etl/pipeline.py --full
```

### Step 3. Spin Up The Control Room
Run the Streamlit frontend locally:
```bash
./start_dashboard.sh
# Alternatively: streamlit run app.py
```

### Step 4. Continuous Integration (Airflow)
To set up completely automated daily ETL runs so your data is always fresh:
```bash
docker-compose up -d
# Access the Airflow UI at http://localhost:8080 to trigger the master DAG.
```

---

## 📂 6. Directory Structure
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
├── app.py                 # Main Streamlit Tactical Dashboard UI
├── run.py                 # Pipeline trigger entry point
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

---
*Architected and Engineered by GIA LUONG DO.*
