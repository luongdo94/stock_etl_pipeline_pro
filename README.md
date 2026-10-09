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
- **Automation**: `python run.py` is the single entry point; schedule it with Windows Task Scheduler (`register_daily_etl.ps1`) or cron. Each run is locked against overlap, validated against the universe and the previous production warehouse, and swapped in atomically (see *ETL reliability* below).

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

## 🧮 3. Scoring Engine (Quality · Value · Momentum)

`core/scoring.py` (pure pandas; every threshold and weight in `config/scoring_rules.yaml`) gives each stock **three
independent 0-100 scores**. They answer different questions and are never mixed:

| Score | Question | What goes in |
| :--- | :--- | :--- |
| **Quality** | Is it a good business? | Return on capital · operating/gross/FCF margin · growth & stability · net debt/EBITDA · FCF conversion. Banks/insurers: ROE, net margin, stability. Red flags subtract points (loss-making, debt without EBITDA, high leverage, uncovered dividend, negative equity). |
| **Value** | Is the price attractive? | FCF yield (after stock-based compensation) · EV/EBITDA · earnings yield (trailing + 3-year median) · PEG on *realised* growth · shareholder yield · P/S, each half *percentile within the industry/sector*, half an absolute band. |
| **Momentum** | What has price been doing? | 12-1 month return rank + trend. **Timing only** — never part of Quality or Value. |

- **Peers, not one yardstick**: margins and multiples are ranked against the industry (sector, then universe, when under 5 peers).
- **Unknown ≠ 0**: missing inputs are excluded and the rest re-weighted; with under 80% of the weight observable the score is pulled toward 50 and *Coverage* lowers Decision confidence.
- **No analyst forecasts anywhere in Quality or Value** (not even forward P/E); ratings/targets are not used at all. A separate **Revisions** score reads the *change* in analysts' EPS estimates (context only). Beta, RSI, Z-score are not scored.
- **Signal** (STRONG SETUP … UNFAVOURABLE) is context for the Decision: trend, quality, value, the Decision's own reward/risk and volume flow. The 52-week position is shown, not scored.
- **Each number appears once** in Stock Analysis: Quality, Value, Momentum and Revisions (with the 30-day upgrade/downgrade counts) in the score tiles; reward/risk, stop and value in the Decision Summary; the 52-week range in its meter; forward P/E in the Analyst section; short interest and ownership in the Ownership section. The Timing context box only shows trend and volume flow, plus a one-line recap of what the Signal counted.
- **One label per stock.** The Scanner shows the Decision with a timing arrow, e.g. `BUY CANDIDATE ▲`: ▲ = the Signal's context (trend, quality, value, reward/risk, volume flow) is supportive (STRONG SETUP / FAVOURABLE), · = neutral, ▼ = against (WEAKENING / UNFAVOURABLE). The arrow never changes the recommendation; the full Signal label is the optional `Action` column and the *Timing context* box in Stock Analysis.
- **Decision and Signal never contradict each other.** Every tab builds the Decision with the same function (`core.decision.decide`) and the same inputs (including the track-record state, which changes confidence). The Signal reads the Decision's stance: when the Decision says AVOID / TRIM the reward/risk pillar shows *OVERVALUED vs DCF* and costs a point, and the label cannot be favourable; a BUY CANDIDATE is never shown as UNFAVOURABLE. When the Value score (peer multiples) and the DCF disagree sharply, the Decision says so and lowers its confidence.
- **Valuation model by business type:** FCFE DCF for operating companies and for asset-light fee businesses (exchanges, index and data providers) with positive free cash flow; justified P/B for banks, insurers, brokers and asset managers, where book value drives value.
- **Valuation (Decision)**: cash flows are discounted at the risk-free rate and long-run growth of *their* currency (live US 10Y for USD; configured for EUR/JPY/GBP/CHF…, `config/decision_rules.yaml`); banks and insurers use a justified price-to-book model ((ROE − g)/(r − g), ROE normalised over 4 years and faded toward the cost of equity); free cash flow is taken after stock-based compensation.
- **In the Decision**: a BUY candidate needs Quality ≥ 50 (cheap and weak is a value trap); a DCF "cheap" contradicted by Value < 30 lowers confidence.
- **Validation**: each daily snapshot stores all three scores stamped with `score_version`; the Track Record tab reports the information coefficient of each. Until it shows t > 2 the weights are judgement, not evidence.

### 🚀 Fundamental Momentum Index (FMI) (0-100)
A CANSLIM-style growth accelerator index. Because free APIs often suffer from delayed/sparse data, FMI utilizes a hyper-dynamic *Live-Computed Engine*:
- **Earnings & Revenue Acceleration**: Compares the latest available quarter against the baseline of the most recent FULL year's growth. 
- **QoQ Fallback Engine**: If Year-over-Year (YoY) data is missing due to API limits, it automatically falls back to extrapolating Quarter-over-Quarter (QoQ) metrics.
- **Margin Expansion**: Mathematically checks if $EPS\_Growth > Revenue\_Growth$, indicating rising operating leverage.
- Outputs actionable labels: `Accelerating`, `Slowing`, `Turning Around`, `Bottoming`.

### 📉 Z-Score Mean Reversion & Deep Value
- Warehouse / screener Z-Score (`price_z_score`): $Z = (Price - MEAN_{5Y}) / STD_{5Y}$ over a rolling 1260-trading-day window.
- The Strategy Lab's Z-Score strategy uses a faster $Z = (Price - MA_{60}) / STD_{60}$.
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
- **Radar Charts**: Powered by Plotly, breaks down Quality (returns, margins, stability, balance sheet, cash conversion) next to Value and Momentum.
- **Progress Panels**: Neon-colored metric bars displaying the real-time FMI Acceleration breakdown.
- Evaluates Short Interest vulnerability and Institutional accumulation flow.

### Tab 3: AI Market Scanner
- A dynamic, multi-condition screener. Filter thousands of stocks in milliseconds using DuckDB's backend.
- **14 Strategy Presets** (`core/scan_presets.py`, tested in `tests/test_scan_presets.py`). Each answers a distinct question; RSI, Z-Score, PEG and the volume-flow heuristic remain available as Custom Refinement sliders.

#### Opportunity
- `🏆 Institutional Pulse` - Quality ≥75 (ELITE) + uptrend (MA50 > MA200)
- `💎 Quality at a Fair Price` - Quality ≥75 + Value ≥50
- `🏷️ Deep Value` - Value ≥70 + Quality ≥60
- `📈 Rising Estimates` - Revisions ≥65 + Quality ≥60
- `🌱 GARP` - 0 < PEG < 1.0 + Quality ≥60
- `⚙️ Both Accelerating` - EPS and revenue both up >10% QoQ for 2 quarters
- `🚀 Buy on Dip` - uptrend + RSI < 40
- `⚡ Strong Breakout` - uptrend, >5% above MA200, Momentum ≥70, RSI 50-70
- `💰 Quality Dividend` - yield >2.5%, Quality ≥60, dividend covered by FCF, net debt/EBITDA <3x (banks kept)

#### Risk & Warning
- `🪤 Value Trap Risk` - Value ≥65 but Quality <45
- `🚩 Red Flags` - any Quality penalty (loss-making, leverage, uncovered dividend…)
- `📉 Downtrend` - MA50 < MA200 and (RSI <50 or Quality <45)
- `⚠️ Earnings Deterioration` - EPS down >10% QoQ for 2 quarters, revenue not accelerating
- `🎈 Overextended` - RSI >70 and Z-Score >+2 (a price statistic, not a valuation)

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

### Decision Summary (Stock Analysis tab)
One answer per stock instead of many competing badges: **stance** (BUY CANDIDATE / HOLD-WATCH /
AVOID-TRIM / NOT ENOUGH DATA), **confidence**, **expected return net of costs**, **downside**
(worse of bear-case value and stop), **reward/risk**, **suggested position size** and the
**conditions that would invalidate the thesis**. A BUY candidate needs a ≥25% margin of safety,
reward/risk ≥ 2 and non-low confidence. Confidence is capped at MEDIUM until the Track Record shows
statistical evidence. Logic: `core/decision.py`; assumptions (commission, FX spread, dividend
withholding, risk per trade, max position, earnings blackout) in `config/decision_rules.yaml`.

- **Valuation** (`core/valuation.py`): 10-year DCF on normalised cash-flow-statement FCF (median of
  the last 3 years, OCF − capex, EUR) discounted at the CAPM cost of equity with Blume-adjusted beta
  (no double-counting of debt). Growth is anchored on the 3-year revenue CAGR; earnings and FCF
  growth may move it by ±5pp; it fades to terminal. Bear/base/bull scenarios, sensitivity table,
  **reverse DCF**, and **relative valuation** (percentile vs industry peers, P/E vs own 5-year average).
  The DCF is reported as *not informative* — and never drives BUY/AVOID — for banks/insurers, when
  the price implies growth beyond the model's range, or when the value is implausibly high (>2.5x).
  BUY needs a 25% margin of safety on the base case; AVOID needs the price above the bull case.
  ERP and margin of safety are set in `config/decision_rules.yaml`.
- **One recommendation**: the Scanner's **Decision** column and the Decision Summary use the same
  function and inputs. The old 6-pillar label is now shown as **Signal** (an input, not a call).
- **Portfolio fit**: correlation with current holdings and sector / currency weight before → after.
- **Timing**: warning when earnings are due within the blackout window.

### Track Record tab
Every ETL run stores the score and action shown for each stock that day (`etl/snapshot.py` →
`warehouse/track_record.duckdb`, mirrored to `marts.score_snapshots`). The tab compares them with
what happened next vs SPY: IC per horizon, returns by score quintile, hit rate per action, and a
log of every change of call. Logic: `core/track_record.py`.

### Alerts & sell discipline
Alert rules are stored per user in Supabase (create the table once with `docs/sql/stock_alerts.sql`)
and evaluated in-app on every load. The Watchlist tab flags ideas whose invalidation level, take
profit, entry zone or intrinsic value has been reached, and holdings/ideas reporting soon.

> Decision support for your own judgement — not investment advice.

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
- Docker & Docker Compose (optional: run the ETL in a container)

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
    - In `.streamlit/secrets.toml`: `COOKIE_SECRET` (a long random string, e.g. `python -c "import secrets; print(secrets.token_hex(32))"`) — signs the 7-day login cookie. Without it, users must log in again in every new browser session. Optionally `SUPABASE_ANON_KEY` for the login client.
4. The dashboard will now stream data directly from the cloud via HTTP Parquet querying.

### Step 5. Scheduling
```powershell
powershell -ExecutionPolicy Bypass -File .\register_daily_etl.ps1     # Windows, Mon–Sat 07:00
```
or `docker compose run --rm pipeline` from cron. Failures are alerted (see *ETL reliability*).

---

## 📂 7. Directory Structure
```text
stock_etl_pipeline/
│
├── etl/                   # Data extraction, normalization, and Quant Engine functions
│   ├── extract.py         # ThreadPool API scrapers (yfinance)
│   ├── transform.py       # DuckDB Star Schema generation 
│   ├── load.py            # Local warehouse persistor
│   └── utils.py           # Watermarks, refresh rules, upside cleaning, email report (scores live in core/scoring.py)
│
├── core/                  # Pure analytics — no Streamlit, unit-testable
│   ├── indicators.py      # Wilder RSI (shared by ETL + dashboard)
│   ├── smart_money.py     # Institutional flow engine
│   ├── rating.py          # 6-pillar institutional rating
│   ├── levels.py          # Swing support/resistance, tactical trade metrics
│   ├── backtest.py        # Strategy Lab simulator
│   └── symbols.py         # Yahoo → TradingView symbol mapping
│
├── services/              # I/O: warehouse, live market data, AI, user store
│   ├── db.py              # get_db_connection (local / parquet cache / S3), load_data
│   ├── market_data.py     # Macro, FX, dividend calendar (with DB fallbacks)
│   ├── screener.py        # Master screener table
│   ├── ai.py              # Cohere narratives, FinBERT sentiment
│   └── user_store.py      # Watchlist / portfolio persistence (Supabase)
│
├── ui/                    # Theme CSS, icons, metric tiles
├── views/                 # One module per dashboard tab: render(ctx)
│
├── warehouse/             # Local database location
│   └── stock_dw.duckdb    # Compiled Analytics Database
│
├── tests/                 # pytest suite (+ synthetic_warehouse.py, app smoke tests)
├── app.py                 # Dashboard shell: auth, data load, sidebar, KPI header, tab routing
├── auth.py                # Supabase login + signed session cookie
├── run.py                 # Pipeline trigger entry point
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

`app.py` loads the data and computes the shared state (scores, regime, KPIs), then calls
`views.<tab>.render(globals())` for the selected tab. Each view unpacks only the shared names it uses
at the top of `render()`.

Run the tests (the `smoke` ones render every tab against a synthetic warehouse):
```bash
pytest                    # everything offline
pytest -m "not smoke"     # fast unit tests only
RUN_LIVE=1 pytest -m live # real Yahoo data: checks EUR units of market cap / revenue / FCF
```

**Currency handling** (`etl/extract.py`): one `fx_to_eur()` for every extractor. Statement amounts
use the statement's own `currencyCode` (e.g. CNY for Xiaomi, DKK for Novo ADRs); quotes may be in a
minor unit (GBp = GBP/100) but company-level amounts are always in the major unit. A missing FX
rate leaves the value empty instead of storing it unconverted. Annual/quarterly FCF is stored in
EUR. After upgrading, run one full refresh (`python run.py --full`) so older rows are re-converted.

**Daily schedule** (needed for the Track Record): `powershell -ExecutionPolicy Bypass -File .\register_daily_etl.ps1`
registers a Windows task (Mon–Sat 07:00, never overlapping) that runs `run.py` and logs to `logs/scheduled_etl.log`.

### ETL reliability

| Concern | What the pipeline does |
| :--- | :--- |
| **Overlapping runs** | One run at a time (OS file lock `warehouse/etl.lock`); a second `run.py` exits immediately. |
| **Dividend / split restatements** | Prices are downloaded split- and dividend-adjusted, so Yahoo restates history after every corporate action. Each run re-downloads a few overlap days; a ticker whose overlap differs from what is stored (> 0.2%) has its whole history re-pulled, and every ticker is re-pulled weekly (`price_integrity` in `config/etl_config.yaml`). |
| **Universe** | Resolved once per run (config tickers + TradingView screens) and recorded in `raw.universe` with `last_seen`. A TradingView outage never shrinks it; a discovered ticker expires only after 30 days unseen. Nothing touches the network at import time. |
| **Release gates** | Before the swap the new warehouse is compared with the universe and with the previous production file (ticker/row counts, freshness, price and market-cap continuity, partial downloads). Critical findings abort the swap and leave production untouched; warnings appear in the dashboard's data-quality list. The structural DQ audit is fail-closed. |
| **Swap** | Shadow file → production with retries while the dashboard holds the file (up to 2 min). If it still cannot swap, the validated warehouse is kept as `stock_dw_pending.duckdb` and promoted by the next run or by `python run.py --promote`. |
| **Cloud sync** | Immutable snapshots under `snapshots/<version>/` plus a `manifest.json` uploaded last, so readers never mix two runs. Large tables are chunked deterministically (250k rows). The warehouse is opened read-only; any failed upload fails the sync (exit code 2) and the previous snapshot stays live. |
| **Provenance** | Every price row stores `currency`, `fx_rate` and `price_scale`; company and statement rows store the FX rate used, so a conversion can be audited or redone without downloading again. A price that cannot be converted is skipped, never stored as EUR. |
| **Alerts** | `logs/last_run.json` always records the outcome. A failed run sends an alert by e-mail (`SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL_TO`) and/or webhook (`ETL_WEBHOOK_URL`); `ETL_NOTIFY_SUCCESS=1` also sends the morning report. |
| **Exit codes** | `0` success · `1` ETL failed or refused (previous data untouched) · `2` ETL ok, cloud sync failed. |

All thresholds live in `config/etl_config.yaml`.

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
- Scores are computed for the whole universe at once (~800 tickers in about a second)
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
python run.py --full

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

#### 5. **Scores Missing or Looking Neutral**

**Symptoms:**
- Quality / Value show 50 or very similar values for many stocks
- `Coverage (%)` is low in the scanner; the Decision lists "Only N% of the scoring inputs are available"

**Why:** unknown inputs are excluded (never scored as 0) and, with under 80% of a score's weight observable, the
score is pulled toward 50. Peer percentiles also need at least 5 stocks per industry/sector (otherwise the whole
universe is the peer group).

**Check what the engine sees:**
```python
from core.scoring import build_features, score_universe
f = build_features(companies, annual_fin, prices)      # derived inputs per ticker
f[["roc", "net_debt_ebitda", "fcf_yield", "ev_ebitda", "ret_12_1"]].isna().mean()   # share unknown
score_universe(companies, annual_fin, prices)[["quality", "value", "momentum", "quality_coverage", "missing"]]
```
Momentum needs 253 trading days of prices; return on capital and the growth/stability pillar need ≥3 annual statements.

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
# Score the whole universe at once (peer percentiles need the full set)
from core.scoring import score_universe
scores = score_universe(companies, annual_fin, prices)   # quality / value / momentum
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
python run.py --full

# 6. Restart dashboard
streamlit run app.py
```

---

*For additional support, check the test files in `tests/` for working examples of all major functions.*

## Local run without login

Set LOCAL_DEV_MODE=1 (environment variable only) to skip Supabase login on your own machine. It is refused unless Streamlit is served from localhost and `SUPABASE_REMOTE_MODE` is not true. Watchlist, portfolio and alerts are then saved in `warehouse/local_user/` (git-ignored). Never set it on a deployed app.

