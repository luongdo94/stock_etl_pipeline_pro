# Data Dictionary & Data Lineage

This documentation clarifies the origin of all data points within the `stock_dw.duckdb` warehouse. It specifies which fields are sourced directly from the upstream provider (Yahoo Finance) and which fields are calculated locally via the ETL pipeline.

## 1. Raw Layer (`raw` schema)

All tables in the `raw` schema are 100% sourced from **Yahoo Finance (`yfinance` API)**. No financial calculations or transformations occur in this layer aside from basic data typing and cleaning.

| Table | Source | Description |
| :--- | :--- | :--- |
| `raw.company_info` | Yahoo Finance (`.info`) | Core company metadata, static financial ratios (P/E, P/B, Debt, Margins, Share Float, Analyst Targets), and descriptive data (Sector, Industry, Region). Note: Sector mapping is partially overridden by the local `config/tickers.yaml`. |
| `raw.stock_prices` | Yahoo Finance (`.history`) | Daily EOD (End of Day) OHLCV prices (Open, High, Low, Close, Volume). |
| `raw.quarterly_financials` | Yahoo Finance (`.quarterly_financials`) | Selected items from quarterly Income Statements and Balance Sheets (Revenue, Net Income, EPS, Total Equity). |
| `raw.historical_financials` | Yahoo Finance (`.financials`) | Selected items from annual Income Statements and Balance Sheets. |
| `raw.cashflows` | Yahoo Finance (`.cashflow`) | Trailing 12-month (TTM) cash flow items, specifically Buybacks and Dividends Paid. |

---

## 2. Presentation Layer (`marts` schema)

The `marts` schema blends raw Yahoo Finance data with local calculations (Technical Indicators, Momentum Scores, and fundamental growth metrics) via SQL transformations (`etl/transform.py`).

### A. `marts.dim_companies`

This table serves as the primary index for screener filtering.

> [!NOTE] 
> Most snapshot metrics like `forward_pe`, `roe`, `debt_to_equity`, and `target_mean_price` are passed directly from Yahoo Finance. The fields listed below are **calculated locally**.

| Column | Source | Calculation / Formula |
| :--- | :--- | :--- |
| `cap_category` | Calculated | Built from `market_cap`: <br> `≥ $1T` = Mega-Cap <br> `$200B-$1T` = Large-Cap <br> `$10B-$200B` = Mid-Cap <br> `< $10B` = Small-Cap |
| `buyback_yield_pct` | Calculated | `(buyback_ttm / market_cap) * 100` |
| `dividends_paid_yield_pct` | Calculated | `(ABS(dividends_paid_ttm) / market_cap) * 100` |
| `net_payout_yield_pct` | Calculated | `buyback_yield_pct + dividends_paid_yield_pct` |
| `fcf_margin` | Calculated | `(free_cashflow / revenue_ttm) * 100` (computed if missing from raw data) |
| `fmi_rev_acceleration` | Calculated | Revenue QoQ Growth (Latest Quarter) - Revenue QoQ Growth (Prior Quarter). <br> *Example:* Q3 +15%, Q2 +5% -> Acceleration = +10%. |
| `fmi_eps_acceleration` | Calculated | EPS QoQ Growth (Latest Quarter) - EPS QoQ Growth (Prior Quarter). |
| `fmi_margin_trend` | Calculated | `eps_yoy_recent` (EPS YoY Growth) - `rev_yoy_recent` (Revenue YoY Growth). <br> *Tests operating leverage (EPS growing faster than revenue).* |
| `fmi_quarters_of_growth` | Calculated | Number of consecutive recent quarters where both EPS and Revenue YoY growth were > 0%. |

### B. `marts.fct_daily_returns`

This table holds the daily time-series performance and all **Technical Indicators**.

| Column | Source | Calculation / Formula |
| :--- | :--- | :--- |
| `price_close`, `volume` | Yahoo Finance | Direct pass-through from `raw.stock_prices`. |
| `daily_return_pct` | Calculated | `((close - prev_close) / prev_close) * 100` |
| `ma_7`, `ma_20`, `ma_50`, `ma_200` | Calculated | Simple Moving Averages computed over looking-back windows (7, 20, 50, and 200 trading days). |
| `rsi` | Calculated | 14-day Wilder's Relative Strength Index. <br> Uses EWMA (Exponential Weighted Moving Average). |
| `ma_signal` | Calculated | Trend Indicator: <br> `BULLISH` if `ma_20 > ma_50`, else `BEARISH` |
| `price_z_score` | Calculated | Number of standard deviations the current price is from its 200-day mean. |

### C. `marts.dim_quarterly_financials` & `dim_annual_financials`

| Column | Source | Calculation / Formula |
| :--- | :--- | :--- |
| `revenue_growth_qoq_pct` | Calculated | `((revenue / prev_quarter_revenue) - 1) * 100` |
| `eps_growth_qoq_pct` | Calculated | `((eps - prev_quarter_eps) / ABS(prev_quarter_eps)) * 100` |
| `revenue_growth_yoy_pct` | Calculated | `((revenue / same_quarter_prev_year_revenue) - 1) * 100` |

---

## 3. Web Dashboard In-Memory Calculations (`app.py`)

| Dashboard Column | Source | Calculation / Formula |
| :--- | :--- | :--- |
| `Quality Score (0-100)` | Calculated | Composite grading score based on financial ratios (ROE, Margins, Debt/Equity). |
| `FMI Score (0-100)` | Calculated | Normalized ranking of fundamental growth parameters. |
| `EPS Momentum` | Calculated | `Accelerating` if last 2 quarters of **QoQ** EPS growth > +10%. |
