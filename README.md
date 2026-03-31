# 🚀 Elite Pro Stock Diagnostic Engine

A professional-grade, institutional-level stock analytics and diagnostic engine. This project transforms raw market data into deep, actionable insights using a modern data stack (DuckDB, dbt, Airflow) and advanced AI forecasting (LSTM Deep Learning).

## 🌟 Key Features

### 1. **Institutional-Grade Metrics**
Go beyond basic price tracking with institutional metrics including:
- **Valuation**: Forward P/E, PEG Ratio (trailing/forward), EV/EBITDA, Price-to-Sales.
- **Solvency & Risk**: Current Ratio, Quick Ratio, Debt-to-Equity, Short Interest %, Institutional Ownership.
- **Growth Analysis**: 4-year historical trends for Revenue, EPS, and Margins.

### 2. **AI Predictive Suite**
Powered by a custom **LSTM (Long Short-Term Memory)** neural network:
- **Deep Learning Forecasts**: Time-series prediction for the next 30 days.
- **Monte Carlo Simulations**: 500+ path-simulations for risk-adjusted price projections.
- **Sentiment-Driven Drift**: Hybrid forecasting that incorporates technical signals and news sentiment.

### 3. **Strategic Control Room**
A high-density Streamlit dashboard for "God-Mode" market awareness:
- **Alpha Trends**: Track relative strength vs SPY to find true market leaders.
- **Diagnostic Matrices**: Map Quality vs Return and Risk vs Return across sectors.
- **Prescriptive Action Plans**: Heuristic-based recommendations (Buy/Sell/Hold) based on multi-factor scores.
- **Sentiment Hub**: AI-summarized news sentiment and top market movers.

### 4. **Automated ETL Pipeline**
A robust, production-ready data architecture:
- **Extraction**: Multi-source ingestion (US, EU, JP markets) via `yfinance`.
- **Warehousing**: High-performance local storage using **DuckDB**.
- **Transformation**: Multi-layered modeling (Raw -> Staging -> Marts) with `dbt`-style logic.
- **Orchestration**: Fully containerized **Apache Airflow** DAGs with daily schedules and email notifications.

## 🛠️ Tech Stack

- **Dashboard**: Streamlit (Python)
- **Database**: DuckDB
- **Transformation**: dbt / Python
- **Orchestration**: Airflow (Docker)
- **Deep Learning**: PyTorch (LSTM)
- **Data Source**: Yahoo Finance (yfinance)

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline (Manual)
```bash
python run.py
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

### 4. Start Airflow (Docker)
```bash
docker-compose up -d
# Access UI at http://localhost:8080
```

## 📂 Project Structure
- `/etl`: Data extraction and loading logic.
- `/airflow`: DAGs and orchestration configuration.
- `/warehouse`: DuckDB data storage.
- `/dbt`: SQL models for the production layer.
- `app.py`: Main Streamlit dashboard.
- `walkthrough.md`: Detailed feature demonstration.

---
*Created by GIA LUONG DO.*
