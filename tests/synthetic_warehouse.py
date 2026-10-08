"""
Builds a small but realistic DuckDB warehouse (raw → staging → marts) from synthetic data,
so the dashboard can be smoke-tested offline without Yahoo / Supabase.

    python -m tests.synthetic_warehouse warehouse/stock_dw.duckdb
"""
import sys
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from etl.load import create_raw_schema
from etl.transform import run_transforms

# ticker: (company, sector, region, start price, market cap)
UNIVERSE = {
    "SPY":    ("SPDR S&P 500 ETF", "Benchmark", "US", 400.0, 5e11),
    "^VIX":   ("CBOE Volatility Index", "Index", "US", 18.0, 0),
    "AAPL":   ("Apple Inc.", "Consumer Electronics", "US", 150.0, 3e12),
    "MSFT":   ("Microsoft Corporation", "Platform Software", "US", 300.0, 2.8e12),
    "NVDA":   ("NVIDIA Corporation", "Semiconductors", "US", 200.0, 2.5e12),
    "JNJ":    ("Johnson & Johnson", "Healthcare", "US", 160.0, 4e11),
    "SAP.DE": ("SAP SE", "Platform Software", "EU", 120.0, 2e11),
    "KO":     ("Coca-Cola Company", "Consumer Staples", "US", 60.0, 2.6e11),
}
N_DAYS = 800


def _prices(rng: np.random.Generator) -> pd.DataFrame:
    days = pd.bdate_range(end=pd.Timestamp(date.today()) - pd.Timedelta(days=1), periods=N_DAYS)
    rows = []
    for i, (t, (name, sector, region, p0, _)) in enumerate(UNIVERSE.items()):
        drift = 0.0004 + 0.0002 * (i % 3)
        rets = rng.normal(drift, 0.015, N_DAYS)
        close = p0 * np.exp(np.cumsum(rets))
        vol = rng.integers(1_000_000, 5_000_000, N_DAYS)
        rows.append(pd.DataFrame({
            "date": days, "ticker": t, "company": name, "sector": sector, "region": region,
            "open": close * (1 + rng.normal(0, 0.003, N_DAYS)),
            "high": close * 1.01, "low": close * 0.99, "close": close, "volume": vol,
        }))
    df = pd.concat(rows, ignore_index=True)
    df["_extracted_at"] = pd.Timestamp.now()
    return df


def _company_info(prices: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    last = prices.groupby("ticker")["close"].last()
    rows = []
    for t, (name, sector, region, _, mcap) in UNIVERSE.items():
        is_equity = t not in ("SPY", "^VIX")
        rows.append({
            "ticker": t, "quote_type": "EQUITY" if is_equity else "ETF", "company": name,
            "sector": sector, "industry": sector, "region": region, "market_cap": int(mcap),
            "pe_ratio": float(rng.uniform(12, 40)) if is_equity else None,
            "forward_pe": float(rng.uniform(10, 35)) if is_equity else None,
            "revenue_ttm": int(mcap / 8), "employees": 50_000, "country": "United States",
            "currency": "EUR" if t.endswith(".DE") else "USD",
            "total_debt": int(mcap / 20), "ebitda": int(mcap / 25),
            "gross_margin": 0.45, "operating_margin": 0.25, "trailing_eps": 5.0, "forward_eps": 5.6,
            "roe": 0.22, "free_cashflow": mcap / 30, "price_to_book": 8.0, "beta": 1.1,
            "target_mean_price": float(last[t] * 1.1), "recommendation_key": "buy",
            "peg_ratio": 1.4, "price_to_sales": 6.0, "ev_to_ebitda": 18.0,
            "revenue_growth": 0.08, "earnings_growth": 0.12, "current_ratio": 1.5, "quick_ratio": 1.2,
            "debt_to_equity": 60.0, "short_ratio": 2.0, "short_percent_of_float": 0.01,
            "inst_ownership": 0.7, "insider_ownership": 0.01, "dividend_yield": 0.012,
            "ex_dividend_date": str(date.today() + timedelta(days=20)),
            "pay_date": str(date.today() + timedelta(days=35)),
            "_extracted_at": pd.Timestamp.now(),
        })
    return pd.DataFrame(rows)


def _financials(quarterly: bool) -> pd.DataFrame:
    rows = []
    periods = pd.date_range(end=pd.Timestamp(date.today()), periods=12 if quarterly else 5,
                            freq="QE" if quarterly else "YE")
    for t, (_, _, _, _, mcap) in UNIVERSE.items():
        if t in ("SPY", "^VIX"):
            continue
        base_rev = mcap / (32 if quarterly else 8)
        for k, d in enumerate(periods):
            growth = 1.03 ** k
            rows.append({"ticker": t, "date": d.date(), "revenue": base_rev * growth,
                         "net_income": base_rev * 0.2 * growth, "total_equity": mcap / 10,
                         "eps": 1.2 * growth * (1 if quarterly else 4),
                         "eps_diluted": 1.18 * growth * (1 if quarterly else 4)})
    return pd.DataFrame(rows)


def build(db_path: str, seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    prices = _prices(rng)
    with duckdb.connect(db_path) as conn:
        create_raw_schema(conn)
        conn.register("p", prices)
        conn.execute("""INSERT INTO raw.stock_prices (date, open, high, low, close, volume, ticker,
                        company, sector, region, _extracted_at)
                        SELECT date, open, high, low, close, volume, ticker, company, sector, region,
                        _extracted_at FROM p""")
        info = _company_info(prices, rng)
        conn.register("ci", info)
        cols = ", ".join(info.columns)
        conn.execute(f"INSERT INTO raw.company_info ({cols}) SELECT {cols} FROM ci")
        for table, quarterly in (("raw.quarterly_financials", True), ("raw.historical_financials", False)):
            conn.register("fin", _financials(quarterly))
            conn.execute(f"""INSERT INTO {table} (ticker, date, revenue, net_income, total_equity, eps, eps_diluted)
                             SELECT ticker, date, revenue, net_income, total_equity, eps, eps_diluted FROM fin""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raw.insider_summary (
                ticker VARCHAR, insider_purchases_6m BIGINT, insider_sales_6m BIGINT, net_shares BIGINT,
                pct_buy DOUBLE, pct_sell DOUBLE, _extracted_at TIMESTAMP)""")
        run_transforms(conn)
    return db_path


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "warehouse/stock_dw.duckdb"))
