"""
Live-data regression test: extract a few representative tickers from Yahoo and check that the
EUR-normalised fundamentals are the right ORDER OF MAGNITUDE. Catches FX / unit bugs like
UK pounds treated as pence (100x too small), JPY left unconverted (~160x too big) or CNY
statements stored as EUR (~8x too big) — all of which happened before this test existed.

Needs network, so it is skipped unless RUN_LIVE=1:
    RUN_LIVE=1 pytest -m live
"""
import os

import pandas as pd
import pytest

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 (network)")]

# ticker: (min, max) plausible range in EUR billions — deliberately wide, it only guards units
MCAP_BN = {"AAPL": (1000, 10000), "SAP.DE": (80, 600), "RR.L": (20, 400), "6758.T": (40, 500),
           "1810.HK": (30, 600), "NESN.SW": (100, 600)}
REVENUE_BN = {"AAPL": (250, 900), "SAP.DE": (20, 80), "RR.L": (10, 60), "6758.T": (40, 200),
              "1810.HK": (20, 150), "NESN.SW": (60, 150)}


@pytest.fixture(scope="module")
def live():
    from etl import extract as ex
    tickers = {t: ex.TICKERS[t] for t in MCAP_BN if t in ex.TICKERS}
    return {
        "info": ex.extract_company_info(tickers).set_index("ticker"),
        "annual": ex.extract_historical_financials(tickers),
        "fcf": ex.extract_historical_fcf(tickers),
    }


@pytest.mark.parametrize("ticker", list(MCAP_BN))
def test_market_cap_and_revenue_in_eur(live, ticker):
    info = live["info"]
    assert ticker in info.index, f"{ticker} missing from company_info"
    mcap = info.at[ticker, "market_cap"] / 1e9
    rev = info.at[ticker, "revenue_ttm"] / 1e9
    lo, hi = MCAP_BN[ticker]
    assert lo <= mcap <= hi, f"{ticker} market cap €{mcap:,.1f}bn outside [{lo}, {hi}]"
    lo, hi = REVENUE_BN[ticker]
    assert lo <= rev <= hi, f"{ticker} TTM revenue €{rev:,.1f}bn outside [{lo}, {hi}]"


@pytest.mark.parametrize("ticker", list(REVENUE_BN))
def test_annual_statements_in_eur(live, ticker):
    a = live["annual"]
    rows = a[a["ticker"] == ticker].sort_values("date")
    assert not rows.empty, f"{ticker} has no annual statements"
    rev = rows["revenue"].dropna().iloc[-1] / 1e9
    lo, hi = REVENUE_BN[ticker]
    assert lo * 0.7 <= rev <= hi * 1.3, f"{ticker} annual revenue €{rev:,.1f}bn — FX/unit error?"


def test_fcf_margin_plausible(live):
    """Statement FCF (EUR) vs TTM revenue (EUR): a currency mix-up shows up as a silly margin."""
    fcf = live["fcf"].sort_values("year").groupby("ticker")["free_cash_flow"].last()
    info = live["info"]
    for t, v in fcf.items():
        if t in info.index and pd.notna(info.at[t, "revenue_ttm"]) and info.at[t, "revenue_ttm"]:
            margin = v / info.at[t, "revenue_ttm"]
            assert -0.5 < margin < 0.8, f"{t} FCF margin {margin:.0%} — FX/unit error?"
