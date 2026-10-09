"""
test_extract.py — Tests for ETL extraction functions.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from etl.extract import (
    extract_stock_prices,
    extract_company_info,
    extract_historical_financials,
    extract_quarterly_financials,
    extract_cashflows,
    extract_historical_fcf,
    extract_quarterly_fcf,
    extract_earnings_calendar,
    extract_earnings_history,
    _guess_currency,
    _fetch_statement,
    _safe_float,
    get_equity_tickers,
    load_tickers_config
)


class TestCurrencyGuessing:
    """Test currency heuristics."""
    
    def test_guess_currency_japanese(self):
        assert _guess_currency("7203.T") == "JPY"
        assert _guess_currency("SONY.T") == "JPY"
    
    def test_guess_currency_european(self):
        assert _guess_currency("SAP.DE") == "EUR"
        assert _guess_currency("MC.PA") == "EUR"
        assert _guess_currency("ASML.AS") == "EUR"
    
    def test_guess_currency_uk(self):
        # LSE prices are quoted in pence; extract converts with GBPEUR=X / 100
        assert _guess_currency("BARC.L") == "GBp"
        assert _guess_currency("BP.L") == "GBp"
    
    def test_guess_currency_hongkong(self):
        assert _guess_currency("0700.HK") == "HKD"
    
    def test_guess_currency_default_usd(self):
        assert _guess_currency("AAPL") == "USD"
        assert _guess_currency("MSFT") == "USD"


class TestSafeFloat:
    """Test safe float conversion."""
    
    def test_safe_float_valid(self):
        assert _safe_float(42.5) == 42.5
        assert _safe_float("123.45") == 123.45
        assert _safe_float(100) == 100.0
    
    def test_safe_float_none(self):
        assert _safe_float(None) is None
    
    def test_safe_float_nan(self):
        assert _safe_float(float('nan')) is None
        assert _safe_float(np.nan) is None
    
    def test_safe_float_invalid(self):
        assert _safe_float("invalid") is None
        assert _safe_float([1, 2, 3]) is None


class TestTickerFiltering:
    """Test ticker filtering logic."""
    
    def test_get_equity_tickers_filters_indices(self):
        tickers = {
            "AAPL": {"name": "Apple", "sector": "Technology", "region": "US"},
            "^VIX": {"name": "VIX", "sector": "Index", "region": "US"},
            "SPY": {"name": "SPY", "sector": "Benchmark", "region": "US"},
            "MSFT": {"name": "Microsoft", "sector": "Technology", "region": "US"},
        }
        
        equities = get_equity_tickers(tickers)
        
        assert "AAPL" in equities
        assert "MSFT" in equities
        assert "^VIX" not in equities  # Index
        assert "SPY" not in equities   # Benchmark


def _price_frame(start, periods, closes):
    """Single-ticker yf.download shape: OHLCV columns, DatetimeIndex named 'Date'."""
    idx = pd.date_range(start, periods=periods, name="Date")
    return pd.DataFrame({
        'Open': closes, 'High': [c + 2 for c in closes], 'Low': [c - 2 for c in closes],
        'Close': closes, 'Volume': [1_000_000] * periods,
    }, index=idx)


def _fake_download(prices, usd_eur=0.9):
    """yf.download stub: price frame for stock tickers, constant USDEUR=X for FX tickers."""
    def _dl(tickers, start=None, end=None, **kwargs):
        if any(str(t).endswith("=X") for t in tickers):
            idx = pd.date_range("2020-01-01", "2030-01-01", name="Date")
            return pd.DataFrame({("Close", "USDEUR=X"): usd_eur}, index=idx)
        return prices
    return _dl


def _usd_ticker_stub():
    """yf.Ticker stub: reports USD and has no history (keeps tests off the network)."""
    t = Mock()
    t.fast_info = {"currency": "USD"}
    t.history.return_value = pd.DataFrame()
    return t


@patch('etl.extract.yf.Ticker', side_effect=lambda *_a, **_k: _usd_ticker_stub())
class TestStockPricesExtraction:
    """Test stock price extraction."""

    @patch('etl.extract.yf.download')
    def test_extract_stock_prices_basic(self, mock_download, _mock_ticker):
        """Test basic price extraction + USD→EUR normalisation."""
        mock_download.side_effect = _fake_download(_price_frame('2024-01-01', 3, [100.0, 110.0, 120.0]))

        tickers = {
            "AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}
        }

        result = extract_stock_prices(tickers, lookback_days=30)

        assert not result.empty
        assert 'ticker' in result.columns
        assert 'date' in result.columns
        assert 'close' in result.columns
        # Prices are stored in EUR: 100 USD * 0.9 = 90 EUR
        assert result['close'].tolist() == pytest.approx([90.0, 99.0, 108.0])

    @patch('etl.extract.yf.download')
    def test_extract_stock_prices_empty_response(self, mock_download, _mock_ticker):
        """Full refresh with no data at all is a hard error (pipeline must not swap)."""
        mock_download.return_value = pd.DataFrame()

        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}

        with pytest.raises(ValueError):
            extract_stock_prices(tickers, lookback_days=30)

    @patch('etl.extract.yf.download')
    def test_extract_stock_prices_empty_incremental(self, mock_download, _mock_ticker):
        """Incremental run with no new rows (weekend) returns empty without FX calls."""
        mock_download.return_value = pd.DataFrame()

        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        result = extract_stock_prices(tickers, watermarks={"AAPL": datetime(2024, 4, 25).date()})

        assert result.empty
        assert mock_download.call_count == 1  # price batch only — no FX download


class TestCompanyInfoExtraction:
    """Test company metadata extraction."""
    
    @patch('etl.extract.YQTicker')
    def test_extract_company_info_basic(self, mock_yq):
        """Test basic company info extraction."""
        # Mock yahooquery response
        mock_instance = Mock()
        mock_instance.all_modules = {
            "AAPL": {
                "summaryDetail": {"marketCap": 3000000000000, "trailingPE": 30},
                "assetProfile": {"sector": "Technology", "country": "US"},
                "financialData": {"totalRevenue": 400000000000},
                "price": {"shortName": "Apple Inc."}
            }
        }
        mock_yq.return_value = mock_instance
        
        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        
        result = extract_company_info(tickers)
        
        assert not result.empty
        assert 'ticker' in result.columns
        assert 'market_cap' in result.columns
        assert 'sector' in result.columns


class TestFinancialsExtraction:
    """Test financial statements extraction."""
    
    @patch('etl.extract.YQTicker')
    def test_extract_historical_financials(self, mock_yq):
        """Test annual financials extraction."""
        mock_instance = Mock()
        mock_df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'asOfDate': ['2023-12-31', '2022-12-31'],
            'TotalRevenue': [400000000000, 380000000000],
            'BasicEPS': [6.5, 6.0]
        })
        mock_instance.income_statement.return_value = mock_df
        mock_instance.balance_sheet.return_value = pd.DataFrame()
        mock_yq.return_value = mock_instance
        
        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        
        result = extract_historical_financials(tickers)
        
        # Should handle gracefully even with partial data
        assert isinstance(result, pd.DataFrame)


class TestCashflowExtraction:
    """Test cashflow extraction."""
    
    @patch('etl.extract.YQTicker')
    @patch('etl.extract.yf.download')
    def test_extract_cashflows_basic(self, mock_download, mock_yq):
        """Test cashflow extraction."""
        # Mock FX rates
        mock_download.return_value = pd.Series([1.0], index=pd.date_range('2024-01-01', periods=1))
        
        # Mock yahooquery cashflow
        mock_instance = Mock()
        mock_cf = pd.DataFrame({
            'RepurchaseOfCapitalStock': [-5000000000],
            'CashDividendsPaid': [-3000000000]
        }, index=[0])
        mock_instance.cash_flow.return_value = mock_cf
        mock_instance.summary_detail = {"AAPL": {"marketCap": 3000000000000}}
        mock_yq.return_value = mock_instance
        
        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        
        result = extract_cashflows(tickers)
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert 'ticker' in result.columns
            assert 'buyback_ttm' in result.columns


class TestEarningsExtraction:
    """Test earnings calendar and history extraction."""
    
    @patch('etl.extract.YQTicker')
    def test_extract_earnings_calendar(self, mock_yq):
        """Test earnings calendar extraction."""
        mock_instance = Mock()
        mock_instance.calendar_events = {
            "AAPL": {
                "earnings": {
                    "earningsDate": ["2024-04-30"],
                    "earningsAverage": 1.5,
                    "revenueAverage": 90000000000
                }
            }
        }
        mock_yq.return_value = mock_instance
        
        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        
        result = extract_earnings_calendar(tickers)
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert 'ticker' in result.columns
            assert 'earnings_date' in result.columns
    
    @patch('etl.extract.YQTicker')
    def test_extract_earnings_history(self, mock_yq):
        """Test earnings surprise history extraction."""
        mock_instance = Mock()
        mock_df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'quarter': ['2024-03-31', '2023-12-31'],
            'epsActual': [1.55, 1.50],
            'epsEstimate': [1.50, 1.45],
            'epsDifference': [0.05, 0.05],
            'surprisePercent': [3.3, 3.4]
        })
        mock_instance.earning_history = mock_df
        mock_yq.return_value = mock_instance
        
        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        
        result = extract_earnings_history(tickers)
        
        assert isinstance(result, pd.DataFrame)


class TestFCFExtraction:
    """Test Free Cash Flow extraction."""
    
    @patch('etl.extract.YQTicker')
    def test_extract_historical_fcf(self, mock_yq):
        """Test annual FCF extraction."""
        mock_instance = Mock()
        mock_df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'asOfDate': ['2023-12-31', '2022-12-31'],
            'FreeCashFlow': [100000000000, 95000000000],
            'OperatingCashFlow': [120000000000, 115000000000],
            'CapitalExpenditure': [-20000000000, -20000000000]
        })
        mock_instance.cash_flow.return_value = mock_df
        mock_yq.return_value = mock_instance
        
        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        
        result = extract_historical_fcf(tickers)
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert 'ticker' in result.columns
            assert 'free_cash_flow' in result.columns
    
    @patch('etl.extract.YQTicker')
    def test_extract_quarterly_fcf(self, mock_yq):
        """Test quarterly FCF extraction."""
        mock_instance = Mock()
        mock_df = pd.DataFrame({
            'symbol': ['AAPL'],
            'asOfDate': ['2024-03-31'],
            'FreeCashFlow': [25000000000],
            'OperatingCashFlow': [30000000000],
            'CapitalExpenditure': [-5000000000]
        })
        mock_instance.cash_flow.return_value = mock_df
        mock_yq.return_value = mock_instance
        
        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        
        result = extract_quarterly_fcf(tickers)
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert 'quarter' in result.columns


@patch('etl.extract.yf.Ticker', side_effect=lambda *_a, **_k: _usd_ticker_stub())
class TestIncrementalLoad:
    """Test incremental loading logic."""

    @patch('etl.extract.yf.download')
    def test_extract_stock_prices_with_watermarks(self, mock_download, _mock_ticker):
        """Incremental load starts 2 days before the watermark."""
        mock_download.side_effect = _fake_download(_price_frame('2024-04-28', 2, [103.0, 104.0]))

        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        watermarks = {"AAPL": datetime(2024, 4, 25).date()}

        result = extract_stock_prices(tickers, lookback_days=365, watermarks=watermarks)

        assert len(result) == 2
        price_call = mock_download.call_args_list[0]
        assert price_call.kwargs["start"] == datetime(2024, 4, 23)


class TestErrorHandling:
    """Test error handling and resilience."""

    @patch('etl.extract.yf.Ticker', side_effect=lambda *_a, **_k: _usd_ticker_stub())
    @patch('etl.extract.yf.download')
    def test_extract_handles_api_failure(self, mock_download, _mock_ticker):
        """Test graceful handling of API failures."""
        mock_download.side_effect = Exception("API Error")
        
        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        
        # Should not crash
        try:
            result = extract_stock_prices(tickers, lookback_days=30)
            # Either returns empty or raises ValueError
            assert isinstance(result, pd.DataFrame)
        except ValueError:
            # Acceptable behavior for complete failure
            pass
    
    @patch('etl.extract.YQTicker')
    def test_extract_company_info_handles_missing_data(self, mock_yq):
        """Test handling of missing company data."""
        mock_instance = Mock()
        mock_instance.all_modules = {"AAPL": None}  # Missing data
        mock_yq.return_value = mock_instance
        
        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        
        result = extract_company_info(tickers)
        
        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)


class _FakeYQ:
    """yahooquery 2.4.x statement signatures: balance_sheet has NO `trailing` parameter."""
    calls = []

    def __init__(self, symbols, asynchronous=False, session=None, **_kw):
        self._with_session = session is not None

    def _respond(self, name, frame):
        # Only the Pass-3 client (built with a session) gets real data; earlier passes get an error payload
        type(self).calls.append(name)
        return frame if self._with_session else {"AAPL": "No fundamentals data found for symbol: AAPL"}

    def income_statement(self, frequency='a', trailing=True):
        return self._respond("income_statement", pd.DataFrame({
            'symbol': ['AAPL'], 'asOfDate': ['2023-12-31'], 'currencyCode': ['USD'],
            'TotalRevenue': [400.0], 'BasicEPS': [6.5]}))

    def balance_sheet(self, frequency='a'):
        return self._respond("balance_sheet", pd.DataFrame({
            'symbol': ['AAPL'], 'asOfDate': ['2023-12-31'], 'currencyCode': ['USD'],
            'StockholdersEquity': [50.0]}))

    def cash_flow(self, frequency='a', trailing=True):
        return self._respond("cash_flow", pd.DataFrame({
            'symbol': ['AAPL'], 'asOfDate': ['2023-12-31'], 'currencyCode': ['USD'],
            'FreeCashFlow': [90.0], 'OperatingCashFlow': [110.0], 'CapitalExpenditure': [-20.0]}))


@pytest.fixture
def offline_extract():
    """No sleeps, no network FX, no real evasion session."""
    _FakeYQ.calls = []
    with patch('time.sleep'), \
         patch('etl.extract._backoff_sleep', return_value=0.0), \
         patch('etl.extract._make_evasion_session', return_value=(object(), {})), \
         patch('etl.extract.fx_to_eur', return_value=1.0):
        yield


TICKER = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}


@pytest.mark.usefixtures("offline_extract")
class TestStatementSignatureAndErrorPayloads:
    """yahooquery signature drift (balance_sheet(trailing=...)) and dict/str error payloads."""

    def test_fetch_statement_passes_trailing_only_when_supported(self):
        yq = _FakeYQ("AAPL", session=object())
        # Would raise TypeError if `trailing` were forwarded to balance_sheet
        assert not _fetch_statement(yq, 'balance_sheet', 'a').empty
        assert not _fetch_statement(yq, 'income_statement', 'q').empty
        assert not _fetch_statement(yq, 'cash_flow', 'a').empty

    def test_fetch_statement_forwards_trailing_false_when_supported(self):
        seen = {}

        def cash_flow(frequency='a', trailing=True):
            seen.update(frequency=frequency, trailing=trailing)
            return pd.DataFrame({'x': [1]})

        _fetch_statement(Mock(cash_flow=cash_flow), 'cash_flow', 'a')
        assert seen == {'frequency': 'a', 'trailing': False}

    @pytest.mark.parametrize("payload", [
        {"AAPL": "No fundamentals data found"},      # dict error payload
        "No fundamentals data found",                # str payload (str.index is a builtin method)
        None,
    ])
    def test_fetch_statement_non_dataframe_is_no_data(self, payload):
        yq = Mock()
        yq.balance_sheet = Mock(side_effect=lambda frequency='a': payload)
        out = _fetch_statement(yq, 'balance_sheet', 'a')
        assert isinstance(out, pd.DataFrame) and out.empty

    @patch('etl.extract.YQTicker', _FakeYQ)
    def test_pass3_historical_financials_new_signature(self, caplog):
        """Pass 3 must recover via the new signature instead of 'unexpected keyword argument trailing'."""
        with caplog.at_level("WARNING", logger="etl.extract"):
            result = extract_historical_financials(TICKER)
        assert "unexpected keyword argument" not in caplog.text
        assert "Pass 3 Error" not in caplog.text
        assert list(result["ticker"]) == ["AAPL"]
        assert result["revenue"].iloc[0] == 400.0

    @patch('etl.extract.YQTicker', _FakeYQ)
    def test_pass3_quarterly_financials_new_signature(self, caplog):
        with caplog.at_level("WARNING", logger="etl.extract"):
            result = extract_quarterly_financials(TICKER)
        assert "Pass 3 Error" not in caplog.text
        assert "AAPL" in set(result["ticker"])

    @patch('etl.extract.YQTicker')
    def test_dict_error_payload_financials_is_no_data(self, mock_yq, caplog):
        mock_yq.return_value = Mock(
            income_statement=Mock(return_value={"AAPL": "Unauthorized"}),
            balance_sheet=Mock(return_value={"AAPL": "Unauthorized"}))
        with caplog.at_level("WARNING", logger="etl.extract"):
            assert extract_historical_financials(TICKER).empty
            assert extract_quarterly_financials(TICKER).empty
        assert "Pass 3 Error" not in caplog.text

    @patch('etl.extract.YQTicker')
    def test_dict_error_payload_cashflow_does_not_crash(self, mock_yq, caplog):
        mock_yq.return_value = Mock(cash_flow=Mock(return_value={"AAPL": "No fundamentals data found"}))
        with caplog.at_level("WARNING", logger="etl.extract"):
            cf = extract_cashflows(TICKER)
            fcf = extract_historical_fcf(TICKER)
            qfcf = extract_quarterly_fcf(TICKER)
        assert cf.empty and fcf.empty and qfcf.empty
        assert "Cashflow fetch failed" not in caplog.text
        assert "has no attribute" not in caplog.text
        assert "Pass 3 Error" not in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
