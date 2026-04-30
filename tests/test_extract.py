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
        assert _guess_currency("BARC.L") == "GBP"
        assert _guess_currency("BP.L") == "GBP"
    
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


class TestStockPricesExtraction:
    """Test stock price extraction."""
    
    @patch('etl.extract.yf.download')
    def test_extract_stock_prices_basic(self, mock_download):
        """Test basic price extraction."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        mock_download.return_value = mock_data
        
        tickers = {
            "AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}
        }
        
        result = extract_stock_prices(tickers, lookback_days=30)
        
        assert not result.empty
        assert 'ticker' in result.columns
        assert 'date' in result.columns
        assert 'close' in result.columns
    
    @patch('etl.extract.yf.download')
    def test_extract_stock_prices_empty_response(self, mock_download):
        """Test handling of empty API response."""
        mock_download.return_value = pd.DataFrame()
        
        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        
        # Should handle gracefully
        result = extract_stock_prices(tickers, lookback_days=30)
        assert result.empty or len(result) == 0


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


class TestIncrementalLoad:
    """Test incremental loading logic."""
    
    @patch('etl.extract.yf.download')
    def test_extract_stock_prices_with_watermarks(self, mock_download):
        """Test incremental load with watermarks."""
        mock_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [99, 100],
            'Close': [103, 104],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2024-04-28', periods=2))
        
        mock_download.return_value = mock_data
        
        tickers = {"AAPL": {"name": "Apple", "sector": "Technology", "region": "US"}}
        watermarks = {"AAPL": datetime(2024, 4, 25).date()}
        
        result = extract_stock_prices(tickers, lookback_days=365, watermarks=watermarks)
        
        # Should only fetch recent data
        assert isinstance(result, pd.DataFrame)


class TestErrorHandling:
    """Test error handling and resilience."""
    
    @patch('etl.extract.yf.download')
    def test_extract_handles_api_failure(self, mock_download):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
