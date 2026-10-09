"""
test_app.py — Tests for dashboard logic and data processing.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestDataLoading:
    """Test data loading and caching."""
    
    def test_load_data_returns_correct_structure(self):
        """Test that load_data returns expected tuple structure."""
        # This would require mocking the database connection
        # For now, we test the structure expectation
        pass


class TestScoreCalculation:
    """Test scoring integration in dashboard."""
    
    def test_scoring_degrades_gracefully_on_sparse_input(self):
        """Only a handful of columns: scores stay bounded and the stock is not scored as the worst case."""
        from core.scoring import score_universe
        df = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(8)], "sector": "Technology", "industry": "Software",
            "market_cap": 1e10, "pe_ratio": [30, 35, 12, 18, 25, 40, 22, 28],
            "operating_margin": [0.1, 0.2, 0.3, 0.15, 0.25, 0.05, 0.12, 0.18],
        })
        scores = score_universe(df)
        assert len(scores) == 8
        assert scores["quality"].between(0, 100).all() and scores["value"].between(0, 100).all()
        assert (scores["quality_coverage"] < 100).all()


def _yf_close_frame(columns, rows):
    """yf.download shape for several tickers: MultiIndex ('Close', ticker) columns."""
    idx = pd.date_range("2024-04-29", periods=len(rows), name="Date")
    return pd.DataFrame(rows, index=idx, columns=pd.MultiIndex.from_product([["Close"], columns]))


class TestMacroDataFetching:
    """Test macro data fetching and fallback."""

    @patch('services.market_data.get_forex_rates', return_value=0.9)
    @patch('services.market_data.yf.download')
    def test_fetch_macro_data_success(self, mock_download, _fx):
        tickers = ["SPY", "DX-Y.NYB", "^TNX", "^IRX", "^VIX", "CL=F", "GC=F"]
        mock_download.return_value = _yf_close_frame(
            tickers, [[500, 104, 4.5, 5.2, 15, 80, 2300], [505, 105, 4.6, 5.2, 16, 81, 2310]])

        from services.market_data import fetch_macro_data
        fetch_macro_data.clear()
        result = fetch_macro_data()

        assert result["VIX"]["val"] == 16 and result["VIX"]["pct"] == pytest.approx(100 / 15)
        assert result["SPY"]["val"] == pytest.approx(505 * 0.9)   # USD → EUR
        assert result["US10Y"]["chg"] == pytest.approx(0.1)


class TestCurrencyNormalization:
    """FX conversion, incl. the UK pence (GBp) vs pound (GBP) distinction."""

    @patch('services.market_data.yf.download')
    def test_get_forex_rates(self, mock_download):
        mock_download.return_value = pd.DataFrame(
            {"Close": [0.92]}, index=pd.date_range('2024-04-30', periods=1))
        from services.market_data import get_forex_rates
        get_forex_rates.clear()
        assert get_forex_rates(target="EUR", source="USD") == pytest.approx(0.92)

    @patch('services.market_data.yf.download')
    def test_gbp_pounds_are_not_treated_as_pence(self, mock_download):
        mock_download.return_value = pd.DataFrame(
            {"Close": [1.17]}, index=pd.date_range('2024-04-30', periods=1))
        from services.market_data import get_forex_rates
        get_forex_rates.clear()
        assert get_forex_rates(target="EUR", source="GBP") == pytest.approx(1.17)    # pounds
        assert get_forex_rates(target="EUR", source="GBp") == pytest.approx(0.0117)  # pence
        assert get_forex_rates(target="GBP", source="GBP") == 1.0


class TestSmartMoneyAnalysis:
    """Test Smart Money flow analysis."""
    
    def test_smart_money_with_valid_data(self):
        """Test Smart Money calculation with valid price/volume data."""
        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=150, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'price_close': np.cumsum(np.random.randn(150)) + 100,
            'volume': np.random.randint(1000000, 5000000, 150),
            'price_high': np.cumsum(np.random.randn(150)) + 102,
            'price_low': np.cumsum(np.random.randn(150)) + 98
        })
        
        from core.smart_money import get_sm_spirit_unified_v2
        
        result = get_sm_spirit_unified_v2(df)
        
        assert isinstance(result, dict)
        assert "signal" in result
        assert "strength" in result
        assert "layer" in result
        assert result["signal"] in ["ACCUMULATION", "DISTRIBUTION", "NEUTRAL"]
        assert 0 <= result["strength"] <= 100
        assert result["layer"] in ["DIVERGENCE", "TREND", "NONE"]
    
    def test_smart_money_with_insufficient_data(self):
        """Test Smart Money with insufficient data."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-04-01', periods=10),
            'price_close': [100] * 10,
            'volume': [1000000] * 10,
            'price_high': [102] * 10,
            'price_low': [98] * 10
        })
        
        from core.smart_money import get_sm_spirit_unified_v2
        
        result = get_sm_spirit_unified_v2(df)
        
        assert isinstance(result, dict)
        assert result["signal"] == "NEUTRAL"
        assert result["strength"] == 0
        assert result["layer"] == "NONE"


class TestRSICalculation:
    """Test RSI calculation."""
    
    def test_rsi_vectorized(self):
        """Test vectorized RSI calculation."""
        # Create sample price data
        df = pd.DataFrame({
            'price_close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
        })
        
        from core.indicators import get_rsi_vectorized
        
        rsi = get_rsi_vectorized(df)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(df)
        # RSI should be between 0 and 100
        assert all((rsi.isna()) | ((rsi >= 0) & (rsi <= 100)))


class TestTacticalMetrics:
    """Test tactical metrics calculation."""
    
    def test_tactical_metrics_calculation(self):
        """Support below price, resistance above, stop below support."""
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 1, 120))
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=120, freq='D'),
            'price_open': close, 'price_close': close,
            'price_high': close + 1.5, 'price_low': close - 1.5,
            'volume': rng.integers(1_000_000, 2_000_000, 120),
        })
        cur_p = float(close[-1])

        from core.levels import get_tactical_metrics

        result = get_tactical_metrics(df, cur_p, analyst_target=cur_p * 1.15)

        assert result['s1'] < cur_p < result['r1']
        assert result['stop_loss'] < result['s1']
        assert 0 <= result['rsi'] <= 100


# Higher = more bullish. Labels documented in compute_institutional_rating's docstring.
_ACTION_RANK = {"SELL": 0, "REDUCE": 1, "HOLD": 2, "BUY": 3, "STRONG BUY": 4}


def _rank(label: str) -> int:
    return next(v for k, v in sorted(_ACTION_RANK.items(), key=lambda kv: -len(kv[0])) if k in label.upper())


class TestInstitutionalRating:
    """Test institutional rating calculation."""

    def _rate(self, **kw):
        from core.rating import compute_institutional_rating
        base = dict(ai_score=50, ma_sig="NEUTRAL", latest_rsi=50, upside=0, pe_v=20, peg_v=1.5,
                    sector="Technology", w52_pos=50, rr=1.5)
        base.update(kw)
        return compute_institutional_rating(**base)

    def test_institutional_rating_strong_setup_beats_weak_setup(self):
        strong = self._rate(ai_score=85, ma_sig="STRONG BULL", latest_rsi=55, upside=25, pe_v=18,
                            peg_v=0.9, w52_pos=70, rr=3.0, sm_status="ACCUMULATION", sm_strength=80)
        weak = self._rate(ai_score=25, ma_sig="STRONG BEAR", latest_rsi=75, upside=-15, pe_v=60,
                          peg_v=4.0, w52_pos=10, rr=0.4, sm_status="DISTRIBUTION", sm_strength=80)
        assert {"action_label", "action_color"} <= set(strong)
        assert _rank(strong["action_label"]) >= _ACTION_RANK["BUY"]
        assert _rank(weak["action_label"]) <= _ACTION_RANK["REDUCE"]

    def test_strong_buy_requires_quality(self):
        """v15 rule: STRONG BUY needs AI Score >= 65, whatever the other pillars say."""
        r = self._rate(ai_score=50, ma_sig="STRONG BULL", latest_rsi=55, upside=40, pe_v=12,
                       peg_v=0.5, w52_pos=70, rr=4.0, sm_status="ACCUMULATION", sm_strength=90)
        assert "STRONG BUY" not in r["action_label"].upper()


class TestPortfolioManagement:
    """Test portfolio management functions."""
    
    def test_portfolio_metrics_calculation(self):
        """Test portfolio performance metrics."""
        # Create sample portfolio data
        portfolio_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'shares': [10, 5, 3],
            'avg_cost': [150, 300, 2500],
            'current_price': [180, 350, 2800]
        })
        
        # Calculate metrics
        portfolio_df['position_value'] = portfolio_df['shares'] * portfolio_df['current_price']
        portfolio_df['cost_basis'] = portfolio_df['shares'] * portfolio_df['avg_cost']
        portfolio_df['gain_loss'] = portfolio_df['position_value'] - portfolio_df['cost_basis']
        portfolio_df['gain_loss_pct'] = (portfolio_df['gain_loss'] / portfolio_df['cost_basis']) * 100
        
        assert all(portfolio_df['position_value'] > 0)
        assert len(portfolio_df) == 3


class TestDataQuality:
    """Test data quality checks."""
    
    def test_data_quality_warnings(self):
        """Test data quality warning generation."""
        # Create sample data with quality issues
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'INVALID'],
            'price_close': [180, 350, None],
            'market_cap': [3000000000000, 2500000000000, 100]
        })
        
        # Check for missing prices
        missing_prices = df[df['price_close'].isna()]
        assert len(missing_prices) > 0
        
        # Check for suspiciously low market caps
        low_mcap = df[df['market_cap'] < 1000000]
        assert len(low_mcap) > 0


class TestPerformanceOptimizations:
    """Test performance optimizations."""
    
    def test_memory_optimization(self):
        """Test DataFrame memory optimization."""
        from etl.performance_utils import optimize_dataframe_memory
        
        # Create large DataFrame
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 10000),
            'float_col': np.random.random(10000),
            'str_col': ['test'] * 10000
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = optimize_dataframe_memory(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Should reduce memory usage
        assert optimized_memory <= original_memory
    
    def test_scoring_scales_to_the_full_universe(self):
        """~800 tickers (the real universe size) are scored in a couple of seconds."""
        import time
        from core.scoring import score_universe
        n = 800
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "ticker": [f"TICK{i}" for i in range(n)], "company": "x",
            "sector": rng.choice(["Software", "Banks", "Retail", "Regulated Utilities"], n),
            "industry": rng.choice(["a", "b", "c", "d", "e", "f"], n),
            "market_cap": rng.uniform(1e9, 1e12, n), "pe_ratio": rng.uniform(8, 50, n),
            "forward_pe": rng.uniform(8, 40, n), "roe": rng.uniform(0.02, 0.4, n),
            "fcf_margin": rng.uniform(-5, 30, n), "free_cashflow": rng.uniform(-1e9, 5e10, n),
            "total_debt": rng.uniform(1e8, 1e11, n), "ebitda": rng.uniform(1e8, 1e11, n),
            "ev_to_ebitda": rng.uniform(5, 25, n), "peg_ratio": rng.uniform(0.4, 3, n),
        })
        start = time.time()
        scores = score_universe(df)
        assert time.time() - start < 10
        assert len(scores) == n and scores["quality"].between(0, 100).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
