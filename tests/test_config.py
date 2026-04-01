# tests/test_config.py
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTickerConfig:
    """Test ticker configuration file."""
    
    def test_config_exists(self):
        """Verify config file exists."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'tickers.yaml'
        )
        assert os.path.exists(config_path), f"Config not found at {config_path}"
    
    def test_config_valid_yaml(self):
        """Verify config is valid YAML."""
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'tickers.yaml'
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert 'tickers' in config
        assert isinstance(config['tickers'], dict)
        assert len(config['tickers']) > 0
    
    def test_all_tickers_have_required_fields(self):
        """Verify all tickers have name, sector, region."""
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'tickers.yaml'
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        for ticker, data in config['tickers'].items():
            assert 'name' in data, f"{ticker} missing 'name'"
            assert 'sector' in data, f"{ticker} missing 'sector'"
            assert 'region' in data, f"{ticker} missing 'region'"
