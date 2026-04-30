"""
config_manager.py — Centralized configuration management.
Eliminates hardcoded business logic and magic numbers.
"""
import yaml
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

_CONFIG_CACHE: Dict[str, Any] = {}


def load_config(config_name: str, reload: bool = False) -> dict:
    """
    Load configuration from YAML file with caching.
    
    Args:
        config_name: Name of config file (without .yaml extension)
        reload: Force reload from disk
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = load_config("scoring_rules")
        >>> pe_threshold = config["valuation"]["pe_good"]
    """
    if config_name in _CONFIG_CACHE and not reload:
        return _CONFIG_CACHE[config_name]
    
    config_path = Path(__file__).parent.parent / "config" / f"{config_name}.yaml"
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        _CONFIG_CACHE[config_name] = config
        logger.info(f"Loaded config: {config_name}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config {config_name}: {e}")
        return {}


def get_scoring_config() -> dict:
    """Get scoring rules configuration with defaults."""
    default_config = {
        "valuation": {
            "pe_excellent": 15,
            "pe_good": 20,
            "pe_fair": 35,
            "pe_poor": 50,
            "peg_excellent": 0.8,
            "peg_good": 1.5,
            "peg_fair": 2.5,
            "pb_excellent_tech": 3.5,
            "pb_excellent_value": 1.0,
            "pb_good": 6.0,
            "pb_fair": 10.0,
        },
        "profitability": {
            "fcf_margin_excellent": 20,
            "fcf_margin_good": 12,
            "fcf_margin_fair": 5,
            "roe_excellent": 0.30,
            "roe_good": 0.18,
            "roe_fair": 0.10,
            "roe_poor": 0.05,
        },
        "financial_health": {
            "debt_ebitda_excellent": 2.5,
            "debt_ebitda_good": 4.5,
            "debt_ebitda_fair": 7.0,
            "debt_ebitda_poor": 12.0,
            "debt_ebitda_critical": 12.0,
        },
        "momentum": {
            "rsi_oversold": 30,
            "rsi_neutral_low": 35,
            "rsi_neutral_high": 60,
            "rsi_overbought": 70,
            "z_score_deep_value": -2.0,
            "z_score_fair": 0.0,
            "z_score_expensive": 1.8,
            "z_score_bubble": 3.0,
        },
        "growth": {
            "revenue_growth_excellent": 0.50,
            "revenue_growth_good": 0.15,
            "revenue_growth_fair": 0.05,
            "earnings_growth_excellent": 0.50,
            "earnings_growth_good": 0.20,
            "earnings_growth_fair": 0.10,
        },
        "red_flags": {
            "negative_pe_early_stage": -3,
            "negative_pe_high_growth": -5,
            "negative_pe_stagnant": -12,
            "high_debt_moderate": -10,
            "high_debt_critical": -15,
            "value_trap": -5,
        },
        "sector_adjustments": {
            "tech_profitability_cap": 30,
            "tech_payout_cap": 5,
            "financial_pb_low": 0.5,
            "financial_pb_ideal_low": 1.0,
            "financial_pb_ideal_high": 1.8,
        }
    }
    
    config = load_config("scoring_rules")
    
    # Merge with defaults (config overrides defaults)
    if config:
        for category, values in default_config.items():
            if category in config:
                values.update(config[category])
    
    return default_config


def get_etl_config() -> dict:
    """Get ETL pipeline configuration with defaults."""
    default_config = {
        "extraction": {
            "batch_size": 40,
            "max_workers": 8,
            "retry_attempts": 3,
            "backoff_base": 2.0,
            "request_delay": 1.0,
        },
        "incremental_load": {
            "lookback_days_full": 1825,  # 5 years
            "lookback_days_incremental": 7,
            "overlap_buffer_days": 2,
        },
        "refresh_intervals": {
            "prices_hours": 0,  # Always refresh
            "fundamentals_hours": 168,  # 7 days
            "metadata_hours": 168,  # 7 days (reduced from 720)
            "earnings_hours": 168,  # 7 days
        },
        "coverage_thresholds": {
            "metadata_min_pct": 0.95,
            "fundamentals_min_pct": 0.90,
            "earnings_min_pct": 0.95,
        },
        "data_quality": {
            "min_price": 0.01,
            "max_pe_ratio": 1000,
            "max_debt_ebitda": 50,
            "min_market_cap": 1_000_000,
        }
    }
    
    config = load_config("etl_config")
    
    if config:
        for category, values in default_config.items():
            if category in config:
                values.update(config[category])
    
    return default_config


def get_api_config() -> dict:
    """Get API configuration (rate limits, timeouts, etc.)."""
    default_config = {
        "yahoo_finance": {
            "rate_limit_per_minute": 2000,
            "timeout_seconds": 30,
            "max_retries": 3,
        },
        "yahooquery": {
            "rate_limit_per_minute": 1000,
            "timeout_seconds": 45,
            "max_retries": 3,
        },
        "cohere": {
            "rate_limit_per_minute": 100,
            "timeout_seconds": 60,
            "max_tokens": 1000,
        }
    }
    
    config = load_config("api_config")
    
    if config:
        for service, values in default_config.items():
            if service in config:
                values.update(config[service])
    
    return default_config
