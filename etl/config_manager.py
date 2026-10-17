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
        >>> config = load_config("etl_config")
        >>> config["extraction"]["batch_size"]
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


def get_etl_config() -> dict:
    """Pipeline configuration: config/etl_config.yaml merged over these defaults (section by section)."""
    default_config = {
        "extraction": {"max_workers": 8, "insiders": True},
        "incremental_load": {"lookback_days_full": 1825, "overlap_buffer_days": 2},
        "refresh_intervals": {"fundamentals_hours": 168, "metadata_hours": 168, "earnings_hours": 168, "insider_hours": 168},
        "price_integrity": {"drift_tolerance": 0.002, "rebase_every_days": 7, "min_rebase_ratio": 0.5},
        "universe": {"discovery_retention_days": 30},
        "lifecycle": {"inactive_after_days": 10, "unexplained_jump_pct": 40, "market_quiet_pct": 5},
        "provider": {"block_share": 0.3, "cooldown_base_s": 30, "cooldown_cap_s": 300, "cooldown_budget_s": 900,
                     "retry_pause_s": 1.5, "no_data_ttl_days": 14},
        "gates": {
            "min_universe_coverage": 0.90, "min_vs_previous_tickers": 0.95, "min_vs_previous_rows": 0.97,
            "max_stale_days": 5, "min_fresh_share": 0.90, "critical_fresh_share": 0.50, "min_compared": 20, "min_moved": 3,
            "max_jump_pct_systemic": 0.02,
            "continuity_move": 0.25, "mcap_ratio_bounds": [0.5, 2.0], "max_mcap_share": 0.05,
            "min_company_coverage": 0.80, "min_vs_previous_fundamentals": 0.95,
        },
        "reconcile": {"enabled": True, "price_tolerance_pct": 1.5, "fundamental_tolerance_pct": 3.0,
                      "sec_max_tickers_per_run": 60, "min_compared": 30, "warn_mismatch_share": 0.05,
                      "critical_mismatch_share": 0.25},
        "filings": {"enabled": True, "max_tickers_per_run": 40, "recheck_days": 7, "annual_reports": 3,
                    "earnings_releases": 4, "embed": True, "embed_model": "embed-v4.0", "embed_dimension": 512,
                    "rerank_model": "rerank-v3.5", "extract_leases": True, "embed_tokens_per_minute": 90000,
                    "max_lease_readings_per_run": 40, "llm_calls_per_minute": 18, "llm_model": "command-r-plus-08-2024",
                    "answer_model": "command-a-03-2025"},
        "swap": {"attempts": 60, "wait_seconds": 2.0},
        "data_quality": {"min_price": 0.01, "max_pe_ratio": 1000, "max_debt_ebitda": 50,
                         "min_market_cap": 1_000_000, "jump_pct": 60, "jump_systemic_tickers": 3},
    }

    config = load_config("etl_config")
    if config:
        for category, values in default_config.items():
            if isinstance(config.get(category), dict):
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
