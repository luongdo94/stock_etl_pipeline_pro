"""
performance_utils.py — Performance optimization utilities.
Vectorized operations and caching strategies.
"""
import pandas as pd
import numpy as np
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def get_cached_config(config_name: str):
    """
    Cached config loading to avoid repeated file I/O.
    
    Args:
        config_name: Name of config file
        
    Returns:
        Configuration dictionary
    """
    from etl.config_manager import load_config
    return load_config(config_name)


def batch_process_dataframe(
    df: pd.DataFrame,
    process_func,
    batch_size: int = 1000,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Process large DataFrame in batches to reduce memory usage.
    
    Args:
        df: Input DataFrame
        process_func: Function to apply to each batch
        batch_size: Number of rows per batch
        show_progress: Whether to log progress
        
    Returns:
        Processed DataFrame
    """
    results = []
    n_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        result = process_func(batch)
        results.append(result)
        
        if show_progress and (i // batch_size) % 10 == 0:
            logger.info(f"Processed batch {i // batch_size + 1}/{n_batches}")
    
    return pd.concat(results, ignore_index=True)


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Optimized DataFrame
        
    Performance:
        - Can reduce memory usage by 50-70%
        - Especially effective for large price history DataFrames
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if str(col_type)[:3] == 'int':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    logger.info(f"Memory optimized: {start_mem:.2f}MB → {end_mem:.2f}MB ({reduction:.1f}% reduction)")
    
    return df
