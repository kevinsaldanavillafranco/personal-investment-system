import numpy as np
import pandas as pd

def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns from price data."""
    return prices.pct_change()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price data."""
    return np.log(prices / prices.shift(1))

def compute_cumulative_wealth(simple_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative wealth (growth factor) from simple returns."""
    return (1 + simple_returns).cumprod()

def compute_cumulative_wealth_from_log(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative wealth from log returns."""
    return np.exp(log_returns.cumsum())