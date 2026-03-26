import numpy as np
import pandas as pd


def compute_variance(returns: pd.Series) -> float:
    """
    Compute the variance of a return series.

    Parameters
    ----------
    returns : pd.Series
        Series of returns.

    Returns
    -------
    float
        Variance of the returns.
    """
    if returns.empty:
        raise ValueError("Returns series is empty")
    mean_return = returns.mean()
    squared_diff = (returns - mean_return) ** 2
    return float(squared_diff.mean())


def compute_volatility(returns: pd.Series) -> float:
    """
    Compute the volatility of a return series.

    Parameters
    ----------
    returns : pd.Series
        Series of returns.

    Returns
    -------
    float
        Volatility of the returns.
    """
    if returns.empty:
        raise ValueError("Returns series is empty")
    variance = compute_variance(returns)
    return float(np.sqrt(variance))