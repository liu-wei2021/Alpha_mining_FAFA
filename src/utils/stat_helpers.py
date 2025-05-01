import pandas as pd
import numpy as np
from typing import Union


def stddev(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation (population, ddof=0) over `window` bars."""
    return series.rolling(window, min_periods=1).std(ddof=0)

def correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation between `x` and `y` over `window` bars."""
    return x.rolling(window, min_periods=1).corr(y)

def covariance(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling covariance between `x` and `y` over `window` bars."""
    return x.rolling(window, min_periods=1).cov(y)

def rank(series: pd.Series) -> pd.Series:
    """
    Cross-sectional rank of a Series (1 = smallest, N = largest),
    then scaled to [0,1]. If you just want raw 1…N, remove the final division.
    """
    r = series.rank(method="average", na_option="keep")
    return (r - 1) / (r.max() - 1)

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over `window` bars."""
    return series.rolling(window, min_periods=1).mean()

def scale(series: pd.Series) -> pd.Series:
    """Z-score: subtract mean, divide by std (sample std, ddof=0)."""
    return (series - series.mean()) / series.std(ddof=0)

def sign(series: pd.Series) -> pd.Series:
    """Sign of each element: +1, 0, or -1."""
    return series.apply(np.sign)
