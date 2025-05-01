import numpy as np, pandas as pd
from scipy.stats import rankdata

def ts_sum(df: pd.Series, window: int = 10) -> pd.Series:
    return df.rolling(window).sum()

def ts_rank(df: pd.Series, window: int = 10) -> pd.Series:
    return df.rolling(window)\
             .apply(lambda arr: rankdata(arr, method="min")[-1], raw=True)

def delta(df: pd.Series, period: int = 1) -> pd.Series:
    return df.diff(period)

def delay(df: pd.Series, period: int = 1) -> pd.Series:
    return df.shift(period)

def decay_linear(series: pd.Series, window: int = None, period: int = None) -> pd.Series:
    """
    Rolling linear‐decay over the last `window` bars (alias `period` for legacy calls).
    If there are fewer than `window` past values, weights are scaled to that shorter length.
    """
    # allow either window=… or period=… (period takes precedence)
    w = period if period is not None else window
    if w is None:
        w = 10

    def apply_decay(arr: np.ndarray) -> float:
        n = len(arr)
        # build weights 1/n, 2/n, …, n/n
        weights = np.arange(1, n + 1) / n
        return float(np.dot(arr, weights))

    return series.rolling(w, min_periods=1).apply(apply_decay, raw=True)



def ts_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).sum()

def ts_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).min()

def ts_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).max()

def ts_argmax(series: pd.Series, window: int) -> pd.Series:
    """For each date, index of the maximum within the last `window` bars."""
    return series.rolling(window, min_periods=1).apply(lambda x: np.argmax(x), raw=True)

def ts_argmin(series: pd.Series, window: int) -> pd.Series:
    """For each date, index of the minimum within the last `window` bars."""
    return series.rolling(window, min_periods=1).apply(lambda x: np.argmin(x), raw=True)
