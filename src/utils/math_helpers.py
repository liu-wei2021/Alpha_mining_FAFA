import numpy as np
import pandas as pd

def rolling_prod(df: pd.Series, window: int = 10) -> pd.Series:
    return df.rolling(window).apply(lambda arr: np.prod(arr), raw=True)
def product(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling product over `window` bars.
    If you want the product of the last `window` values at each date,
    use this function.  It returns a Series of the same length.
    """
    # apply numpy.prod to each rolling window
    return series.rolling(window, min_periods=1).apply(lambda x: np.prod(x), raw=True)