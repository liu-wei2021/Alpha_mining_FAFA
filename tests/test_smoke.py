# tests/test_smoke.py
import numpy as np
import pandas as pd
from src.alpha_functions import AlphaFactory

def test_all_alphas_run_without_errors():
    # build a simple monotonic DataFrame
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    base  = np.arange(20)   # array([0,1,2,…,19])

    df = pd.DataFrame({
        "open":    1.00 + 0.01 * base,
        "high":    1.05 + 0.01 * base,
        "low":     0.95 + 0.01 * base,
        "close":   1.00 + 0.01 * base,
        "volume":  100   + base,
        "vwap":    1.00 + 0.01 * base,
        "returns": [0.0] * 20,
    }, index=dates)


    # just verify that everything executes and returns a Series
    factors = AlphaFactory.all_alphas(df)
    assert isinstance(factors, dict)
    # expect at least one alpha
    assert len(factors) > 0
    for name, series in factors.items():
        assert hasattr(series, "values")  # it’s a Series
        assert len(series) == len(df)      # same length
