import os
import pandas as pd
import numpy as np
import pytest

from src.data_fetch import download_spx_spy

@pytest.fixture
def dummy_ohlcv():
    # five days of made-up OHLCV
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Open":   np.arange(5) + 1.0,
        "High":   np.arange(5) + 1.5,
        "Low":    np.arange(5) + 0.5,
        "Close":  np.arange(5) + 1.1,
        "Volume": np.arange(5) * 10 + 100,
    }, index=idx)
    return df

def test_download_spx_spy_writes_file_and_returns_df(tmp_path, monkeypatch, dummy_ohlcv):
    # 1) monkeypatch yfinance.download inside our data_fetch module:
    def fake_download(ticker, start, end):
        # ignore ticker, start, end, always return our dummy OHLCV
        return dummy_ohlcv.copy()
    monkeypatch.setattr("src.data_fetch.yf.download", fake_download)

    # 2) call download_spx_spy, pointing it at a tmp file
    out_file = tmp_path / "out.parquet"
    df = download_spx_spy(
        start_date="2021-01-01",
        end_date="2021-01-06",
        output_path=str(out_file)
    )

    # 3) it should have written the parquet
    assert out_file.exists(), "Parquet file was not created"

    # 4) returned DataFrame should contain all tickers & columns
    #    we patched to download two tickers by default (^GSPC, SPY)
    #    so shape = (5 days × 2 tickers =) 10 rows
    assert len(df) == 10

    # must have lower-cased OHLCV + our computed VWAP
    for col in ("open","high","low","close","volume","vwap"):
        assert col in df.columns

    # must have at least one of our ADV columns (e.g. adv5)
    assert "adv5" in df.columns
    # and adv5 for each ticker = rolling‐mean of int volume, so all >= 100
    assert (df["adv5"] >= 100).all()
