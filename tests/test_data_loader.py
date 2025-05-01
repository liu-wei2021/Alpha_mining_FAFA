# tests/test_data_loader.py

import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_and_clean

def make_sample_df(tmp_path):
    """
    Build a tiny DataFrame with two tickers, three dates each,
    with open/high/low/close/volume columns.
    """
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    data = []
    for t in ["AAA", "BBB"]:
        for i, d in enumerate(dates):
            data.append({
                "date": d,
                "ticker": t,
                "open": 100 + i,
                "high": 105 + i,
                "low":  95 + i,
                "close":100 + 2*i,
                "volume": 1000 + 10*i,
                # deliberately leave out vwap so it gets computed
            })
    df = pd.DataFrame(data).set_index(["date","ticker"])
    fn = tmp_path / "sample.parquet"
    df.to_parquet(fn)
    return fn

def test_load_and_clean_creates_returns(tmp_path):
    path = make_sample_df(tmp_path)
    raw = pd.read_parquet(path)

    df = load_and_clean(str(path))

    # vwap computed, volume→vol, and returns added
    assert "vwap" in df.columns
    assert "vol"  in df.columns
    assert "returns" in df.columns

    # grab the second date for AAA from the raw sample
    raw_aaa = raw.xs("AAA", level="ticker")
    expected = raw_aaa["close"].pct_change().iloc[1]  # (102/100 - 1) = 0.02

    grp = df.xs("AAA", level="ticker")
    first_ret = grp["returns"].iloc[0]
    assert np.isclose(first_ret, expected)


def test_load_and_clean_keeps_existing_vwap(tmp_path):
    path = make_sample_df(tmp_path)
    raw = pd.read_parquet(path)
    raw["vwap"] = raw["high"] + raw["low"]
    raw.to_parquet(path)

    df = load_and_clean(str(path))

    # only compare on the rows that remain after clean()
    # i.e. raw.vwap.loc[df.index]
    assert (df["vwap"] == raw["vwap"].loc[df.index]).all()


def test_load_and_clean_fills_nans_and_drops_all_nan_cols(tmp_path):
    # make sample and then introduce a column of all NaN
    path = make_sample_df(tmp_path)
    raw = pd.read_parquet(path)
    raw["allnan"] = np.nan
    raw.to_parquet(path)
    df = load_and_clean(str(path))
    assert "allnan" not in df.columns
