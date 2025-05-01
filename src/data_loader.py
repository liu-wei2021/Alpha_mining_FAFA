import pandas as pd

def load_and_clean(path: str) -> pd.DataFrame:
    """
    1) Read raw Parquet (indexed by date,ticker).
    2) Lower-case & standardize columns.
    3) Compute VWAP if missing.
    4) Rename volume → vol.
    5) Forward/back-fill missing data.
    6) Compute pct-change returns per ticker and drop the NaNs.
    """
    df = pd.read_parquet(path)

    # 1) Normalization
    df.columns = [c.lower() for c in df.columns]
    if "vwap" not in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    #df = df.rename(columns={"volume": "vol"})
    df["vol"] = df["volume"]

     # compute and drop on returns
    df["returns"] = df.groupby("ticker")["close"].pct_change()
    df = df.dropna(subset=["returns"])

    # 4) Drop the first row of each ticker (which will have NaN returns)
    df = df.dropna(axis=1, how="all").ffill().bfill()

    return df
