# src/data_fetch.py
import pandas as pd
from pandas_datareader.data import DataReader
from tqdm import tqdm
from pathlib import Path

def download_spx_spy(start_date: str, end_date: str, output_path: str):
    """
    Download S&P 500 index (^SPX) and SPY ETF OHLCV from Stooq,
    then write a single Parquet at output_path.
    """
    tickers = {
        "^SPX": "^SPX",  # Stooq uses ^SPX for the S&P 500 index
        "SPY": "SPY"
    }
    parts = []

    for label, symbol in tqdm(tickers.items(), desc="Fetching from Stooq"):
        try:
            df = DataReader(symbol, "stooq", start=start_date, end=end_date)
        except Exception as e:
            tqdm.write(f"⚠️  Failed to fetch {label} ({symbol}): {e}")
            continue

        if df.empty:
            tqdm.write(f"⚠️  No data for {label}, skipping.")
            continue

        # Stooq returns descending dates—sort ascending
        df = df.sort_index()

        # Keep & lowercase the columns you need
        df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
        df["ticker"] = label
        df.index.name = "date"
        parts.append(df)

    if not parts:
        raise RuntimeError(
            "❌ No market data was fetched for SPX or SPY from Stooq."
        )

    # Concatenate, re-index, write out
    out = (
        pd.concat(parts)
          .reset_index()
          .set_index(["date", "ticker"])
    )
    # ensure parent dirs exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    print(f"✅ Market data saved to {output_path}")
