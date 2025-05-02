import logging
import argparse
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data_loader import load_and_clean
from src.alpha_functions import AlphaFactory
from src.factor_matrix import compute_ic_matrix
from src.clustering import cluster_factors
from src.data_fetch import download_spx_spy
from src.utils.generate_rankics import write_rankic_jsons  # ← JSON helper
from src.constants.formula_map import FORMULA_MAP as formula_map
import json
from src.deepseek_agent import run as alpha_mine  # DeepSeek integration


from src.utils.clustered_formulas import write_clustered_formulas

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fetch-data", action="store_true",
        help="Download raw market data (SPX & SPY) before running the pipeline"
    )
    p.add_argument(
        "--output-dir", "-o",
        default="outputs",
        help="Directory where JSON rank-IC files and clusters.csv will be written"
    )
    p.add_argument(
        "--alpha-mine", action="store_true",
        help="Run DeepSeek alpha-mining iterations after clustering"
    )
    p.add_argument(
        "--iters", type=int, default=50,
        help="Number of DeepSeek mining iterations"
    )
    args = p.parse_args()

    if args.fetch_data:
        download_spx_spy(
            start_date="2015-01-01",
            end_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
            output_path=settings.input_parquet
        )
        print("Downloaded data to", settings.input_parquet)
        return

    setup_logging()
    #out_dir = Path(args.output_dir)
    out_dir = Path(args.output_dir or settings.outputs_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    logging.info("Loading & cleaning data…")
    df = load_and_clean(settings.input_parquet)

    logging.info("Generating factor exposures…")
    ex_list, ret_list = [], []
    for ticker, grp in df.groupby("ticker"):
        alphas = AlphaFactory.all_alphas(grp)
        ex_list.append(
            pd.DataFrame(alphas, index=grp.index)
              .assign(ticker=ticker)
        )
        ret_list.append(
            grp[["returns"]]
              .assign(ticker=ticker)
        )

    #exposures = pd.concat(ex_list).set_index(["date", "ticker"])
    #returns   = pd.concat(ret_list).set_index(["date", "ticker"])

    # exposures & returns already have a MultiIndex (date, ticker) from grp.index
    exposures = pd.concat(ex_list)
    returns   = pd.concat(ret_list)

    logging.info("Computing IC matrix…")
    ic_df = compute_ic_matrix(exposures, returns)

    # ─── write your two key JSON files ────────────────
    logging.info("Writing RankIC JSONs…")
    write_rankic_jsons(ic_df, formula_map, out_dir)

    logging.info("Clustering factors…")
    clusters = cluster_factors(
        ic_df,
        settings.n_clusters,
        settings.random_state
    )

    # save clusters to CSV
    clusters.to_csv(out_dir / "factor_clusters.csv", header=True)
    logging.info("Wrote factor_clusters.csv & JSON files to %s", out_dir)

    ic_map = json.load(open(out_dir / "time_series_rankic_all_factors.json"))
    write_clustered_formulas(clusters, formula_map, ic_map, out_dir)

    if args.alpha_mine:
        logging.info("Starting DeepSeek alpha-mining (%d iterations)…", args.iters)
        alpha_mine(num_iters=args.iters)

if __name__ == "__main__":
    main()
