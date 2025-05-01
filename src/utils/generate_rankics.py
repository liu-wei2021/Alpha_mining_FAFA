# src/utils/generate_rankics.py

import json
from pathlib import Path
import pandas as pd

def write_rankic_jsons(ic_df: pd.DataFrame, formula_map: dict, out_dir: Path):
    """
    - ic_df: DataFrame, rows = dates, cols = factor names (e.g. alpha001, alpha002, …, plus maybe 'ticker')
    - formula_map: maps alphaXXX -> human formula
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) average IC per factor
    avg_ic = ic_df.mean(axis=0).fillna(0)
    avg_ic = avg_ic.to_dict()

    # 2) write the time-series JSON
    (out_dir / "time_series_rankic_all_factors.json") \
        .write_text(json.dumps(avg_ic, indent=2))
    print("Wrote", out_dir / "time_series_rankic_all_factors.json")

    # 3) build formula→IC, but only for your alpha factors
    #    and default to using the factor name if no formula is provided
    formula_ic = {}
    for f, ic in avg_ic.items():
        # skip any accidental extra columns
        if not f.startswith("alpha"):
            continue
        label = formula_map.get(f, f)
        formula_ic[label] = ic

    # 4) write the formulas JSON
    (out_dir / "formula_rankic.json") \
        .write_text(json.dumps(formula_ic, indent=2))
    print("Wrote", out_dir / "formula_rankic.json")
