# src/utils/generate_rankics.py

import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..alpha_functions import AlphaFactory  # for calling alphaXXX methods
from .ts_functions import (
    ts_rank, delta, ts_sum, ts_min, ts_max, decay_linear, ts_argmax,
    ts_argmin, delay
)
from .stat_helpers import (
    stddev, correlation, rank, sma, scale, sign, covariance
)
from .math_helpers import product


def write_rankic_jsons(ic_df: pd.DataFrame, formula_map: dict, out_dir: Path):
    """
    - ic_df: DataFrame, rows = dates, cols = factor names (e.g. alpha001, alpha002, …)
    - formula_map: maps alphaXXX -> human formula
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) average IC per factor, drop NaNs and guard small samples
    avg_ic = ic_df.mean(axis=0).fillna(0)
    avg_ic_dict = avg_ic.to_dict()

    # 2) write the time-series JSON
    (out_dir / "time_series_rankic_all_factors.json").write_text(
        json.dumps(avg_ic_dict, indent=2), encoding='utf-8'
    )
    print("Wrote", out_dir / "time_series_rankic_all_factors.json")

    # 3) build formula->IC map for alpha factors only
    formula_ic = {}
    for f, ic in avg_ic_dict.items():
        if not f.startswith("alpha"):
            continue
        label = formula_map.get(f, f)
        formula_ic[label] = ic

    # 4) write the formulas JSON
    (out_dir / "formula_rankic.json").write_text(
        json.dumps(formula_ic, indent=2), encoding='utf-8'
    )
    print("Wrote", out_dir / "formula_rankic.json")


def factor_series_fn(df: pd.DataFrame, expr: str) -> pd.Series:
    """
    Evaluate an Alpha101 expression or factory method name against df.
    """
    # 1) direct call for alphaXXX methods
    if hasattr(AlphaFactory, expr):
        return getattr(AlphaFactory, expr)(df)

    # 2) prepare safe namespace
    func_ns = {
        "ts_rank": ts_rank, "delta": delta, "ts_sum": ts_sum,
        "ts_min": ts_min, "ts_max": ts_max, "decay_linear": decay_linear,
        "ts_argmax": ts_argmax, "ts_argmin": ts_argmin, "delay": delay,
        "stddev": stddev, "correlation": correlation, "rank": rank,
        "sma": sma, "scale": scale, "sign": sign, "covariance": covariance,
        "product": product, "np": np
    }
    # inject all DataFrame columns
    local_ns = {col: df[col] for col in df.columns}

    # detect advXX patterns and add moving-average series
    adv_windows = {int(w) for w in
                   re.findall(r"adv(\d+)", expr)}
    for w in adv_windows:
        local_ns[f"adv{w}"] = df["volume"].rolling(w).mean()

    # 3) evaluate
    try:
        result = eval(expr, func_ns, local_ns)
    except NameError as e:
        raise ValueError(f"Unknown name in expression {expr!r}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to eval expression {expr!r}: {e}")

    # 4) ensure pd.Series
    if not isinstance(result, pd.Series):
        raise TypeError(f"Expression did not return a Series: got {type(result)}")
    return result


def compute_rankic(series: pd.Series, returns: pd.Series) -> float:
    """
    Compute Spearman rank-IC between a factor series and the return series.
    """
    dfv = pd.concat([series, returns], axis=1).dropna()
    # need at least 2 points
    if dfv.shape[0] < 2:
        return 0.0
    # compute rank correlation
    ic = spearmanr(dfv.iloc[:, 0], dfv.iloc[:, 1]).correlation
    return float(ic) if not np.isnan(ic) else 0.0
