import pandas as pd
from scipy.stats import spearmanr

def compute_ic_matrix(
    exposures: pd.DataFrame,
    returns: pd.DataFrame
) -> pd.DataFrame:
    """
    exposures: MultiIndex [date,ticker] × factors
    returns:   MultiIndex [date,ticker] × returns
    """
    dates = exposures.index.get_level_values("date").unique()
    ic = {}
    for factor in exposures.columns:
        ic_vals = []
        for d in dates:
            x = exposures.xs(d, level="date")[factor]
            y = returns.xs(d, level="date")["returns"]
            ic_vals.append(spearmanr(x, y)[0])
        ic[factor] = ic_vals
    return pd.DataFrame(ic, index=dates)
