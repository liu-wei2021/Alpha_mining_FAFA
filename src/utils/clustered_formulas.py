import json
import pandas as pd
from pathlib import Path

def write_clustered_formulas(
    cluster_series: pd.Series,
    formula_map: dict[str,str],
    ic_map: dict[str,float],
    out_dir: Path
):
    """
    Produce JSON of the form:
    {
      "Cluster_0": [ "formulaA", "formulaB", … ],
      "Cluster_1": [ … ],
      …
    }
    where within each cluster the formulas are sorted by descending |IC|.
    """
    # Build DataFrame
    df = (
        cluster_series
          .rename("cluster")
          .reset_index()  # index was factor name
          .rename(columns={"index":"factor"})
    )
    df["formula"] = df["factor"].map(formula_map)
    df["ic"]      = df["factor"].map(ic_map)
    df["abs_ic"]  = df["ic"].abs()

    # For each cluster, grab its formulas sorted by abs_ic desc
    out: dict[str,list[str]] = {}
    for cluster_label, group in df.groupby("cluster"):
        # sort descending by absolute IC
        sorted_formulas = (
            group
              .sort_values("abs_ic", ascending=False)
              ["formula"]
              .tolist()
        )
        out[f"Cluster_{cluster_label}"] = sorted_formulas

    # write JSON
    dest = out_dir / "clustered_formulas.json"
    Path(dest).write_text(json.dumps(out, indent=2))
    print(f"✅ Wrote clustered formulas to {dest}")
