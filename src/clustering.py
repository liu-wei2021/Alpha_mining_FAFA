import pandas as pd
from sklearn.cluster import KMeans

def cluster_factors(ic_df, n_clusters, random_state):
    # 1) pick only your alphaXXX columns
    factor_cols = [c for c in ic_df.columns if c.startswith("alpha")]
    # 2) transpose so each factor is a sample
    X = ic_df[factor_cols].fillna(0).T

    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)

    return pd.Series(labels, index=X.index, name="cluster")
