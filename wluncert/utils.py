from datetime import datetime
import pandas as pd
import numpy as np
import networkx as nx
import sys


def get_date_time_uuid():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _sample_df(df: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
    return df.sample(n=max_rows, random_state=0) if len(df) > max_rows else df


def _pairwise_exclusive_mask(arr: np.ndarray) -> np.ndarray:
    on_counts = arr.sum(axis=0)
    co_occurrence = arr.T @ arr
    return (co_occurrence == 0) & (on_counts[:, None] > 0) & (on_counts[None, :] > 0)


def factorize_if_not_int(df: pd.DataFrame) -> np.ndarray:
    df_factorized = df.copy()

    for col in df.columns:
        if not pd.api.types.is_integer_dtype(df[col]):
            # Factorize with NaNs filled as placeholder
            df_factorized[col] = pd.factorize(df[col].fillna("MISSING"))[0]
        else:
            # For integer columns, fill NaN with 0
            df_factorized[col] = df[col].fillna(0)

    return df_factorized.astype(int).to_numpy()


def remove_multicollinearity_limited(
    df: pd.DataFrame,
    sample_rows: int = 100,
    max_clique_size: int = 5,
    skip_entirely=True,
) -> pd.DataFrame:
    """Fast multicollinearity removal for very wide data frames.

    This heuristic drops one column per group of mutually exclusive features.
    It avoids the expensive clique detection used in the full implementation.
    """

    print("⚡ Fast multicollinearity removal...", flush=True)

    # Drop constant/index columns first
    nunique = df.nunique()
    drop_cols = list(nunique[nunique == 1].index)
    if "Unnamed: 0" in df.columns:
        drop_cols.append("Unnamed: 0")
    df = df.drop(columns=drop_cols, errors="ignore")
    if df.empty:
        return df
    if skip_entirely:
        return df

    df_sampled = _sample_df(df, max_rows=sample_rows)
    arr = factorize_if_not_int(df_sampled)

    n_features = arr.shape[1]
    parent = list(range(n_features))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Detect pairs of mutually exclusive features and union them
    for i in range(n_features):
        col_i = arr[:, i]
        for j in range(i + 1, n_features):
            col_j = arr[:, j]
            if np.all(col_i + col_j <= 1):
                union(i, j)

    comps = {}
    features = list(df.columns)
    for idx in range(n_features):
        root = find(idx)
        comps.setdefault(root, []).append(idx)

    cols_to_drop = [features[min(idxs)] for idxs in comps.values() if len(idxs) > 1]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


def remove_multicollinearity(df: pd.DataFrame, sample_rows: int = 100) -> pd.DataFrame:
    if len(df.columns) > 100:
        return remove_multicollinearity_limited(df, sample_rows=sample_rows)

    print("⏳ Removing multicollinearity...", end=" ")

    # Drop constant/index columns
    nunique = df.nunique()
    drop_cols = list(nunique[nunique == 1].index)
    if "Unnamed: 0" in df.columns:
        drop_cols.append("Unnamed: 0")
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    print(f"\n🧹 Dropped {len(drop_cols)} constant/index columns.", end=" ")

    if df.empty:
        print("⚠️ Empty DataFrame after cleanup. Skipping.")
        return df

    print("\n🔍 Sampling ...", end=" ", flush=True)
    df_sampled = _sample_df(df, max_rows=sample_rows)
    # arr = df_sampled.astype(int).to_numpy()
    print(" and computing exclusive masks...", end=" ", flush=True)
    arr = factorize_if_not_int(df_sampled)  # .astype(int)#.to_numpy()

    features = list(df.columns)
    mask = _pairwise_exclusive_mask(arr)
    print("Done.")

    print("🧱 Building exclusivity graph...", end=" ")
    G = nx.Graph()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if mask[i, j]:
                G.add_edge(features[i], features[j])
    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    print("🧹 Removing redundant cliques...", end=" ")
    removed = 0
    while True:
        cliques = list(nx.find_cliques(G))
        for clique in cliques:
            if df_sampled[list(clique)].sum(axis=1).eq(1).all():
                to_drop = sorted(clique)[0]
                df.drop(columns=[to_drop], inplace=True)
                df_sampled.drop(columns=[to_drop], inplace=True)
                G.remove_node(to_drop)
                removed += 1
                break
        else:
            break
    print(f"{removed} features removed.")

    print("✅ Done.")
    return df
