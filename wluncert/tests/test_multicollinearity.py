import unittest
import pandas as pd
import numpy as np
import time
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "utils", Path(__file__).resolve().parents[1] / "utils.py"
)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
import sys
sys.modules["utils"] = utils
remove_multicollinearity = utils.remove_multicollinearity
remove_multicollinearity_limited = utils.remove_multicollinearity_limited


def make_large_df(n_groups=50, group_size=3, n_rows=200):
    rng = np.random.default_rng(0)
    data = {}
    for i in range(n_groups):
        choices = rng.integers(0, group_size, size=n_rows)
        for j in range(group_size):
            col_name = f"G{i}_{j}"
            data[col_name] = (choices == j).astype(int)
    return pd.DataFrame(data)


class TestRemoveMulticollinearity(unittest.TestCase):
    def _baseline_original(self, df):
        """Original implementation using nx.find_cliques."""
        from utils import _sample_df, factorize_if_not_int, _pairwise_exclusive_mask
        import networkx as nx

        nunique = df.nunique()
        drop_cols = list(nunique[nunique == 1].index)
        if "Unnamed: 0" in df.columns:
            drop_cols.append("Unnamed: 0")
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        if df.empty:
            return df

        df_sampled = _sample_df(df, max_rows=100)
        arr = factorize_if_not_int(df_sampled)
        features = list(df.columns)
        mask = _pairwise_exclusive_mask(arr)

        G = nx.Graph()
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if mask[i, j]:
                    G.add_edge(features[i], features[j])

        while True:
            cliques = list(nx.find_cliques(G))
            for clique in cliques:
                if df_sampled[list(clique)].sum(axis=1).eq(1).all():
                    to_drop = sorted(clique)[0]
                    df.drop(columns=[to_drop], inplace=True)
                    df_sampled.drop(columns=[to_drop], inplace=True)
                    G.remove_node(to_drop)
                    break
            else:
                break

        return df

    def test_large_dataframe_speed(self):
        df = make_large_df()

        t0 = time.time()
        limited = remove_multicollinearity_limited(df.copy())
        t_limited = time.time() - t0

        t0 = time.time()
        baseline = self._baseline_original(df.copy())
        t_baseline = time.time() - t0

        self.assertLess(limited.shape[1], df.shape[1])
        self.assertEqual(limited.shape[1], baseline.shape[1])
        self.assertLess(t_limited, t_baseline)
        self.assertLess(t_limited, 5)


if __name__ == "__main__":
    unittest.main()
