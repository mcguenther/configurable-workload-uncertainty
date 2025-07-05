import unittest
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from wluncert.dal.utils.general import recursive_dividing

class TestRecursiveDividing(unittest.TestCase):
    def test_divisions_match_expected_samples(self):
        X = np.array([[0],[1],[2],[3],[4],[5]], dtype=float)
        y = np.array([0,0,1,1,1,1], dtype=float)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)
        clusters = recursive_dividing(
            0,
            1,
            dt.tree_,
            X,
            [1,2,4,5],
            max_depth=1,
            min_samples=0,
            cluster_indexes_all=None,
        )
        self.assertEqual(clusters, [[1], [2,4,5]])

    def test_no_empty_division_when_split_possible(self):
        X = np.arange(8).reshape(-1,1)
        y = np.arange(8)
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(X, y)
        clusters = recursive_dividing(
            0,
            1,
            dt.tree_,
            X,
            [1,2,3,6,7],
            max_depth=2,
            min_samples=0,
            cluster_indexes_all=None,
        )
        for cl in clusters:
            self.assertGreater(len(cl), 0)

if __name__ == '__main__':
    unittest.main()
