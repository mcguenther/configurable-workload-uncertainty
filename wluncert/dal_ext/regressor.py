import os
import sys
import numpy as np
from sklearn.base import BaseEstimator


class DaLExtRegressor(BaseEstimator):
    """Wrapper around the DaL-ext implementation."""

    def __init__(self, repo_path="DaL-ext", test_mode=True):
        self.repo_path = repo_path
        self.test_mode = test_mode
        self._data = None

    def _import_dal(self):
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)
            sys.path.append(os.path.join(self.repo_path, "utils"))
        import utils.runHINNPerf as dal_run

        return dal_run

    def fit(self, X, y):
        self._data = (X.reset_index(drop=True), np.array(y))
        return self

    def predict(self, X):
        dal_run = self._import_dal()
        X_train, y_train = self._data
        train_np = X_train.to_numpy(dtype=float)
        y_train_np = y_train.reshape(-1, 1).astype(float)
        test_np = X.to_numpy(dtype=float)
        fake_y = np.zeros((len(test_np), 1), dtype=float)
        whole = np.concatenate(
            [np.vstack([train_np, test_np]), np.vstack([y_train_np, fake_y])], axis=1
        )
        train_idx = list(range(len(train_np)))
        test_idx = list(range(len(train_np), len(train_np) + len(test_np)))
        _, preds = dal_run.get_HINNPerf_MRE_and_predictions(
            [whole, train_idx, test_idx, self.test_mode, []]
        )
        return np.array(preds)
