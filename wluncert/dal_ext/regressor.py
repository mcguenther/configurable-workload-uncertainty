import os
import sys
import tempfile
from typing import List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression


class DaLExtRegressor(BaseEstimator):
    """Wrapper around the DaL-ext implementation."""

    def __init__(
        self,
        repo_path: str = "DaL-ext",
        test_mode: bool = True,
        min_samples_division: int = 4,
        seed: int = 2,
        max_epoch: int = 2000,
    ) -> None:
        self.repo_path = repo_path
        self.test_mode = test_mode
        self.min_samples_division = min_samples_division
        self.seed = seed
        self.max_epoch = max_epoch

        self._data = None
        self._forest: RandomForestClassifier | None = None
        self._cluster_models: List[tuple] = []

    def _recursive_dividing(
        self,
        node: int,
        depth: int,
        tree,
        X: np.ndarray,
        samples: List[int],
        max_depth: int,
    ) -> List[List[int]]:
        """Recursively divide samples according to the trained tree."""
        from sklearn.tree import _tree

        cluster_indexes_all: List[List[int]] = []

        if depth <= max_depth:
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                left_samples: List[int] = []
                right_samples: List[int] = []
                name = tree.feature[node]
                threshold = tree.threshold[node]
                for i_sample in range(len(samples)):
                    if X[i_sample, name] <= threshold:
                        left_samples.append(samples[i_sample])
                    else:
                        right_samples.append(samples[i_sample])
                if (
                    len(left_samples) <= self.min_samples_division
                    or len(right_samples) <= self.min_samples_division
                ):
                    cluster_indexes_all.append(samples)
                else:
                    cluster_indexes_all.extend(
                        self._recursive_dividing(
                            tree.children_left[node],
                            depth + 1,
                            tree,
                            X,
                            left_samples,
                            max_depth,
                        )
                    )
                    cluster_indexes_all.extend(
                        self._recursive_dividing(
                            tree.children_right[node],
                            depth + 1,
                            tree,
                            X,
                            right_samples,
                            max_depth,
                        )
                    )
            else:
                cluster_indexes_all.append(samples)
        else:
            cluster_indexes_all.append(samples)

        return cluster_indexes_all

    def _import_dal(self):
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)
            sys.path.append(os.path.join(self.repo_path, "utils"))
        import utils.runHINNPerf as dal_run
        from utils.adapting_depth import get_depth_AvgHV
        from utils.HINNPerf_model_runner import ModelRunner
        from utils.HINNPerf_data_preproc import DataPreproc
        from utils.HINNPerf_models import MLPHierarchicalModel

        return dal_run, get_depth_AvgHV, ModelRunner, DataPreproc, MLPHierarchicalModel

    def fit(self, X, y):
        dal_run, get_depth_AvgHV, ModelRunner, DataPreproc, MLPHierarchicalModel = (
            self._import_dal()
        )

        X_train = X.reset_index(drop=True).to_numpy(dtype=float)
        y_train = np.array(y, dtype=float).reshape(-1, 1)
        self._data = (X_train, y_train)

        whole = np.concatenate([X_train, y_train], axis=1)
        train_idx = list(range(len(X_train)))

        # adapt depth by creating a temporary csv for the util function
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            np.savetxt(f.name, whole, delimiter=",")
            max_depth = get_depth_AvgHV(
                f.name, len(train_idx), 0, self.seed, self.min_samples_division
            )
        os.unlink(f.name)

        dt = DecisionTreeRegressor(
            random_state=self.seed, splitter="best", criterion="squared_error"
        )
        dt.fit(X_train, y_train)
        tree_ = dt.tree_

        cluster_indexes = self._recursive_dividing(
            0, 1, tree_, X_train, train_idx, max_depth
        )

        # prepare classifier and local models
        k = len(cluster_indexes)
        total_index = []
        cluster_labels = []
        for i, idxs in enumerate(cluster_indexes):
            total_index.extend(idxs)
            cluster_labels.extend([i] * len(idxs))

        total_index = np.array(total_index)
        cluster_labels = np.array(cluster_labels)

        weights = mutual_info_regression(
            X_train[total_index], y_train[total_index, 0], random_state=0
        )
        max_X = np.amax(X_train[total_index], axis=0)
        max_X[max_X == 0] = 1
        X_scaled = X_train[total_index] / max_X
        X_scaled = X_scaled * weights

        forest = RandomForestClassifier(random_state=self.seed, criterion="gini")
        forest.fit(X_scaled, cluster_labels)
        self._forest = forest

        self._cluster_models = []
        for idxs in cluster_indexes:
            data_gen = DataPreproc(whole, idxs, idxs)
            best_config = dal_run.get_HINNPerf_best_config(
                [whole, idxs, idxs, self.test_mode, []]
            )
            runner = ModelRunner(
                data_gen, MLPHierarchicalModel, max_epoch=self.max_epoch
            )
            runner.test(best_config)
            self._cluster_models.append((runner, best_config))

        return self

    def predict(self, X):
        if self._forest is None:
            raise RuntimeError("Model has not been fitted.")

        dal_run, _, ModelRunner, DataPreproc, MLPHierarchicalModel = self._import_dal()
        X_train, y_train = self._data
        test_np = X.to_numpy(dtype=float)
        preds = np.zeros(len(test_np))

        # scale using training max and weights used during training
        max_X = np.amax(X_train, axis=0)
        max_X[max_X == 0] = 1
        weights = mutual_info_regression(X_train, y_train.ravel(), random_state=0)

        for i, row in enumerate(test_np):
            temp_X = (row / max_X) * weights
            cluster = int(self._forest.predict(temp_X.reshape(1, -1))[0])
            runner, best_config = self._cluster_models[cluster]
            # use DataPreproc to create temporary dataset for single prediction
            whole = np.concatenate([X_train, y_train], axis=1)
            temp_index = list(range(len(X_train))) + [len(X_train)]
            whole = np.vstack([whole, np.concatenate([row, [0.0]])])
            data_gen = DataPreproc(whole, list(range(len(X_train))), [len(X_train)])
            runner = ModelRunner(
                data_gen, MLPHierarchicalModel, max_epoch=self.max_epoch
            )
            pred, _ = dal_run.get_HINNPerf_MRE_and_predictions(
                [
                    whole,
                    list(range(len(X_train))),
                    [len(X_train)],
                    self.test_mode,
                    best_config,
                ]
            )
            preds[i] = pred[-1]

        return preds
