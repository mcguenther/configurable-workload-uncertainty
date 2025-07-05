import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from imblearn.over_sampling import SMOTE

from .utils.general import recursive_dividing

from .utils.HINNPerf_data_preproc import DataPreproc
from .utils.HINNPerf_model_runner import ModelRunner
from .utils.HINNPerf_models import MLPHierarchicalModel
from .utils.HINNPerf_args import list_of_param_dicts

def _depth_search_avg_hv(X, y, min_samples):
    """Return optimal depth using averaging hypervolume."""
    dt = DecisionTreeRegressor(
        random_state=0, criterion="squared_error", splitter="best"
    )
    dt.fit(X, y)
    tree_ = dt.tree_
    selected_depthes = range(1, tree_.max_depth + 1)
    points_divisions = []
    for max_depth in selected_depthes:
        clusters = recursive_dividing(
            0,
            1,
            tree_,
            X,
            list(range(len(X))),
            max_depth=max_depth,
            min_samples=min_samples,
            cluster_indexes_all=[],
        )
        pts = []
        for idx in clusters:
            cur_y = y[idx]
            sq_loss = np.mean((cur_y - cur_y.mean()) ** 2)
            pts.append([sq_loss, -len(idx)])
        points_divisions.append(pts)

    from pymoo.indicators.hv import HV

    ref_point_rate = 1.1
    max_loss = 0
    min_size = -len(y)
    for pts in points_divisions:
        if pts:
            temp_max = np.max(np.array(pts)[:, 0])
            temp_min = np.max(np.array(pts)[:, 1])
            max_loss = max(max_loss, temp_max)
            min_size = max(min_size, temp_min)
    ref_point = np.array(
        [max_loss * ref_point_rate, min_size * (1 - (ref_point_rate - 1))]
    )
    indicator = HV(ref_point=ref_point)

    best_depth = 1
    best_hv = -np.inf
    for depth, pts in zip(selected_depthes, points_divisions):
        if not pts:
            continue
        hv = np.mean([indicator(np.array(p)) for p in pts])
        if hv > best_hv:
            best_hv = hv
            best_depth = depth
    return best_depth


class _HINNPerfModel:
    """Minimal wrapper around MLPHierarchicalModel for training and prediction."""

    def __init__(self, config, model_cls, max_epoch=1000):
        self.config = config
        self.max_epoch = max_epoch
        self.model_cls = model_cls
        self.model = None
        self.scaler_ = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if self.config["gnorm"]:
            mean_X = np.mean(X, axis=0)
            std_X = np.std(X, axis=0)
            std_X[std_X == 0] = 1
            X_norm = (X - mean_X) / std_X
            mean_Y = np.mean(y)
            std_Y = np.std(y)
            if std_Y == 0:
                std_Y = 1
            Y_norm = (y - mean_Y) / std_Y
            self.scaler_ = {
                "gnorm": True,
                "mean_X": mean_X,
                "std_X": std_X,
                "mean_Y": mean_Y,
                "std_Y": std_Y,
            }
        else:
            max_X = np.amax(X, axis=0)
            max_X[max_X == 0] = 1
            X_norm = X / max_X
            max_Y = np.max(y) / 100
            if max_Y == 0:
                max_Y = 1
            Y_norm = y / max_Y
            self.scaler_ = {
                "gnorm": False,
                "max_X": max_X,
                "max_Y": max_Y,
            }

        self.model = self.model_cls(self.config)
        self.model.build_train()
        lr = self.config["lr"]
        decay = lr / 1000
        for epoch in range(1, self.max_epoch + 1):
            self.model.sess.run(
                [self.model.train_op],
                {self.model.X: X_norm, self.model.Y: Y_norm, self.model.lr: lr},
            )
            lr = lr * 1 / (1 + decay * epoch)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.scaler_["gnorm"]:
            X_norm = (X - self.scaler_["mean_X"]) / self.scaler_["std_X"]
        else:
            X_norm = X / self.scaler_["max_X"]
        y_pred = self.model.sess.run(self.model.output, {self.model.X: X_norm})
        if self.scaler_["gnorm"]:
            y_pred = y_pred * self.scaler_["std_Y"] + self.scaler_["mean_Y"]
        else:
            y_pred = y_pred * self.scaler_["max_Y"]
        return y_pred.ravel()

    def finalize(self):
        if self.model is not None:
            self.model.finalize()
            self.model = None


class DaLRegressor(BaseEstimator, RegressorMixin):
    """Divide-and-Learn regression model with scikit-learn interface.

    Parameters
    ----------
    depth : int or None, default=None
        Maximum depth for the decision tree used to split the data.
        If ``None`` the depth is determined automatically.
    base_regressor : object, default=None
        Regressor used for each division. When ``None`` a minimal
        ``HINNPerf`` model is trained.
    classifier : object, default=None
        Classifier used to assign unseen samples to a division.
        ``RandomForestClassifier`` is used when ``None``.
    min_samples_division : int, default=4
        Minimum number of samples for a leaf division.
    random_state : int or None, default=None
        Random seed used across the procedure.
    tune : bool, default=True
        When ``True`` perform hyperparameter tuning of the local models
        and the classifier via grid search.
    max_epoch : int, default=100
        Maximum number of epochs for ``HINNPerf`` models.
    """

    def __init__(
        self,
        depth=None,
        base_regressor=None,
        classifier=None,
        min_samples_division=4,
        random_state=None,
        tune=True,
        max_epoch=100,
    ):
        self.depth = depth
        self.base_regressor = base_regressor
        self.classifier = classifier
        self.min_samples_division = min_samples_division
        self.random_state = random_state
        self.tune = tune
        self.max_epoch = max_epoch

    def _init_models(self):
        self.use_hinnperf_ = False
        if self.base_regressor is None:
            self.use_hinnperf_ = True
            self.base_regressor_ = None
            self.DataPreproc = DataPreproc
            self.ModelRunner = ModelRunner
            self.MLPHierarchicalModel = MLPHierarchicalModel
            self.list_of_param_dicts = list_of_param_dicts
        else:
            self.base_regressor_ = clone(self.base_regressor)
        if self.classifier is None:
            self.classifier_ = RandomForestClassifier(
                random_state=self.random_state, criterion="gini"
            )
        else:
            self.classifier_ = clone(self.classifier)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._init_models()
        self.n_features_in_ = X.shape[1]

        if self.depth is None:
            self.depth = _depth_search_avg_hv(X, y, self.min_samples_division)

        dt = DecisionTreeRegressor(
            random_state=self.random_state, criterion="squared_error", splitter="best"
        )
        dt.fit(X, y)
        tree_ = dt.tree_

        clusters = recursive_dividing(
            0,
            1,
            tree_,
            X,
            list(range(len(X))),
            max_depth=self.depth,
            min_samples=self.min_samples_division,
            cluster_indexes_all=[],
        )
        self.k_ = len(clusters)
        self.cluster_models_ = []
        for idx in clusters:
            if self.use_hinnperf_:
                config = dict(
                    input_dim=self.n_features_in_,
                    num_neuron=128,
                    num_block=2,
                    num_layer_pb=2,
                    lamda=0.001,
                    linear=False,
                    gnorm=True,
                    lr=0.001,
                    decay=None,
                    verbose=False,
                )
                if self.tune:
                    init_config = dict(
                        input_dim=[self.n_features_in_],
                        num_neuron=[128],
                        num_block=[2, 3, 4],
                        num_layer_pb=[2, 3, 4],
                        lamda=[0.001, 0.01, 0.1, 1],
                        linear=[False],
                        gnorm=[True, False],
                        lr=[0.0001, 0.001, 0.01],
                        decay=[None],
                        verbose=[False],
                    )
                    config_list = self.list_of_param_dicts(init_config)
                    whole = np.concatenate([X, y[:, None]], axis=1)
                    data_gen = self.DataPreproc(whole, idx, idx)
                    runner = self.ModelRunner(
                        data_gen, self.MLPHierarchicalModel, max_epoch=self.max_epoch
                    )
                    best_err = float("inf")
                    best_conf = config
                    for conf in config_list:
                        _, err_val = runner.train(conf)
                        if err_val < best_err:
                            best_err = err_val
                            best_conf = conf
                    config = best_conf
                model = _HINNPerfModel(
                    config, self.MLPHierarchicalModel, max_epoch=self.max_epoch
                )
                model.fit(X[idx], y[idx])
            else:
                model = clone(self.base_regressor_)
                model.fit(X[idx], y[idx])
            self.cluster_models_.append(model)

        # classifier and scaling info
        total_index = np.concatenate(clusters)
        self.max_X_ = np.amax(X[total_index], axis=0)
        self.max_X_[self.max_X_ == 0] = 1
        self.weights_ = mutual_info_regression(
            X[total_index], y[total_index], random_state=self.random_state
        )
        if self.k_ > 1:
            labels = []
            for i, idx in enumerate(clusters):
                labels.extend([i] * len(idx))
            labels = np.array(labels)
            X_smo = X[total_index] / self.max_X_
            for j in range(self.n_features_in_):
                X_smo[:, j] *= self.weights_[j]
            enough_data = all(len(idx) >= 5 for idx in clusters)
            if enough_data:
                smo = SMOTE(random_state=1, k_neighbors=3)
                X_smo, labels = smo.fit_resample(X_smo, labels)
            if self.tune and enough_data:
                param = {"n_estimators": np.arange(10, 100, 10)}
                gridS = GridSearchCV(self.classifier_, param)
                gridS.fit(X_smo, labels)
                self.classifier_.set_params(**gridS.best_params_)
            self.classifier_.fit(X_smo, labels)
        else:
            self.classifier_ = None
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_models_")
        X = check_array(X)
        if self.k_ == 1:
            return self.cluster_models_[0].predict(X)
        X_scaled = X / self.max_X_
        for j in range(self.n_features_in_):
            X_scaled[:, j] *= self.weights_[j]
        clusters = self.classifier_.predict(X_scaled)
        preds = np.empty(X.shape[0])
        for i in range(self.k_):
            idx = np.where(clusters == i)[0]
            if len(idx) == 0:
                continue
            preds[idx] = self.cluster_models_[i].predict(X[idx])
        return preds
