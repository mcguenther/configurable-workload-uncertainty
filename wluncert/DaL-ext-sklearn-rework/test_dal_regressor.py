import unittest
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_percentage_error

from dal_regressor import DaLRegressor
from utils.general import recursive_dividing


def original_dal_predict(X_train, y_train, X_test):
    dt = DecisionTreeRegressor(
        random_state=0, criterion="squared_error", splitter="best"
    )
    dt.fit(X_train, y_train)
    tree_ = dt.tree_

    clusters = recursive_dividing(
        0,
        1,
        tree_,
        X_train,
        list(range(len(X_train))),
        max_depth=1,
        min_samples=4,
        cluster_indexes_all=[],
    )
    k = len(clusters)
    regressors = []
    for idx in clusters:
        reg = RandomForestRegressor(random_state=0)
        reg.fit(X_train[idx], y_train[idx])
        regressors.append(reg)

    total_index = np.concatenate(clusters)
    max_X = np.amax(X_train[total_index], axis=0)
    max_X[max_X == 0] = 1
    weights = mutual_info_regression(
        X_train[total_index], y_train[total_index], random_state=0
    )

    if k > 1:
        labels = []
        for i, idx in enumerate(clusters):
            labels.extend([i] * len(idx))
        labels = np.array(labels)
        X_smo = X_train[total_index] / max_X
        for j in range(X_train.shape[1]):
            X_smo[:, j] *= weights[j]
        clf = RandomForestClassifier(random_state=0)
        clf.fit(X_smo, labels)
    else:
        clf = None

    if k == 1:
        return regressors[0].predict(X_test)

    X_scaled = X_test / max_X
    for j in range(X_train.shape[1]):
        X_scaled[:, j] *= weights[j]
    clusters_test = clf.predict(X_scaled)
    preds = np.empty(X_test.shape[0])
    for i in range(k):
        idx = np.where(clusters_test == i)[0]
        if len(idx) == 0:
            continue
        preds[idx] = regressors[i].predict(X_test[idx])
    return preds


class TestDaLRegressor(unittest.TestCase):

    def test_dal_regressor_matches_original_with_RF_model(self):
        # Lade Daten
        data = np.genfromtxt("data/Lrzip.csv", delimiter=",", skip_header=1)
        X = data[:, :-1]
        y = data[:, -1]

        # Aufteilen
        X_train, y_train = X[:100], y[:100]
        X_test = X[100:120]

        # Originale Vorhersage
        preds_original = original_dal_predict(X_train, y_train, X_test)

        # Neue Vorhersage mit DaLRegressor
        model = DaLRegressor(
            depth=1,
            random_state=0,
            base_regressor=RandomForestRegressor(random_state=0),
        )
        model.fit(X_train, y_train)
        preds_new = model.predict(X_test)

        # Vergleich
        np.testing.assert_allclose(
            preds_original,
            preds_new,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Vorhersagen vom Original und DaLRegressor stimmen nicht überein.",
        )

    def test_dal_regressor_matches_original(self):
        # Lade Daten
        data = np.genfromtxt("data/Lrzip.csv", delimiter=",", skip_header=1)
        X = data[:, :-1]
        y = data[:, -1]

        # Aufteilen
        X_train, y_train = X[:100], y[:100]
        X_test = X[100:120]

        # Originale Vorhersage
        preds_original = original_dal_predict(X_train, y_train, X_test)

        # Neue Vorhersage mit DaLRegressor
        model = DaLRegressor(
            random_state=0,
        )
        model.fit(X_train, y_train)
        preds_new = model.predict(X_test)

        # Vergleich
        np.testing.assert_allclose(
            preds_original,
            preds_new,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Vorhersagen vom Original und DaLRegressor stimmen nicht überein.",
        )

    def test_dal_mape_vs_baselines(self):
        datasets = [
            "Lrzip.csv",
            "BDBC_AllNumeric.csv",
            "BDBJ_AllNumeric.csv",
            "nginx.csv",
        ]
        results = []
        for fname in datasets:
            data = np.genfromtxt(f"data/{fname}", delimiter=",", skip_header=1)
            X, y = data[:, :-1], data[:, -1]
            X_train, y_train = X[:100], y[:100]
            X_test, y_test = X[100:150], y[100:150]

            # model = DaLRegressor(
            #     depth=1,
            #     random_state=0,
            #     base_regressor=RandomForestRegressor(random_state=0),
            # )
            model = DaLRegressor(
                random_state=0,
            )
            model.fit(X_train, y_train)
            mape_dal = mean_absolute_percentage_error(y_test, model.predict(X_test))

            lasso = Lasso(alpha=0.001, random_state=0, max_iter=10000)
            lasso.fit(X_train, y_train)
            mape_lasso = mean_absolute_percentage_error(y_test, lasso.predict(X_test))

            rf = RandomForestRegressor(random_state=0)
            rf.fit(X_train, y_train)
            mape_rf = mean_absolute_percentage_error(y_test, rf.predict(X_test))

            system_results = {
                "dataset": fname,
                "DaL": mape_dal,
                "Lasso": mape_lasso,
                "RandomForest": mape_rf,
            }
            print(system_results)
            results.append(system_results)

            best_baseline = min(mape_lasso, mape_rf)
            self.assertLessEqual(
                mape_dal - best_baseline,
                0.25,
                msg=f"{fname}: DaL MAPE differs too much from best baseline",
            )

        df = pd.DataFrame(results)
        print("\nAggregated MAPEs:\n", df)


if __name__ == "__main__":
    unittest.main()
