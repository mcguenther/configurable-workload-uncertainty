import argparse
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import dask.dataframe as dd
import dask
from dask.distributed import Client, LocalCluster

import arviz as az

from wluncert.data import SingleEnvData
from wluncert.models import NumPyroRegressor
from dask.distributed import Client


class Evaluation:
    def __init__(self, default_ci_width=0.5):
        self.default_ci_width = default_ci_width

    def get_errors(self, df: dd.DataFrame):
        pred_arr = np.atleast_2d(df["y_pred"].to_numpy()).T
        modes = float(NumPyroRegressor.get_mode_from_samples(pred_arr)[0])
        y_true = df["y_true"].iloc[0]  # np.array((df["y_true"]))[0]
        ape = self.compute_ape(y_true, modes)
        # R2 = self.R2(y_true, modes)
        mape_ci = self.mape_ci(y_true, pred_arr)

        series = pd.Series([y_true, ape, mape_ci])
        return series

    def scalar_accuracy(self, test_sets: List[SingleEnvData], predictions):
        merged_y_true = []
        merged_predictions = []
        merged_predictions_samples = []

        col_names = ["err_type", "env", "err"]
        tups = []
        err_type_mape = "mape"
        err_type_r2 = "R2"
        for test_set, y_pred in zip(test_sets, predictions):
            y_true = test_set.get_y()
            merged_y_true.extend(y_true)
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred_samples = y_pred
                y_pred = NumPyroRegressor.get_mode_from_samples(y_pred_samples.T)
                merged_predictions_samples.extend(y_pred_samples)
                mape_ci = self.mape_ci(y_true, y_pred_samples)
            merged_predictions.extend(y_pred)
            mape = Evaluation.mape_100(y_true, y_pred)

            r2 = Evaluation.R2(y_true, y_pred)
            env_id = test_set.env_id
            tups.append((err_type_mape, env_id, mape))
            tups.append((err_type_r2, env_id, r2))
        merged_err_mape_ci = self.mape_ci(np.atleast_2d(merged_y_true).T, np.atleast_2d(merged_predictions).T)
        merged_err_mape = Evaluation.mape_100(np.atleast_2d(merged_y_true).T, np.atleast_2d(merged_predictions).T)
        merged_err_R2 = Evaluation.R2(merged_y_true, merged_predictions)
        overall_tup_mape = err_type_mape, "overall", merged_err_mape
        overall_tup_R2 = err_type_r2, "overall", merged_err_R2
        tups.append(overall_tup_mape)
        tups.append(overall_tup_R2)
        df = pd.DataFrame(tups, columns=col_names)
        return df

    # def scalar_accuracy(self, test_sets: List[SingleEnvData], predictions):
    def scalar_accuracy_on_dask(self, df_dd: dd.DataFrame):
        df_dd = df_dd.drop(columns=["exp_id"], errors="ignore")
        # get back to pandas because pivot_table is awkward in dask
        # df = df_dd.compute()
        merged_y_true = []
        merged_predictions = []
        col_names = "err_type", "env", "err"
        tups = []
        err_type_mape = "mape"
        err_type_r2 = "R2"
        df_index = ["model", "env_id", "budget_abs", "rnd", "subject_system", "testing_sample_idx"]
        # debug_df = df.sample(frac=0.01)
        # mcmc_rows_mask = np.array(debug_df["model"].str.contains("mcmc"))
        # debug_df_samples = debug_df[mcmc_rows_mask]
        # debug_df_samples = debug_df.iloc[mcmc_rows_mask, :]
        # debug_df_scalar = debug_df[~mcmc_rows_mask]
        # debug_df_scalar = debug_df.iloc[~mcmc_rows_mask, :]
        # debug_df_samples["index-combined"] = debug_df_samples["model"].astype(str) + debug_df_samples["env_id"].astype(str) + debug_df_samples["budget_abs"].astype(str) + debug_df_samples["rnd"].astype(str) + debug_df_samples["subject_system"].astype(str) + debug_df_samples["testing_sample_idx"].astype(str)
        # print("starting pandas")
        # pivot_table_samples = debug_df_samples.pivot_table(values=["y_true", "y_pred"],
        #                                            index=["model", "env_id", "budget_abs", "rnd", "subject_system",
        #                                                   "testing_sample_idx"], aggfunc=self.enforce_scalar)
        # pivot_table_scalar = debug_df_scalar.pivot_table(values=["y_true", "y_pred"],
        #                                            index=["model", "env_id", "budget_abs", "rnd", "subject_system",
        #                                                   "testing_sample_idx"])
        # combined_pivot_table = pd.concat([pivot_table_samples, pivot_table_scalar])
        # print(combined_pivot_table)

        # debug_df.set_index(df_index)
        # prediction_groups  = df.groupby(by=df_index)
        # debug_df = df.sample(200_000)
        # print(debug_df.head())
        # modes = prediction_groups.aggregate(self.enforce_scalar, )
        print("starting dask")
        t_start = time.time()
        df_result = df_dd.groupby(by=df_index).apply(lambda df: self.get_errors(df)).compute()
        print(f"Aggregating samples took {time.time() - t_start}s")
        print("done dask")
        print(df_result)
        predictions = list(np.array(df_result))
        return df_result


def main():
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--results', type=str, help='path to results parquet file or similar')
    args = parser.parse_args()
    results = args.results
    client = Client(n_workers=10, threads_per_worker=2, processes=True, memory_limit='auto')
    print(client)

    print("Start reading data")
    idx = ["model", "env_id", "budget_abs", "rnd", "subject_system", "testing_sample_idx"]

    # df = dd.read_parquet(results)
    df = dd.read_parquet(results)  # index=idx,) #split_row_groups='adaptive', calculate_divisions=True)
    print("Done reading")
    err_type = "mape"
    # err_type = "R2"
    eval = Evaluation()
    err_df = eval.scalar_accuracy_on_dask(df)

    selected_error_df = df[df["err_type"] == err_type]
    sns.relplot(data=selected_error_df, x="exp_id", y="err",
                hue="model", col="env", kind="line", col_wrap=4, )  # row="setting", )
    # plt.yscale("log")
    plt.ylim((0, 50))
    plt.savefig("./results/multitask-result.png", )
    plt.show()


if __name__ == "__main__":
    main()


class ModelEvaluation:
    def __init__(self, predictions, test_list, train_list, meta_df, default_ci_width=0.5):
        self.predictions_samples = None
        self.default_ci_width = default_ci_width
        self.predictions = predictions
        self.test_list = test_list
        self.train_list = train_list
        self.meta_df = meta_df
        self.eval_mape = False
        self.eval_mape_ci = False
        self.eval_R2 = False
        self.model_wise_dict = {}
        # env_id = train_data.get_env_id()
        # df["env_id"] = env_id

    def prepare_sample_modes(self):
        self.predictions_samples = self.predictions
        self.predictions = [NumPyroRegressor.get_mode_from_samples(samples.T) for samples in self.predictions_samples]

    @classmethod
    def mape_100(cls, y_true, predictions):
        flattened_y_true, reshaped_predictions = cls.get_clean_y_true_y_pred(predictions, y_true)
        return mean_absolute_percentage_error(flattened_y_true, reshaped_predictions) * 100

    @classmethod
    def get_clean_y_true_y_pred(cls, predictions, y_true):
        reshaped_predictions = np.atleast_2d(predictions).reshape(len(y_true), -1)
        flattened_y_true = np.array(y_true).ravel()
        return flattened_y_true, reshaped_predictions

    @classmethod
    def R2(cls, y_true, predictions):
        flattened_y_true, reshaped_predictions = cls.get_clean_y_true_y_pred(predictions, y_true)
        return r2_score(flattened_y_true, reshaped_predictions)

    def compute_ape(self, y_true, y_pred):
        return np.abs(y_true - y_pred / y_true)

    def mape_ci(self, y_true, pred_arr, ci=None):
        ci = ci or self.default_ci_width
        interval = az.hdi(pred_arr, hdi_prob=ci).ravel()
        lower, upper = interval[0], interval[1]

        if lower < y_true < upper:
            err = 0
        elif y_true > upper:
            err = self.compute_ape(y_true, upper)
        else:
            err = self.compute_ape(y_true, lower)
        return err

    def add_mape(self):
        self.eval_mape = True

    def add_R2(self):
        self.eval_R2 = True

    def add_mape_CI(self):
        self.eval_mape_ci = True


    def add_custom_model_dict(self, d: Dict):
        self.model_wise_dict = d

    def get_metadata(self):
        df = pd.DataFrame(self.model_wise_dict.items(), columns=["metric", "score"])
        return df

    def get_scores(self):
        merged_y_true = []
        merged_predictions = []
        merged_predictions_samples = []

        col_names = ["err_type", "env", "err"]
        tups = []
        err_type_mape = "mape"
        err_type_r2 = "R2"
        err_type_mape_ci = "mape_ci"
        pred_has_samples = self.predictions_samples is not None
        for test_set, y_pred in zip(self.test_list, self.predictions):
            y_true = test_set.get_y()
            merged_y_true.extend(y_true)
            env_id = test_set.env_id
            pred_has_samples = len(y_pred.shape) > 1 and y_pred.shape[1] > 1
            if pred_has_samples:
                y_pred_samples = y_pred
                y_pred = NumPyroRegressor.get_mode_from_samples(y_pred_samples.T)
                merged_predictions_samples.extend(y_pred_samples)
                mape_ci = self.mape_ci(y_true, y_pred_samples)
                tups.append((err_type_mape_ci, env_id, mape_ci))
            merged_predictions.extend(y_pred)
            mape = self.mape_100(y_true.ravel(), y_pred)

            r2 = self.R2(y_true, y_pred)
            tups.append((err_type_mape, env_id, mape))
            tups.append((err_type_r2, env_id, r2))
        if pred_has_samples:
            merged_err_mape_ci = self.mape_ci(np.atleast_2d(merged_y_true).T, np.atleast_2d(merged_predictions).T)
            overall_tup_mape_ci = err_type_mape_ci, "overall", merged_err_mape_ci
            tups.append(overall_tup_mape_ci)
        merged_err_mape = self.mape_100(np.atleast_2d(merged_y_true).T, np.atleast_2d(merged_predictions).T)
        merged_err_R2 = self.R2(merged_y_true, merged_predictions)
        overall_tup_mape = err_type_mape, "overall", merged_err_mape
        overall_tup_R2 = err_type_r2, "overall", merged_err_R2
        tups.append(overall_tup_mape)
        tups.append(overall_tup_R2)
        df = pd.DataFrame(tups, columns=col_names)
        return df
