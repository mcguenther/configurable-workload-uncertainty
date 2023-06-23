import argparse
import os
import os.path
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
from wluncert.utils import get_date_time_uuid
from wluncert.models import NumPyroRegressor
from dask.distributed import Client


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


class Analysis:
    def __init__(self, base_path):
        self.results_base_path = base_path
        print("Start reading data")
        self.idx = ["model", "env_id", "budget_abs", "rnd", "subject_system"]

        scores_file_path = os.path.join(self.results_base_path, "scores.csv")
        meta_file_path = os.path.join(self.results_base_path, "model-meta.csv")
        my_id = get_date_time_uuid()
        self.output_base_path = os.path.join(self.results_base_path, f"{my_id}-analysis")
        os.makedirs(self.output_base_path)

        self.score_df = pd.read_csv(scores_file_path)
        self.meta_df = pd.read_csv(meta_file_path)

        print("Done reading")
        self.err_type = "mape"

    def plot_errors(self, score_df=None, err_type=None):
        print("start plotting errors")
        score_df = score_df or self.score_df
        err_type = err_type or self.err_type
        selected_error_df = score_df[score_df["err_type"] == err_type]
        sns.relplot(data=selected_error_df, x="exp_id", y="err",
                    hue="model", col="env", kind="line", col_wrap=4, )  # row="setting", )
        # plt.yscale("log")
        plt.ylim((0, 50))
        multitask_file = os.path.join(self.output_base_path, "multitask-result.png")
        plt.savefig(multitask_file)
        plt.show()
        print("done")

    def plot_metadata(self, meta_df=None):
        print("start plotting metadata")
        meta_df = meta_df or self.meta_df
        # ignore some metrics
        meta_df = meta_df.drop(meta_df[meta_df['metric'].isin(["warning", "scale"])].index)
        meta_df["score"] = meta_df["score"].astype(float)
        sns.relplot(data=meta_df, x="budget_abs", y="score",
                    hue="model", col="metric", kind="line", col_wrap=4, facet_kws={'sharey': False, 'sharex': True})
        metadata_file = os.path.join(self.output_base_path, "metadata.png")
        plt.savefig(metadata_file)
        plt.show()
        # time.sleep(0.1)
        print("done")

    def run(self):
        self.plot_metadata()
        self.plot_errors()


def main():
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--results', type=str, help='path to results parquet file or similar')
    args = parser.parse_args()
    results_base_path = args.results

    al = Analysis(results_base_path)
    al.run()


    # plot_metadata(meta_df, output_base_path)
    # plot_errors(err_type, output_base_path, score_df)


if __name__ == "__main__":
    main()
