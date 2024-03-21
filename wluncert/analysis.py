import argparse
import os
import os.path
import time
from typing import List, Dict

import localflow as mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score

import arviz as az

from utils import get_date_time_uuid
from models import NumPyroRegressor


class ModelEvaluation:
    def __init__(self, predictions, test_list, default_ci_width=0.68):
        self.predictions_samples = None
        self.default_ci_width = default_ci_width
        self.predictions = predictions
        self.test_list = test_list
        self.eval_mape = False
        self.eval_mape_ci = False
        self.eval_R2 = False
        self.custom_pred_eval = None
        self.model_wise_dict = {}

    def get_test_list(self):
        return self.test_list

    def prepare_sample_modes(self):
        self.predictions_samples = self.predictions
        # for samples in self.predictions_samples:
        # if type(samples) == list:
        #     print(samples)
        #     print(self.predictions_samples)
        self.predictions = [
            NumPyroRegressor.get_mode_from_samples(samples.T)
            if samples is not None and len(samples) > 0
            else None
            for samples in self.predictions_samples
        ]

    @classmethod
    def mape_100(cls, y_true, predictions):
        flattened_y_true, reshaped_predictions = cls.get_clean_y_true_y_pred(
            predictions, y_true
        )
        return (
            mean_absolute_percentage_error(flattened_y_true, reshaped_predictions) * 100
        )

    @classmethod
    def get_clean_y_true_y_pred(cls, predictions, y_true):
        reshaped_predictions = np.atleast_2d(predictions).reshape(len(y_true), -1)
        flattened_y_true = np.array(y_true).ravel()
        return flattened_y_true, reshaped_predictions

    @classmethod
    def R2(cls, y_true, predictions):
        flattened_y_true, reshaped_predictions = cls.get_clean_y_true_y_pred(
            predictions, y_true
        )
        return r2_score(flattened_y_true, reshaped_predictions)

    def compute_ape(self, y_true, y_pred):
        return np.abs((y_true - y_pred) / y_true) * 100

    def mape_ci(self, y_true, y_pred, pred_arr, ci=None):
        ci = ci or self.default_ci_width
        intervals = az.hdi(pred_arr.T, hdi_prob=ci)  # .ravel()
        lowers, uppers = intervals[:, 0], intervals[:, 1]

        mask_y_true_in_lower_bound = lowers < y_true
        mask_y_true_in_upper_bound = y_true < uppers
        ape_cis = np.zeros_like(mask_y_true_in_lower_bound).astype(float)
        ape_cis[~mask_y_true_in_lower_bound] = self.compute_ape(
            y_true[~mask_y_true_in_lower_bound], lowers[~mask_y_true_in_lower_bound]
        )
        ape_cis[~mask_y_true_in_upper_bound] = self.compute_ape(
            y_true[~mask_y_true_in_upper_bound], uppers[~mask_y_true_in_upper_bound]
        )
        interval_widths = uppers - lowers
        relative_ci_widths = np.abs(interval_widths / y_pred)
        mean_width = relative_ci_widths.mean()
        mape = ape_cis.mean()
        return mape, mean_width

    def add_mape(self):
        self.eval_mape = True

    def add_custom_pred_eval(self, custom_pred_eval):
        self.custom_pred_eval = custom_pred_eval

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
        y_true_env_list = []
        col_names = ["err_type", "env", "err"]
        tups = []
        err_type_mape = "mape"
        err_type_r2 = "R2"
        err_type_mape_ci = "mape_ci"
        err_type_rel_ci_width = "rel_pred_ci_width"
        pred_has_samples = self.predictions_samples is not None
        for i, (test_set, y_pred) in enumerate(zip(self.test_list, self.predictions)):
            if test_set is None:
                continue
            y_true = test_set.get_y()
            merged_y_true.extend(y_true)
            y_true_env_list.append(y_true)
            env_id = test_set.env_id
            # pred_has_samples = len(y_pred.shape) > 1 and y_pred.shape[1] > 1
            if pred_has_samples:
                raw_samples = self.predictions_samples[i]
                # y_pred_samples = y_pred
                # y_pred = NumPyroRegressor.get_mode_from_samples(raw_samples.T)
                merged_predictions_samples.extend(raw_samples)
                if self.eval_mape_ci:
                    mape_ci, relative_ci_widths = self.mape_ci(y_true, y_pred, raw_samples)
                    tups.append((err_type_mape_ci, env_id, mape_ci))
                    tups.append((err_type_rel_ci_width, env_id, relative_ci_widths))
            merged_predictions.extend(y_pred)
            if self.eval_mape:
                mape = self.mape_100(y_true.ravel(), y_pred)
                tups.append((err_type_mape, env_id, mape))
            if self.eval_R2:
                r2 = self.R2(y_true, y_pred)
                tups.append((err_type_r2, env_id, r2))

        if self.custom_pred_eval is not None:
            self.custom_pred_eval(
                y_true_env_list,
                self.predictions_samples,
            )
        if pred_has_samples and self.eval_mape_ci:

            merged_err_mape_ci, relative_ci_widths = self.mape_ci(
                np.atleast_1d(merged_y_true).T,
                np.atleast_1d(merged_predictions),
                np.atleast_2d(merged_predictions_samples),
            )
            overall_tup_mape_ci = err_type_mape_ci, "overall", merged_err_mape_ci
            overall_tup_rel_ci_width = err_type_rel_ci_width, "overall", relative_ci_widths
            tups.append(overall_tup_mape_ci)
            tups.append(overall_tup_rel_ci_width)
            mlflow.log_metric("mape_ci_overall", merged_err_mape_ci)
            mlflow.log_metric("relative_ci_width_overall", relative_ci_widths)
        if self.eval_mape:
            merged_err_mape = self.mape_100(
                np.atleast_2d(merged_y_true).T, np.atleast_2d(merged_predictions).T
            )
            overall_tup_mape = err_type_mape, "overall", merged_err_mape
            mlflow.log_metric("mape_overall", merged_err_mape)
            tups.append(overall_tup_mape)
        if self.eval_R2:
            merged_err_R2 = self.R2(merged_y_true, merged_predictions)
            overall_tup_R2 = err_type_r2, "overall", merged_err_R2
            mlflow.log_metric("R2_overall", merged_err_R2)
            tups.append(overall_tup_R2)
        mlflow.log_metrics(self.model_wise_dict)
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
        self.output_base_path = os.path.join(
            self.results_base_path, f"{my_id}-analysis"
        )
        os.makedirs(self.output_base_path)
        print(f"plotting to {self.output_base_path}")

        self.score_df = pd.read_csv(scores_file_path)
        self.meta_df = pd.read_csv(meta_file_path)

        print("Done reading")
        self.err_type = "mape"

    def plot_errors(self, score_df=None, err_type=None):
        score_df = score_df or self.score_df
        if len(score_df["exp_id"].unique()) < 2:
            print("skipping errors because not enough training sets in data")
            return
        print("start plotting errors")
        # err_type = err_type or self.err_type
        for err_type in score_df["err_type"].unique():
            selected_error_df = score_df[score_df["err_type"] == err_type]
            sns.relplot(
                data=selected_error_df,
                x="exp_id",
                y="err",
                hue="model",
                col="env",
                kind="line",
                col_wrap=4,
            )  # row="setting", )
            plt.yscale("log")
            plt.suptitle(err_type)
            if "mape" in err_type:
                y_min, y_max = plt.ylim()
                new_y_max = min(y_max, 270)
                pass
                # plt.ylim((0, new_y_max))
            elif "R2" in err_type:
                pass
                # plt.ylim((-0.5, 1.1))
            multitask_file = os.path.join(
                self.output_base_path, f"multitask-result-{err_type}.png"
            )
            plt.savefig(multitask_file)
            plt.show()

        print(self.score_df)
        print("done")

    def plot_metadata(self, meta_df=None):
        print("start plotting metadata")
        meta_df = meta_df or self.meta_df
        # ignore some metrics
        meta_df = self.get_meta_df(meta_df)
        sns.relplot(
            data=meta_df,
            x="budget_abs",
            y="score",
            hue="model",
            col="metric",
            kind="line",
            col_wrap=4,
            facet_kws={"sharey": False, "sharex": True},
        )
        metadata_file = os.path.join(self.output_base_path, "metadata.png")
        plt.savefig(metadata_file)
        plt.show()
        # time.sleep(0.1)
        print("done")

    def get_meta_df(self, meta_df=None):
        meta_df = meta_df if meta_df is not None else self.meta_df
        meta_df = meta_df.drop(
            meta_df[meta_df["metric"].isin(["warning", "scale"])].index
        )
        meta_df["score"] = meta_df["score"].astype(float)
        return meta_df

    def run(self):
        self.plot_metadata()
        self.plot_errors()


def main():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument(
        "--results", type=str, help="path to results parquet file or similar"
    )
    args = parser.parse_args()
    results_base_path = args.results

    al = Analysis(results_base_path)
    al.run()

    # plot_metadata(meta_df, output_base_path)
    # plot_errors(err_type, output_base_path, score_df)


if __name__ == "__main__":
    main()
