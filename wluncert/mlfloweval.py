import argparse
import os
import os.path
import time
from typing import List, Dict

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import json
import arviz as az

from utils import get_date_time_uuid
from models import NumPyroRegressor
from dataclasses import dataclass

RESULTS_EXP = "jdorn-multilevel-eval"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


class Evaluation:
    def __init__(self, parent_run, tracking_url, experiment_name=None):
        self.experiment_name = experiment_name or "jdorn-tmp"
        self.parent_run = parent_run
        self.tracking_url = tracking_url
        self.csv_path = "mlfloweval-last.csv"

        mlflow.set_tracking_uri(self.tracking_url)
        self.idx = ["model", "env_id", "budget_abs", "rnd", "subject_system"]
        experiment = mlflow.search_experiments(
            filter_string=("attribute.name = '%s'" % self.experiment_name)
        )[0]
        experiment_id = experiment.experiment_id
        self.experiment_id = experiment_id
        #
        # self.output_base_path = os.path.join(
        #     self.results_base_path, f"{my_id}-analysis"
        # )
        # os.makedirs(self.output_base_path)
        # print(f"plotting to {self.output_base_path}")

    def plot_errors(self):
        df = pd.read_csv(self.csv_path)
        print(df)

        # large_init_train_df = df.loc[df["params.relative_train_size"] == 5.0]
        metric = "metrics.mape_overall"

        sns.relplot(
            data=df,
            x="params.loo_budget_rel",
            y=metric,
            col="params.loo_idx",
            kind="line",
            hue="params.model",
            style="params.pooling_cat",
            row="params.relative_train_size",  # col_wrap=4,
        )

        # plt.yscale("log")

        # Get the current y-axis limits
        current_ylim = plt.ylim()
        # Check if the current ymax is above 200
        # if current_ylim[1] > 200:
        #     # Set the ymax to 200
        #     plt.ylim(current_ylim[0], 200)
        plt.ylim(0, 10)
        plt.tight_layout()
        fig_pdf_path = "lastplot-errors.pdf"
        plt.savefig(fig_pdf_path)

        mlflow.log_figure(plt.gcf(), "errors.png")
        mlflow.log_artifact(fig_pdf_path)
        plt.show()

        # score_df = score_df or self.score_df
        # if len(score_df["exp_id"].unique()) < 2:
        #     print("skipping errors because not enough training sets in data")
        #     return
        # print("start plotting errors")
        # # err_type = err_type or self.err_type
        # for err_type in score_df["err_type"].unique():
        #     selected_error_df = score_df[score_df["err_type"] == err_type]
        #     sns.relplot(
        #         data=selected_error_df,
        #         x="exp_id",
        #         y="err",
        #         hue="model",
        #         col="env",
        #         kind="line",
        #         col_wrap=4,
        #     )  # row="setting", )
        #     plt.yscale("log")
        #     plt.suptitle(err_type)
        #     if "mape" in err_type:
        #         y_min, y_max = plt.ylim()
        #         new_y_max = min(y_max, 270)
        #         pass
        #         # plt.ylim((0, new_y_max))
        #     elif "R2" in err_type:
        #         pass
        #         # plt.ylim((-0.5, 1.1))
        #     multitask_file = os.path.join(
        #         self.output_base_path, f"multitask-result-{err_type}.png"
        #     )
        #     plt.savefig(multitask_file)
        #     plt.show()
        #
        # print(self.score_df)
        # print("done")

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

    def get_params_dict(self, child_run):
        filter_str = "param"
        return self.get_filtered_row_dict(child_run, filter_str)

    def get_scores_dict(self, child_run):
        filter_str = "metric"
        return self.get_filtered_row_dict(child_run, filter_str)

    def get_filtered_row_dict(self, child_run, filter_str):
        return dict(
            child_run[[col for col in dict(child_run) if col.startswith(filter_str)]]
        )

    def run(self):
        mlflow.set_experiment(experiment_name=RESULTS_EXP)
        # mlflow.set_experiment(experiment_name=self.experiment_name)
        with mlflow.start_run(
            run_name="aggregation"  # self.experiment_name.replace(" ", ""),
        ):
            # Initialize an empty list to store the data
            data_list = []

            # Fetch the parent run and its children
            # parent_run = mlflow.get_run(self.parent_run)
            child_runs = self.get_sub_runs(self.parent_run)

            all_runs_in_experiment = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"status='FINISHED'",
            )

            # Iterate over the first-level nested runs
            for (row_id, child_run) in child_runs.iterrows():
                lvl_1_run_id = child_run["run_id"]
                model = child_run["params.model"]
                print(f"fetching new model {model}")
                lvl_1_params = self.get_params_dict(child_run)
                # Fetch the second-level nested runs for each environment
                relative_transfer_budgets = lvl_1_params["params.transfer_budgets"]
                relative_transfer_budgets = json.loads(relative_transfer_budgets)

                env_runs = self.get_sub_runs(lvl_1_run_id)
                for (_, env_run) in env_runs.iterrows():
                    env_idx = env_run["params.loo_idx"]
                    print(f"getting budget runs for env idx {env_idx}")
                    lvl_2_params = self.get_params_dict(env_run)
                    lvl_2_run_id = env_run["run_id"]
                    lvl_3_child_runs = self.get_sub_runs(lvl_2_run_id)
                    env_data = []
                    for (_, transfer_budget_run) in lvl_3_child_runs.iterrows():
                        lvl_3_params = self.get_params_dict(transfer_budget_run)
                        lvl_3_metrics = self.get_scores_dict(transfer_budget_run)
                        joined_dict = {
                            **lvl_1_params,
                            **lvl_2_params,
                            **lvl_3_params,
                            **lvl_3_metrics,
                        }
                        # Append data to the list
                        env_data.append(joined_dict)
                    abs_transfer_budgets = [
                        int(run_dict["params.loo_budget"]) for run_dict in env_data
                    ]
                    unique_abs_budgets = list(np.unique(abs_transfer_budgets))
                    budget_map = {
                        absolute: relative
                        for absolute, relative in zip(
                            sorted(unique_abs_budgets),
                            sorted(relative_transfer_budgets),
                        )
                    }
                    for run_dict in env_data:
                        run_dict["params.loo_budget_rel"] = budget_map[
                            int(run_dict["params.loo_budget"])
                        ]
                    data_list.extend(env_data)

            # Create a pandas DataFrame from the collected data
            data_df = pd.DataFrame(data_list)

            # Print the DataFrame
            print(data_df)
            csv_path = self.csv_path
            data_df.to_csv(csv_path)
            mlflow.log_artifact(csv_path)
            # mlflow.log_table(data_df, csv_path)
            self.plot_errors()

    def get_sub_runs(self, parent_run_id):
        return mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' AND status='FINISHED'",
        )


def main():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--run", type=str, help="mlflow parent run")
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
    )
    args = parser.parse_args()
    parent_run_id = args.run
    skip_aggregation = args.skip_aggregation
    tracking_url = "http://185.209.223.218:5000"
    # tracking_url = "https://mlflow.sws.informatik.uni-leipzig.de"
    # parent_run_id = "d843627702ba4dadb2d7e08e99da8720"
    # parent_run_id = "224331c23c4b4575ba5dfc3ef2d30c04"
    # parent_run_id = "355878e4baae4be3a2792978e5643026" # jump3r
    parent_run_id = "15466c9a7134451b89e49fbc0244d29e"
    from experiment import EXPERIMENT_NAME

    al = Evaluation(parent_run_id, tracking_url, experiment_name=EXPERIMENT_NAME)
    if not skip_aggregation:
        al.run()


if __name__ == "__main__":
    main()
