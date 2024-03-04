import argparse
import os
import os.path
import time
from typing import List, Dict

import traceback
import localflow as mlflow
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import json
import arviz as az

from scipy.spatial import cKDTree as KDTree
from utils import get_date_time_uuid
from models import NumPyroRegressor
from dataclasses import dataclass

RESULTS_EXP = "jdorn-multilevel-eval"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


class Plotter:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def log_figure(self, lbl):
        fig_pdf_path = f"lastplot-errors-{lbl}.pdf"
        print(f"Saving plt fig {lbl}")
        plt.savefig(fig_pdf_path)
        print(f"logging artifact at {fig_pdf_path}")
        mlflow.log_artifact(fig_pdf_path)
        print("done logging")
        # fig_png_path = f"lastplot-errors-{lbl}.png"
        # plt.savefig(fig_png_path)
        # mlflow.log_artifact(fig_png_path)


class MultitaskPlotter(Plotter):
    def plot_errors(self):
        df = pd.read_csv(self.csv_path)
        mlflow.log_artifact(self.csv_path)
        print(df)

        print("Plotting relative multitask performance")
        metric_columns = [col for col in df.columns if col.startswith("metrics.")]

        # Melt the DataFrame
        melted_df = df.melt(
            id_vars=[col for col in df.columns if col not in metric_columns],
            value_vars=metric_columns,
            var_name="Metric",
            value_name="Value",
        )

        metric = "metrics.mape_overall"
        plot = sns.relplot(
            data=melted_df,
            x="params.relative_train_size",
            # x="params.loo_budget_rel",
            y="Value",
            row="params.software-system",
            col="Metric",
            # col_wrap=4,
            # sharey=False,
            kind="line",
            hue="params.model",
            style="params.pooling_cat",
            facet_kws={"sharey": False, "sharex": False},
        )
        # setting boundaries that make sense
        plt.tight_layout()

        for ax in plt.gcf().axes:
            title = ax.get_title()
            lower_title = str(title).lower()
            if "R2" in title:
                ax.set_ylim(-1, 1)
                print("set R2 ylims")
            if "mape" in str(title).lower():
                y_min, y_max = ax.get_ylim()
                print("old y limits", y_min, y_max)
                ax.set_ylim(0, min(y_max, 250))

                y_min, y_max = ax.get_ylim()
                print("new y limits", y_min, y_max)
                # ax.set_yscale('log')

            if "test_set_log" in lower_title:
                ax.set_yscale("symlog")
        # plt.suptitle("Absolute training size")
        self.log_figure("abs-training-size")
        plt.show()
        # plt.yscale("log")
        # Get the current y-axis limits
        current_ylim = plt.ylim()
        # Check if the current ymax is above 200
        # if current_ylim[1] > 200:
        #     # Set the ymax to 200
        #     plt.ylim(current_ylim[0], 200)
        # plt.ylim(0, 12)


class TransferPlotter(Plotter):
    def plot_errors(self):
        # kwargs = {"run_id": self.run_id} if self.run_id else {"run_name": self.run_name}
        # with mlflow.start_run(
        #     **kwargs  # self.experiment_name.replace(" ", ""),
        # ) as run:
        df = pd.read_csv(self.csv_path)
        mlflow.log_artifact(self.csv_path)
        print(df)

        print("Plotting relative transfer budget")
        metric = "metrics.mape_overall"
        sns.relplot(
            data=df,
            x="params.loo_budget",
            # x="params.loo_budget_rel",
            y=metric,
            col="params.loo_idx",
            kind="line",
            hue="params.model",
            style="params.pooling_cat",
            row="params.relative_train_size",  # col_wrap=4,
        )
        plt.suptitle("Absolute training size")
        # plt.yscale("log")
        # Get the current y-axis limits
        current_ylim = plt.ylim()
        # Check if the current ymax is above 200
        # if current_ylim[1] > 200:
        #     # Set the ymax to 200
        #     plt.ylim(current_ylim[0], 200)
        plt.ylim(0, 12)
        self.log_figure("abs-training-size")
        print("Plotting absolute transfer budget")
        metric = "metrics.mape_overall"
        sns.relplot(
            data=df,
            x="params.loo_budget_rel",
            # x="params.loo_budget_rel",
            y=metric,
            col="params.loo_idx",
            kind="line",
            hue="params.model",
            style="params.pooling_cat",
            row="params.relative_train_size",  # col_wrap=4,
        )
        plt.suptitle("Relative training size ")
        # plt.yscale("log")
        # Get the current y-axis limits
        current_ylim = plt.ylim()
        # Check if the current ymax is above 200
        # if current_ylim[1] > 200:
        #     # Set the ymax to 200
        #     plt.ylim(current_ylim[0], 200)
        plt.ylim(0, 12)
        self.log_figure("rel-training-size")


class Evaluation:
    def __init__(self, parent_run, tracking_url, experiment_name=None):
        self.experiment_name = experiment_name or "jdorn-tmp"
        self.parent_run = parent_run
        self.tracking_url = tracking_url
        self.csv_path = "mlfloweval-last.csv"
        self.run_id = None
        self.run_name = "aggregation"
        mlflow.set_tracking_uri(self.tracking_url)
        self.idx = ["model", "env_id", "budget_abs", "rnd", "subject_system"]
        self.experiment_id = mlflow.get_short_valid_id(experiment_name)
        mlflow.set_experiment(experiment_name=RESULTS_EXP)
        #
        # self.output_base_path = os.path.join(
        #     self.results_base_path, f"{my_id}-analysis"
        # )
        # os.makedirs(self.output_base_path)
        # print(f"plotting to {self.output_base_path}")

    def plot_metadata(self, meta_df=None):
        kwargs = {"run_id": self.run_id} if self.run_id else {"run_name": self.run_name}
        with mlflow.start_run(
            **kwargs  # self.experiment_name.replace(" ", ""),
        ) as run:
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
            run_name=self.run_name  # self.experiment_name.replace(" ", ""),
        ) as run:
            self.run_id = run.info.run_id
            # Initialize an empty list to store the data
            data_list = {"transfer": [], "multitask": []}

            # Fetch the parent run and its children
            # parent_run = mlflow.get_run(self.parent_run)
            print("fetching child runs of parent run", self.parent_run)
            exp_result = mlflow.ExperimentResult(self.experiment_id)
            parent_run_result = exp_result.get_run_by_id(self.parent_run)
            child_runs = parent_run_result.get_sub_runs()

            # Iterate over the first-level nested runs
            for child_run in child_runs:
                lvl_1_params = child_run.get_params()
                experiment_type = lvl_1_params["params.experiment-type"]
                if experiment_type == "transfer":
                    results = self.eval_transfer_learning(child_run)
                    data_list["transfer"].extend(results)
                elif experiment_type == "multitask":
                    results = self.eval_multitask_learning(child_run)
                    data_list["multitask"].extend(results)
                else:
                    raise AssertionError

            # Create a pandas DataFrame from the collected data
            if data_list[("transfer")]:
                data_df = pd.DataFrame(data_list[("transfer")])
                print(data_df)
                csv_path = prepend_to_filename("transfer-", self.csv_path)
                self.store_csvs(csv_path, data_df)
                plotter = TransferPlotter(csv_path)
                plotter.plot_errors()

            if data_list[("multitask")]:
                data_df = pd.DataFrame(data_list[("multitask")])
                print(data_df)
                csv_path = prepend_to_filename("multitask-", self.csv_path)
                self.store_csvs(csv_path, data_df)
                plotter = MultitaskPlotter(csv_path)
                plotter.plot_errors()

    def store_csvs(self, csv_path, data_df):
        data_df.to_csv(csv_path)
        mlflow.log_artifact(csv_path)
        xlsx_path = csv_path.replace(".csv", ".xlsx")
        data_df.to_excel(xlsx_path)
        mlflow.log_artifact(xlsx_path)

    def eval_multitask_learning(self, child_run):
        child_run_params = child_run.get_params()
        model = child_run_params["params.model"]
        print(f"fetching new model {model}")
        metrics = child_run.get_metrics()
        log = {**child_run_params, **metrics}

        az_data = child_run.get_arviz_data()
        if az_data:
            "this modle has az data"

        return [log]

    def eval_transfer_learning(self, child_run):
        lvl_1_params = child_run.get_params()
        lvl_1_run_id = child_run.run_id
        model = lvl_1_params["params.model"]
        print(f"fetching new model {model}")
        # Fetch the second-level nested runs for each environment
        relative_transfer_budgets = lvl_1_params["params.transfer_budgets"]
        # relative_transfer_budgets = json.loads(relative_transfer_budgets)
        env_runs = child_run.get_sub_runs()
        data_list = []
        for number_of_transfer_samples_env_run in env_runs:
            lvl_2_params = number_of_transfer_samples_env_run.get_params()
            try:
                lvl_2_child_runs = number_of_transfer_samples_env_run.get_sub_runs()
                for number_of_source_env_run in lvl_2_child_runs:
                    lvl_3_params = number_of_source_env_run.get_params()
                    # env_idx = lvl_2_params["params.loo_idx"]
                    # print(f"getting budget runs for env idx {env_idx}")
                    # lvl_2_params = self.get_params_dict(env_run)
                    lvl_3_child_runs = number_of_source_env_run.get_sub_runs()
                    # env_data = []
                    for source_env_permutation_run in lvl_3_child_runs:
                        lvl_4_params = source_env_permutation_run.get_params()
                        lvl_4_metrics = source_env_permutation_run.get_metrics()
                        joined_dict = {
                            **lvl_1_params,
                            **lvl_2_params,
                            **lvl_3_params,
                            **lvl_4_params,
                            **lvl_4_metrics,
                        }
                        az_data = source_env_permutation_run.get_arviz_data()
                        if az_data:
                            joined_dict["az_data"] = az_data
                        # Append data to the list
                        data_list.append(joined_dict)
            except Exception as e:
                tb = traceback.format_exc()
                print("[LFlow]", e)
                print(tb)

            # abs_transfer_budgets = [
            #     int(run_dict["params.loo_budget"]) for run_dict in env_data
            # ]
            # unique_abs_budgets = list(np.unique(abs_transfer_budgets))
            # budget_map = {
            #     absolute: relative
            #     for absolute, relative in zip(
            #         sorted(unique_abs_budgets, reverse=True),
            #         sorted(relative_transfer_budgets, reverse=True),
            #     )
            # }
            # for run_dict in env_data:
            #     run_dict["params.loo_budget_rel"] = budget_map[
            #         int(run_dict["params.loo_budget"])
            #     ]
            #
            # data_list.extend(env_data)
        return data_list

    # def get_sub_runs(self, parent_run_id):
    #     exp_folder = mlflow.get_experiment_folder(self.experiment_name)
    #     os.listd
    #     return mlflow.search_runs(
    #         experiment_ids=[self.experiment_id],
    #         filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' AND status='FINISHED'",
    #     )

    def download_netcdf(self, child_run):
        # MLflow Client initialisieren
        client = mlflow.tracking.MlflowClient()

        # Liste aller Artefakte für diesen Run
        run_id = child_run.run_id
        artifacts = client.list_artifacts(run_id)

        # Jetzt filtern wir nach der Dateiendung
        netcdf_artifacts = [a.path for a in artifacts if a.path.endswith(".netcdf")]

        # Checken, ob wir was gefunden haben
        if netcdf_artifacts:
            # Nehmen wir an, du willst das erste .netcdf Artefakt herunterladen
            local_path = client.download_artifacts(run_id, netcdf_artifacts[0])
            az_data = az.from_netcdf(local_path)
            filtered_variables = [
                var for var in az_data.posterior.data_vars if "hyper" in var
            ]
            feature_names = list(
                np.array(az_data.posterior.coords.variables["features"])
            )
            env_names = list(np.array(az_data.posterior.coords.variables["envs"]))
            n_chain, n_draws, n_envs, n_features = az_data.posterior["influences"].shape
            hyperior_mean_samples = az_data.posterior[
                "influences-mean-hyperior"
            ].values.reshape(-1, n_features)
            hyperior_stddevs_samples = az_data.posterior[
                "influences-stddevs-hyperior"
            ].values.reshape(-1, n_features)
            env_specific_samples = az_data.posterior["influences"].values.reshape(
                -1, n_envs, n_features
            )
            ft_id = 5
            feauture = feature_names[ft_id]
            env_id = env_names[ft_id]
            specific_infs = env_specific_samples[:, :, ft_id]
            az_data.posterior["influences-mean-hyperior"]
            az_data.posterior["influences-stddevs-hyperior"]
            # results in n_chains x n_samples x n_simulations samples for simulated influence
            n_simulations = 100
            simulated_infl = np.concatenate(
                [
                    np.random.normal(
                        hyperior_mean_samples[:, 0], hyperior_stddevs_samples[:, 0]
                    )
                    for _ in range(n_simulations)
                ]
            )
            # for feature in feature_names:

            # chain, draw, env, feature
            kl_divergence(
                np.array([0, 1, 1, 1, 1, 2, 3, 4]),
                np.array([0, 1, 1, 1, 1, 2, 6, 8]) * 1.00001,
            )
            kl_divergence(
                np.array([0, 1, 1, 1, 1, 2, 3, 4]),
                np.array([0, 1, 1, 1, 1, 2, 3, 4]) * 1.00001,
            )
            # kl_divergence(specific_infs[:, 0], specific_infs[:, 0] * 1.00001)
            scipy.stats.entropy(
                list(specific_infs[:, 0]), list(specific_infs[:, 0]), base=2
            )
            scipy.stats.entropy(
                list(specific_infs[:, 0]), list(specific_infs[:, 0]), base=2
            )
            plot_trace = False
            if plot_trace:
                az.plot_trace(az_data, var_names=filtered_variables, filter_vars=None)
                plt.show()
            print(f"\tArtefakt heruntergeladen nach: {local_path}")
            return az_data


def prepend_to_filename(string_to_prepend, path):
    # Split the path into directory and filename
    dir_name, file_name = os.path.split(path)

    # Prepend the string to the filename
    new_file_name = string_to_prepend + file_name

    # Reconstruct the full path with the new filename
    new_path = os.path.join(dir_name, new_file_name)

    return new_path


# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html
# https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
def kl_divergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
      Parameters
      ----------
      x : 2D array (n,d)
        Samples from distribution P, which typically represents the true
        distribution.
      y : 2D array (m,d)
        Samples from distribution Q, which typically represents the approximate
        distribution.
      Returns
      -------
      out : float
        The estimated Kullback-Leibler divergence D(P||Q).
      References
      ----------
      Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """

    if np.all(x == y):
        return 0.0

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    x, y = x.T, y.T
    n, d = x.shape
    m, dy = y.shape
    assert d == dy

    # get distance to nearest non-identical neighbour for every sample
    r = get_distances(x, x)
    s = get_distances(x, y)

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.0))


def get_distances(x, y):
    x_remain = np.copy(x)
    y_remain = np.copy(y)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    tree = KDTree(x_remain)
    k = 3
    neighbour_results = tree.query(y_remain, k=k, eps=0.01, p=2)
    neighbour_ids = neighbour_results[1]
    r = np.zeros(x.shape[0])
    for i in range(k):
        mask_for_existing_samples = r == 0
        if not np.any(mask_for_existing_samples):
            break
        r[mask_for_existing_samples] = neighbour_results[0][
            mask_for_existing_samples, i
        ]
    while np.any(r == 0):
        mask_r_zero = r == 0.0
        for idx in np.where(mask_r_zero)[0]:
            sample = y[idx]
            x_remain = np.atleast_2d(x[x != sample]).T
            tree = KDTree(x_remain)
            neighbour_results = tree.query(sample, k=1, eps=0.01, p=2)
            r[idx] = neighbour_results[0]
    return r


def get_fitting_evaluator(parent_run: str, tracking_url: str, experiment_name: str):
    experiment_name = experiment_name or "jdorn-tmp"
    parent_run = parent_run
    tracking_url = tracking_url
    csv_path = "mlfloweval-last.csv"
    run_id = None
    run_name = "aggregation"
    mlflow.set_tracking_uri(tracking_url)
    idx = ["model", "env_id", "budget_abs", "rnd", "subject_system"]
    experiment = mlflow.search_experiments(
        filter_string=("attribute.name = '%s'" % experiment_name)
    )[0]
    experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name=RESULTS_EXP)

    run = mlflow.get_run(parent_run)


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
    import experiment

    tracking_url = experiment.MLFLOW_URI
    # tracking_url = "https://mlflow.sws.informatik.uni-leipzig.de"
    # parent_run_id = "d843627702ba4dadb2d7e08e99da8720"
    # parent_run_id = "224331c23c4b4575ba5dfc3ef2d30c04"
    # parent_run_id = "355878e4baae4be3a2792978e5643026" # jump3r
    # parent_run_id = "ec5fe58c918046d4a20b8f497c348576"
    # parent_run_id = "5fbb9d52019a42fba015a4b840ec2b2d"
    # parent_run_id = "26fcff0b056e4ba5b262c28ef47dc4f9"
    parent_run_id = "231219-16-04-08-uncertainty-learning-2023-EDnxMVNhCg"
    parent_run_id = "231220-10-53-08-uncertainty-learning-2023-JEsu9PFWxJ"  # multitask
    parent_run_id = (
        "231220-14-06-10-uncertainty-learning-2023-aqSe3L6nWD"  # transfer mini
    )
    # parent_run_id = (
    #     "231220-21-59-44-uncertainty-learning-2023-2yWWUcd6GN"  # transfer gigantic
    # )
    parent_run_id = (
        "240228-17-59-03-uncertainty-learning-2024-fPjWZCZrCa"  # transfer gigantic
    )
    from experiment import EXPERIMENT_NAME

    # al = get_fitting_evaluator(
    #     parent_run_id, tracking_url, experiment_name=EXPERIMENT_NAME
    # )
    al = Evaluation(parent_run_id, tracking_url, experiment_name=EXPERIMENT_NAME)
    if not skip_aggregation:
        al.run()
    else:
        al.plot_errors()


if __name__ == "__main__":
    main()
