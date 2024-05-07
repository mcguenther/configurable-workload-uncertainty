import argparse
import os
import os.path
import time
from typing import List, Dict
import json
import localflow as mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score

import matplotlib.gridspec as gridspec
import arviz as az
from scipy import stats as sps
from utils import get_date_time_uuid
from models import NumPyroRegressor
from mlfloweval import kl_divergence
from pprint import pprint
EXP_ID = "jdorn-modelinsights"

class NumpyroModelInsight:
    def __init__(self, path_to_netcdf):
        self.path_to_netcdf = path_to_netcdf
        self.az_data = az.from_netcdf(self.path_to_netcdf)

    def plot_overview(self):
        # az.plot_posterior(self.az_data)
        # plt.show()
        self.sus_out_options()
        self.plot_single_hyperiors()
        self.plot_multiple_hyperior()

    def get_RV_names(self):
        posterior = self.get_posterior_data()
        # Get all random variable names
        rv_names = list(posterior.var_names)
        return rv_names

    def get_coordinates(self):
        # Access the posterior group
        posterior = self.get_posterior_data()
        # Extract dimensions
        dimensions = list(posterior.dims)
        # Extract coordinate labels for each dimension
        coordinates = {dim: list(posterior.coords[dim].values) for dim in dimensions}
        return coordinates

    def get_posterior_data(self):
        return self.az_data.posterior

    def plot_all_options(self):
        coords = self.get_coordinates()
        feature_names = coords["features"]
        self.plot_base()
        for feature_name in feature_names:
            print(feature_name)
            self.plot_option(feature_name)

    def plot_option(self, option_name):
        posterior = self.get_posterior_data()
        env_lbls = list(np.array(self.get_posterior_data()["base"].envs))
        infl_samples = posterior["influences"]
        option_samples = infl_samples.loc[:, :, :, option_name]
        env_specific_dict = {
            lbl: np.array(option_samples[:, :, i]).flatten()
            for i, lbl in enumerate(env_lbls)
        }

        df = pd.DataFrame(env_specific_dict)

        option_hyper_samples = np.array(
            posterior["influences-mean-hyperior"].loc[:, :, option_name]
        ).flatten()
        self.plot_hyperior_over_specifics(option_hyper_samples, df, option_name)

    def sus_out_options(self):
        (
            feature_hyper_sanmples_list,
            feature_df_list,
            feature_names,
        ) = self.get_feature_hypers_and_influences()

        tups = []
        sus_kls = {}
        conformal_envs = {}
        for f_name, hyper_samples, influences_by_env in zip(
            feature_names, feature_hyper_sanmples_list, feature_df_list
        ):
            std = np.std(hyper_samples)
            kurt = sps.kurtosis(hyper_samples)
            skew = sps.skew(hyper_samples)
            median = float(az.hdi(hyper_samples, 0.95).mean())
            new_tup = (
                f_name,
                median,
                std,
                skew,
                kurt,
            )
            tups.append(new_tup)
            kl_divergence_results = list(
                (t, max(0, kl_divergence(hyper_samples, influences_by_env[t].values)))
                for t in influences_by_env
            )
            conformal_workloads = {
                opt: kl for opt, kl in kl_divergence_results if kl < 0.1
            }
            conformal_envs[f_name] = conformal_workloads
            kl_df = pd.DataFrame(kl_divergence_results, columns=["option", "kl"])
            kl_outliers = self.get_inter_quartile_outliers(kl_df, "kl")
            kl_dict = pd.Series(
                kl_outliers.kl.values, index=kl_outliers.option
            ).to_dict()
            sus_kls[f_name] = kl_dict
            col_names = "name", "median", "std", "skewness", "excesskurtosis"

        df = pd.DataFrame(tups, columns=col_names)
        outlier_value_col = "std"
        df["skewness"] = df["skewness"].abs()
        pprint(df)
        log_dataframe(df, "hyperior-moments")
        outliers = self.get_inter_quartile_outliers(df, outlier_value_col)
        print()
        pprint(outliers)
        log_dataframe(outliers, "outliers-hyperiors-most-iqr")
        print()
        pprint(sus_kls)
        mlflow.log_dict(sus_kls, "sus-kl-divergence-workloads-per-feature")
        print()
        print("conformal")
        pprint(conformal_envs)
        mlflow.log_dict(conformal_envs, "env-conformality")
        file_counts = {}
        for category in conformal_envs:
            for file in conformal_envs[category]:
                if file not in file_counts:
                    file_counts[file] = 1
                else:
                    file_counts[file] += 1
        # Sort file_counts by value in descending order
        sorted_file_counts = dict(
            sorted(file_counts.items(), key=lambda item: item[1], reverse=True)
        )
        mlflow.log_dict(sorted_file_counts, "wl-conformality-counts")
        print("Conformal envs counts:")
        for file, count in sorted_file_counts.items():
            print(f"{file}: {count}")
        print("finished quantitative analysis")

    def get_inter_quartile_outliers(self, df, outlier_value_col):
        # Schritt 1: IQR berechnen
        Q1 = df[outlier_value_col].quantile(0.25)
        Q3 = df[outlier_value_col].quantile(0.75)
        IQR = Q3 - Q1
        # Schritt 2: Grenzen für Ausreißer definieren
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Schritt 3: Ausreißer identifizieren
        outliers = df[
            (df[outlier_value_col] < lower_bound)
            | (df[outlier_value_col] > upper_bound)
        ]
        return outliers

    def plot_single_hyperiors(self, columns=8, rows=6):
        base_df, base_hyper_samples = self.get_base_hyperior_and_influences()

        (
            feature_hyper_sanmples_list,
            feature_df_list,
            feature_names,
        ) = self.get_feature_hypers_and_influences()

        var_names = ["base", *feature_names]

        hyper_samples_list = [base_hyper_samples, *feature_hyper_sanmples_list]
        df_list = [base_df, *feature_df_list]
        # Ensure that var_names has the same length as the hyper_samples_list and df_list
        if len(var_names) != len(hyper_samples_list) or len(var_names) != len(df_list):
            raise ValueError(
                "Length of var_names, hyper_samples_list, and df_list must be the same"
            )

        # Loop over each pair of plots and populate the grid
        for i, (base_hyper_samples, df, var_name) in enumerate(
                zip(hyper_samples_list, df_list, var_names)
        ):

            scale=1.5
            aspect = 1.2
            # Create a figure with the appropriate number of subplots
            fig, axes = plt.subplots(2, 1, figsize=(scale*aspect, scale * 2), sharex=True, sharey=False)
            # plt.suptitle(var_name)
            # Flatten the axes array for easy iteration
            axes = axes.flatten()
            # Melt the DataFrame for the current pair
            size_kw = {
                "fill":True,
                "thresh": 0.05,
            }
            df_long = df.melt(var_name="Workload", value_name="Standard influence")

            # The top plot - single distribution without hue
            top_index = 0
            lower_index = 1
            # Check if we've filled all the subplots
            # if lower_index >= total_subplots:
            #     break
            ax_top = axes[top_index]
            sns.kdeplot(base_hyper_samples, ax=ax_top, color="gray", **size_kw)
            ax_top.set_title(f"Hyper Prior")
            ax_top.set_xlabel("")  # Hide x-axis label for the top plot
            ax_top.set_yticks([])

            # The bottom plot - displot with different hues
            ax_bottom = axes[lower_index]
            do_legend = False #True # i==0
            sns_ax = sns.kdeplot(
                data=df_long, x="Standard influence", hue="Workload", ax=ax_bottom,legend=do_legend,
                palette="colorblind", **size_kw

            )

            # sns.move_legend(
            #     sns_ax,
            #     "center right",
            #     bbox_to_anchor=(-0.005, 0.5),
            #     # ncol=3,
            #     # title=None,
            #     frameon=True,
            #     fancybox=True,
            # )

            ax_bottom.set_title(f"By Workload")
            ax_bottom.set_xlabel("Influence")
            ax_bottom.set_yticks([])

            # # Hide x-axis label for all but the bottom row plots
            # if (2 * i + 1) // columns < (rows - 1) * 2:
            #     ax_bottom.set_xlabel("")

            plt.tight_layout()
            log_figure_pdf(f"hyperiors-{var_name}")
            # plt.show()
            plt.close(fig)


        # plt.show()
    def plot_multiple_hyperior(self, columns=8, rows=6):
        base_df, base_hyper_samples = self.get_base_hyperior_and_influences()

        (
            feature_hyper_sanmples_list,
            feature_df_list,
            feature_names,
        ) = self.get_feature_hypers_and_influences()

        var_names = ["base", *feature_names]

        hyper_samples_list = [base_hyper_samples, *feature_hyper_sanmples_list]
        df_list = [base_df, *feature_df_list]
        # Ensure that var_names has the same length as the hyper_samples_list and df_list
        if len(var_names) != len(hyper_samples_list) or len(var_names) != len(df_list):
            raise ValueError(
                "Length of var_names, hyper_samples_list, and df_list must be the same"
            )

        # Determine global min and max for the shared x-axis
        global_min = min(
            [df.min().min() for df in df_list] + [base_hyper_samples.min()]
        )
        global_max = max(
            [df.max().max() for df in df_list] + [base_hyper_samples.max()]
        )

        # Calculate the total number of plots (pairs of plots)
        total_pairs = len(var_names)

        # Calculate total subplots needed (each pair requires two subplots)
        total_subplots = min(total_pairs * 2, rows * columns)

        scale=2.2
        aspect = 1.5
        # Create a figure with the appropriate number of subplots
        fig, axes = plt.subplots(rows, columns, figsize=(scale*aspect * columns, scale * rows))

        # If there's only one row and one column, put axes in a list for consistency
        if rows == columns == 1:
            axes = [axes]

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Loop over each pair of plots and populate the grid
        for i, (base_hyper_samples, df, var_name) in enumerate(
            zip(hyper_samples_list, df_list, var_names)
        ):
            # Melt the DataFrame for the current pair
            size_kw = {
                "fill":True,
            }
            df_long = df.melt(var_name="Workload", value_name="Standard influence")

            # The top plot - single distribution without hue
            top_index = i % columns + 2 * columns * (i // columns)
            lower_index = i % columns + 2 * columns * (i // columns) + columns
            # Check if we've filled all the subplots
            if lower_index >= total_subplots:
                break
            ax_top = axes[top_index]
            sns.kdeplot(base_hyper_samples, ax=ax_top, color="gray", **size_kw)
            ax_top.set_title(f"{var_name} Location Hyperior")
            ax_top.set_xlabel("")  # Hide x-axis label for the top plot

            # The bottom plot - displot with different hues
            ax_bottom = axes[lower_index]
            do_legend = i==0
            sns.kdeplot(
                data=df_long, x="Standard influence", hue="Workload", ax=ax_bottom,legend=do_legend, **size_kw

            )
            ax_bottom.set_title(f"{var_name} per workload")

            # Hide x-axis label for all but the bottom row plots
            if (2 * i + 1) // columns < (rows - 1) * 2:
                ax_bottom.set_xlabel("")

        last_used_subplot_index = lower_index if lower_index < total_subplots else top_index
        unused_axes_start_index = last_used_subplot_index

        # Loop through the remaining axes and delete them
        for ax in axes[unused_axes_start_index:]:
            ax.remove()

        plt.tight_layout()
        log_figure_pdf("hyperiors")
        # plt.show()

    def get_base_hyperior_and_influences(self):
        posterior = self.get_posterior_data()
        env_lbls = list(np.array(self.get_posterior_data()["base"].envs))
        base_samples = posterior["base"]
        env_specific_dict = {
            lbl: np.array(base_samples[:, :, i]).flatten()
            for i, lbl in enumerate(env_lbls)
        }
        base_df = pd.DataFrame(env_specific_dict)
        base_hyper_samples = np.array(posterior["base-hyper"]).flatten()
        return base_df, base_hyper_samples

    def get_feature_hypers_and_influences(self):
        coords = self.get_coordinates()
        posterior = self.get_posterior_data()
        env_lbls = list(np.array(self.get_posterior_data()["base"].envs))
        infl_samples = posterior["influences"]
        feature_names = coords["features"]
        feature_hyper_sanmples_list = []
        feature_df_list = []
        for option_name in feature_names:
            option_samples = infl_samples.loc[:, :, :, option_name]
            env_specific_dict = {
                lbl: np.array(option_samples[:, :, i]).flatten()
                for i, lbl in enumerate(env_lbls)
            }

            df = pd.DataFrame(env_specific_dict)
            feature_df_list.append(df)

            option_hyper_samples = np.array(
                posterior["influences-mean-hyperior"].loc[:, :, option_name]
            ).flatten()
            feature_hyper_sanmples_list.append(option_hyper_samples)
        return feature_hyper_sanmples_list, feature_df_list, feature_names

    def plot_base(self):
        base_df, base_hyper_samples = self.get_base_hyperior_and_influences()
        self.plot_hyperior_over_specifics(base_hyper_samples, base_df)

    def plot_hyperior_over_specifics(self, base_hyper_samples, df, var_name="Base"):
        # Melt the DataFrame to have 'variable' and 'value' columns
        evn_lbl = "Workload"
        value_name = "Standard influence"
        df_long = df.melt(var_name=evn_lbl, value_name=value_name)
        # # Plotting with Seaborn
        # sns.displot(data=df_long, x="Value", hue="Key", kind="hist", aspect=1.5)
        # plt.show()
        plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        # The top plot - single distribution without hue
        ax0 = plt.subplot(gs[0])
        # sns.histplot(df_long["Value"], ax=ax0, color="gray")
        sns.histplot(base_hyper_samples, ax=ax0, color="gray")
        ax0.set_title("%s Location Hyperior" % var_name)
        ax0.set_xlabel("")  # Hide x-axis label for the top plot
        # The bottom plot - displot with different hues
        ax1 = plt.subplot(gs[1], sharex=ax0)
        sns.histplot(data=df_long, x=value_name, hue=evn_lbl, ax=ax1)
        ax1.set_title("%s per workload" % var_name)
        # Adjust layout
        plt.tight_layout()
        # Show the plot
        plot_name = "hyperiors-vs-concrete"
        log_figure_pdf(plot_name)

        plt.show()




# This function will search for .netcdf files that are in directories containing the word "partial"
def find_insight_files(path):
    netcdf_files = []
    params = []
    metrics_list = []
    for root, dirs, files in os.walk(path):
        if len(root) < 45:
            for file in files:
                relative_path = os.path.join(root, file)
                if file.endswith('params.json'):
                    with open(relative_path, 'r') as json_file:
                        params_dict = json.load(json_file)
                        params.append(params_dict)
                if file.endswith('metrics.json'):
                    with open(relative_path, 'r') as json_file:
                        metrics_dict = json.load(json_file)
                        metrics_list.append(metrics_dict)
                if file.endswith('.netcdf'):
                    # Get the relative path from the given directory path
                    # relative_path = os.path.relpath(os.path.join(root, file), path)
                    netcdf_files.append(relative_path)
    return netcdf_files, params, metrics_list



# Get the list of relevant .netcdf files

def log_dataframe(df, df_name):
    df.to_csv("%s.csv" % df_name)
    mlflow.log_artifact("%s.csv" % df_name)
    time.sleep(0.1)
    os.remove("%s.csv" % df_name)

def main():
    mlflow.set_experiment(EXP_ID)
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument(
        "--folder",
        type=str,
        help="path to results parquet file or similar",
        # default="single-trace-debugger/arviz_data-6e5a3d1d-032d-428f-aea7-16bccdf79158.netcdf",
        default="insights/240322-00-47-34-2024-03-22_00-47-34-doLXggV4xj",
        # default="single-trace-debugger/240321-13-42-40-uncertainty-learning-2024-DnfMtUuQWk",
    )
    args = parser.parse_args()

    netcdf_parefolder = args.folder
    partial_netcdf_files, all_params, all_metrics = find_insight_files(netcdf_parefolder)

    replication_lbl = get_date_time_uuid()
    with mlflow.start_run(run_name=replication_lbl):
        for i, (netcdf, params, metrics) in enumerate(zip(partial_netcdf_files, all_params, all_metrics)):
            sws = params["params.software-system"]
            job_name = f"{str(i)}-{sws}"
            with mlflow.start_run(run_name=job_name):
                mlflow.log_dict(params, "training-params")
                mlflow.log_dict(metrics, "training-metrics")
                al = NumpyroModelInsight(netcdf)
                al.plot_overview()
    # al.plot_all_options()

    # plot_metadata(meta_df, output_base_path)
    # plot_errors(err_type, output_base_path, score_df)

def log_figure_pdf(plot_name):
    file_name = "%s.pdf" % plot_name
    plt.savefig(file_name, bbox_inches="tight")
    mlflow.log_artifact(file_name)

if __name__ == "__main__":
    main()
