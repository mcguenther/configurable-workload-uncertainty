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

from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

EXP_ID = "jdorn-modelinsights"

class NumpyroModelInsight:
    def __init__(self, path_to_netcdf):
        self.path_to_netcdf = path_to_netcdf
        self.az_data = az.from_netcdf(self.path_to_netcdf)

    def plot_overview(self):
        # az.plot_posterior(self.az_data)
        # plt.show()
        self.plot_single_hyperiors()
        # kld_threshold = 0.5
        # self.plot_representativeness_matrices(kld_threshold=kld_threshold)
        # kld_threshold_varying_from_hyperior = 1.0
        # self.sus_out_options(kl_threshold=kld_threshold_varying_from_hyperior)
        # self.plot_multiple_hyperior()

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


    def plot_loss_differences(self, df, df_results_screening):
        """Plot a line graph showing the differences in information loss."""
        # Calculate the difference in information loss
        # loss_red_lbl = 'Loss Reduction'
        for loss_red_lbl in ['Information Loss Reduction', "Information Loss Remaining", "Unfinished Options",
                             "Number of Represented Workload-Expcific Influences", "Relative Number of Represented Workload-Expcific Influences"]:
            # df[loss_red_lbl] = -1 * df['Information Loss'].diff() # df['Information Loss'].diff().fillna(df['Information Loss']).abs()
            # Plot the loss differences using seaborn
            df["Step"] = df["Step"].astype(int)
            df_results_screening["Step"] = df_results_screening["Step"].astype(int)
            bad_values = [None, np.inf, np.nan]
            metric_df = df.loc[~df[loss_red_lbl].isin(bad_values)]
            metric_df_results_screening = df_results_screening.loc[df_results_screening["Step"].isin(metric_df["Step"])]

            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 6))
            # Line plot using 'Step' for x-axis
            # Violin plot using 'Step' for x-axis
            # sns.violinplot(data=df_results_screening, x='Step', y=loss_red_lbl, scale='width', inner='quartile', palette='muted')

            sns.lineplot(data=metric_df, x='Step', y=loss_red_lbl, marker='o')
            if not metric_df_results_screening.empty:
                sns.swarmplot(data=metric_df_results_screening, x='Step', y=loss_red_lbl, size=5, color='brown', alpha=0.7, ax=plt.gca())

            # Set the title and labels with the existing column name used dynamically
            plt.title(loss_red_lbl)
            plt.xlabel('Step')
            plt.ylabel(loss_red_lbl)
            # Customizing x-axis ticks to reflect 'Selected Workload' categories corresponding to each 'Step'
            # First, we need to ensure that the 'Selected Workload' column is sorted according to 'Step' or aligned accordingly
            workload_labels = metric_df.sort_values(by='Step')['Selected Workload'].unique()
            #plt.xticks(ticks=list(metric_df["Step"].unique()), labels=workload_labels, rotation=45, ha='right')
            # Adjust layout to ensure all components are visible without overlap
            plt.tight_layout()
            log_figure_pdf(f"rep-set-builder-{loss_red_lbl}")

            # plt.close()
            # plt.show()

    def plot_representativeness_matrices(self, kld_threshold=0.5):
        df = self.compute_representativeness_tensor()
        self.generate_solar_plots(df, kld_threshold)
        log_dataframe(df, "information-loss-per-workload-per-option")
        inform_loss_per_wl = dict(df.groupby(["represented_by_env"])["kld"].sum())
        print(inform_loss_per_wl)
        number_of_representing_wl = dict(df[df["kld"] < kld_threshold].groupby(["represented_by_env"])["kld"].count())
        print(number_of_representing_wl)
        log_df, minimum_rep_set_size_from_saturation, df_results_screening = self.greedy_representation_set_generation(df, kld_threshold)
        # print(log_df)
        # self.plot_loss_differences(log_df, df_results_screening)


    def compute_representativeness_tensor(self):
        (
            feature_hyper_sanmples_list,
            feature_df_list,
            feature_names,
        ) = self.get_feature_hypers_and_influences()
        feature_tups = []
        for f_name, hyper_samples, influences_by_env in zip(
                feature_names, feature_hyper_sanmples_list, feature_df_list
        ):
            env_lbls = list(influences_by_env)
            for env_a in env_lbls:
                samples_a = influences_by_env[env_a].values
                for env_possible_representant in env_lbls:
                    samples_approximation = influences_by_env[env_possible_representant].values
                    kld = max(0, kl_divergence(samples_a, samples_approximation))
                    tup = f_name, env_a, env_possible_representant, kld
                    feature_tups.append(tup)
        col_names = "option", "env", "represented_by_env", "kld"
        df = pd.DataFrame(feature_tups, columns=col_names)
        # Define the threshold value below which values will be blue
        return df

    def generate_solar_plots(self, df, threshold):
        # Create the custom colormap
        reds_blues = sns.diverging_palette(240, 10, n=5, as_cmap=False)
        colors = [reds_blues[0], reds_blues[1], "white", reds_blues[3], reds_blues[4]]
        cmap = LinearSegmentedColormap.from_list("CustomRedBlue", colors)

        # Group by the "option" column
        options = df['option'].unique()
        with sns.plotting_context("talk"):
            for option in options:
                df_option = df[df['option'] == option]
                # Create the heatmap data pivot table
                heatmap_data = pd.pivot_table(df_option, values='kld', index='represented_by_env', columns='env', fill_value=0)
                # Calculate the number of cells below the threshold for each row
                row_counts_below_threshold = (heatmap_data < threshold).sum(axis=1)
                # Calculate the average KLD for each row
                average_kld = heatmap_data.mean(axis=1)
                # Combine these two metrics into a DataFrame for sorting
                sorting_criteria = pd.DataFrame({
                    'count_below_threshold': row_counts_below_threshold,
                    'average_kld': average_kld
                })
                # Sort by count of cells below the threshold and then by average KLD
                sorted_rows = sorting_criteria.sort_values(by=['count_below_threshold', 'average_kld'], ascending=[False, True]).index
                # Reindex the heatmap data
                heatmap_data = heatmap_data.reindex(index=sorted_rows, columns=sorted_rows)
                # Adjust the normalization with a tighter range for the upper threshold
                vmax = threshold * (1 + 10 ** -10)
                norm = TwoSlopeNorm(vmin=0, vcenter=threshold, vmax=vmax)
                ratio = 7./11
                ratio = 5.5/11
                scale = 0.55
                plt.figure(figsize=(11*scale, 11*ratio*scale))
                ax = sns.heatmap(
                    heatmap_data, annot=True, fmt=".1f", cmap=cmap, norm=norm, cbar_kws={'label': 'KLD'},
                    linewidths=0.5,
                    linecolor='white',
                    annot_kws={"size": 14}
                )
                # Remove ".wav" from tick labels
                # ax.set_xticklabels([label.get_text().replace(".wav", "") for label in ax.get_xticklabels()])
                ax.set_yticklabels([label.get_text().replace(".wav", "") for label in ax.get_yticklabels()])
                plt.xlabel("Workload")
                plt.ylabel("Represented By")
                # plt.xticks(rotation=45, ha='right')
                plt.xticks([])
                plt.yticks(rotation=0)
                plt.tight_layout()
                log_figure_pdf(f"heatmap_{option}")
                # Generate KDE and histogram plot for the KLD values of the current option
                plt.figure(figsize=(18, 8))
                sns.kdeplot(df_option['kld'], fill=True, color='blue', bw_adjust=0.07, label="KDE")
                bin_edges = np.arange(0, df_option['kld'].max() + 0.05, 0.05)
                # Set x-ticks to match histogram bin edges
                plt.xticks(bin_edges)
                plt.hist(df_option['kld'], bins=bin_edges, alpha=0.5, label='Histogram')
                plt.title(f"KDE and Histogram of KLD values - option: {option}", fontsize=16, fontweight='bold')
                plt.xlabel("KLD", fontsize=14, fontweight='bold')
                plt.ylabel("Density / Frequency", fontsize=14, fontweight='bold')
                plt.xlim((0.05, 2.0))
                plt.legend()
                plt.tight_layout()
                log_figure_pdf(f"kde_histogram_{option}")



    def sus_out_options(self, kl_threshold = 1.0):
        (
            feature_hyper_sanmples_list,
            feature_df_list,
            feature_names,
        ) = self.get_feature_hypers_and_influences()

        tups = []
        sus_kls = {}
        conformal_envs = {}
        invariant_options = []
        kl_divs_tups = []
        for f_name, hyper_samples, influences_by_env in zip(
            feature_names, feature_hyper_sanmples_list, feature_df_list
        ):
            std = np.std(hyper_samples)
            kurt = sps.kurtosis(hyper_samples)
            skew = sps.skew(hyper_samples)
            credible_interval_width = abs(np.subtract(*az.hdi(hyper_samples, 0.95)))
            median = float(az.hdi(hyper_samples, 0.02).mean())
            new_tup = (
                f_name,
                median,
                std,
                credible_interval_width,
                skew,
                kurt,
            )
            tups.append(new_tup)
            kl_divergence_results = list(
                # (t, max(0, kl_divergence(hyper_samples, influences_by_env[t].values)))
                (t, max(0, kl_divergence(influences_by_env[t].values, hyper_samples)))
                for t in influences_by_env
            )
            new_tups = [(f_name, wl, kldiv) for wl, kldiv in kl_divergence_results]
            kl_divs_tups.extend(new_tups)

            conformal_workloads = {
                opt: kl for opt, kl in kl_divergence_results if kl < kl_threshold
            }
            if len(conformal_workloads) == len(kl_divergence_results):
                invariant_options.append(f_name)
            conformal_envs[f_name] = conformal_workloads
            kl_df = pd.DataFrame(kl_divergence_results, columns=["option", "kl"])
            kl_outliers = self.get_inter_quartile_outliers(kl_df, "kl")
            kl_dict = pd.Series(
                kl_outliers.kl.values, index=kl_outliers.option
            ).to_dict()
            sus_kls[f_name] = kl_dict

        kldivs_df = pd.DataFrame(kl_divs_tups, columns=["option", "env", "kldiv"])
        log_dataframe(kldivs_df, "kldivs-towards-hyperior-per-option")

        col_names = "name", "median", "std", "credible_interval_width", "skewness", "excesskurtosis"
        df = pd.DataFrame(tups, columns=col_names)
        outlier_value_col = "credible_interval_width"
        df["skewness"] = df["skewness"].abs()
        pprint(df)

        invar_d = {"options_with_no_influence_var": invariant_options,
                   "num_invar_opts": len(invariant_options),
                   "ratio_invar_options": float(len(invariant_options) / float(len(feature_names)))
                   }
        print()
        pprint(invar_d)
        mlflow.log_dict(invar_d, "invariant-options")

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
            (df[outlier_value_col] > upper_bound)
            # | (df[outlier_value_col] < lower_bound)
        ]
        return outliers

    def plot_single_hyperiors(self):
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

        with sns.plotting_context("paper"):
            # Loop over each pair of plots and populate the grid
            for i, (base_hyper_samples, df, var_name) in enumerate(
                    zip(hyper_samples_list, df_list, var_names)
            ):

                # Melt the DataFrame for the current pair
                df_long = df.melt(var_name="Workload", value_name="Standard influence")
                # Combine the base hyper samples and the specific influence data
                all_samples = np.concatenate([base_hyper_samples, df_long["Standard influence"].values])
                # Calculate the 99% HDI
                hdi = az.hdi(all_samples, hdi_prob=0.998)
                scale=1.2
                aspect = 1.2
                # Create a figure with the appropriate number of subplots
                fig, axes = plt.subplots(2, 1, figsize=(scale*aspect, scale * 2 * 0.9), sharex=True, sharey=False)
                axes = axes.flatten()
                size_kw = {
                    "fill":True,
                }
                df_long = df.melt(var_name="Workload", value_name="Standard influence")
                # The top plot - single distribution without hue
                top_index = 0
                lower_index = 1
                ax_top = axes[top_index]
                sns.kdeplot(base_hyper_samples, ax=ax_top, color="gray",
                            **size_kw)
                # ax_top.set_title(f"Hyper Prior")
                ax_top.set_xlabel("")  # Hide x-axis label for the top plot
                ax_top.set_yticks([])
                # The bottom plot - displot with different hues
                ax_bottom = axes[lower_index]
                do_legend = False #True # i==0
                sns_ax = sns.kdeplot(
                    data=df_long, x="Standard influence", hue="Workload", ax=ax_bottom,legend=do_legend,
                    palette="colorblind",
                    #multiple="fill",
                    **size_kw
                )
                ax_top.set_ylabel(f"General")
                ax_bottom.set_ylabel(f"Specific")

                ax_bottom.set_xlabel("Influence")
                ax_bottom.set_yticks([])

                for ax in (ax_top, ax_bottom):
                    # Ensure 0 is included in the x-axis limits
                    # ax.set_xlim(hdi[0], hdi[1])
                    x_min, x_max = hdi
                    x_min = min(x_min, 0)
                    x_max = max(x_max, 0)
                    ax.set_xlim(x_min, x_max)

                # Get x-tick positions and labels from the bottom plot
                x_ticks = ax_bottom.get_xticks()
                x_tick_labels = ax_bottom.get_xticklabels()

                # Set the same x-tick positions and labels for the top plot
                ax_top.set_xticks(x_ticks)
                ax_top.set_xticklabels(x_tick_labels)

                sns.despine(left=True)
                # # Hide x-axis label for all but the bottom row plots
                # if (2 * i + 1) // columns < (rows - 1) * 2:
                #     ax_bottom.set_xlabel("")
                plt.tight_layout()
                log_figure_pdf(f"hyperiors-{var_name}-y-labels", close=False)
                ax_top.set_ylabel("")
                ax_bottom.set_ylabel("")
                plt.tight_layout()
                log_figure_pdf(f"hyperiors-{var_name}")





                scale = 2.0
                aspect = 1.0 / 1.05

                # Melt the DataFrame for the current pair
                df_long = df.melt(var_name="Workload", value_name="Standard influence")
                # Combine the base hyper samples and the specific influence data
                all_samples = np.concatenate([base_hyper_samples, df_long["Standard influence"].values])
                # Calculate the 99% HDI
                hdi = az.hdi(all_samples, hdi_prob=0.998)

                # Create a figure with a single subplot
                fig, ax = plt.subplots(figsize=(scale*aspect, scale))

                # Plot the KDE plot for the DataFrame with different hues
                do_legend = False
                lw = 1.05
                sns_ax = sns.kdeplot(
                    data=df_long, x="Standard influence", hue="Workload", ax=ax, legend=do_legend,
                    palette="colorblind",
                    linewidth=lw,
                    common_norm=False,
                    fill=True,
                )

                # Plot the base hyper samples with a white contour
                sns.kdeplot(base_hyper_samples, ax=ax, color="white",
                            linewidth=lw,  # Make this line thicker
                            fill=False)

                # Plot the base hyper samples as a black dashed line on top of the white contour
                sns.kdeplot(base_hyper_samples, ax=ax, color="black",
                            linewidth=lw,
                            linestyle='--',
                            fill=False)

                # Set the x-axis limits to cover the 99% HDI
                x_min, x_max = hdi
                x_min = min(x_min, 0)
                x_max = max(x_max, 0)
                ax.set_xlim(x_min, x_max)

                # Get x-tick positions and labels
                x_ticks = ax.get_xticks()
                x_tick_labels = ax.get_xticklabels()

                # Set the x-tick positions and labels
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_tick_labels)

                ax.set_xlabel("Influences")
                ax.set_yticks([])
                ax.set_ylabel("")

                sns.despine(left=True)
                plt.tight_layout()
                log_figure_pdf(f"hyperiors-combined-{var_name}-y-labels")

                ax_top.set_ylabel("")
                ax_bottom.set_ylabel("")
                plt.tight_layout()
                log_figure_pdf(f"hyperiors-{var_name}")











        # plt.show()
    def plot_multiple_hyperior(self, columns=8, rows=8):
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
        base_hyper_loc_samples = np.array(posterior["base-hyper"]).flatten()
        base_hyper_std_samples = np.array(posterior["base-hyper_var"]).flatten()
        base_hyper_samples = self.simulate_hyperior_expectation(base_hyper_loc_samples,
                                                                  base_hyper_std_samples)

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

            option_hyper_location_samples = np.array(
                posterior["influences-mean-hyperior"].loc[:, :, option_name]
            ).flatten()
            option_hyper_spread_samples = np.array(
                posterior["influences-stddevs-hyperior"].loc[:, :, option_name]
            ).flatten()
            option_hyper_samples = self.simulate_hyperior_expectation(option_hyper_location_samples,
                                                                      option_hyper_spread_samples)

            feature_hyper_sanmples_list.append(option_hyper_samples)
        return feature_hyper_sanmples_list, feature_df_list, feature_names

    def simulate_hyperior_expectation(self, option_hyper_location_samples, option_hyper_spread_samples):
        return np.array([float(sps.norm(mean, stddev).rvs(1)[0]) for mean, stddev in
                         zip(option_hyper_location_samples, option_hyper_spread_samples)])

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
        # plt.show()


    def calculate_loss(self, df, selected_workloads):
        """Calculate the sum of the lowest KLD per workload per option for unrepresented workloads."""
        # Create a mask for workloads that are yet unrepresented
        mask = ~(df['env'].isin(selected_workloads))

        unrepresented_df = df[mask]

        # Create a mask for rows with selected workloads in the `represented_by_env` column
        mask_selected = unrepresented_df['represented_by_env'].isin(selected_workloads)

        # Filter to find the lowest KLD per workload per option among those having a selected workload
        lowest_kld_per_workload_option = unrepresented_df[mask_selected].groupby(['option', 'env'])['kld'].min()

        # Sum up all the lowest KLDs
        return lowest_kld_per_workload_option.sum()


    def greedy_representation_set_generation(self, df, threshold):
        """Select workloads progressively until all workloads are represented and log the loss."""
        results_screening = []
        no_influences_original = len(df)
        for workload in df['represented_by_env'].unique():
            selected_workloads = []
            remaining_df = df.copy()
            selected_workloads.append(workload)
            covered_influences = remaining_df[(remaining_df['kld'] < threshold) & (remaining_df['represented_by_env'] == workload)]
            id_cols = ["option", "env"]
            to_be_removed_combinations = covered_influences[id_cols]
            n_covered = len(to_be_removed_combinations)
            # Merging on the columns of interest and marking rows to keep
            merged = remaining_df.merge(to_be_removed_combinations, on=id_cols, how='left', indicator=True)
            # Filtering out rows found in df
            remaining_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
            # remaining_df = remaining_df.loc[remaining_df["env"] != best_workload]

            # kld_reduction = float(merged[merged['_merge'] == 'both']["kld"].sum())
            if not remaining_df.empty:
                # we first determine the minimum kld for each env compared to the selected workloads.
                # then, we average over these minimum klds to get an expectation of the information loss for that option when encountering a new environment
                # the last mean averages over all options to account for variations in option count per system
                kld_remaining = remaining_df.loc[remaining_df["represented_by_env"].isin(selected_workloads)].groupby(["option", "env"])["kld"].min().groupby(["option"]).sum().sum()

                kld_remaining_expected_per_env = remaining_df.loc[remaining_df["represented_by_env"].isin(selected_workloads)].groupby(["option", "env"])["kld"].mean().mean()
            else:
                kld_remaining = 0
                kld_remaining_expected_per_env = 0
            kld_remaining_whole_df = df.loc[df["represented_by_env"].isin(selected_workloads)].groupby(["option", "env"])["kld"].min().groupby(["option"]).sum().sum()

            kld_reduction = None
            loss_after_first_env = kld_remaining
            minimum_rep_set_size_from_saturation = 1
            prev_cum_kld = kld_remaining
            no_remaining_opt = len(remaining_df["option"].unique())
            no_remaining_envs = len(remaining_df["env"].unique())
            # Log the step number, best workload, and information loss
            results_screening.append({
                'Step': 1,
                'Selected Workload': workload,
                'Number of Represented Workload-Expcific Influences': n_covered,
                'Information Loss Within Remaining Influences': kld_remaining,
                'Information Loss': kld_remaining_whole_df,
                'Expected Information Loss Remaining Per Option Per Env': kld_remaining_expected_per_env,
                'Information Loss Reduction': kld_reduction,
                'Unfinished Options': no_remaining_opt,
                'Unrepresented Envs': no_remaining_envs,
            })

        df_results_screening = pd.DataFrame(results_screening)
        df_results_screening["Relative Number of Represented Workload-Expcific Influences"] = df_results_screening["Number of Represented Workload-Expcific Influences"]/no_influences_original
        log_dataframe(df_results_screening, "representation-selection-log-screening-single-rep-sets")

        selected_workloads = []
        remaining_df = df.copy()
        n_opts = len(df["option"].unique())
        n_envs = len(df["env"].unique())

        results = [
            {
                'Step': 0,
                'Selected Workload': "No workloads",
                'Number of Represented Workload-Expcific Influences': None,
                'Information Loss Within Remaining Influences': np.inf,
                'Information Loss': np.inf,
                'Information Loss Reduction': None,
                'Unfinished Options': n_opts,
                'Unrepresented Envs': n_envs,
            }

        ]
        # matrix = pd.pivot_table(remaining_df, values='kld', index='env', columns='represented_by_env', fill_value=0)
        step = 1
        prev_cum_kld = np.inf
        minimum_rep_set_size_from_saturation = None
        loss_after_first_env = None
        saturation_percentage = 0.10
        while not remaining_df.empty:
            # Find the workload that represents the most others
            counts = remaining_df[remaining_df['kld'] < threshold].groupby('represented_by_env')['kld'].count()
            if counts.empty:
                break


            remaining_klds = {}
            for workload in df['represented_by_env'].unique():
                if workload not in selected_workloads:
                    kld_remaining = remaining_df.loc[remaining_df["represented_by_env"].isin([*selected_workloads, workload])].groupby(["option", "env"])["kld"].min().groupby(["option"]).sum().sum()
                    remaining_klds[workload] = kld_remaining
            best_workload = min(remaining_klds, key=remaining_klds.get)


            # best_workload = counts.idxmax()

            # Add the best workload to the selected list
            selected_workloads.append(best_workload)
            # Calculate the loss after each step
            # loss = self.calculate_loss(remaining_df, selected_workloads)
            # own calculation
            covered_influences = remaining_df[(remaining_df['kld'] < threshold) & (remaining_df['represented_by_env'] == best_workload)]
            id_cols = ["option", "env"]
            to_be_removed_combinations = covered_influences[id_cols]
            n_covered = len(to_be_removed_combinations)
            # Merging on the columns of interest and marking rows to keep
            merged = remaining_df.merge(to_be_removed_combinations, on=id_cols, how='left', indicator=True)
            # Filtering out rows found in df
            remaining_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
            # remaining_df = remaining_df.loc[remaining_df["env"] != best_workload]

            # kld_reduction = float(merged[merged['_merge'] == 'both']["kld"].sum())
            if not remaining_df.empty:
                # we first determine the minimum kld for each env compared to the selected workloads.
                # then, we average over these minimum klds to get an expectation of the information loss for that option when encountering a new environment
                # the last mean averages over all options to account for variations in option count per system
                kld_remaining = remaining_df.loc[remaining_df["represented_by_env"].isin(selected_workloads)].groupby(["option", "env"])["kld"].min().groupby(["option"]).sum().sum()
                kld_remaining_expected_per_env = remaining_df.loc[remaining_df["represented_by_env"].isin(selected_workloads)].groupby(["option", "env"])["kld"].mean().mean()
            else:

                kld_remaining = 0
                kld_remaining_expected_per_env = 0

            kld_remaining_whole_df = df.loc[df["represented_by_env"].isin(selected_workloads)].groupby(["option", "env"])["kld"].min().groupby(["option"]).sum().sum()
            if prev_cum_kld == np.inf:
                kld_reduction = None
                loss_after_first_env = kld_remaining
                minimum_rep_set_size_from_saturation = 1
            else:
                kld_reduction = prev_cum_kld -  kld_remaining
                relative_reduction = kld_reduction / prev_cum_kld
                if minimum_rep_set_size_from_saturation == 1 and kld_reduction < loss_after_first_env * saturation_percentage:
                    minimum_rep_set_size_from_saturation = len(selected_workloads)
            prev_cum_kld = kld_remaining
            no_remaining_opt = len(remaining_df["option"].unique())
            no_remaining_envs = len(remaining_df["env"].unique())
            # else:
            #     # kld_remaining = 0.0
            #     kld_reduction = prev_cum_kld
            #     no_remaining_opt = 0

            # Log the step number, best workload, and information loss
            results.append({
                'Step': step,
                'Selected Workload': best_workload,
                'Number of Represented Workload-Expcific Influences': n_covered,

                'Information Loss Within Remaining Influences': kld_remaining,
                'Information Loss': kld_remaining_whole_df,
                'Expected Information Loss Remaining Per Option Per Env': kld_remaining_expected_per_env,
                'Information Loss Reduction': kld_reduction,
                'Unfinished Options': no_remaining_opt,
                'Unrepresented Envs': no_remaining_envs,
            })

            # Remove all workloads that the best workload represents well (kld < threshold)
            # remaining_df = remaining_df[~((remaining_df['represented_by_env'] == best_workload) & (remaining_df['kld'] < threshold))]

            step += 1
        log_df = pd.DataFrame(results)
        log_df["Relative Number of Represented Workload-Expcific Influences"] = log_df["Number of Represented Workload-Expcific Influences"]/no_influences_original

        log_dataframe(log_df, "representation-selection-log")
        mlflow.log_metric("minimum_rep_set_size_from_saturation", minimum_rep_set_size_from_saturation)
        rel_number_of_workloads = minimum_rep_set_size_from_saturation / n_opts if minimum_rep_set_size_from_saturation is not None else None
        mlflow.log_metric("relative_minimum_rep_set_size_from_saturation", rel_number_of_workloads)
        return log_df, minimum_rep_set_size_from_saturation, df_results_screening



# This function will search for .netcdf files that are in directories containing the word "partial"
def find_partial_netcdf_files(path):
    netcdf_files = []
    params = []
    metrics_list = []
    for root, dirs, files in os.walk(path):
        if "partial" in root:
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
        "--netcdf",
        type=str,
        help="path to results parquet file or similar",
        # default="single-trace-debugger/arviz_data-6e5a3d1d-032d-428f-aea7-16bccdf79158.netcdf",
        default="single-trace-debugger/240321-10-01-40-uncertainty-learning-2024-FKQHtsnsk8",
        # default="single-trace-debugger/240321-13-42-40-uncertainty-learning-2024-DnfMtUuQWk",
    )
    args = parser.parse_args()

    netcdf_parent = args.netcdf
    partial_netcdf_files, all_params, all_metrics = find_partial_netcdf_files(netcdf_parent)

    # Combine the lists and sort them
    combined = list(zip(partial_netcdf_files, all_params, all_metrics))
    combined.sort(key=lambda x: x[1]["params.software-system"] != "jump3r")

    replication_lbl = get_date_time_uuid()
    with mlflow.start_run(run_name=replication_lbl):
        for i, (netcdf, params, metrics) in enumerate(combined):
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

def log_figure_pdf(plot_name, close=True):
    file_name = "%s.pdf" % plot_name
    plt.savefig(file_name, bbox_inches="tight")
    if close:
        plt.close()
    mlflow.log_artifact(file_name)

if __name__ == "__main__":
    main()
