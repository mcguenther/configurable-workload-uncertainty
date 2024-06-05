import copy
import json

import matplotlib

matplotlib.use("Agg")
from datetime import datetime
import streamlit as st
import pandas as pd
import seaborn as sns
import os
import glob
import matplotlib.pyplot as plt
import time
from fractions import Fraction
import numpy as np
import streamlit.components.v1 as components
import base64
from microRQdashboard import get_subfolders, read_and_combine_csv, embed_pdf, bayes_palette, comparison_palette


def read_sws_insights(root_dir):
    interesting_files = [
        "representation-selection-log.csv",
        "representation-selection-log-screening-single-rep-sets.csv",
        "kldivs-towards-hyperior-per-option.csv",
        "outliers-hyperiors-most-iqr.csv",
        "invariant-options.json",
        "metrics.json",
        "training-params.json",
    ]

    data = {}
    for main_dir in os.listdir(root_dir):
        main_dir_path = os.path.join(root_dir, main_dir)
        if os.path.isdir(main_dir_path):
            for subdir, _, files in os.walk(main_dir_path):
                subdir_data = {}
                kdes = []

                for file_name in files:
                    file_path = os.path.join(subdir, file_name)
                    if file_name in interesting_files:
                        if file_name.endswith('.csv'):
                            subdir_data[file_name] = pd.read_csv(file_path)
                        elif file_name.endswith('.json'):
                            with open(file_path, 'r') as json_file:
                                subdir_data[file_name] = json.load(json_file)
                    elif file_name.endswith('.pdf') and "hyperiors-" in file_name:
                        kdes.append(file_path)

                if kdes:
                    subdir_data['kdes'] = kdes

            if 'training-params.json' in subdir_data:
                sws = subdir_data['training-params.json']['params.software-system']
                data[sws] = subdir_data
                # data[subdir] = subdir_data

    return data


def main():
    st.set_page_config(
        page_title="MCMC Multilevel Models Insights",
        page_icon="📊",  # Update path accordingly
        layout="wide"
    )
    # st.title("CSV File Processor and Visualizer from Subfolders")

    config_ok = False
    with st.sidebar:
        st.write("## Folder Selection")
        parent_folder = st.text_input(
            "Enter the path of the parent folder",
            value="/home/jdorn/results/localflow/jdorn-modelinsights/",
        )
        if parent_folder:
            subfolders = get_subfolders(parent_folder)
            folder_names = [
                os.path.basename(f) for f in sorted(subfolders, reverse=True)
            ]
            selected_subfolders = st.multiselect(
                "Select Subfolders",
                folder_names,
                # default=["240316-20-16-41-aggregation-bWn627g4jN",
                #          "240317-14-51-05-aggregation-dNPobw6xky",
                #          "240318-14-27-26-aggregation-WLFgXnWyqc",
                #          "240319-11-56-40-aggregation-gn5W8tJhaY"],
                default=[
                    # "240601-22-15-26-2024-06-01_22-15-26-3hEyEdUK8d",
                    "240603-23-57-17-2024-06-03_23-57-17-bQ232vvGXt",
                ],
            )

            if (
                    selected_subfolders
            ):  # st.button("Process CSV Files in Selected Subfolders"):
                whole_folders = [
                    os.path.join(parent_folder, f) for f in selected_subfolders
                ]
                results_list = [read_sws_insights(f) for f in whole_folders][0]
                if not results_list:
                    st.error(
                        "No CSV files found in the selected subfolders or the combined DataFrame is empty."
                    )
                else:
                    config_ok = True
    if not config_ok:
        st.error("please check config in sidebar")
        exit(21)
    else:

        selected_systems = st.multiselect("Subselect", results_list,
                                          default=[f for f in results_list if "artif" not in f and "kanzi" not in f])
        results_list = {k: v for k, v in results_list.items() if k in selected_systems}

        st.write("# hello world")

        # pdf_paths = [(sws, r["kde_paths"]) for sws, r in results_list.items()]
        # for (sws, pdf_paths), i in zip(pdf_paths, range(3)):
        #     f"# {sws}"
        #     for pdf_path in pdf_paths:
        #         start = pdf_path.find("hyperiors-") + len("hyperiors-")
        #         end = pdf_path.find(".pdf")
        #         option_lbl= pdf_path[start:end]
        #         st.write(f"## {option_lbl}")
        #         embed_pdf(
        #             pdf_path
        #         )

        analyses = {
            "representativeness_plot": representativeness_plot,
            "hyper level": hyper_level_analysis,
            "invariance": plot_invariance_option_results,

        }
        tabs = st.tabs(tabs=analyses, )

        for tab, func in zip(tabs, analyses.values()):
            with tab:
                func(results_list)

        # plot_invariance_option_results(results_list)

        st.divider()
        with st.expander("Full data", expanded=False):
            results_list


def get_first_value_below_threshold(df, threshold, label):
    for index, value in enumerate(df[label]):
        if value:
            if value < threshold:
                return index, value
    return len(df), 0


def representativeness_plot(results_list):
    rep_dfs = [(sws, r["representation-selection-log.csv"]) for sws, r in results_list.items()]
    rep_screening_dfs = [(sws, r["representation-selection-log-screening-single-rep-sets.csv"]) for sws, r in
                         results_list.items()]
    res_df = pd.DataFrame(columns=['sws', 'Average Initial Information Loss', 'Optimal Initial Information Loss',
                                   'Opt Init Loss per Influence', 'Loss reduce 2nd Workload in proz', 'Rep. Set Size',
                                   'Loss with rep set', 'Loss with rep set per Infl.', 'no. of unfinished options',
                                   'unf opt proz', 'no. of Unrep Envs', 'unrep envs proz', 'max rep set size'])
    # st.write(rep_dfs)

    loss_str = 'Information Loss'
    for pos in range(len(rep_dfs)):
        sws = rep_dfs[pos][0]
        rep_df = rep_dfs[pos][1]
        rep_df['newStep'] = rep_df['Step'] - 1
        rep_screening_df = rep_screening_dfs[pos][1]
        min_start = rep_screening_df['Information Loss'].min()
        num_infl = (rep_df.loc[0]["Unfinished Options"] * rep_df.loc[0]["Unrepresented Envs"])
        max_start = rep_screening_df['Information Loss'].max()
        st.write(f"Max start: {max_start}")
        reduce2nd = round(100 * (rep_df.loc[2]['Information Loss Reduction']) / rep_df.loc[1][loss_str])

        mean_start_loss = rep_screening_df[loss_str].mean()
        stop_step, stop_value = get_first_value_below_threshold(rep_df, (0.1 * rep_screening_df[loss_str].min()),
                                                                'Information Loss Reduction')
        res_df.loc[pos] = [sws,  # sws
                           mean_start_loss,  # Average Initial Information Loss
                           min_start,  # optimal initial information loss
                           min_start / num_infl,  # opt. init loss per influence
                           reduce2nd,  # Loss reduce 2nd Workload in proz
                           (stop_step - 1),  # rep set size
                           rep_df.loc[stop_step - 1][loss_str],  # loss with rep set size
                           rep_df.loc[stop_step - 1][loss_str] / num_infl,  # loss with rep set size per infl
                           rep_df.loc[stop_step - 1]["Unfinished Options"],  # no. of unfinished options
                           round(100 * rep_df.loc[stop_step - 1]["Unfinished Options"] / rep_df.loc[0][
                               "Unfinished Options"]),  # no unf opt proz
                           rep_df.loc[stop_step - 1]["Unrepresented Envs"],  # no of unrep envs
                           round(100 * rep_df.loc[stop_step - 1]["Unrepresented Envs"] / rep_df.loc[0][
                               "Unrepresented Envs"]),  # no of unrep envs proz
                           (len(rep_df) - 1)]  # max rep set size

    # Compute the mean for each column
    average_row = res_df.mean(numeric_only=True).to_frame().T

    # Append the average row to the DataFrame
    average_row['sws'] = 'Average'  # Or any other label you want to give this row
    res_df = pd.concat([res_df, average_row], ignore_index=True)

    st.write("# Result Table")
    st.dataframe(res_df)

    st.write("## Latex")
    latex_str = res_df.to_latex(index=True, multirow=True, multicolumn=True,
                                multicolumn_format='c', column_format='r' + 'r' * res_df.shape[1],
                                escape=False,
                                float_format="{:0.1f}".format)

    st.latex(latex_str)

    st.write("# PLOTS")

    for pos in range(len(rep_dfs)):
        sns.set_style("whitegrid")
        # sns.set_context("talk", rc={"axes.labelsize": "0.7", "xtick.labelsize": "0.7", "ytick.labelsize": "0.7", "legend.fontsize": "0.7", "axes.titlesize": "0.7"})
        sns.set_context("talk", font_scale=0.8)

        sws = rep_dfs[pos][0]
        st.write(f"# SWS: {sws}")
        rep_df = rep_dfs[pos][1]
        rep_df['newStep'] = rep_df['Step'] - 1  # fix position swarmplot
        rep_screening_df = rep_screening_dfs[pos][1]
        mean_start_loss = rep_screening_df[loss_str].mean()
        stop_step, stop_value = get_first_value_below_threshold(rep_df, (0.1 * rep_screening_df[loss_str].min()),
                                                                'Information Loss Reduction')
        # fig = plt.figure(figsize=(5,1.75))
        fig = plt.figure(figsize=(6, 2.05))
        st.dataframe(rep_df)
        line_color = bayes_palette[0]
        swarm_color = comparison_palette[0]
        sns.swarmplot(data=rep_screening_df, x="Step", y=loss_str, color=swarm_color)

        sns.lineplot(data=rep_df, x="newStep", y=loss_str, marker="o", color=line_color)
        # sns.scatterplot(data=rep_screening_df, x="Step", y=loss_str, color=swarm_color)
        # plt.axhline(y=mean_start_loss, color="orange", linestyle='--', label="Mean Starting Information Loss")
        set_size_color = "#444444"
        plt.axvline(x=(stop_step - 2), color=set_size_color, linestyle='--',
                    label=f"Number Envs in Set")  # -1 for last step, -1 for fix swarmplot
        plt.xticks(ticks=rep_df['newStep'], labels=rep_df['Step'])
        # plt.title(f"{sws}")
        plt.xlabel("Step")
        plt.ylabel("Information Loss")
        plt.ylim(bottom=0)
        plt.xlabel("Representation Set Size")
        plt.xlim(left=-0.85)
        sns.despine(left=True)
        # plt.legend()
        tmp_file = f"log_{sws}.pdf"
        plt.savefig(tmp_file, bbox_inches='tight')
        # plt.show()
        # st.pyplot(plt.gcf())
        with st.expander(label="Get Your PDF Now COMPLETELY FREE!!!1!11!!", expanded=False):
            embed_pdf(
                tmp_file
            )
        st.pyplot(fig=fig)
        min_start = rep_screening_df['Information Loss'].min()
        # res_df.loc[pos] = [sws, mean_start_loss, min_start,
        #                    (stop_step-1),
        #                    rep_df.loc[stop_step-1]["Unfinished Options"], round(100 * rep_df.loc[stop_step-1]["Unfinished Options"] / rep_df.loc[0]["Unfinished Options"]),
        #                    rep_df.loc[stop_step-1]["Unrepresented Envs"], round(100 * rep_df.loc[stop_step-1]["Unrepresented Envs"] / rep_df.loc[0]["Unrepresented Envs"]),
        #                    (len(rep_df)-1)]


def hyper_level_analysis(results_list):
    df_hyper_iqr_outliers = [(sws, r["outliers-hyperiors-most-iqr.csv"]) for sws, r in results_list.items()]

    total_number = sum([len(df) for sws, df in df_hyper_iqr_outliers])
    st.metric("Total Hyper Poster Outliers", total_number)
    st.metric("Average Hyper Poster Outliers Per System", round(total_number / len(df_hyper_iqr_outliers), 2))

    # Create a new list of dataframes with the 'sws' column added
    df_hyper_iqr_outliers_with_sws = [(sws, df.assign(sws=sws)) for sws, df in df_hyper_iqr_outliers]

    # Merge all dataframes
    merged_df_with_sws = pd.concat([df for _, df in df_hyper_iqr_outliers_with_sws])

    # Get the top 5 options with the highest credible_interval_width
    top_5_df_with_sws = merged_df_with_sws.nlargest(5, "credible_interval_width")
    st.dataframe(top_5_df_with_sws)

    for sws, df in df_hyper_iqr_outliers:
        f"# {sws}"
        st.dataframe(df)


def plot_invariance_option_results(results_list):
    kldivs_raw_data = {sws: r["kldivs-towards-hyperior-per-option.csv"] for sws, r in results_list.items()}
    tups = []
    plt.close()
    for sws, sws_kld_df in kldivs_raw_data.items():
        st.write(f"## {sws} for KLDs")

        n_envs = sws_kld_df.groupby("option")["kldiv"].count()[0]
        st.write(n_envs)
        count_variant_influences = sws_kld_df.loc[sws_kld_df["kldiv"] > 1.].groupby("option")["kldiv"].count() / n_envs
        st.write("Number of informative influences")
        st.write(count_variant_influences.to_dict())

        new_tups = [(sws, option, kld) for option, kld in count_variant_influences.items()]
        tups.extend(new_tups)

    df_kld_informs = pd.DataFrame(tups, columns=["Software System", "Option", "KLD"])
    st.dataframe(df_kld_informs)
    n_only_informative = np.sum(df_kld_informs["KLD"] == 1.0)
    st.write(f"Number of options with only informative influences: {n_only_informative}")
    st.dataframe(df_kld_informs[df_kld_informs["KLD"] == 1.0].groupby("Option").count())

    plt.figure(figsize=(8.5, 3.5))
    sns.swarmplot(data=df_kld_informs, x="KLD", hue="Software System", size=8)
    plt.xlabel("Share of Informative Influences")

    # Customize legend
    plt.legend(title='Software System', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    st.pyplot(plt.gcf())
    tmp_file = "sws-kldevs.pdf"
    plt.savefig(tmp_file, bbox_inches='tight')

    with st.expander(label="Get Your PDF Now COMPLETELY FREE!!!1!11!!", expanded=False):
        embed_pdf(tmp_file)

    invariant_ratios = [(sws, r["invariant-options.json"]["ratio_invar_options"]) for sws, r in results_list.items()]
    ratio_lbl = "Ratio of Invariant Options"
    df_opt_invar = pd.DataFrame(invariant_ratios, columns=["Software System", ratio_lbl])
    df_opt_invar_sorted = df_opt_invar.sort_values(by=ratio_lbl)
    st.dataframe(df_opt_invar_sorted)
    df = df_opt_invar
    # Set the style for the plot
    sns.set(style="whitegrid")
    # Create the violin plot with swarm scatters

    scale = 3
    ratio = 1 / 5
    plt.figure(figsize=(2 * scale, 3 * scale * ratio), dpi=300)
    violin_color = bayes_palette[0]
    sns.violinplot(x=ratio_lbl, data=df,
                   inner=None,
                   bw_adjust=0.35,
                   # cut=1,
                   scale='width',
                   linewidth=1.25,
                   color=violin_color)

    sns.swarmplot(x=ratio_lbl, data=df, color='k', alpha=1.0, edgecolor='w', linewidth=1.0)
    # Customize the plot
    plt.xlim(0.0, 1.0)
    plt.xlabel("%s" % ratio_lbl)  # , fontsize=14)
    plt.ylabel("")
    # Add grid lines
    plt.grid(True, linestyle='--', linewidth=0.5)
    # Remove the top and right spines for a cleaner look
    sns.despine()
    plt.tight_layout()
    tmp_file = "invariant-poptions.pdf"
    plt.savefig(tmp_file, bbox_inches='tight')
    # plt.show()
    st.pyplot(plt.gcf())
    with st.expander(label="Get Your PDF Now COMPLETELY FREE!!!1!11!!", expanded=False):
        embed_pdf(
            tmp_file
        )


def draw_multitask_paper_plot(combined_df, system_col="params.software-system",
                              model_col="params.model",
                              cat_col="params.pooling_cat", ):
    st.dataframe(combined_df)
    st.write("Filter models ...")
    wanted_models = {
        "mcmc": "Bayesian",
        # "mean-pred": "Mean",
        "mcmc-adaptive-shrinkage": "Bayesian",
        # "model_lasso_reg_no_pool": "Lasso",
        # "model_lasso_reg_cpool": "Lasso",
        "model_lassocv_reg_no_pool": "$\\hat{\\Pi}^\\mathit{np}_\\mathit{Lasso}$",
        "model_lassocv_reg_cpool": "$\\hat{\\Pi}^\\mathit{cp}_\\mathit{Lasso}$",
        # "dummy": "mean",
    }
    all_models = list(combined_df[model_col].unique())
    # st.write(all_models)
    filtered_df = combined_df[combined_df[model_col].isin(wanted_models)]
    filtered_df["params.model"] = filtered_df["params.model"].map(wanted_models)

    col_mapper = {
        "mape": "MAPE",
        # "mape_ci":"$\\text{MAPE}_\\text{CI}$"
        "mape_ci": "MAPEci",
        "pmape_ci": "pMAPE",
        # "test_set_log-likelihood":"ell",

    }
    # filtered_df = filtered_df[filtered_df[cat_col].isin(s_poolings)]
    st.write("Filter metrics ...")
    metrics_raw = get_metrics_in_df(filtered_df)
    melted_df = filtered_df.melt(
        id_vars=[col for col in filtered_df.columns if col not in metrics_raw],
        value_vars=metrics_raw,
        var_name="Metric",
        value_name="Value",
    )
    melted_df["Metric"] = melted_df["Metric"].str.replace("metrics.", "")
    melted_df = melted_df.loc[melted_df["Metric"].isin(col_mapper)]
    melted_df["Metric"] = melted_df["Metric"].replace(col_mapper)
    # st.dataframe(melted_df)

    pooling_cat_lbl = "Pooling"
    relative_train_size_lbl = "Relative Train Size"
    model_lbl = "Model"
    subject_system_lbl = "Subject System"
    params_mapper = {
        "params.model": model_lbl,
        "params.software-system": subject_system_lbl,
        "params.relative_train_size": relative_train_size_lbl,
        "params.pooling_cat": pooling_cat_lbl,
    }
    melted_df = melted_df.rename(columns=params_mapper)

    bnp = "$\\tilde{\Pi}^\\mathit{np}_B$"
    bpp = "$\\tilde{\\Pi}^\\mathit{pp}_B$"
    bcp = "$\\tilde{\\Pi}^\\mathit{cp}_B$"
    melted_df[model_lbl].loc[(melted_df[model_lbl] == "Bayesian") & (melted_df[pooling_cat_lbl] == "no")] = bnp
    melted_df[model_lbl].loc[(melted_df[model_lbl] == "Bayesian") & (melted_df[pooling_cat_lbl] == "partial")] = bpp
    melted_df[model_lbl].loc[(melted_df[model_lbl] == "Bayesian") & (melted_df[pooling_cat_lbl] == "complete")] = bcp

    # Wrapping "Subject System" column values in {}

    melted_df = melted_df.loc[~melted_df[subject_system_lbl].isin(["artificial", "kanzi"])]
    plot_df = copy.deepcopy(melted_df)
    melted_df = melted_df.drop(columns=["Pooling"])
    melted_df[subject_system_lbl] = melted_df[subject_system_lbl].apply(lambda x: f'\\sws{{{x}}}')
    st.dataframe(melted_df)

    # melted_df = melted_df.loc[melted_df[relative_train_size_lbl].isin([0.25,0.5,0.75,1,3])]

    melted_df["Value"] = melted_df["Value"].astype(float)
    melted_df["Value"] = melted_df["Value"].astype(float)
    replacements = {
        # "0.125000":"\\multicolumn{1}{c}{$\\sfrac{1}{8}$}",
        # "0.250000":"\\multicolumn{1}{c}{$\\sfrac{1}{4}$}",
        # "0.500000" :"\\multicolumn{1}{c}{$\\sfrac{1}{2}$}",
        # "0.750000":"\\multicolumn{1}{c}{$\\sfrac{3}{4}$}",
        # "1.000000":"\\multicolumn{1}{c}{$1$}",
        # "2.000000":"\\multicolumn{1}{c}{$2$}",
        # "3.000000":"\\multicolumn{1}{c}{$3$}",
        "0.125000": "$\\sfrac{1}{8} \\vert \\mathcal{O} \\vert$",
        "0.250000": "$\\sfrac{1}{4} \\vert \\mathcal{O} \\vert$",
        "0.500000": "$\\sfrac{1}{2} \\vert \\mathcal{O} \\vert$",
        "0.750000": "$\\sfrac{3}{4} \\vert \\mathcal{O} \\vert$",
        "1.000000": "$1 \\vert \\mathcal{O} \\vert$",
        "2.000000": "$2 \\vert \\mathcal{O} \\vert$",
        "3.000000": "$3 \\vert \\mathcal{O} \\vert$",
        r"Subject System": "",
        r"Relative Train Size": "$\\vert \\mathcal{D}^\mathit{train} \\vert$",
        r"\\\\ \& \& \& \& \& \& \& \& \& \& \& \& ": "",
        r" &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  \\": "",
    }
    mape_df = melted_df[['Subject System', 'Relative Train Size', 'Model', "Metric", "Value"]]
    st.write("## Latex Tables.")
    with st.expander("all MAPES!", expanded=False):
        # grouped_mape = mape_df.groupby(['Subject System', 'Relative Train Size', 'Model', pooling_cat_lbl, "Metric",]).mean().reset_index()
        # st.dataframe(grouped_mape)
        all_mapes_ape = mape_df.loc[mape_df[relative_train_size_lbl].isin([0.25, 0.5, 0.75, 1])]
        initial_pivot = all_mapes_ape.pivot_table(index=['Subject System'],
                                                  columns=['Model', 'Metric', 'Relative Train Size'],
                                                  values='Value',
                                                  aggfunc='mean')
        st.dataframe(initial_pivot)
        rounded_scores = initial_pivot.applymap(lambda x: float(round(x, 1)) if isinstance(x, (int, float)) else x)
        st.dataframe(rounded_scores)
        rounded_scores.to_csv("./results-rq1.csv")
        latex_str = rounded_scores.to_latex(index=True, multirow=True, multicolumn=True,
                                            multicolumn_format='c', column_format='r' + 'r' * rounded_scores.shape[1],
                                            escape=False,
                                            float_format="{:0.1f}".format)

        for pattern, replacement in replacements.items():
            latex_str = latex_str.replace(pattern, replacement)
        st.latex(latex_str)
        st.write(os.getcwd())

    with st.expander("MAPE ONLY", expanded=False):
        # grouped_mape = mape_df.groupby(['Subject System', 'Relative Train Size', 'Model', pooling_cat_lbl, "Metric",]).mean().reset_index()
        # st.dataframe(grouped_mape)

        mapes_only_df = melted_df.loc[melted_df[relative_train_size_lbl].isin([0.125, 0.25, 0.5, 0.75, 1, 2, 3])]
        mape_only_df = mapes_only_df.loc[mapes_only_df["Metric"].isin([col_mapper["mape"]])]
        mape_only_df.drop(columns=["Metric"])
        # initial_pivot = mape_only_df.pivot_table(index=['Subject System'],
        #                                  columns=['Model', 'Metric', 'Relative Train Size'],
        #                                  values='Value',
        #                                  aggfunc='mean')
        initial_pivot = mape_only_df.pivot_table(index=['Relative Train Size'],
                                                 columns=['Model', 'Metric', ],
                                                 values='Value',
                                                 aggfunc='mean')
        st.dataframe(initial_pivot)
        rounded_scores = initial_pivot.applymap(lambda x: float(round(x, 1)) if isinstance(x, (int, float)) else x)
        st.dataframe(rounded_scores)
        rounded_scores.to_csv("./results-rq1.csv")
        latex_str = rounded_scores.to_latex(index=True, multirow=True, multicolumn=True,
                                            multicolumn_format='c', column_format='r' + 'r' * rounded_scores.shape[1],
                                            escape=False,
                                            float_format="{:0.1f}".format)
        for pattern, replacement in replacements.items():
            latex_str = latex_str.replace(pattern, replacement)

        st.text_area("output", latex_str)

    with st.expander("pMAPE ONLY", expanded=True):
        mapeci_only_df = mape_df.loc[melted_df["Metric"].isin([col_mapper["pmape_ci"]])]
        # st.dataframe(mapeci_only_df)
        mapeci_only_df = mapeci_only_df.drop(columns=["Metric"])

        mapeci_only_df = mapeci_only_df.loc[
            mapeci_only_df[relative_train_size_lbl].isin([0.125, 0.25, 0.5, 0.75, 1, 2, 3])]
        pivot_df = mapeci_only_df.pivot_table(index=['Subject System'],
                                              columns=['Relative Train Size', 'Model'],
                                              values='Value',
                                              aggfunc='mean')

        # Calculate the mean for each column, skipping non-numeric data automatically
        column_means = pivot_df.mean()

        # 2. Append the mean row to the DataFrame.
        # Note: Given your DataFrame's complexity, especially with multi-index columns, adjust as needed.
        pivot_df.loc['Mean'] = column_means
        rounded_scores = pivot_df.applymap(lambda x: float(round(x, 1)) if isinstance(x, (int, float)) else x)

        for rel_train_size in rounded_scores.columns.levels[0]:
            # For each 'Relative Train Size', find the model with the minimum error for each 'Subject System'
            min_error_models = rounded_scores[rel_train_size].idxmin(axis=1)
            # Iterate through each 'Subject System' and the corresponding model with the lowest error
            for system, min_model in min_error_models.items():
                # Prepend "X" to the value of the cell with the minimum error
                rounded_scores.loc[system, (rel_train_size, min_model)] = "\\cellcolor{tabSignal}" + str(
                    rounded_scores.loc[system, (rel_train_size, min_model)])
                # pivot_df.loc[system, (rel_train_size, min_model)] = "X" + str(pivot_df.loc[system, (rel_train_size, min_model)])
        st.dataframe(rounded_scores)
        rounded_scores.to_csv("./results-rq1.csv")
        # column_format = 'l' + ('|' + 'r' * subcols_per_model) * num_models

        column_format = 'r'
        num_columns = rounded_scores.shape[1]
        # Loop through each column, starting from the first data column (ignoring the index column)
        for i in range(1, num_columns + 1):
            # For every third colu mn starting from the second, use '||' instead of '|'
            if i % 3 == 1:
                column_format += '||r'
            else:
                column_format += '|r'
        latex_str = rounded_scores.to_latex(index=True, multirow=True, multicolumn=True,
                                            multicolumn_format='c',
                                            column_format=column_format,  # 'r' + '|r' * rounded_scores.shape[1],
                                            escape=False,
                                            float_format="{:0.1f}".format)
        for pattern, replacement in replacements.items():
            latex_str = latex_str.replace(pattern, replacement)
        # latex_str = latex_str.replace("\\\\", "\\\\"+os.linesep)
        # st.latex(latex_str)
        st.text_area("output", latex_str)

    # wanted_models = {
    #     "mcmc": "Bayesian",
    #     # "mean-pred": "Mean",
    #     "mcmc-adaptive-shrinkage": "Bayesian",
    #     # "model_lasso_reg_no_pool": "Lasso",
    #     # "model_lasso_reg_cpool": "Lasso",
    #     "model_lassocv_reg_no_pool": "$\\hat{\\Pi}^\\mathit{np}_\\mathit{Lasso}$",
    #     "model_lassocv_reg_cpool": "$\\hat{\\Pi}^\\mathit{cp}_\\mathit{Lasso}$",
    #     "dummy": "mean",
    # }

    with st.spinner("Waiting for plot to be rendered"):
        model_order = ["Lasso", "Bayesian", "Mean"]
        # bayes_palette = sns.color_palette("YlOrBr", 3)
        bayes_palette = ["#47AEED", "#398CBF", "#2F729C"]  # ["blue", "green", "red"] # sns.color_palette("flare", 3)
        model_colors = {
            # "Lasso": "blue",
            # "Bayesian": "green",
            bnp: bayes_palette[0],
            bpp: bayes_palette[1],
            bcp: bayes_palette[2],
            # "Mean": "dimgrey",
            wanted_models["model_lassocv_reg_no_pool"]: "#BF393A",
            wanted_models["model_lassocv_reg_cpool"]: "#BF393A",
            # wanted_models["dummy"]: "black",

        }

        model_order = list(model_colors)
        plot_df_filtered = plot_df.loc[plot_df["Metric"].isin([col_mapper["pmape_ci"]])]
        st.dataframe(plot_df_filtered)
        plot = sns.relplot(
            data=plot_df_filtered,
            x=relative_train_size_lbl,
            # x="params.loo_budget_rel",
            y="Value",
            kind="line",
            hue="Model",
            style="Pooling",
            # style_order=["MAPE", "MAPEci"],
            style_order=["complete", "partial", "no"],
            facet_kws={"sharey": False, "sharex": True},
            # palette=palette_,
            hue_order=model_order,  # Ensuring the order is applied
            palette=model_colors,
            aspect=1.2,
            height=2.9,
            col="Subject System",
            col_wrap=5,
            legend=True,
        )
    st.write("## Plot")

    for ax in plt.gcf().axes:
        title = ax.get_title()
        _, y_max = ax.get_ylim()
        if "x264" in title:
            upper_pMAPE = 120
        else:
            upper_pMAPE = 400
        y_max = min(upper_pMAPE, y_max)
        ax.set_ylim(0, y_max)
        # ax.set_xticks([0.5,1,3])
        ax.set_xticks([0, 1, 2, 3])
        title = ax.get_title()
        new_title = title.replace("Subject System = ", "")
        ax.set_title(new_title)
        ax.set_ylabel("")
        # if ax.legend_:
        #     ax.legend_.remove()
    fig = plt.gcf()
    #     fig.canvas.draw()
    #     time.sleep(0.1)

    handles, labels = plot.axes[0].get_legend_handles_labels()
    padded_handles = [*handles[:4], None, None, *handles[4:]]
    padded_labels = [*labels[:4], None, None, *labels[4:]]
    fig.legend(
        padded_handles,
        padded_labels,
        loc='upper center',  # Adjusts legend position relative to the anchor.
        ncol=len(padded_handles),  # Assumes you want all items in one row; adjust as needed.
        frameon=True,
        bbox_to_anchor=(0.5, -0.0015)  # Centers the legend below the plot. Adjust Y-offset as needed.
    )
    plot._legend.remove()

    tmp_file = "streamlit-last-results-multitask.pdf"
    plt.savefig(tmp_file, bbox_inches='tight')
    fig.savefig("temp_plot.png", bbox_inches="tight", dpi=300)
    st.image("temp_plot.png")

    with st.expander(label="Get Your PDF Now COMPLETELY FREE!!!1!11!!", expanded=False):
        embed_pdf(tmp_file)


def draw_multitask_dashboard(combined_df):
    st.write("## Plot configuration")

    plot_type = st.selectbox("Do you want to get the paper plots more extensive plots?", ["Paper", "Custom"])

    if plot_type == "Paper":
        draw_multitask_paper_plot(combined_df)
    else:

        filtered_df, score_columns, systems, y_lim_max_mape = filter_result_df(combined_df)
        sns.set_context("talk")
        share_y = False
        share_x = True
        only_one_metric = False
        only_one_system = False
        col_wrap = 5
        if len(systems) == 1:
            col_dict = {
                "col": "Metric",
                "col_wrap": col_wrap,
            }
            st.info("Wrapping columns because only one system was selected!")
            only_one_system = True
        elif len(score_columns) == 1:
            col_dict = {
                "col": "params.software-system",
                "col_wrap": col_wrap,
            }
            st.info("Wrapping columns because only one score was selected!")
            only_one_metric = True

        else:
            col_dict = {
                "col": "Metric",
                "col_wrap": None,
                "row": "params.software-system",
            }
        plot_multitask(
            col_dict,
            filtered_df,
            only_one_metric,
            only_one_system,
            score_columns,
            share_x,
            share_y,
            systems,
            y_lim_max_mape,
        )


def plot_multitask(
        col_dict,
        filtered_df,
        only_one_metric,
        only_one_system,
        score_columns,
        share_x,
        share_y,
        systems,
        y_lim_max_mape,
):
    # Start with known values
    known_values = ["mcmc", "rf", "dummy"]
    # Extract unique values from the DataFrame's hue column
    additional_values = filtered_df["params.model"].dropna().unique()
    # Combine known values with additional unique values, excluding duplicates
    possible_values = known_values + [
        val for val in additional_values if val not in known_values
    ]
    # Generate a color palette with enough colors for all possible values
    palette = sns.color_palette("husl", len(possible_values))
    # st.write(additional_values)
    # st.write("possible:")
    # st.write(possible_values)
    #
    # st.write(palette)
    with st.spinner("Waiting for plot to be rendered"):
        palette_ = dict(zip(possible_values, palette))
        filtered_df = filtered_df.sort_values(
            by="params.software-system", ascending=True
        )

        plot = sns.relplot(
            data=filtered_df,
            x="params.relative_train_size",
            # x="params.loo_budget_rel",
            y="Value",
            kind="line",
            hue="params.model",
            style="params.pooling_cat",
            style_order=["complete", "partial", "no"],
            facet_kws={"sharey": share_y, "sharex": share_x},
            # palette=palette_,
            hue_order=possible_values,
            # palette="colorblind",
            # aspect=1.035,
            aspect=0.8,
            height=5.8,
            **col_dict,
        )
        # setting boundaries that make sense

        sns.move_legend(
            plot,
            "center right",
            bbox_to_anchor=(-0.005, 0.5),
            # ncol=3,
            # title=None,
            frameon=True,
            fancybox=True,
        )

        if only_one_system:
            plt.suptitle(str(systems[0]))
        if only_one_metric:
            plt.suptitle(str(score_columns[0]))

        sup_title = (
            "" if plt.gcf()._suptitle is None else plt.gcf()._suptitle.get_text()
        )

        for ax in plt.gcf().axes:
            title = ax.get_title()
            if "R2" in title or "R2" in sup_title:
                ax.set_ylim(-1, 1)
                print("set R2 ylims")
            lower_title = str(title).lower()
            if "mape" in lower_title or "mape" in sup_title:
                y_min, y_max = ax.get_ylim()
                print("old y limits", y_min, y_max)
                ax.set_ylim(0, min(y_max, y_lim_max_mape))

                y_min, y_max = ax.get_ylim()
                print("new y limits", y_min, y_max)
                # ax.set_yscale('log')
            if "test_set_log" in lower_title or "test_set_log" in sup_title:
                ax.set_yscale("symlog")
                y_min, _ = ax.get_ylim()
                ax.set_ylim(y_min, 0)
            new_title = title
            new_title = new_title.replace("params.software-system = ", "")
            new_title = new_title.replace("Metric = ", "")
            ax.set_title(new_title)
            # Adjusting the legend position
        # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.suptitle("")
        plt.tight_layout()
        fig = plt.gcf()
        # fig.canvas.draw()
        time.sleep(0.1)
        tmp_file = "streamlit-last-results-multitask.pdf"
        plt.savefig(tmp_file, bbox_inches='tight')
        fig.savefig("temp_plot.png", bbox_inches="tight", dpi=300)
        st.image("temp_plot.png")

        with st.expander(label="Get Your PDF Now COMPLETELY FREE!!!1!11!!", expanded=False):
            embed_pdf(tmp_file)

        # st.pyplot(fig)


def filter_result_df(
        combined_df,
        system_col="params.software-system",
        model_col="params.model",
        cat_col="params.pooling_cat",
):
    col1, col2, col3, col4 = st.columns(4)
    metrics_raw = get_metrics_in_df(combined_df)
    metrics = [m.replace("metrics.", "") for m in metrics_raw]
    all_systems = combined_df[system_col].unique()
    all_models = combined_df[model_col].unique()
    all_poolings = combined_df[cat_col].unique()
    with col1:
        defaut_metrics = [
            "mape",
            # "mape_ci",
            # "relative_DOF",
            # "test_set_log-likelihood",
        ]
        score_columns = st.multiselect(
            "Select Score Columns", metrics, default=defaut_metrics
        )
        if not score_columns:
            score_columns = metrics
    with col2:
        common_sys = [s for s in all_systems if s not in ["H2", "kanzi"]]
        systems = st.multiselect("Select Systems", all_systems, default=common_sys)
        if not systems:
            systems = all_systems
    with col3:
        s_models = st.multiselect("Select Models", all_models, default=all_models[:2])
        if not s_models:
            s_models = all_models
    with col4:
        s_poolings = st.multiselect(
            "Pooling Methods", all_poolings, default=all_poolings
        )
        if not s_poolings:
            s_poolings = all_poolings
    with st.status("Filtering data..."):
        st.write("Filter systems ...")
        filtered_df = combined_df[combined_df[system_col].isin(systems)]
        st.write("Filter models ...")
        filtered_df = filtered_df[filtered_df[model_col].isin(s_models)]
        filtered_df = filtered_df[filtered_df[cat_col].isin(s_poolings)]
        st.write("Filter metrics ...")
        melted_df = filtered_df.melt(
            id_vars=[col for col in filtered_df.columns if col not in metrics_raw],
            value_vars=metrics_raw,
            var_name="Metric",
            value_name="Value",
        )
        melted_df["Metric"] = melted_df["Metric"].str.replace("metrics.", "")
        filtered_df = melted_df[melted_df["Metric"].isin(score_columns)]
        # plot_scores(filtered_df, score_columns)
        st.write("## Result")
        st.dataframe(filtered_df)
    with st.expander(label="cherry-pick", expanded=False):
        # Assuming 'df' is your original DataFrame and you want to group by these columns:
        all_columns_to_group_by = [
            model_col,
            cat_col,
            "Metric",
            system_col,
        ]
        pick_cols = st.multiselect(
            "Select Columns For Picking",
            all_columns_to_group_by,
            default=all_columns_to_group_by[:2],
        )
        if not pick_cols:
            pick_cols = all_columns_to_group_by

        # Grouping by the specified columns
        grouped = filtered_df.groupby(pick_cols)

        # Extracting the unique combinations of values as a list of tuples
        unique_combinations = grouped.groups.keys()

        # Converting the unique combinations into a new DataFrame
        new_df = pd.DataFrame(unique_combinations, columns=pick_cols)
        new_df["Picked"] = True
        new_df = st.data_editor(new_df)

        # Merge the original DataFrame with new_df
        merged_df = pd.merge(filtered_df, new_df, how="left", on=pick_cols)

        # Filter out rows where 'active' is False
        filtered_df = merged_df[merged_df["Picked"] != False]

        # Drop the 'active' column to return to the original DataFrame's structure
        filtered_df = filtered_df.drop("Picked", axis=1)

        y_lim_max_mape = st.select_slider(
            "Y Limit for MAPEs", [50, 100, 150, 200, 250, 300, 350, 400], value=300
        )
    return filtered_df, score_columns, systems, y_lim_max_mape


@st.cache_data
def get_metrics_in_df(df):
    return [col for col in df.columns if col.startswith("metrics.")]


if __name__ == "__main__":
    main()
