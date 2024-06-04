import copy

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

import streamlit.components.v1 as components
import base64

def get_subfolders(parent_folder):
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    return subfolders

bayes_palette = ["#47AEED", "#398CBF",  "#2F729C"]
comparison_palette = ["#BF393A"]
@st.cache_data
def read_and_combine_csv(subfolders):
    all_dfs = []
    for folder in subfolders:
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                all_dfs.append(df)
            except Exception as e:
                st.error(f"Error reading {file}: {e}")
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        st.error("No dataframes found!")


@st.cache_data
def plot_scores(df, score_columns):
    for column in score_columns:
        st.subheader(f"Plot for {column}")
        sns_plot = sns.histplot(df[column].dropna())
        st.pyplot(sns_plot.figure)


@st.cache_data
def replace_strings(filtered_df):
    model_replacements = {
        "no-pooling-": "",
        "complete-pooling-": "",
        "partial-pooling-": "",
        "-robust": "",
        "cpooling-": "",
        "-1model": "",
        "dummy": "mean-pred",
    }

    filtered_df["params.model"] = filtered_df["params.model"].replace(
        model_replacements, regex=True
    )
    pooling_replacements = {
        "NO_POOLING": "no",
        "COMPLETE_POOLING": "complete",
        "PARTIAL_POOLING": "partial",
    }

    filtered_df["params.pooling_cat"] = filtered_df["params.pooling_cat"].replace(
        pooling_replacements, regex=True
    )
    metrics_replacements = {
        "_overall": "",
    }

    # Creating the rename mapping using dictionary comprehension
    rename_mapping = {
        col: col.replace(old, new)
        for col in filtered_df.columns
        for old, new in metrics_replacements.items()
    }

    # Renaming the columns in the DataFrame
    filtered_df = filtered_df.rename(columns=rename_mapping)
    return filtered_df


@st.cache_data
def filter_by_first_unique_value(df, col_name):
    """
    Filters a DataFrame such that it only contains rows that have the first of all values for a given column name.
    When there are two different unique values for the column, it returns only the rows containing the first value.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be filtered.
    col_name (str): The name of the column to filter by its first unique value.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    # Identify the first unique value in the specified column
    first_unique_value = df[col_name].unique()[0]

    # Filter the DataFrame to only include rows with this first unique value
    filtered_df = df[df[col_name] == first_unique_value]

    return filtered_df


def embed_pdf(file_path, width=700, height=700):
    with open(file_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')

    pdf_display = f"""
    <style>
    .pdf-container {{
        width: 100%;
        height: {height}px;
    }}
    iframe {{
        width: 100%;
        height: 100%;
    }}
    </style>
    <div class="pdf-container">
        <iframe src="data:application/pdf;base64,{base64_pdf}" type="application/pdf"   ></iframe>
    </div>
    """
    components.html(pdf_display, height=height)

def draw_transfer_dashboard(combined_df):
    sws_col = "params.subject_system"
    filtered_df, score_columns, systems, y_lim_max_mape = filter_result_df(
        combined_df, system_col=sws_col
    )

    sws_list = list(filtered_df[sws_col].unique())
    sns.set_theme(
        rc={
            "figure.dpi": 200,
        }
    )
    sns.set_context("talk")  # Makes labels and lines larger
    sns.set_style("whitegrid")  # Adds a grid for easier data visualization

    # Create tabs
    tabs = st.tabs(sws_list)
    len(tabs)
    # Iterate through each tab and build content

    # Iterate through each tab and build content
    for tab, sws_lbl in zip(tabs, sws_list):
        with tab:
            filtered_df = filter_by_first_unique_value(filtered_df, "Metric")
            metric = list(filtered_df["Metric"].unique())[0]
            sws_df = filtered_df.loc[filtered_df[sws_col] == sws_lbl]
            sws_df = sws_df.loc[sws_df["params.n_transfer_samples"] >= 3]

            # st.dataframe(filtered_df)

            st.write(f"## Results {sws_lbl}")
            g = sns.relplot(
                data=sws_df,
                x="params.n_transfer_samples",
                y="Value",
                col="params.number_of_source_envs",
                row="params.relative_train_size",
                hue="params.model",
                kind="line",
                style="params.pooling_cat",
                style_order=["complete", "partial", "no"],
                # palette="crest",
                # linewidth=4,
                # zorder=5,
                # col_wrap=3,
                height=5,
                aspect=1.015,
                legend=True,
            )
            suptitle = f"{sws_lbl}, {metric}"
            plt.suptitle(suptitle)

            sns.move_legend(
                g,
                "center right",
                bbox_to_anchor=(-0.005, 0.5),
                # ncol=3,
                # title=None,
                frameon=True,
                fancybox=True,
            )

            for ax in plt.gcf().axes:
                title = ax.get_title()
                if "R2" in suptitle:
                    ax.set_ylim(-1, 1)
                    print("set R2 ylims")
                # lower_title = str(title).lower()
                if "mape" in suptitle:
                    y_min, y_max = ax.get_ylim()
                    ax.set_ylim(0, min(y_max, y_lim_max_mape))
            plt.tight_layout()

            # sns.set_theme(
            #     rc={
            #         "figure.dpi": 200,
            #     }
            # )
            tmp_file = "streamlit-last-results.pdf"
            plt.savefig(tmp_file, bbox_inches='tight')
            fig = plt.gcf()
            st.pyplot(
                fig,
            )

            with st.expander(label=f"{sws_lbl} PDF Now COMPLETELY FREE!!!1!11!!", expanded=False):
                embed_pdf(tmp_file)


def main():
    st.set_page_config(
        page_title="MCMC Multilevel Models Dashboard",
        page_icon="📊",  # Update path accordingly
        layout="wide"
    )
    # st.title("CSV File Processor and Visualizer from Subfolders")

    config_ok = False
    with st.sidebar:
        st.write("## Folder Selection")
        parent_folder = st.text_input(
            "Enter the path of the parent folder",
            value="/home/jdorn/results/localflow/jdorn-multilevel-eval/",
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
                        "240321-19-59-19-aggregation-bTLYxz3uFg", # old bayesian multitask
                         # "240319-22-38-13-aggregation-dwnJojszkX", # new lasso grid
                         "240322-13-08-04-aggregation-NKUie5gttU", # 0.9 CI
                         ],
            )

            if (
                selected_subfolders
            ):  # st.button("Process CSV Files in Selected Subfolders"):
                whole_folders = [
                    os.path.join(parent_folder, f) for f in selected_subfolders
                ]
                combined_df = read_and_combine_csv(whole_folders)
                if combined_df is None or combined_df.empty:
                    st.error(
                        "No CSV files found in the selected subfolders or the combined DataFrame is empty."
                    )
                else:
                    config_ok = True
    if not config_ok:
        st.error("please check config in sidebar")
        exit(21)
    else:
        combined_df = replace_strings(combined_df)
        exp_types = combined_df["params.experiment-type"].unique()
        if len(exp_types) > 1:
            st.error("Not more than one experiment type supported!")
            exit(22)
        else:
            total_pred_time_cost = int(combined_df["metrics.pred_time_cost"].sum())
            total_fitting_time_cost = int(combined_df["metrics.fitting_time_cost"].sum())
            total_preproc_fitting_time_cost = int(combined_df["metrics.preproc-fittin_time_cost"].sum())
            total_preproc_pred_time_cost = int(combined_df["metrics.preproc-pred-_time_cost"].sum())
            total_time = total_pred_time_cost + total_fitting_time_cost + total_preproc_fitting_time_cost + total_preproc_pred_time_cost


            days, hours, minutes, seconds_remaining  = seconds_to_days(total_time)

            st.write("## Time Cost in Compute Time")
            col_days, col1, col2, col3 = st.columns(4)
            with col_days:
                st.metric("Experiment Days", f"{days}d")
            with col1:
                st.metric("Experiment Hours", f"{hours}h")
            with col2:
                st.metric("Experiment Minutes", f"{minutes}m")
            with col3:
                st.metric("Experiment Seconds", f"{seconds_remaining}s")



            st.write("## Experiment Filters")
            exp_type = exp_types[0]
            if exp_type == "multitask":
                draw_multitask_dashboard(combined_df)
            elif exp_type == "transfer":
                draw_transfer_dashboard(combined_df)


def seconds_to_days(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    seconds_remaining = seconds % 60
    return days, hours, minutes, seconds_remaining


def convert_to_frac(value):

    if value < 1:

        denominator = 1 / value

        return r'$\frac{1}{' + f'{int(denominator)}' + r'}$'

    return str(int(value))


def draw_multitask_paper_plot(combined_df,     system_col="params.software-system",
    model_col="params.model",
    cat_col="params.pooling_cat",):
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
    #st.write(all_models)
    filtered_df = combined_df[combined_df[model_col].isin(wanted_models)]
    filtered_df["params.model"] = filtered_df["params.model"].map(wanted_models)

    col_mapper = {
        "mape":"MAPE",
        # "mape_ci":"$\\text{MAPE}_\\text{CI}$"
        "mape_ci":"MAPEci",
        "pmape_ci":"pMAPE",
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
    #st.dataframe(melted_df)

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
    melted_df=melted_df.drop(columns=["Pooling"])
    melted_df[subject_system_lbl] = melted_df[subject_system_lbl].apply(lambda x: f'\\sws{{{x}}}')
    st.dataframe(melted_df)

    # melted_df = melted_df.loc[melted_df[relative_train_size_lbl].isin([0.25,0.5,0.75,1,3])]

    melted_df["Value"] = melted_df["Value"].astype(float)
    melted_df["Value"] = melted_df["Value"].astype(float)
    replacements =  {
        # "0.125000":"\\multicolumn{1}{c}{$\\sfrac{1}{8}$}",
        # "0.250000":"\\multicolumn{1}{c}{$\\sfrac{1}{4}$}",
        # "0.500000" :"\\multicolumn{1}{c}{$\\sfrac{1}{2}$}",
        # "0.750000":"\\multicolumn{1}{c}{$\\sfrac{3}{4}$}",
        # "1.000000":"\\multicolumn{1}{c}{$1$}",
        # "2.000000":"\\multicolumn{1}{c}{$2$}",
        # "3.000000":"\\multicolumn{1}{c}{$3$}",
        "0.125000":"$\\sfrac{1}{8} \\vert \\mathcal{O} \\vert$",
        "0.250000":"$\\sfrac{1}{4} \\vert \\mathcal{O} \\vert$",
        "0.500000":"$\\sfrac{1}{2} \\vert \\mathcal{O} \\vert$",
        "0.750000":"$\\sfrac{3}{4} \\vert \\mathcal{O} \\vert$",
        "1.000000":"$1 \\vert \\mathcal{O} \\vert$",
        "2.000000":"$2 \\vert \\mathcal{O} \\vert$",
        "3.000000":"$3 \\vert \\mathcal{O} \\vert$",
        r"Subject System":"",
        r"Relative Train Size":"$\\vert \\mathcal{D}^\mathit{train} \\vert$",
            r"\\\\ \& \& \& \& \& \& \& \& \& \& \& \& ":"",
            r" &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  &  \\":"",
    }
    mape_df = melted_df[['Subject System', 'Relative Train Size', 'Model', "Metric", "Value"]]
    st.write("## Latex Tables.")
    with st.expander("all MAPES!", expanded=False):
        # grouped_mape = mape_df.groupby(['Subject System', 'Relative Train Size', 'Model', pooling_cat_lbl, "Metric",]).mean().reset_index()
        # st.dataframe(grouped_mape)
        all_mapes_ape  = mape_df.loc[mape_df[relative_train_size_lbl].isin([0.25,0.5,0.75,1])]
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

        mapes_only_df = melted_df.loc[melted_df[relative_train_size_lbl].isin([0.125,0.25,0.5,0.75,1,2,3])]
        mape_only_df = mapes_only_df.loc[mapes_only_df["Metric"].isin([col_mapper["mape"]])]
        mape_only_df.drop(columns=["Metric"])
        # initial_pivot = mape_only_df.pivot_table(index=['Subject System'],
        #                                  columns=['Model', 'Metric', 'Relative Train Size'],
        #                                  values='Value',
        #                                  aggfunc='mean')
        initial_pivot = mape_only_df.pivot_table(index=['Relative Train Size'],
                                         columns=['Model', 'Metric',],
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

        mapeci_only_df = mapeci_only_df.loc[mapeci_only_df[relative_train_size_lbl].isin([0.125,0.25,0.5,0.75,1,2,3])]
        pivot_df = mapeci_only_df.pivot_table(index=['Subject System'],
                                            columns=['Relative Train Size', 'Model'],
                                            values='Value',
                                            aggfunc='mean')


        #Calculate the mean for each column, skipping non-numeric data automatically
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
                rounded_scores.loc[system, (rel_train_size, min_model)] = "\\cellcolor{tabSignal}" + str(rounded_scores.loc[system, (rel_train_size, min_model)])
                # pivot_df.loc[system, (rel_train_size, min_model)] = "X" + str(pivot_df.loc[system, (rel_train_size, min_model)])
        st.dataframe(rounded_scores)
        rounded_scores.to_csv("./results-rq1.csv")
        # column_format = 'l' + ('|' + 'r' * subcols_per_model) * num_models

        column_format = 'r'
        num_columns=rounded_scores.shape[1]
        # Loop through each column, starting from the first data column (ignoring the index column)
        for i in range(1, num_columns + 1):
            # For every third colu mn starting from the second, use '||' instead of '|'
            if i % 3 == 1:
                column_format += '||r'
            else:
                column_format += '|r'
        latex_str = rounded_scores.to_latex(index=True, multirow=True, multicolumn=True,
                                            multicolumn_format='c',
                                            column_format=column_format,#'r' + '|r' * rounded_scores.shape[1],
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
         # ["blue", "green", "red"] # sns.color_palette("flare", 3)
        model_colors = {
            # "Lasso": "blue",
            # "Bayesian": "green",
            bnp: bayes_palette[0],
            bpp: bayes_palette[1],
            bcp: bayes_palette[2],
            # "Mean": "dimgrey",
            wanted_models["model_lassocv_reg_no_pool"]: comparison_palette[0],
            wanted_models["model_lassocv_reg_cpool"]: comparison_palette[0],
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
            col_wrap= 5,
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
        ax.set_xticks([0,1,2,3])
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
    padded_handles = [*handles[:4], None, None,  *handles[4:]]
    padded_labels =[*labels[:4], None, None, *labels[4:]]
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
