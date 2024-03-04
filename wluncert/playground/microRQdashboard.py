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


def get_subfolders(parent_folder):
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    return subfolders


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


def draw_transfer_dashboard(combined_df):
    sws_col = "params.subject_system"
    filtered_df, score_columns, systems, y_lim_max_mape = filter_result_df(
        combined_df, system_col=sws_col
    )

    sws_list = list(filtered_df[sws_col].unique())

    for sws_lbl in sws_list:
        filtered_df = filter_by_first_unique_value(filtered_df, "Metric")
        metric = list(filtered_df["Metric"].unique())[0]
        sws_df = filtered_df.loc[filtered_df[sws_col] == sws_lbl]

        # st.dataframe(filtered_df)

        g = sns.relplot(
            data=sws_df,
            x="params.relative_train_size",
            y="Value",
            col="params.number_of_source_envs",
            row="params.n_transfer_samples",
            hue="params.model",
            kind="line",
            style="params.pooling_cat",
            # palette="crest",
            # linewidth=4,
            # zorder=5,
            # col_wrap=3,
            height=8,
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
        fig = plt.gcf()
        st.pyplot(fig)


def main():
    st.title("CSV File Processor and Visualizer from Subfolders")

    config_ok = False
    with st.sidebar:
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
                default=["240214-08-28-44-aggregation-EjU8UAfn4w"],
            )

            if (
                selected_subfolders
            ):  # st.button("Process CSV Files in Selected Subfolders"):
                whole_folders = [
                    os.path.join(parent_folder, f) for f in selected_subfolders
                ]
                combined_df = read_and_combine_csv(whole_folders)
                if combined_df.empty:
                    st.error(
                        "No CSV files found in the selected subfolders or the combined DataFrame is empty."
                    )
                else:
                    config_ok = True
                combined_df = replace_strings(combined_df)
    if not config_ok:
        "please check config in sidebar"
    else:
        exp_types = combined_df["params.experiment-type"].unique()
        if len(exp_types) > 1:
            st.error("Not more than one experiment type supported!")
            exit(22)
        else:
            exp_type = exp_types[0]
            if exp_type == "multitask":
                draw_multitask_dashboard(combined_df)
            elif exp_type == "transfer":
                draw_transfer_dashboard(combined_df)


def draw_multitask_dashboard(combined_df):
    st.write("## Plot configuration")
    filtered_df, score_columns, systems, y_lim_max_mape = filter_result_df(combined_df)
    sns.set_context("talk")
    share_y = False
    share_x = True
    only_one_metric = False
    only_one_system = False
    if len(systems) == 1:
        col_dict = {
            "col": "Metric",
            "col_wrap": 3,
        }
        st.info("Wrapping columns because only one system was selected!")
        only_one_system = True
    elif len(score_columns) == 1:
        col_dict = {
            "col": "params.software-system",
            "col_wrap": 3,
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
        plt.tight_layout()
        fig = plt.gcf()
        fig.canvas.draw()
        time.sleep(0.1)
        fig.savefig("temp_plot.png", bbox_inches="tight", dpi=300)
        st.image("temp_plot.png")
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
