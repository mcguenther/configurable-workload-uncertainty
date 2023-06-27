import argparse
import os

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from analysis import Analysis


def hash_func(obj: Analysis) -> int:
    return obj.results_base_path


sns.set_style("white")


@st.cache_data
def get_analysis_object(path):
    al = Analysis(path)
    return al


@st.cache_data(persist="disk")
# @st.cache_data(hash_funcs={Analysis: hash_func})
def plot_metadata(meta_df):
    # plt.figure(dpi=1200)
    g = sns.relplot(data=meta_df, x="budget_abs", y="score",
                    hue="model", col="metric", kind="line", col_wrap=2, facet_kws={'sharey': False, 'sharex': True}, )

    fig = plt.gcf()
    return fig


@st.cache_data(persist="disk")
# @st.cache_data(hash_funcs={Analysis: hash_func})
def plot_error(err_df):
    # plt.figure(dpi=1200)
    sns.relplot(data=err_df, x="exp_id", y="err",
                hue="model", col="env", kind="line", col_wrap=2, )  # row="setting", )
    fig = plt.gcf()
    return fig


def main():
    # parser = argparse.ArgumentParser(description='Script description')
    # parser.add_argument('--results', type=str, help='path to results parquet file or similar')
    # args = parser.parse_args()
    # results_base_path = args.results

    st.title("Multilevel Experiment Explorer")
    with st.sidebar:
        st.header("Experiment Results Parent Folder")
        folder_path = st.text_input("Experiment Results Parent Folder", "./wluncert/results")

        # subfolders = [os.path.join(results_base_path, item) for item in os.listdir(results_base_path) if
        #               os.path.isdir(os.path.join(results_base_path, item))]
        subfolders = sorted(
            [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))], reverse=True)
        chosen_exp = st.selectbox("Experiment", subfolders)
        exp_path = os.path.join(folder_path, chosen_exp)
        # all_exps = os.listdir(results_base_path)
        # all_exps
        # chosen_experiment =

    al = get_analysis_object(exp_path)
    meta_df = al.get_meta_df()
    models = meta_df["model"].unique()
    chosen_models = st.multiselect("models", models, models)
    with st.expander("Model Meta Data", expanded=True):
        metrics = meta_df["metric"].unique()
        default_metrics = [m for m in metrics if m not in ["se", "elpd_loo"]]
        chosen_metrics = st.multiselect("metric", metrics, default_metrics)
        if chosen_models:
            meta_df_filtered = meta_df.loc[meta_df["model"].isin(chosen_models)]
            meta_df_filtered = meta_df_filtered.loc[meta_df_filtered["metric"].isin(chosen_metrics)]
            fig = plot_metadata(meta_df_filtered)
            st.pyplot(fig)

    with st.expander("Errors", expanded=True):
        show_only_overall = st.checkbox("Only show overall scores")
        x_range = st.slider('X range', 0.0, 300.0, (0.0, 50.0))
        y_lim_boundaries = {
            "mape": (0., 500.),
            "mape_ci": (0., 500.),
            "R2": (-1., 1.05),
        }
        y_lim_defaults = {
            "mape": (0., 150.),
            "mape_ci": (0., 150.),
            "R2": (0.5, 1.05),
        }
        err_df = al.score_df
        if show_only_overall:
            err_df = err_df.loc[err_df["env"] == "overall"]

        err_df = err_df.loc[err_df["model"].isin(chosen_models)]

        err_types = list(err_df["err_type"].unique())
        first_elem = "mape"
        lst = move_to_front(err_types, first_elem)
        tabs = st.tabs(lst)
        for err_type, tab in zip(err_types, tabs):
            with tab:
                if err_type in y_lim_defaults:
                    y_limit = y_lim_defaults[err_type]
                else:
                    y_limit = None
                if err_type in y_lim_boundaries:
                    y_boundary = y_lim_boundaries[err_type]
                else:
                    y_boundary = None

                y_range = st.slider('Y range', min_value=y_boundary[0], max_value=y_boundary[1], value=y_limit, key=f"y-range-{err_type}")
                selected_error_df = err_df.loc[err_df["err_type"] == err_type]
                fig = plot_error(selected_error_df)
                str(plt.xlim())
                current_xlim = plt.xlim()

                # Adjust the given xrange if it extends beyond the data limits
                new_xlim = (
                    max(x_range[0], current_xlim[0]),
                    min(x_range[1], current_xlim[1])
                )
                plt.xlim(new_xlim)
                plt.ylim(y_range)
                st.pyplot(fig)

                # plt.yscale("log")
                # plt.suptitle(err_type)


def move_to_front(err_types, first_elem):
    return [first_elem] + [x for x in err_types if x != first_elem]


# al.plot_metadata()


if __name__ == "__main__":
    main()
