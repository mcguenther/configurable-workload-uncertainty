import numpyro

from wluncert.analysis import Analysis

# must be run before any JAX imports
numpyro.set_host_device_count(6)

import argparse
from wluncert.experiment import Replication
import os
from wluncert.data import DataLoaderStandard, DataAdapterJump3r, WorkloadTrainingDataSet, DataAdapterXZ, \
    DataLoaderDashboardData, DataAdapterH2, Standardizer, PaiwiseOptionMapper
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.dummy import DummyRegressor
from wluncert.models import MCMCMultilevelPartial, NoPoolingEnvModel, CompletePoolingEnvModel, MCMCCombinedNoPooling, \
    MCMCPartialRobust, MCMCPartialHorseshoe, MCMCCombinedCompletePooling


def main():
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--jobs', type=int, default=None, help='Number of jobs for parallel mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--plot', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-store', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    n_jobs = args.jobs
    debug = args.debug
    plot = args.plot
    do_store = not args.no_store

    print("pwd", os.getcwd())
    data_providers = get_all_datasets()

    selected_data = "jump3r",
    data_providers = {data_providers[key]: data_providers[key] for key in selected_data}

    models = get_all_models(debug, n_jobs, plot)

    rep_lbl = "full-run"
    if debug:
        chosen_model_lbls = []
        # chosen_model_lbls.extend(["cpooling-rf"])
        chosen_model_lbls.extend(["no-pooling-rf"])
        # chosen_model_lbls.extend(["cpooling-mcmc-1model", "partial-pooling-mcmc-extra", "no-pooling-rf"])
        chosen_model_lbls.extend(["partial-pooling-mcmc-extra"])
        chosen_model_lbls.extend(["partial-pooling-mcmc-extra-pw"])
        # chosen_model_lbls.extend(["cpooling-mcmc-1model"])
        # chosen_model_lbls.extend(["no-pooling-lin"])
        # chosen_model_lbls.extend(["cpooling-lin"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-extra-pw"])
        # chosen_model_lbls.extend(["no-pooling-lin-pw"])
        chosen_model_lbls.extend(["no-pooling-dummy"])
        chosen_model_lbls.extend(["no-pooling-mcmc-1model"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-robust"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-horseshoe"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-extra", "partial-pooling-mcmc-robust", "partial-pooling-mcmc-horseshoe"])

        models = {k: v for k, v in models.items() if k in chosen_model_lbls}
        train_sizes = 0.75,
        train_sizes = 0.25, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 3
        rnds = list([11, 222, 333, 44, 55, 666, 77, 888, 99])
        train_sizes = 0.5, 0.75, 1.0, 1.5, 2, 5, 10
        rnds = 1, 55, 635 ,65 ,84
        # rnds = list(range(3))

        rep_lbl = "debug-1modelvs partial"
    else:
        # train_sizes = 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5
        # train_sizes = 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 3, 4, 8, 10, #20, 30, 50 # 80, 100, 150,  # 250, 500,
        train_sizes = 0.75, 1, 1.25, 1.5, 3, 5, 8, 10,  # 20, 30, 50 # 80, 100, 150,  # 250, 500,
        rnds = 23, 24, 25, 26, 27,  # list(range(3))
    print("created models")

    rep = Replication(models, data_providers, train_sizes, rnds, n_jobs=n_jobs, replication_lbl=rep_lbl)
    merged_df_scores, merged_df_metas = rep.run()
    if do_store:
        experiment_base_path = rep.store()
        al = Analysis(experiment_base_path)
        al.run()

    # eval = Evaluation()

    print("DONE with experiment.")

    print("running analysis")


def get_all_models(debug, n_jobs, plot):
    if debug:
        mcmc_num_warmup = 750
        mcmc_num_samples = 500
        mcmc_num_chains = 3
    else:
        mcmc_num_warmup = 750
        mcmc_num_samples = 750
        mcmc_num_chains = 3
    progress_bar = False if n_jobs else True
    mcmc_kwargs = {"num_warmup": mcmc_num_warmup, "num_samples": mcmc_num_samples, "num_chains": mcmc_num_chains,
                   "progress_bar": progress_bar}
    rf_proto = RandomForestRegressor()
    model_rf = NoPoolingEnvModel(rf_proto, preprocessings=[Standardizer()])

    lin_reg_proto = LinearRegression()
    model_lin_reg = NoPoolingEnvModel(lin_reg_proto, preprocessings=[Standardizer()])
    model_lin_reg_cpool = CompletePoolingEnvModel(lin_reg_proto, preprocessings=[Standardizer()])
    # model_lin_reg_cpool = CompletePoolingEnvModel(lin_reg_proto, preprocessings=[])
    model_lin_reg_pw = NoPoolingEnvModel(lin_reg_proto, preprocessings=[PaiwiseOptionMapper(), Standardizer()])

    # model_lin_reg_poly = Poly
    dummy_proto = DummyRegressor()
    model_dummy = NoPoolingEnvModel(dummy_proto)
    model_partial_extra_standardization = MCMCMultilevelPartial(plot=plot, **mcmc_kwargs,
                                                                return_samples_by_default=True,
                                                                preprocessings=[Standardizer()])

    model_partial_extra_standardization_pw = MCMCMultilevelPartial(plot=plot, **mcmc_kwargs,
                                                                   return_samples_by_default=True,
                                                                   preprocessings=[PaiwiseOptionMapper(),
                                                                                   Standardizer()])

    model_multilevel_partial_robust = MCMCPartialRobust(plot=plot, **mcmc_kwargs,
                                                        return_samples_by_default=True,
                                                        preprocessings=[Standardizer()])
    model_no_pooling_combined = MCMCCombinedNoPooling(plot=plot, **mcmc_kwargs,
                                                      return_samples_by_default=True,
                                                      preprocessings=[Standardizer()])
    model_multilevel_partial_horseshoe = MCMCPartialHorseshoe(plot=plot, **mcmc_kwargs,
                                                              return_samples_by_default=True,
                                                              preprocessings=[Standardizer()])
    model_complete_pooling_combined = MCMCCombinedCompletePooling(plot=plot, **mcmc_kwargs,
                                                                  return_samples_by_default=True,
                                                                  preprocessings=[Standardizer()])
    complete_pooling_rf = CompletePoolingEnvModel(rf_proto)
    models = {
        "partial-pooling-mcmc-extra": model_partial_extra_standardization,
        "partial-pooling-mcmc-extra-pw": model_partial_extra_standardization_pw,
        "partial-pooling-mcmc-robust": model_multilevel_partial_robust,
        "no-pooling-rf": model_rf,
        "no-pooling-lin": model_lin_reg,
        "cpooling-lin": model_lin_reg_cpool,
        "no-pooling-lin-pw": model_lin_reg_pw,
        "no-pooling-dummy": model_dummy,
        "cpooling-rf": complete_pooling_rf,
        "no-pooling-mcmc-1model": model_no_pooling_combined,
        "cpooling-mcmc-1model": model_complete_pooling_combined,
        "partial-pooling-mcmc-horseshoe": model_multilevel_partial_horseshoe,
    }
    return models


def get_all_datasets():
    print("loading data")

    # path_x264 = "./training-data/dashboard-resources/x264/"
    # x264_data_raw = DataLoaderDashboardData(path_x264)
    # data_x264 = DataAdapterX264(x264_data_raw)
    # h2_wl_data: WorkloadTrainingDataSet = data_x264.get_wl_data()

    path_h2 = "./training-data/dashboard-resources/h2/"
    h2_data_raw = DataLoaderDashboardData(path_h2)
    data_H2 = DataAdapterH2(h2_data_raw)
    h2_wl_data: WorkloadTrainingDataSet = data_H2.get_wl_data()

    path_xz = "./training-data/dashboard-resources/xz/"
    xz_data_raw = DataLoaderDashboardData(path_xz)
    data_xz = DataAdapterXZ(xz_data_raw)
    xz_wl_data: WorkloadTrainingDataSet = data_xz.get_wl_data()

    path_jump3r = "./training-data/jump3r.csv"
    jump3r_data_raw = DataLoaderStandard(path_jump3r)
    data_jump3r = DataAdapterJump3r(jump3r_data_raw)
    wl_data: WorkloadTrainingDataSet = data_jump3r.get_wl_data()

    data_providers = {
        "jump3r": wl_data,
        "H2": h2_wl_data,
        "xz": xz_wl_data,
    }
    print("loaded data")
    return data_providers


if __name__ == "__main__":
    main()
