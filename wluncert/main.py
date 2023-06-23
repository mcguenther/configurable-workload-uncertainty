import numpyro

from wluncert.analysis import Analysis

# must be run before any JAX imports
numpyro.set_host_device_count(6)

import argparse
from wluncert.experiment import Replication
import os
from wluncert.data import DataLoaderStandard, DataAdapterJump3r, WorkloadTrainingDataSet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from wluncert.models import MCMCMultilevelPartial, \
    ExtraStandardizingEnvAgnosticModel, NoPoolingEnvModel, CompletePoolingEnvModel, MCMCCombinedCompletePooling


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
    print("loading data")
    path_jump3r = "./training-data/jump3r.csv"
    jump3r_data_raw = DataLoaderStandard(path_jump3r)
    data_jump3r = DataAdapterJump3r(jump3r_data_raw)
    wl_data: WorkloadTrainingDataSet = data_jump3r.get_wl_data()

    env_lbls = data_jump3r.get_environment_lables()
    feature_names = list(wl_data.get_workloads_data()[0].get_X().columns)
    if debug:
        mcmc_num_warmup = 1000
        mcmc_num_samples = 500
        mcmc_num_chains = 3
    else:
        mcmc_num_warmup = 1000
        mcmc_num_samples = 1000
        mcmc_num_chains = 3
    progress_bar = False if n_jobs else True
    mcmc_kwargs = {"num_warmup": mcmc_num_warmup, "num_samples": mcmc_num_samples, "num_chains": mcmc_num_chains,
                   "progress_bar": progress_bar}
    data_providers = {"jump3r": wl_data}

    print("loaded data")

    rf_proto = RandomForestRegressor()
    model_rf = NoPoolingEnvModel(rf_proto)

    # pairwise_reg_proto = get_pairwise_lasso_reg()
    # model_pairwise_reg = NoPoolingEnvModel(pairwise_reg_proto)

    lin_reg_proto = LinearRegression()
    model_lin_reg = NoPoolingEnvModel(lin_reg_proto)

    dummy_proto = DummyRegressor()
    model_dummy = NoPoolingEnvModel(dummy_proto)

    mcmc_no_pooling_proto = ExtraStandardizingEnvAgnosticModel(plot=plot, feature_names=feature_names, **mcmc_kwargs)
    model_mcmc_no_pooling = NoPoolingEnvModel(mcmc_no_pooling_proto)

    model_partial_extra_standardization = MCMCMultilevelPartial(plot=plot, feature_names=feature_names,
                                                                env_names=env_lbls, **mcmc_kwargs,
                                                                return_samples_by_default=True)
    model_complete_pooling_combined = MCMCCombinedCompletePooling(plot=plot, feature_names=feature_names,
                                                                  env_names=env_lbls, **mcmc_kwargs,
                                                                  return_samples_by_default=True)

    complete_pooling_rf = CompletePoolingEnvModel(rf_proto)
    complete_pooling_mcmc = CompletePoolingEnvModel(mcmc_no_pooling_proto)

    models = {
        "partial-pooling-mcmc-extra": model_partial_extra_standardization,
        "no-pooling-mcmc": model_mcmc_no_pooling,
        "no-pooling-rf": model_rf,
        "no-pooling-lin": model_lin_reg,
        "no-pooling-dummy": model_dummy,
        "cpooling-rf": complete_pooling_rf,
        "cpooling-mcmc": complete_pooling_mcmc,
        "cpooling-mcmc-1model": model_complete_pooling_combined,
    }

    rep_lbl = "last-run"
    if debug:
        # debug_models = ["cpooling-mcmc"]
        # debug_models = ["cpooling-rf"]
        # debug_models = ["no-pooling-mcmc"]
        debug_models = ["cpooling-mcmc-1model"]
        # debug_models = ["partial-pooling-mcmc-extra"]
        models = {k: v for k, v in models.items() if k in debug_models}
        train_sizes = 1.0,  # 1.5, 2
        rnds = [1]
        rep_lbl = "debug4"
    else:
        # train_sizes = 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5
        train_sizes = 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 3, 4, 8, 10, 20, 30, 80, 100, 150, 175  # 250, 500,
        rnds = list(range(5))
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


if __name__ == "__main__":
    main()
