import numpyro
from analysis import Analysis
import matplotlib

from deepperf import DeepPerfModel
from dal import DaLRegressor


# must be run before any JAX imports
numpyro.set_host_device_count(50)


import argparse
from experiment import (
    Replication,
    ExperimentTransfer,
    ExperimentMultitask,
    MLFLOW_URI,
    EXPERIMENT_NAME,
)
import os

print(os.environ)
import localflow as mlflow
from data import (
    DataLoaderStandard,
    DataAdapterJump3r,
    WorkloadTrainingDataSet,
    DataAdapterXZ,
    DataLoaderDashboardData,
    DataAdapterH2,
    Standardizer,
    PaiwiseOptionMapper,
    DataAdapterX264,
    DataAdapterBatik,
    DataAdapterDConvert,
    DataAdapterKanzi,
    DataAdapterZ3,
    DataAdapterLrzip,
    DataAdapterFastdownward,
    DataAdapterArtificial,
    DataAdapterVP9,
)

from datanfp import (
    DataAdapterNFPApache,
    DataAdapterNFP7z,
    DataAdapterNFPbrotli,
    DataAdapterNFPexastencil,
    DataAdapterNFPHSQLDB,
    DataAdapterNFPjump3r,
    DataAdapterNFPkanzi,
    DataAdapterNFPLLVM,
    DataAdapterNFPlrzip,
    DataAdapterNFPMongoDB,
    DataAdapterNFPnginx,
    DataAdapterNFPposgreSQL,
    DataAdapterNFPposgreVP8,
    DataAdapterNFPx264,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.dummy import DummyRegressor
from models import (
    MCMCMultilevelPartial,
    NoPoolingEnvModel,
    CompletePoolingEnvModel,
    MCMCCombinedNoPooling,
    MCMCPartialRobustLasso,
    MCMCPartialHorseshoe,
    MCMCCombinedCompletePooling,
    MCMCPartialSelfStandardizing,
    MCMCPartialRobustLassoAdaptiveShrinkage,
    MCMCPartialSelfStandardizingConstInfl,
    MCMCRHS,
    LassoGridSearchCV,
)
import mlfloweval


def get_rep_ids(default_n_reps, custom_num_reps=None, rep_offset=0):
    if custom_num_reps:
        print("custom num_reps:", custom_num_reps)
        print("custom rep_offset:", rep_offset)
        rep_ids = list(range(rep_offset, custom_num_reps + rep_offset))
    else:
        rep_ids = list(range(default_n_reps))
    print(f"Generated repetidion random seeds:", rep_ids)
    return rep_ids


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)

    experiment_class_labels = {
        "multitask": ExperimentMultitask,
        "transfer": ExperimentTransfer,
    }
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument(
        "--jobs", type=int, default=None, help="Number of jobs for parallel mode"
    )
    parser.add_argument(
        "--reps", type=int, default=None, help="Number of repetitions for experiment"
    )
    parser.add_argument(
        "--rep-offset",
        type=int,
        default=0,
        help="Offsets for the random number generation",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--plot", action="store_true", help="Enable debug mode")
    parser.add_argument("--store", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--training-set-size",
        type=float,
        help="Disables the sweep over different training set sizes and uses the given size",
    )

    parser.add_argument(
        "--experiments",
        default=experiment_class_labels.keys(),
        choices=experiment_class_labels.keys(),
        nargs="*",
        help="allows selecting individual experiments",
    )
    args = parser.parse_args()
    n_jobs = args.jobs
    debug = args.debug
    plot = args.plot
    do_store = args.store
    num_reps = args.reps
    rep_offset = args.rep_offset
    training_set_size = args.training_set_size
    chosen_experiments = [experiment_class_labels[e] for e in args.experiments]
    print("Preparing experiments", chosen_experiments)

    print("pwd", os.getcwd())
    print("Storing arviz data?", do_store)
    models = get_all_models(debug, n_jobs, plot, do_store=do_store)

    rep_lbl = "full-run"
    if debug:
        chosen_model_lbls = []
        # chosen_model_lbls.extend(["no-pooling-lin"])
        # chosen_model_lbls.extend(["cpooling-lin"])
        # chosen_model_lbls.extend(["model_lasso_reg_cpool"])
        # chosen_model_lbls.extend(["model_lasso_reg_no_pool"])
        # chosen_model_lbls.extend(["cpooling-rf"])
        # chosen_model_lbls.extend(["no-pooling-rf"])
        # chosen_model_lbls.extend(["no-pooling-dummy"])

        # chosen_model_lbls.extend(["no-pooling-mcmc-1model"])
        # chosen_model_lbls.extend(["cpooling-mcmc-1model"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-robust"])
        # LAST ACTIVE:
        chosen_model_lbls.extend(["no-pooling-mcmc-1model"])
        chosen_model_lbls.extend(["cpooling-mcmc-1model"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-robust-adaptive-shrinkage"])
        chosen_model_lbls.extend(["partial-pooling-mcmc-robust-adaptive-shrinkage"])

        chosen_model_lbls.extend(["cpooling-rf"])
        chosen_model_lbls.extend(["no-pooling-rf"])

        # chosen_model_lbls.extend(["model_dal_cpooling"])
        # chosen_model_lbls.extend(["model_dal_no_pooling"])

        # chosen_model_lbls.extend(["partial-pooling-mcmc-robust-adaptive-shrinkage-pw"])

        # chosen_model_lbls.extend(["partial-pooling-mcmc-robust-pw"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-selfstd"])
        # chosen_model_lbls.extend(["mcmc-selfstd-const-hyper"])

        # chosen_model_lbls.extend(["partial-pooling-mcmc-extra"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-horseshoe"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-RHS"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-RHS-pw"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-robust-pw"])
        # chosen_model_lbls.extend(["no-pooling-lin-pw"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-extra-pw"])

        # chosen_model_lbls.extend(["cpooling-mcmc-1model", "partial-pooling-mcmc-extra", "no-pooling-rf"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-extra-pw"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-robust"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-horseshoe-pw"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-selfstd"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-extra", "partial-pooling-mcmc-robust", "partial-pooling-mcmc-horseshoe"])

        # chosen_model_lbls.extend(["model_deeperf_cpooling"])
        # chosen_model_lbls.extend(["model_deeperf_no_pooling"])

        # number of pairwise interactions > 10N for N>=22
        # train_sizes = (
        #     0.125,
        #     0.25,
        #     0.375,
        #     0.5,
        #     0.675,
        #     0.75,
        #     0.9,
        #     1,
        #     1.1,
        #     1.25,
        #     1.5,
        #     1.75,
        #     2,
        #     2.5,
        #     3,
        #     4,
        # )
        train_sizes = (
            # 0.001,
            # 0.125,
            # 0.25,
            0.5,
            0.75,
            # # 0.9,
            1.0,
            # # 1.1,
            # 1.25,
            1.5,
            # 1.75,
            # 2,
            # 3.0,
            # 5,
        )

        n_reps = 3
        rnds = get_rep_ids(n_reps, num_reps, rep_offset)

        selected_data = (
            "jump3r",
            # "H2",
            # "xz",  # bad results
            # "x264",  # bad results
            # "batik",
            # "dconvert",
            # "kanzi",
            # "lrzip",  # bad results
            # "z3",
            # "artificial",
            # "VP9",
            # "x265",
            # "tuxkconfig",
            # "nfp-apache",
            # "nfp-7z",
            # "nfp-brotli",
            # "nfp-exastencils",
            # "nfp-HSQLDB",
            # "nfp-jump3r",
            # "nfp-kanzi",
            # "nfp-LLVM",
            # "nfp-lrzip",
            # "nfp-MongoDB",
            # "nfp-nginx",
            # "nfp-PostgreSQL",
            # "nfp-VP8",
            # "nfp-x264",
        )
        rep_lbl = "debug-1modelvs partial"
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        train_sizes = (
            0.125,
            0.25,
            0.5,
            # 0.75,
            1.0,
            # 1.5,
            2,
            # 3,
        )
        if training_set_size is not None:
            train_sizes = (training_set_size,)

        default_n_reps = 3
        rnds = get_rep_ids(default_n_reps, num_reps, rep_offset)

        selected_data = (
            "jump3r",
            "xz",
            "x264",
            "lrzip",
            "z3",
            # "artificial",
            "VP9",
            "x265",
            "batik",
            "dconvert",
            # "kanzi",
            "H2",
        )
        chosen_model_lbls = []

        # chosen_model_lbls.extend(["no-pooling-lin"])
        # chosen_model_lbls.extend(["cpooling-lin"])

        # # chosen_model_lbls.extend(["partial-pooling-mcmc-robust-adaptive-shrinkage-pw"])

        # FINALS
        # chosen_model_lbls.extend(["no-pooling-mcmc-1model"])
        # chosen_model_lbls.extend(["cpooling-mcmc-1model"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-robust-adaptive-shrinkage"])
        # chosen_model_lbls.extend(["cpooling-rf"])
        # chosen_model_lbls.extend(["no-pooling-rf"])
        chosen_model_lbls.extend(["model_dal_cpooling"])
        chosen_model_lbls.extend(["model_dal_no_pooling"])

        # # # LAST UNCOMMENTED!!!!
        # chosen_model_lbls.extend(["model_lasso_reg_cpool"])
        # chosen_model_lbls.extend(["model_lasso_reg_no_pool"])
        # chosen_model_lbls.extend(["no-pooling-dummy"])
        # chosen_model_lbls.extend(["cpooling-dummy"])
        #

        # chosen_model_lbls.extend(["model_lassocv_reg_no_pool"])
        # chosen_model_lbls.extend(["model_lassocv_reg_cpool"])

        # # # LAST UNCOMMENTED END!!!!

        # chosen_model_lbls.extend(["mcmc-selfstd-const-hyper"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-RHS"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-RHS-pw"])

        # chosen_model_lbls.extend(["partial-pooling-mcmc-robust"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-horseshoe"])
        # chosen_model_lbls.extend(["partial-pooling-mcmc-horseshoe-pw"])

        chosen_model_lbls.extend(["model_deeperf_cpooling"])
        chosen_model_lbls.extend(["model_deeperf_no_pooling"])

    models = {k: v for k, v in models.items() if k in chosen_model_lbls}

    print("Using systems:", selected_data)
    print("With training set size N=", train_sizes)
    print(f"And rnd seeds", rnds)
    data_providers = get_datasets(dataset_lbls=selected_data)

    print("created models")

    # data_providers = {key: data_providers[key] for key in selected_data}

    rep = Replication(
        chosen_experiments,
        models,
        data_providers,
        train_sizes,
        rnds,
        n_jobs=n_jobs,
        replication_lbl=rep_lbl,
    )
    run_id = rep.run()

    # eval = Evaluation()
    print("DONE with experiment.")
    print("running analysis")
    eval = mlfloweval.Evaluation(run_id, MLFLOW_URI, EXPERIMENT_NAME)
    eval.run()

    # eval.plot_errors()


def get_all_models(debug, n_jobs, plot, do_store=False):
    if debug:
        mcmc_num_warmup = 500
        mcmc_num_samples = 500
        mcmc_num_chains = 3
    else:
        mcmc_num_warmup = 1000
        mcmc_num_samples = 1000
        mcmc_num_chains = 3
    progress_bar = False if n_jobs else True
    mcmc_kwargs = {
        "num_warmup": mcmc_num_warmup,
        "num_samples": mcmc_num_samples,
        "num_chains": mcmc_num_chains,
        "progress_bar": progress_bar,
    }
    rf_proto = RandomForestRegressor(n_jobs=3)
    model_rf = NoPoolingEnvModel(rf_proto, preprocessings=[Standardizer()])

    complete_pooling_rf = CompletePoolingEnvModel(
        rf_proto, preprocessings=[Standardizer()]
    )
    lin_reg_proto = LinearRegression()
    model_lin_reg = NoPoolingEnvModel(lin_reg_proto, preprocessings=[Standardizer()])
    model_lin_reg_cpool = CompletePoolingEnvModel(
        lin_reg_proto, preprocessings=[Standardizer()]
    )
    model_lin_reg_cpool_pw = CompletePoolingEnvModel(
        lin_reg_proto, preprocessings=[PaiwiseOptionMapper(), Standardizer()]
    )

    model_lin_reg_pw = NoPoolingEnvModel(
        lin_reg_proto, preprocessings=[PaiwiseOptionMapper(), Standardizer()]
    )

    lasso_proto = Lasso(random_state=0)
    model_lasso_reg_no_pool = NoPoolingEnvModel(
        lasso_proto, preprocessings=[Standardizer()]
    )

    lasso_proto = Lasso(random_state=0)
    model_lasso_reg_cpool = CompletePoolingEnvModel(
        lasso_proto, preprocessings=[Standardizer()]
    )

    lassocv_proto = LassoGridSearchCV()
    model_lassocv_reg_no_pool = NoPoolingEnvModel(
        lassocv_proto, preprocessings=[Standardizer()]
    )

    lasso_proto = Lasso(random_state=0)
    model_lassocv_reg_cpool = CompletePoolingEnvModel(
        lassocv_proto, preprocessings=[Standardizer()]
    )

    deep_perf_proto = DeepPerfModel()
    model_deeperf_no_pooling = NoPoolingEnvModel(
        deep_perf_proto, preprocessings=[Standardizer()]
    )
    model_deeperf_cpooling = CompletePoolingEnvModel(
        deep_perf_proto, preprocessings=[Standardizer()]
    )

    # model_lin_reg_poly = Poly
    dummy_proto = DummyRegressor()
    model_dummy = NoPoolingEnvModel(dummy_proto)
    model_dummy_cpool = CompletePoolingEnvModel(dummy_proto)
    model_partial_extra_standardization = MCMCMultilevelPartial(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[Standardizer()],
        persist_arviz=do_store,
    )

    model_partial_extra_standardization_pw = MCMCMultilevelPartial(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[PaiwiseOptionMapper(), Standardizer()],
        persist_arviz=do_store,
    )

    model_multilevel_partial_robust = MCMCPartialRobustLasso(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[Standardizer()],
        persist_arviz=do_store,
    )

    model_multilevel_partial_robust_adaptive_shrinkage = (
        MCMCPartialRobustLassoAdaptiveShrinkage(
            plot=plot,
            **mcmc_kwargs,
            return_samples_by_default=True,
            preprocessings=[Standardizer()],
            persist_arviz=do_store,
        )
    )

    model_multilevel_partial_robust_adaptive_shrinkage_pw = (
        MCMCPartialRobustLassoAdaptiveShrinkage(
            plot=plot,
            **mcmc_kwargs,
            return_samples_by_default=True,
            preprocessings=[PaiwiseOptionMapper(), Standardizer()],
            persist_arviz=do_store,
        )
    )

    model_multilevel_partial_robust_pw = MCMCPartialRobustLasso(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[PaiwiseOptionMapper(), Standardizer()],
        persist_arviz=do_store,
    )
    model_no_pooling_combined = MCMCCombinedNoPooling(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[Standardizer()],
        persist_arviz=do_store,
    )
    model_no_pooling_combined_pw = MCMCCombinedNoPooling(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[PaiwiseOptionMapper(), Standardizer()],
        persist_arviz=do_store,
    )
    model_multilevel_partial_horseshoe = MCMCPartialHorseshoe(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[Standardizer()],
        persist_arviz=do_store,
    )
    model_multilevel_partial_RHS = MCMCRHS(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[Standardizer()],
        persist_arviz=do_store,
    )
    model_multilevel_partial_RHS_pw = MCMCRHS(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[PaiwiseOptionMapper(), Standardizer()],
        persist_arviz=do_store,
    )
    model_multilevel_partial_horseshoe_pw = MCMCPartialHorseshoe(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[PaiwiseOptionMapper(), Standardizer()],
        persist_arviz=do_store,
    )
    model_complete_pooling_combined = MCMCCombinedCompletePooling(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[Standardizer()],
        persist_arviz=do_store,
    )
    model_complete_pooling_combined_pw = MCMCCombinedCompletePooling(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[PaiwiseOptionMapper(), Standardizer()],
        persist_arviz=do_store,
    )
    model_partial_diff = MCMCPartialSelfStandardizing(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[
            Standardizer(standardize_y=False),
        ],
        persist_arviz=do_store,
    )
    model_selfstd_const = MCMCPartialSelfStandardizingConstInfl(
        plot=plot,
        **mcmc_kwargs,
        return_samples_by_default=True,
        preprocessings=[
            Standardizer(standardize_y=False),
        ],
        persist_arviz=do_store,
    )

    models = {
        "partial-pooling-mcmc-extra": model_partial_extra_standardization,
        "partial-pooling-mcmc-extra-pw": model_partial_extra_standardization_pw,
        "partial-pooling-mcmc-robust": model_multilevel_partial_robust,
        "partial-pooling-mcmc-robust-adaptive-shrinkage": model_multilevel_partial_robust_adaptive_shrinkage,
        "partial-pooling-mcmc-robust-adaptive-shrinkage-pw": model_multilevel_partial_robust_adaptive_shrinkage_pw,
        "partial-pooling-mcmc-robust-pw": model_multilevel_partial_robust_pw,
        "no-pooling-rf": model_rf,
        "no-pooling-lin-pw": model_lin_reg_pw,
        "no-pooling-lin": model_lin_reg,
        "cpooling-lin": model_lin_reg_cpool,
        "cpooling-lin-pw": model_lin_reg_cpool_pw,
        "no-pooling-dummy": model_dummy,
        "cpooling-dummy": model_dummy_cpool,
        "cpooling-rf": complete_pooling_rf,
        "no-pooling-mcmc-1model": model_no_pooling_combined,
        "no-pooling-mcmc-1model-pw": model_no_pooling_combined_pw,
        "cpooling-mcmc-1model": model_complete_pooling_combined,
        "cpooling-mcmc-1model-pw": model_complete_pooling_combined_pw,
        "partial-pooling-mcmc-horseshoe": model_multilevel_partial_horseshoe,
        "partial-pooling-mcmc-horseshoe-pw": model_multilevel_partial_horseshoe_pw,
        "partial-pooling-mcmc-selfstd": model_partial_diff,
        "mcmc-selfstd-const-hyper": model_selfstd_const,
        "partial-pooling-mcmc-RHS": model_multilevel_partial_RHS,
        "partial-pooling-mcmc-RHS-pw": model_multilevel_partial_RHS_pw,
        "model_lasso_reg_cpool": model_lasso_reg_cpool,
        "model_lasso_reg_no_pool": model_lasso_reg_no_pool,
        "model_lassocv_reg_no_pool": model_lassocv_reg_no_pool,
        "model_lassocv_reg_cpool": model_lassocv_reg_cpool,
        "model_deeperf_no_pooling": model_deeperf_no_pooling,
        "model_deeperf_cpooling": model_deeperf_cpooling,
    }
    return models


def get_datasets(train_data_folder=None, dataset_lbls=None):
    lbl_jump_r = "jump3r"
    lbl_H2 = "H2"
    lbl_xz = "xz"
    lbl_x264 = "x264"
    lbl_x265 = "x265"
    lbl_batik = "batik"
    lbl_dconvert = "dconvert"
    lbl_kanzi = "kanzi"
    lbl_lrzip = "lrzip"
    lbl_z3 = "z3"
    lbl_fastdownward = "fastdownward"
    lbl_artificial = "artificial"
    lbl_nfp_apache = "nfp-apache"
    lbl_nfp_7z = "nfp-7z"
    lbl_nfp_brotli = "nfp-brotli"
    lbl_nfp_exastencils = "nfp-exastencils"
    lbl_nfp_HSQLDB = "nfp-HSQLDB"
    lbl_nfp_jump3r = "nfp-jump3r"
    lbl_nfp_kanzi = "nfp-kanzi"
    lbl_nfp_LLVM = "nfp-LLVM"
    lbl_nfp_lrzip = "nfp-lrzip"
    lbl_nfp_MongoDB = "nfp-MongoDB"
    lbl_nfp_nginx = "nfp-nginx"
    lbl_nfp_PostgreSQL = "nfp-PostgreSQL"
    lbl_nfp_VP8 = "nfp-VP8"
    lbl_nfp_x264 = "nfp-x264"
    lbl_VP9 = "VP9"
    all_lbls = [
        lbl_jump_r,
        lbl_H2,
        lbl_xz,
        lbl_x264,
        lbl_x265,
        lbl_batik,
        lbl_dconvert,
        lbl_kanzi,
        lbl_lrzip,
        lbl_z3,
        lbl_fastdownward,
        lbl_artificial,
        lbl_VP9,
        lbl_nfp_apache,
        lbl_nfp_7z,
        lbl_nfp_brotli,
        lbl_nfp_exastencils,
        lbl_nfp_HSQLDB,
        lbl_nfp_jump3r,
        lbl_nfp_kanzi,
        lbl_nfp_LLVM,
        lbl_nfp_lrzip,
        lbl_nfp_MongoDB,
        lbl_nfp_nginx,
        lbl_nfp_PostgreSQL,
        lbl_nfp_VP8,
        lbl_nfp_x264,
    ]
    dataset_lbls = dataset_lbls or all_lbls

    data_providers = {}

    print("loading data")
    train_data_folder = train_data_folder or "./training-data"
    if lbl_H2 in dataset_lbls:
        path_h2 = os.path.join(train_data_folder, "dashboard-resources/h2/")
        h2_data_raw = DataLoaderDashboardData(path_h2)
        data_H2 = DataAdapterH2(h2_data_raw)
        h2_wl_data: WorkloadTrainingDataSet = data_H2.get_wl_data()
        data_providers[lbl_H2] = h2_wl_data

    if lbl_xz in dataset_lbls:
        path_xz = os.path.join(train_data_folder, "dashboard-resources/xz/")
        xz_data_raw = DataLoaderDashboardData(path_xz)
        data_xz = DataAdapterXZ(xz_data_raw)
        xz_wl_data: WorkloadTrainingDataSet = data_xz.get_wl_data()
        data_providers[lbl_xz] = xz_wl_data

    if lbl_jump_r in dataset_lbls:
        path_jump3r = os.path.join(train_data_folder, "jump3r.csv")
        jump3r_data_raw = DataLoaderStandard(path_jump3r)
        data_jump3r = DataAdapterJump3r(jump3r_data_raw)
        wl_data_jump3r: WorkloadTrainingDataSet = data_jump3r.get_wl_data()
        data_providers[lbl_jump_r] = wl_data_jump3r

    if lbl_x264 in dataset_lbls:
        path_x264 = os.path.join(train_data_folder, "dashboard-resources/x264/")
        x264_data_raw = DataLoaderDashboardData(path_x264)
        data_x264 = DataAdapterX264(x264_data_raw)
        x264_wl_data: WorkloadTrainingDataSet = data_x264.get_wl_data()
        data_providers[lbl_x264] = x264_wl_data

    if lbl_batik in dataset_lbls:
        path_batik = os.path.join(train_data_folder, "dashboard-resources/batik/")
        batik_data_raw = DataLoaderDashboardData(path_batik)
        data_batik = DataAdapterBatik(batik_data_raw)
        batik_wl_data: WorkloadTrainingDataSet = data_batik.get_wl_data()
        data_providers[lbl_batik] = batik_wl_data

    if lbl_dconvert in dataset_lbls:
        path_dconvert = os.path.join(train_data_folder, "dashboard-resources/dconvert/")
        dconvert_data_raw = DataLoaderDashboardData(path_dconvert)
        data_dconvert = DataAdapterDConvert(dconvert_data_raw)
        dconvert_wl_data: WorkloadTrainingDataSet = data_dconvert.get_wl_data()
        data_providers[lbl_dconvert] = dconvert_wl_data

    if lbl_kanzi in dataset_lbls:
        path_kanzi = os.path.join(train_data_folder, "dashboard-resources/kanzi/")
        kanzi_data_raw = DataLoaderDashboardData(path_kanzi)
        data_kanzi = DataAdapterKanzi(kanzi_data_raw)
        kanzi_wl_data: WorkloadTrainingDataSet = data_kanzi.get_wl_data()
        data_providers[lbl_kanzi] = kanzi_wl_data

    if lbl_lrzip in dataset_lbls:
        path_lrzip = os.path.join(train_data_folder, "dashboard-resources/lrzip/")
        lrzip_data_raw = DataLoaderDashboardData(path_lrzip)
        data_lrzip = DataAdapterLrzip(lrzip_data_raw)
        lrzip_wl_data: WorkloadTrainingDataSet = data_lrzip.get_wl_data()
        data_providers[lbl_lrzip] = lrzip_wl_data

    if lbl_z3 in dataset_lbls:
        path_z3 = os.path.join(train_data_folder, "dashboard-resources/z3/")
        z3_data_raw = DataLoaderDashboardData(path_z3)
        data_z3 = DataAdapterZ3(z3_data_raw)
        z3_wl_data: WorkloadTrainingDataSet = data_z3.get_wl_data()
        data_providers[lbl_z3] = z3_wl_data

    if lbl_fastdownward in dataset_lbls:
        path_fastdownward = os.path.join(
            train_data_folder, "FastDownward_Data/measurements.csv"
        )
        fastdownward_data_raw = DataLoaderStandard(path_fastdownward)
        data_fastdownward = DataAdapterFastdownward(fastdownward_data_raw)
        fastdownward_wl_data: WorkloadTrainingDataSet = data_fastdownward.get_wl_data()
        data_providers[lbl_fastdownward] = fastdownward_wl_data

    if lbl_artificial in dataset_lbls:
        path_artificial = os.path.join(
            train_data_folder, "artificial/artificial_data.csv"
        )
        artificial_data_raw = DataLoaderStandard(path_artificial)
        data_artificial = DataAdapterArtificial(artificial_data_raw, noise_std=1.0)
        artificial_wl_data: WorkloadTrainingDataSet = data_artificial.get_wl_data()
        data_providers[lbl_artificial] = artificial_wl_data

    if lbl_VP9 in dataset_lbls:
        path_vp9 = os.path.join(
            train_data_folder,
            "measurements_VP9_1.13.0-t_wise.csv",
        )
        vp9_data_raw = DataLoaderStandard(path_vp9)
        data_vp9 = DataAdapterVP9(vp9_data_raw)
        vp9_wl_data: WorkloadTrainingDataSet = data_vp9.get_wl_data()
        data_providers[lbl_VP9] = vp9_wl_data

    if lbl_x265 in dataset_lbls:
        path_x265 = os.path.join(
            train_data_folder,
            "measurements_x265_3.5-t_wise.csv",
        )
        x265_data_raw = DataLoaderStandard(path_x265)
        data_x265 = DataAdapterVP9(x265_data_raw)
        x265_wl_data: WorkloadTrainingDataSet = data_x265.get_wl_data()
        data_providers[lbl_x265] = x265_wl_data

    ############
    # NFP DATA #
    # ##########

    if lbl_nfp_apache in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/Apache/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPApache(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_apache] = apache_wl_data

    if lbl_nfp_7z in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/7z/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFP7z(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_7z] = apache_wl_data

    if lbl_nfp_brotli in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/brotli/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPbrotli(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_brotli] = apache_wl_data

    if lbl_nfp_exastencils in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/exastencils/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPexastencil(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_exastencils] = apache_wl_data

    if lbl_nfp_HSQLDB in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/HSQLDB/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPHSQLDB(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_HSQLDB] = apache_wl_data

    if lbl_nfp_jump3r in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/jump3r/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPjump3r(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_jump3r] = apache_wl_data

    if lbl_nfp_kanzi in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/kanzi/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPkanzi(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_kanzi] = apache_wl_data

    if lbl_nfp_LLVM in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/LLVM/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPLLVM(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_LLVM] = apache_wl_data

    if lbl_nfp_lrzip in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/lrzip/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPlrzip(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_lrzip] = apache_wl_data

    if lbl_nfp_MongoDB in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/MongoDB/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPMongoDB(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_MongoDB] = apache_wl_data

    if lbl_nfp_nginx in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/nginx/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPnginx(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_nginx] = apache_wl_data

    if lbl_nfp_PostgreSQL in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/PostgreSQL/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPposgreSQL(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_PostgreSQL] = apache_wl_data

    if lbl_nfp_VP8 in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/VP8/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPposgreVP8(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_VP8] = apache_wl_data

    if lbl_nfp_x264 in dataset_lbls:
        path_apache = os.path.join(
            train_data_folder,
            "twins-paper-data/x264/measurements.csv",
        )
        apache_data_raw = DataLoaderStandard(path_apache, sep=";")
        data_apache = DataAdapterNFPx264(apache_data_raw)
        apache_wl_data: WorkloadTrainingDataSet = data_apache.get_wl_data()
        data_providers[lbl_nfp_x264] = apache_wl_data

    # env_data = list(data_providers.values())[0]
    # env_data.get_nfps()
    # nfp_name = env_data.get_selected_nfp_name()
    # if "time" in nfp_name.lower():
    #     total_runtime += np.sum(env_data.get_y(nfp_name))
    # else:
    #     print(f"Warning: No time-related NFP found for {system_name}")
    #     break

    print("loaded data")
    return data_providers


if __name__ == "__main__":
    main()
