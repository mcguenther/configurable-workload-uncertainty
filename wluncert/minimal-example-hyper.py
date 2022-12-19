import itertools
import os

import arviz as az
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as npdist
import pandas as pd
import scipy.stats
import random as python_random
from jax import random
from matplotlib import pyplot as plt
from numpyro.handlers import condition as npcondition
from numpyro.infer import MCMC as npMCMC, NUTS as npNUTS, BarkerMH as npBMH, HMC as npHMC, SA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam
import numpy
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300

numpyro.set_host_device_count(4)


def workload_model(data, workloads, n_workloads, reference_y):
    workloads = jnp.array(workloads)
    y_order_of_magnitude = jnp.std(reference_y)
    joint_coef_stdev = 0.25 # 2 * y_order_of_magnitude
    num_opts = data.shape[1]
    stddev_exp_prior = 0.250
    with numpyro.plate("options", num_opts):
        hyper_coef_means = numpyro.sample("means-hyper", npdist.Normal(0, joint_coef_stdev), )
        hyper_coef_stddevs = numpyro.sample("stddevs-hyper", npdist.Exponential(stddev_exp_prior), )

    hyper_base_mean = numpyro.sample("base mean hyperior", npdist.Normal(0, joint_coef_stdev), )
    hyper_base_stddev = numpyro.sample("base stddevs hyperior", npdist.Exponential(stddev_exp_prior), )

    with numpyro.plate("options", num_opts):
        with numpyro.plate("workloads", n_workloads):
            rnd_influences = numpyro.sample("influences", npdist.Normal(hyper_coef_means, hyper_coef_stddevs), )

    with numpyro.plate("workloads", n_workloads):
        bases = numpyro.sample("base", npdist.Normal(hyper_base_mean, hyper_base_stddev))

    respective_influences = rnd_influences[workloads]
    respective_bases = bases[workloads]
    result_arr = jnp.multiply(data, respective_influences)
    result_arr = result_arr.sum(axis=1).ravel() + respective_bases
    error_var = numpyro.sample("error", npdist.Exponential(.05))

    with numpyro.plate("data", result_arr.shape[0]):
        obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
    return obs


def workload_model_abs_rel(data, workloads, n_workloads, reference_y):
    workloads = jnp.array(workloads)
    y_order_of_magnitude = jnp.std(reference_y)
    joint_coef_stdev = 0.25 # 2 * y_order_of_magnitude
    num_opts = data.shape[1]
    stddev_exp_prior = 0.250
    with numpyro.plate("options", num_opts):
        hyper_coef_means = numpyro.sample("means-hyper", npdist.Normal(0, joint_coef_stdev), )
        hyper_coef_stddevs = numpyro.sample("stddevs-hyper", npdist.Exponential(stddev_exp_prior), )

    hyper_base_mean = numpyro.sample("base mean hyperior", npdist.Normal(0, joint_coef_stdev), )
    hyper_base_stddev = numpyro.sample("base stddevs hyperior", npdist.Exponential(stddev_exp_prior), )

    with numpyro.plate("options", num_opts):
        with numpyro.plate("workloads", n_workloads):
            rnd_influences = numpyro.sample("influences", npdist.Normal(hyper_coef_means, hyper_coef_stddevs), )

    with numpyro.plate("workloads", n_workloads):
        bases = numpyro.sample("base", npdist.Normal(hyper_base_mean, hyper_base_stddev))

    respective_influences = rnd_influences[workloads]
    respective_bases = bases[workloads]
    result_arr = jnp.multiply(data, respective_influences)
    result_arr = result_arr.sum(axis=1).ravel() + respective_bases
    error_var = numpyro.sample("error", npdist.Exponential(.05))

    with numpyro.plate("data", result_arr.shape[0]):
        obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
    return obs


def ask_oracle(config, workload_id, noise_percent=0.1):
    base = 20 #+ workload_id * 5
    a, b, c = config
    # a, b = config
    a_mean = 10
    a_stddev = a_mean * noise_percent / 100
    influence_a = float(scipy.stats.norm(a_mean, a_stddev).rvs(1)[0]) + workload_id * 4
    # influence_b = 3 if scipy.stats.bernoulli(0.35).rvs(1)[0] else 10
    influence_b = 5
    # influence_c = 10 if workload_id == 0 else 0
    influence_c = 3
    # nfp = influence_a * a + influence_b * b + base
    nfp = base + influence_a * a + influence_b * b  + c * influence_c
    # nfp = nfp * ((workload_id * 0.5)  + 1)
    return nfp


def compute_obs_stddev(X, nfp, group_cols=None):
    if group_cols is None:
        group_cols = list(range(X.shape[1]))

    df = pd.DataFrame(X)
    df["nfp"] = nfp
    grouped_df = df.groupby(group_cols)
    results = grouped_df.agg(["mean", "std"], columns="nfp").reset_index()
    # results.columns.droplevel(0)
    results.columns = [*group_cols, "mean", "std"]
    X_new = jnp.array(results[group_cols])
    means = jnp.array(results["mean"])
    stddevs = jnp.array(results["std"])

    # standardize standard deviation 0_O
    # stddevs = stddevs / jnp.std(stddevs)
    # means = means / jnp.std(means)
    return results, X_new, means, stddevs


def main():
    feature_names = ["A", "B", "C"]
    wl_names = ["WL1", "WL2", "WL3"]
    X, nfp, workloads = generate_training_data(feature_ids=feature_names, workloads_ids=wl_names)
    X_df_agg, X_agg, nfp_mean_agg, nfp_stddev_agg = compute_obs_stddev(X, nfp)
    # nfp = jnp.repeat(jnp.atleast_2d(nfp), 2, 0).T
    print("Training Data with aggregated repetitions")
    print(X_df_agg)
    n_workloads = len(numpy.unique((workloads)))
    graph = numpyro.render_model(workload_model, model_args=(X, workloads, n_workloads, nfp),
                                 # filename="model.pdf",
                                 filename="minimal-workload-model.pdf")
    # graph = numpyro.render_model(workload_model, model_args=(X, workloads, 3, nfp),
    #                                  # filename="model.pdf",
    #                                  render_params=True, render_distributions=True, filename="minimal-model.pdf")

    reparam_config = {
        "influences": LocScaleReparam(0),
        "base": LocScaleReparam(0),
        # "error": LocScaleReparam(0),
        # "base mean hyperior": LocScaleReparam(0),
        # "base stddevs hyperior": LocScaleReparam(0),
        # "stddevs-hyper": LocScaleReparam(0),
        # "means-hyper": LocScaleReparam(0),
    }
    reparam_model = reparam(workload_model, config=reparam_config)

    # nuts_kernel = npNUTS(model, target_accept_prob=0.9,max_tree_depth=20)
    nuts_kernel = npNUTS(reparam_model, target_accept_prob=0.9)
    # nuts_kernel = npNUTS(model_obs_stddev, target_accept_prob=0.9)
    # nuts_kernel = npHMC(model_obs_stddev, )
    n_chains = 3
    progress_bar = False
    mcmc = npMCMC(nuts_kernel, num_samples=2000,
                  num_warmup=2000, progress_bar=progress_bar,
                  num_chains=n_chains, )
    rng_key_ = random.PRNGKey(0)
    rng_key_, rng_key = random.split(rng_key_)
    # mcmc.run(rng_key, X_agg[:, 0], X_agg[:, 1], X_agg[:, 2], given_obs=nfp_mean_agg, obs_stddev=nfp_stddev_agg)
    mcmc.run(rng_key, X, workloads, n_workloads, nfp)
    mcmc.print_summary()

    coords = {
        "features": feature_names,
        # "features": ["A", "B"],
        "workloads": wl_names
    }
    dims = {
        "influences": ["workloads", "features"],
        "influences_decentered": ["workloads", "features"],
        "base": ["workloads"],
        "base_decentered": ["workloads"],
        "means-hyper": ["features"],
        "stddevs-hyper": ["features"],
    }
    idata_kwargs = {
        "dims": dims,
        "coords": coords,
        # "constant_data": {"x": xdata}
    }

    az_data = az.from_numpyro(mcmc, num_chains=n_chains, **idata_kwargs)
    print()
    print("WAIC")
    waic_data = az.waic(az_data)
    print(waic_data)
    print(waic_data.waic_i)
    print()
    print()
    print("ESS")
    print(az.ess(az_data))
    print()
    print()
    print("MCSE")
    print(az.mcse(az_data))

    # az.plot_autocorr(az_data, combined=True)
    # plt.suptitle("Auto Correlation")
    # plt.tight_layout()
    # plt.show()
    #
    # az.plot_ess(az_data, kind="local")
    # plt.suptitle("Effective Sample Size Local")
    # plt.tight_layout()
    # plt.show()
    #
    # az.plot_ess(az_data, kind="quantile")
    # plt.suptitle("Effective Sample Size Quantile")
    # plt.tight_layout()
    # plt.show()

    az.plot_trace(az_data, legend=True, )

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/toy-abc-example-trace.pdf")
    plt.savefig("results/toy-abc-example-trace.png")
    plt.show()
    return


def generate_training_data(feature_ids=None, workloads_ids=None):
    if workloads_ids is None:
        workloads_ids = [0, 1, 2]
    if feature_ids is None:
        feature_ids = ["A", "B", "C"]

    n_features = len(feature_ids)
    workloads_ids = list(range(len(workloads_ids)))
    configs = list(itertools.product([True, False], repeat=n_features))
    # configs = [
    #     [False, False, False],
    #     [True, False, False],
    #     [False, True, False],
    #     [False, False, True],
    #     [True, True, True],
    # ]
    n_reps = 3
    print(f"Simulating {n_reps} measurement repetitions")
    configs = configs * n_reps  # simulating repeated measurements

    # python_random.shuffle(configs)
    nfp = [numpy.atleast_1d(list(map(ask_oracle, configs, [wl_id] * len(configs)))) for wl_id in workloads_ids]
    nfp = [element for sublist in nfp for element in sublist]
    workloads = [[wl_id] * len(configs) for wl_id in workloads_ids]
    workloads = [element for sublist in workloads for element in sublist]
    X = numpy.atleast_2d([c for c in configs for wl_id in workloads_ids])
    # X = jnp.array(MinMaxScaler().fit_transform(X))

    df = pd.DataFrame(X, columns=feature_ids)
    df["workload"] = workloads
    df["nfp"] = nfp
    df_st_by = df.groupby('workload').transform(lambda x: (x - x.mean()) / x.std())

    nfp = jnp.array(df_st_by["nfp"])
    X = jnp.array(df_st_by[feature_ids])
    return X, nfp, workloads


if __name__ == '__main__':
    main()
