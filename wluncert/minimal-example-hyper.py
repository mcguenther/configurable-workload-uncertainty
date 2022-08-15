import itertools
import os

import arviz as az
import jax.numpy as jnp
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

numpyro.set_host_device_count(4)


def workload_model(data, workloads, n_workloads, reference_y):
    workloads = jnp.array(workloads)
    y_order_of_magnitude = jnp.mean(reference_y)
    joint_coef_stdev = 0.5 * y_order_of_magnitude
    num_opts = data.shape[1]

    stddev_exp_prior = 1.0
    with numpyro.plate("hypers_vectorized", num_opts):
        hyper_coef_means = numpyro.sample("hyper_coef_means", npdist.Normal(0, joint_coef_stdev), )
        hyper_coef_stddevs = numpyro.sample("hyper_coef_stddevs", npdist.Exponential(stddev_exp_prior), )

    hyper_base_mean = numpyro.sample("hyper_base_mean", npdist.Normal(0, joint_coef_stdev), )
    hyper_base_stddev = numpyro.sample("hyper_base_stddev", npdist.Exponential(stddev_exp_prior), )

    with numpyro.plate("coefs_vectorized", num_opts):
        with numpyro.plate("workload_plate_coefs", n_workloads):
            rnd_influences = numpyro.sample("coefs", npdist.Normal(hyper_coef_means, hyper_coef_stddevs), )

    with numpyro.plate("workload_plate_bases", n_workloads):
        bases = numpyro.sample("base", npdist.Normal(hyper_base_mean, hyper_base_stddev))

    respective_influences = rnd_influences[workloads]
    respective_bases = bases[workloads]
    result_arr = jnp.multiply(data, respective_influences)
    result_arr = result_arr.sum(axis=1).ravel() + respective_bases
    error_var = numpyro.sample("error", npdist.Exponential(.10))

    with numpyro.plate("data_vectorized", result_arr.shape[0]):
        obs = numpyro.sample("measurements", npdist.Normal(result_arr, error_var), obs=reference_y)
    return obs


def ask_oracle(config, workload_id):
    base = 20
    a, b, c = config
    influence_a = float(scipy.stats.norm(5, 3).rvs(1)[0])
    influence_b = 0.05
    influence_c = 8
    nfp = influence_a * a + influence_b * b + influence_c * c + base
    nfp = nfp + workload_id * 2
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
    X, nfp, workloads = generate_training_data()
    X_df_agg, X_agg, nfp_mean_agg, nfp_stddev_agg = compute_obs_stddev(X, nfp)
    print("Training Data with aggregated repetitions")
    print(X_df_agg)

    reparam_config = {
        "coefs": LocScaleReparam(0),
        "base": LocScaleReparam(0),
        # "error": LocScaleReparam(0),
        "hyper_base_mean": LocScaleReparam(0),
        # "hyper_base_stddev": LocScaleReparam(0),
        "hyper_coef_means": LocScaleReparam(0),
        # "hyper_coef_stddevs": LocScaleReparam(0),
    }
    reparam_model = reparam(workload_model, config=reparam_config)

    # nuts_kernel = npNUTS(model, target_accept_prob=0.9,max_tree_depth=20)
    nuts_kernel = npNUTS(reparam_model, target_accept_prob=0.9)
    # nuts_kernel = npNUTS(model_obs_stddev, target_accept_prob=0.9)
    # nuts_kernel = npHMC(model_obs_stddev, )
    n_chains = 3
    progress_bar = False
    mcmc = npMCMC(nuts_kernel, num_samples=2000,
                  num_warmup=1000, progress_bar=progress_bar,
                  num_chains=n_chains, )
    rng_key_ = random.PRNGKey(10)
    rng_key_, rng_key = random.split(rng_key_)
    # mcmc.run(rng_key, X_agg[:, 0], X_agg[:, 1], X_agg[:, 2], given_obs=nfp_mean_agg, obs_stddev=nfp_stddev_agg)
    mcmc.run(rng_key, X, workloads, 3, nfp)
    mcmc.print_summary()
    az_data = az.from_numpyro(mcmc, num_chains=n_chains)
    print()
    print("ESS")
    print(az.ess(az_data))
    print()
    print("MCSE")
    print(az.mcse(az_data))

    az.plot_autocorr(az_data, combined=True)
    plt.suptitle("Auto Correlation")
    plt.tight_layout()
    plt.show()

    az.plot_ess(az_data, kind="local")
    plt.suptitle("Effective Sample Size Local")
    plt.tight_layout()
    plt.show()

    az.plot_ess(az_data, kind="quantile")
    plt.suptitle("Effective Sample Size Quantile")
    plt.tight_layout()
    plt.show()

    az.plot_trace(az_data, legend=True, )

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/toy-abc-example-trace.pdf")
    plt.savefig("results/toy-abc-example-trace.png")
    plt.show()
    return


def generate_training_data():
    # configs = list(itertools.product([True, False], repeat=3))
    configs = [
        [False, False, False],
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [True, True, True],
    ]
    n_reps = 15
    print(f"Simulating {n_reps} measurement repetitions")
    configs = configs * n_reps  # simulating repeated measurements
    workloads = [0,1,2]
    # python_random.shuffle(configs)
    nfp = [jnp.atleast_1d(list(map(ask_oracle, configs, [wl_id]*len(configs)))) for wl_id in workloads]
    nfp = [element for sublist in nfp for element in sublist]
    workloads = [[wl_id]*len(configs) for wl_id in workloads]
    workloads = [element for sublist in workloads for element in sublist]
    X = jnp.atleast_2d(configs*3)
    X = jnp.array(MinMaxScaler().fit_transform(X))
    nfp = jnp.array(StandardScaler().fit_transform(jnp.atleast_2d(nfp).T)[:, 0])
    return X, nfp, workloads


if __name__ == '__main__':
    main()
