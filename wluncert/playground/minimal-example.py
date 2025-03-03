import itertools
import os
import time

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
from numpyro.infer import MCMC as npMCMC, NUTS as npNUTS, BarkerMH as npBMH, HMC as npHMC, SA, \
    Predictive as npPredictive
from sklearn.linear_model import BayesianRidge
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns

numpyro.set_host_device_count(5)


def model(a, b, given_obs=None):
    mean = 0.0
    stddev = 1.0
    a_is_influential = numpyro.sample("a_is_infl", npdist.Bernoulli(0.5))
    if a_is_influential:
        influence_a = numpyro.sample("influence_a", npdist.Laplace(mean, stddev))
    b_is_influential = numpyro.sample("b_is_infl", npdist.Bernoulli(0.5))
    if b_is_influential:
        influence_b = numpyro.sample("influence_b", npdist.Laplace(mean, stddev))
    if a_is_influential and b_is_influential:
        influence_ab = numpyro.sample("influence_b", npdist.Laplace(mean, stddev))


    base = numpyro.sample("base", npdist.Normal(0, 1.0))

    error_stddev = numpyro.sample("error", npdist.Exponential(0.01))
    result = base + a * influence_a + b * influence_b + a*b*influence_ab
    with numpyro.plate("data", len(a)):
        obs = numpyro.sample("nfp", npdist.Normal(result, error_stddev), obs=given_obs)
    return obs


def model_obs_stddev(a, b, c, given_obs=None, obs_stddev=None):
    mean = 0
    stddev = 1
    influence_a = numpyro.sample("influence_a", npdist.Normal(mean, stddev))
    influence_b = numpyro.sample("influence_b", npdist.Normal(mean, stddev))
    influence_c = numpyro.sample("influence_c", npdist.Normal(mean, stddev))
    # base = numpyro.sample("base", npdist.HalfNormal(2.0))
    base = numpyro.sample("base", npdist.Normal(0, 2.0))
    result = numpyro.deterministic("raw_result", base + a * influence_a + b * influence_b + c * influence_c)
    error_stddev = numpyro.sample("error", npdist.Exponential(0.01))
    # sigma = numpyro.sample('sigma', npdist.Exponential(1.))
    with numpyro.plate("data", len(a)):
        # nfp = numpyro.sample("nfp", npdist.Normal(result, error_stddev))
        nfp = numpyro.sample("nfp", npdist.Normal(result, error_stddev))
        # with numpyro.plate("observations", len(a)):
        sampled_obs = numpyro.sample("obs", npdist.Normal(nfp, obs_stddev), obs=given_obs)
    return sampled_obs


def ask_oracle(config):
    base = 20
    a, b = config
    influence_a = 5 + float(scipy.stats.norm(0, 3).rvs(1)[0])
    influence_b = 0.05
    # influence_c = 8
    nfp = influence_a * a + influence_b * b + base
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
    X, nfp, nfp_scaler = generate_training_data()
    # X_df_agg, X_agg, nfp_mean_agg, nfp_stddev_agg = compute_obs_stddev(X, nfp)
    # print("Training Data with aggregated repetitions")
    # print(X_df_agg)

    # nuts_kernel = npNUTS(model, target_accept_prob=0.9,max_tree_depth=20)
    nuts_kernel = npNUTS(model, target_accept_prob=0.9,  # max_tree_depth=200,
                         dense_mass=True)
    # nuts_kernel = npNUTS(model_obs_stddev, target_accept_prob=0.9)
    # nuts_kernel = npHMC(model, target_accept_prob=0.9)
    n_chains = 3
    mcmc = npMCMC(nuts_kernel, num_samples=2000,
                  num_warmup=5000, progress_bar=False,
                  num_chains=n_chains, )
    rng_key_ = random.PRNGKey(1)
    # mcmc.run(rng_key, X_agg[:, 0], X_agg[:, 1], X_agg[:, 2], given_obs=nfp_mean_agg, obs_stddev=nfp_stddev_agg)
    start = time.time()
    print("Starting MCMC sampling")
    mcmc.run(rng_key_, X[:, 0], X[:, 1], nfp)
    print(f"MCMC Sampling took {time.time() - start:.2f} seconds")
    mcmc.print_summary()
    az_data = az.from_numpyro(mcmc, num_chains=n_chains)
    print()
    # print("MCSE")
    # print(az.mcse(az_data))

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

    pred_conf = [0, 0]
    plot_prediction_distribution(mcmc, nfp_scaler, pred_conf, rng_key_)
    pred_conf = [1, 0]
    plot_prediction_distribution(mcmc, nfp_scaler, pred_conf, rng_key_)


def plot_prediction_distribution(mcmc, nfp_scaler, pred_conf, rng_key_):
    pred = npPredictive(model, mcmc.get_samples())
    pred_samples = np.array(
        pred(rng_key_, jnp.atleast_2d(np.array([pred_conf[0]])),
             jnp.atleast_2d(np.array([pred_conf[1]])))["nfp"].ravel())
    n_samples = 1000
    pred_samples = np.random.choice(pred_samples, n_samples)
    pred_samples_correct_scale = nfp_scaler.inverse_transform(np.atleast_2d(pred_samples)).ravel()
    oracle_preds = [ask_oracle(pred_conf) for _ in range(n_samples)]
    sns.displot({"prediction": pred_samples_correct_scale, "oracle": oracle_preds},
                kind="kde", common_norm=False)
    plt.suptitle(f"Config {str(pred_conf)}")
    plt.show()


def generate_training_data():
    # configs = list(itertools.product([True, False], repeat=3))
    configs = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]
    n_reps = 30
    print(f"Simulating {n_reps} measurement repetitions")
    configs = configs * n_reps  # simulating repeated measurements
    python_random.shuffle(configs)
    list_of_measurements = list(map(ask_oracle, configs))
    nfp = jnp.atleast_1d(list_of_measurements)
    X = jnp.atleast_2d(configs)
    X = jnp.array(MinMaxScaler().fit_transform(X))
    nfp_scaler = StandardScaler()
    nfp = jnp.array(nfp_scaler.fit_transform(jnp.atleast_2d(nfp).T)[:, 0])
    print(f"Generated data")
    return X, nfp, nfp_scaler


if __name__ == '__main__':
    main()
