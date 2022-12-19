import itertools

import numpyro
import numpyro.distributions as npdist

import arviz as az
import jax.numpy as jnp

from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam
import numpy
from numpyro.infer import MCMC as npMCMC, NUTS as npNUTS, BarkerMH as npBMH, HMC as npHMC, SA

import os
from jax import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpyro.infer.reparam import LocScaleReparam

numpyro.set_host_device_count(3)


def workload_model(data, workloads, n_workloads, reference_y):
    workloads = jnp.array(workloads)
    y_order_of_magnitude = jnp.mean(reference_y)
    joint_coef_stdev = y_order_of_magnitude  # 1  # 2 * y_order_of_magnitude
    num_opts = data.shape[1]
    stddev_exp_prior = y_order_of_magnitude  # 1
    with numpyro.plate("options", num_opts):
        hyper_coef_means = numpyro.sample("abs-means-hyper", npdist.Normal(0, joint_coef_stdev), )
        hyper_coef_stddevs = numpyro.sample("abs-stddevs-hyper", npdist.Exponential(stddev_exp_prior), )

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
    error_var = numpyro.sample("error", npdist.Exponential(.1))

    with numpyro.plate("data", result_arr.shape[0]):
        obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
    return obs


def compute_wl_conf_performance(length_in_s, influence_option_static_influence_time_in_s,
                                influence_option_relative_ratio, influence_option_constant_time_modifyer,
                                ft_diagnosis, ft_stereo, ft_logging):
    conf_runtime, conf_runtime_without_modifyer, effect_stereo = wl_conf_model_internal(length_in_s,
                                                                                        influence_option_constant_time_modifyer,
                                                                                        influence_option_relative_ratio,
                                                                                        influence_option_static_influence_time_in_s,
                                                                                        ft_diagnosis, ft_logging,
                                                                                        ft_stereo)
    return conf_runtime


def get_train_df(influence_option_constant_time_modifyer=0.00, influence_option_relative_ratio=0.20,
                 influence_option_static_influence_time_in_s=10, lengths=None):
    lengths = [5, 30, 100] if lengths is None else lengths
    feature_ids = ["diagnosis", "logging", "stereo"]

    n_features = len(feature_ids)
    configs = list(itertools.product([True, False], repeat=n_features))


    tups = []
    for l in lengths:
        for (diag, logging, stereo) in configs:
            perf = compute_wl_conf_performance(l, influence_option_static_influence_time_in_s,
                                               influence_option_relative_ratio, influence_option_constant_time_modifyer,
                                               diag, stereo, logging)
            tup = (diag, stereo, logging, l, perf)
            tups.append(tup)
    df = pd.DataFrame(tups, columns=[*feature_ids, "wl-base-length", "nfp"])

    return df


def infer_model(df_whole):
    reparam_model = wl_model_reparam()
    workloads = list(df_whole["wl-base-length"])

    nfp = jnp.array(df_whole["nfp"].to_numpy())
    feature_names = df_whole.columns[:3]
    wl_names = list(np.unique(workloads))
    n_workloads = len(wl_names)
    nuts_kernel = npNUTS(reparam_model, target_accept_prob=0.9)
    n_chains = 3
    progress_bar = False
    mcmc = npMCMC(nuts_kernel, num_samples=1000,
                  num_warmup=2000, progress_bar=progress_bar,
                  num_chains=n_chains, )
    rng_key_ = random.PRNGKey(0)
    rng_key_, rng_key = random.split(rng_key_)

    X = df_whole[feature_names].to_numpy()
    X = jnp.array(X)
    mcmc.run(rng_key, X, workloads, n_workloads, nfp)
    mcmc.print_summary()

    coords = {
        "features": feature_names,
        "workloads": wl_names
    }
    dims = {
        "influences": ["workloads", "features"],
        "influences_decentered": ["workloads", "features"],
        "base": ["workloads"],
        "base_decentered": ["workloads"],
        "abs-means-hyper": ["features"],
        "abs-stddevs-hyper": ["features"],
    }
    idata_kwargs = {
        "dims": dims,
        "coords": coords,
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


def wl_model_reparam():
    reparam_config = {
        "influences": LocScaleReparam(0),
        "base": LocScaleReparam(0),
    }
    reparam_model = reparam(workload_model, config=reparam_config)
    return reparam_model


def main():
    with st.sidebar:
        st.header("WL Config")
        length_in_s = st.slider("Base WL length in s", 1, 100, 10)  # 10
        hatch_base_length = "."
        st.header("SW Config")

        ft_diagnosis = st.checkbox("Diagnosis", True)
        hatch_ft_diagnosis = "*"
        influence_option_static_influence_time_in_s = st.slider("Static extra time", 1, 50, 5, 2)

        ft_stereo = st.checkbox("Stereo", True)
        hatch_ft_stereo = "/"
        influence_option_relative_ratio = st.slider("Extra time relative to WL length", 0.05, 0.5, 0.1, 0.05)

        ft_logging = st.checkbox("Logging", True)
        hatch_ft_logging = "o"
        influence_option_constant_time_modifyer = st.slider("Extends length until this point", 0.01, 0.25, 0.05, 0.01)

        do_run = st.button("Run Multilevel")

    conf_runtime, conf_runtime_without_modifyer, effect_stereo = wl_conf_model_internal(length_in_s,
                                                                                        influence_option_constant_time_modifyer,
                                                                                        influence_option_relative_ratio,
                                                                                        influence_option_static_influence_time_in_s,
                                                                                        ft_diagnosis, ft_logging,
                                                                                        ft_stereo)

    max_runtime_without_modifyer = (
            length_in_s + influence_option_static_influence_time_in_s + length_in_s * influence_option_relative_ratio)
    max_runtime = (
                          length_in_s + influence_option_static_influence_time_in_s + length_in_s * influence_option_relative_ratio) * (
                          1 + influence_option_constant_time_modifyer)

    plot_modifyer_effect = influence_option_constant_time_modifyer if ft_logging else 0
    fig, ax = plt.subplots()
    ax.add_patch(Rectangle((0, 0), conf_runtime_without_modifyer, 1., fill=True, color="0.95", hatch=hatch_base_length))
    last_x = 0
    if ft_diagnosis:
        ax.add_patch(Rectangle((last_x, 0), influence_option_static_influence_time_in_s, 1., fill=False,
                               hatch=hatch_ft_diagnosis))
        last_x = influence_option_static_influence_time_in_s
    if ft_stereo:
        ax.add_patch(Rectangle((last_x, 0), effect_stereo, 1., fill=False, hatch=hatch_ft_stereo))
        last_x += effect_stereo
    if ft_logging:
        ax.add_patch(
            Rectangle((0, 1), conf_runtime_without_modifyer, plot_modifyer_effect, fill=False, hatch=hatch_ft_logging))

    # ax.add_patch(Rectangle((0, 0), 2, 2, fill=False, hatch=h))
    st.header("Workload Executions Time Simulator")
    ax.set_xlim((0, max_runtime_without_modifyer))
    ax.set_ylim((0, influence_option_constant_time_modifyer + 1))
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Min Runtime", round(length_in_s, 2))
    c2.metric("Conf Runtime", round(conf_runtime, 2))
    c3.metric("Max Runtime", round(max_runtime, 2))

    if do_run or not st._is_running_with_streamlit:
        st.write("starting inference")
        df_whole = get_train_df()
        infer_model(df_whole)


def wl_conf_model_internal(length_in_s, influence_option_constant_time_modifyer, influence_option_relative_ratio,
                           influence_option_static_influence_time_in_s, ft_diagnosis, ft_logging, ft_stereo):
    effect_stereo = ft_stereo * length_in_s * influence_option_relative_ratio
    conf_runtime_without_modifyer = length_in_s + ft_diagnosis * influence_option_static_influence_time_in_s + effect_stereo
    conf_runtime = conf_runtime_without_modifyer * (
            1 + influence_option_constant_time_modifyer) if ft_logging else conf_runtime_without_modifyer
    return conf_runtime, conf_runtime_without_modifyer, effect_stereo


if __name__ == '__main__':
    main()
