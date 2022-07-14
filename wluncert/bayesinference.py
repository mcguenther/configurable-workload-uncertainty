import sys
import time
from matplotlib import pyplot as plt
import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.util
import pyro.optim
from pyro.optim import Adam, ClippedAdam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from abc import ABC, abstractmethod
import pycosa
from pycosa import util
import sklearn
import pprint
import jax.numpy as jnp
import numpyro.distributions as npdist
from numpyro.infer import HMCECS as npHMCECS, MCMC as npMCMC, NUTS as npNUTS, HMC as npHMC, BarkerMH, \
    Predictive as npPredictive
from jax import random
from numpyro.handlers import condition as npcondition, seed as npseed, substitute as npsubstitute, trace as nptrace
import arviz as az
import numpyro
# from eda4uncert.grammar import BaseGrammar
import math
import os
import bokeh
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
# from fuzzingbook.Grammars import Grammar
from joblib import Parallel, delayed
import logging
from copy import deepcopy
import streamlit as st
# from grammar import InfluenceModelGrammar, BaseGrammar, PairwiseGrammar
# from sws import ConfSysData, ArtificialSystem
from multiprocessing import Process, Queue
import io
from contextlib import redirect_stdout

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import gamma

numpyro.set_host_device_count(3)
import copy


class PyroRegressor(ABC, BaseEstimator, RegressorMixin):
    def __init__(self):
        self.coef_ = None
        self.samples = None

    @staticmethod
    def conditionable_model(data):
        num_opts = data.shape[1]
        num_opts_a = int(num_opts / 2)
        num_opts_b = num_opts - num_opts_a
        means = torch.zeros(num_opts)
        var = torch.ones(num_opts) * 40
        with pyro.plate("coefs_vectorized"):
            rnd_influences = pyro.sample("coefs", dist.Normal(means, var))
        mat_infl = rnd_influences.reshape(-1, 1)
        product = torch.mm(data, mat_infl).reshape(-1)
        base = pyro.sample("base", dist.Normal(0.0, 50.0))
        result = product + base
        error_var = pyro.sample("error", dist.HalfNormal(0.01))
        with pyro.plate("data_vectorized"):
            obs = pyro.sample("measurements", dist.Normal(result, error_var))

    def condition(self, y):
        return pyro.condition(self.conditionable_model, data={"measurements": y})

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def get_tuples(self, feature_names):
        pass

    @abstractmethod
    def coef_ci(self, ci: float):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class PyroMCMCRegressor(PyroRegressor):

    def __init__(self, mcmc_samples: int = 500, mcmc_tune: int = 200, ):
        self.coef_ = None
        self.samples = None
        # self.grammar = grammar
        self.mcmc_samples = mcmc_samples
        self.mcmc_tune = mcmc_tune
        self.mcmc_fitted = None

    @staticmethod
    def conditionable_model(data):
        num_opts = data.shape[1]
        num_opts_a = int(num_opts / 2)
        num_opts_b = num_opts - num_opts_a
        means = torch.zeros(num_opts)
        var = torch.ones(num_opts) * 40
        with pyro.plate("coefs_vectorized"):
            rnd_influences = pyro.sample("coefs", dist.Normal(means, var))
        mat_infl = rnd_influences.reshape(-1, 1)
        product = torch.mm(data, mat_infl).reshape(-1)
        base = pyro.sample("base", dist.Normal(0.0, 50.0))
        result = product + base
        error_var = pyro.sample("error", dist.HalfNormal(0.01))
        with pyro.plate("data_vectorized"):
            obs = pyro.sample("measurements", dist.Normal(result, error_var))

    def condition(self, y):
        return pyro.condition(self.conditionable_model, data={"measurements": y})

    def fit(self, X, y):
        obs_conditioned_model = self.condition(y)
        nuts_kernel = NUTS(obs_conditioned_model, adapt_step_size=True)
        mcmc = MCMC(nuts_kernel, num_samples=self.mcmc_samples,
                    warmup_steps=self.mcmc_tune, )
        mcmc.run(X)
        self.samples = mcmc.get_samples()

    def get_tuples(self, feature_names):
        tuples = []
        # what is about the attribute noise for mcmc
        tuples.extend([("mcmc", "base", float(val)) for val in self.samples["base"].numpy()])
        for n, rv_name in enumerate(feature_names):
            rv_samples = self.samples["coefs"][:, n]
            tuples.extend([("mcmc", rv_name, float(val)) for val in rv_samples.numpy()])
        return tuples

    def coef_ci(self, ci: float):
        pass

    def predict(self, X, n_samples: int = None, ci: float = None):
        # Predictive(model_fast, guide=guide, num_samples=100,
        # return_sites=("measurements",))
        pred = Predictive(self.conditionable_model, self.samples)
        x = jnp.atleast_2d(X)
        y_pred = pred(x, None)["measurements"]
        return y_pred


def weighted_avg_and_std(values, weights, gamma=1):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if gamma != 1:
        weights = np.power(weights, gamma)
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    if variance <= 0:
        sqr_var = 0.0
    else:
        sqr_var = math.sqrt(variance)
    return average, sqr_var


class PyroMCMCWorkloadRegressor(PyroMCMCRegressor):

    def get_prior_weighted_normal(self, x, y, gamma=1, stddev_multiplier=10, n_steps=30):
        print("Getting priors from lin regs.")
        reg_dict_final, err_dict = self.get_regression_spectrum(x, y, n_steps=n_steps)
        all_raw_errs = [errs['raw'] for errs in list(err_dict.values())]
        all_abs_errs = np.array([abs(err['y_pred'] - err['y_true']) for err in all_raw_errs])
        mean_abs_errs = all_abs_errs.mean(axis=1)
        all_rel_errs = np.array([abs((err['y_pred'] - err['y_true']) / err['y_true']) for err in all_raw_errs])
        mean_rel_errs = all_rel_errs.mean(axis=1)
        reg_list = list(reg_dict_final.values())

        means_weighted = []
        stds_weighted = []
        weights = 1 - MinMaxScaler().fit_transform(np.atleast_2d(mean_abs_errs).T).ravel()
        err_mean, err_std = weighted_avg_and_std(mean_abs_errs, weights, gamma=gamma)
        noise_sd_over_all_regs = err_mean + 3 * err_std
        root_candidates = np.array([reg.intercept_ for reg in reg_list])
        root_mean, root_std = weighted_avg_and_std(root_candidates, weights, gamma=gamma)
        for coef_id, coef in enumerate(range(x.shape[1])):
            coef_candidates = np.array([reg.coef_[coef_id] for reg in reg_list])
            mean_weighted, std_weighted = weighted_avg_and_std(coef_candidates, weights, gamma=gamma)
            means_weighted.append(mean_weighted)
            stds_weighted.append(stddev_multiplier * std_weighted)

        weighted_errs_per_sample = np.average(all_abs_errs, axis=0, weights=mean_abs_errs)
        weighted_rel_errs_per_sample = np.average(all_rel_errs, axis=0, weights=mean_rel_errs)
        return np.array(means_weighted), np.array(stds_weighted), root_mean, root_std, \
               err_mean, err_std, weighted_errs_per_sample, weighted_rel_errs_per_sample

    def fit(self, X, y, num_chains=3):
        y = jnp.atleast_1d(y)
        X = jnp.atleast_2d(X)
        self.prior_coef_means, self.prior_coef_stdvs, self.prior_root_mean, \
        self.prior_root_std, err_mean, err_std, weighted_errs_per_sample, weighted_rel_errs_per_sample = self.get_prior_weighted_normal(
            X, y, gamma=5, stddev_multiplier=5)
        gamma_prior = gamma.fit(weighted_errs_per_sample, floc=0)
        gamma_shape, gamma_loc, gamma_scale = gamma_prior
        gamma_k = gamma_shape
        gamma_theta = gamma_scale
        self.abs_err_gamma_alpha = gamma_k
        self.abs_err_gamma_beta = 1 / gamma_theta
        rel_gamma_prior = gamma.fit(weighted_rel_errs_per_sample, floc=0)
        rel_gamma_shape, rel_gamma_loc, rel_gamma_scale = rel_gamma_prior
        rel_gamma_k = rel_gamma_shape
        rel_gamma_theta = rel_gamma_scale
        self.rel_err_gamma_alpha = rel_gamma_k
        self.rel_err_gamma_beta = 1 / rel_gamma_theta
        # rel_err = jnp.mean(self.weighted_rel_errs_per_sample)
        obs_conditioned_model = self.condition(y)
        nuts_kernel = npNUTS(obs_conditioned_model, adapt_step_size=True,
                             dense_mass=False, find_heuristic_step_size=True)
        progress_bar = False
        mcmc = npMCMC(nuts_kernel, num_samples=self.mcmc_samples,
                      num_warmup=self.mcmc_tune, progress_bar=progress_bar, num_chains=num_chains, )
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, X)
        mcmc.print_summary()
        self.mcmc_fitted = mcmc
        self.samples = mcmc.get_samples()

    def get_regression_spectrum(self, x, y, n_steps=20, cv=3, n_jobs=-1):
        start = time.time()
        regs = []
        step_list = np.linspace(0, 1, n_steps)
        for l1_ratio in step_list:
            if 0 < l1_ratio < 1:
                reg_prototype = ElasticNetCV(l1_ratio=l1_ratio, cv=cv, n_jobs=n_jobs)
                reg, err = self.fit_and_eval_lin_reg(x, y, reg_proto=reg_prototype)
                regs.append((reg, err))
        ridge = RidgeCV(cv=cv)
        lasso = LassoCV(cv=cv, n_jobs=n_jobs)
        for reg in [ridge, lasso]:
            fitted_reg, err = self.fit_and_eval_lin_reg(x, y, reg_proto=reg)
            regs.append((fitted_reg, err))

        reg_dict = {l1_ratio: tup[0] for tup, l1_ratio in zip(regs, step_list)}
        err_dict = {l1_ratio: tup[1] for tup, l1_ratio in zip(regs, step_list)}

        end = time.time()
        cost = end - start
        # self.prior_spectrum_cost = cost
        print("Prior Spectrum Computation took", cost)
        return reg_dict, err_dict

    def fit_and_eval_lin_reg(self, x, y, reg_proto=None):
        lr = copy.deepcopy(reg_proto)
        lr.fit(x, y)
        errs = get_err_dict(lr, x, y)
        return lr, errs

    def condition(self, y):
        return npcondition(self.conditionable_model, data={"measurements": y})

    # @staticmethod
    def conditionable_model(self, data):
        num_opts = data.shape[1]
        joint_coef_stdev = np.mean(np.abs(self.prior_coef_stdvs) * 2)
        with numpyro.plate("coefs_vectorized", num_opts):
            rnd_influences = numpyro.sample("coefs", npdist.Normal(0, joint_coef_stdev), )
        mat_infl = rnd_influences.reshape(-1, 1)
        product = jnp.matmul(data, mat_infl).reshape(-1)
        base = numpyro.sample("base", npdist.HalfNormal(self.prior_root_std * 2))
        result = product + base
        # error_var = numpyro.sample("error", npdist.Gamma(self.abs_err_gamma_alpha/5, self.abs_err_gamma_beta))
        # error_var = numpyro.sample("error", npdist.Gamma(1.0, 2.9))
        error_var = numpyro.sample("error", npdist.Exponential(1.5))
        # relative_error_var = numpyro.sample("error_rel", npdist.Gamma(self.rel_err_gamma_alpha/5, self.rel_err_gamma_beta))
        # relative_error_var = numpyro.sample("error_rel", npdist.Gamma(1.0, 0.01))

        relative_error_var = numpyro.sample("error_rel", npdist.Laplace(0, 0.001))
        result = result + (relative_error_var * result)

        # error_var = numpyro.sample("error", npdist.HalfNormal(0.01) )
        with numpyro.plate("data_vectorized", len(data)) as ind:
            obs = numpyro.sample("measurements", npdist.Normal(result, error_var))
        return obs

    def get_tuples(self, feature_names):
        tuples = []
        # what is about the attribute noise for mcmc
        tuples.extend([("mcmc", "base", float(val)) for val in self.samples["base"]])
        tuples.extend([("mcmc", "noise", norm.rvs(0, val)) for val in self.samples["error"]])
        for n, rv_name in enumerate(feature_names):
            rv_samples = self.samples["coefs"][:, n]
            tuples.extend([("mcmc", rv_name, float(val)) for val in rv_samples])
        return tuples

    def predict_ci(self, X, n_samples: int = None, ci: float = None):
        # Predictive(model_fast, guide=guide, num_samples=100,
        # return_sites=("measurements",))
        pred = npPredictive(self.conditionable_model, self.samples)
        x = jnp.atleast_2d(X)
        rng_key = random.PRNGKey(42)
        y_pred = pred(rng_key, x)["measurements"]
        return y_pred

    def predict(self, X, **kwargs):
        y_pred = self.predict_ci(X)
        mode_area = az.hdi(np.array(y_pred), hdi_prob=0.01, )
        mode_approx = np.mean(mode_area, axis=1)
        return mode_approx


def get_err_dict(reg, xs, ys):
    y_pred = reg.predict(xs)
    errors = get_err_dict_from_predictions(y_pred, xs, ys)
    return errors


def score_rmse(reg, xs, y_true, y_predicted=None):
    if y_predicted is None:
        y_predicted = reg.predict(np.atleast_2d(xs))
    rms = np.sqrt(mean_squared_error(y_true, y_predicted))
    return rms


def score_mape(reg, xs, y_true, y_predicted=None):
    if y_predicted is None:
        y_predicted = reg.predict(np.atleast_2d(xs))
    mape = np.mean(np.abs((y_true - y_predicted) / y_true)) * 100
    return mape


def get_err_dict_from_predictions(y_pred, xs, ys):
    mape = score_mape(None, xs, ys, y_pred)
    rmse = score_rmse(None, xs, ys, y_pred)
    r2 = r2_score(ys, y_pred)
    errors = {"r2": r2, "mape": mape, "rmse": rmse,
              "raw": {"x": xs, "y_pred": y_pred, "y_true": ys}}
    return errors


class RelativeScalingWorkloadRegressor(PyroMCMCWorkloadRegressor):
    """
    Class to do MCMC with struture finding without categorical variables
    """

    def conditionable_model(self, data):
        joint_coef_stdev = np.mean(np.abs(self.prior_coef_stdvs) * 2)
        num_opts = data.shape[1]
        # wl_scale = numpyro.sample("workload-scaling", npdist.Normal(0, joint_coef_stdev))
        wl_scale = numpyro.sample("workload-scaling", npdist.Normal(1.0, 50))
        with numpyro.plate("coefs_vectorized", num_opts):
            rnd_influences = numpyro.sample("coefs", npdist.Normal(0, joint_coef_stdev), )
        mat_infl = rnd_influences.reshape(-1, 1)
        product = jnp.matmul(data, mat_infl).reshape(-1)
        base = numpyro.sample("base", npdist.HalfNormal(self.prior_root_std * 2))
        result = product + base

        wl_col = data[:, -1]
        relative_workload_diff = wl_col * wl_scale
        result = result * relative_workload_diff

        # error_var = numpyro.sample("error", npdist.Gamma(self.abs_err_gamma_alpha/5, self.abs_err_gamma_beta))
        error_var = numpyro.sample("error", npdist.Exponential(1.5))
        # relative_error_var = numpyro.sample("error_rel", npdist.Gamma(self.rel_err_gamma_alpha/5, self.rel_err_gamma_beta))

        relative_error_var = numpyro.sample("error_rel", npdist.Laplace(0, 0.01))
        result = result + (relative_error_var * result)

        # error_var = numpyro.sample("error", npdist.HalfNormal(0.01) )
        with numpyro.plate("data_vectorized", len(data)) as ind:
            obs = numpyro.sample("measurements", npdist.Normal(result, error_var))
            return obs


class HierarchicalWorkloadRegressor(PyroMCMCWorkloadRegressor):
    """
    Class to do MCMC with struture finding without categorical variables
    """

    def conditionable_model(self, data):
        joint_coef_stdev = np.mean(np.abs(self.prior_coef_stdvs) * 2)
        num_opts = data.shape[1]
        # wl_scale = numpyro.sample("workload-scaling", npdist.Normal(0, joint_coef_stdev))
        wl_scale = numpyro.sample("workload-scaling", npdist.Gamma(1.0, 0.01))
        with numpyro.plate("coefs_vectorized", num_opts):
            rnd_influences = numpyro.sample("coefs", npdist.Normal(0, joint_coef_stdev), )
        mat_infl = rnd_influences.reshape(-1, 1)
        product = jnp.matmul(data, mat_infl).reshape(-1)
        base = numpyro.sample("base", npdist.Gamma(self.prior_root_mean, self.prior_root_std))
        result = product + base

        wl_col = data[:, -1]
        relative_workload_diff = wl_col * wl_scale
        result = result * relative_workload_diff

        # error_var = numpyro.sample("error", npdist.Gamma(self.abs_err_gamma_alpha/5, self.abs_err_gamma_beta))
        error_var = numpyro.sample("error", npdist.Gamma(1.0, 0.9))
        # relative_error_var = numpyro.sample("error_rel", npdist.Gamma(self.rel_err_gamma_alpha/5, self.rel_err_gamma_beta))
        relative_error_var = numpyro.sample("error_rel", npdist.Gamma(1.0, 0.01))
        result = result + (relative_error_var * result)
        # error_var = numpyro.sample("error", npdist.HalfNormal(0.01) )
        with numpyro.plate("data_vectorized", len(data)) as ind:
            obs = numpyro.sample("measurements", npdist.Normal(result, error_var))
            return obs


def get_mape(sk_model, x, y) -> float:
    pred_y = sk_model.predict(x)
    apes = np.abs(pred_y - y) / y * 100
    mape = float(np.mean(apes))
    return mape


def st_create_scores(sk_model, x, y):
    score = sk_model.score(x, y)
    # print("MCMC", score)
    # st.write(score)
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="R²", value=round(score, 2))
    mape = get_mape(sk_model, x, y)
    mape_rounded = round(mape, 1)
    with c2:
        st.metric(label="MAPE", value=f"{mape_rounded} %")


def main():
    no_streamlit = len(sys.argv) > 1

    st.sidebar.write("# Training Data")

    sub_folder_name = "training-data/"

    dirname = os.path.dirname(__file__)
    data_path_parent = os.path.join(dirname, sub_folder_name)
    avail_sws = [s.replace(".csv", "") for s in list(os.listdir(data_path_parent))]
    sws = st.sidebar.selectbox("Software System", avail_sws, index=1)

    data_path = os.path.join(data_path_parent, f"{sws}.csv")
    sys_df = pd.read_csv(data_path)
    cleared_sys_df = util.remove_multicollinearity(sys_df)
    # read workloads and NFPs
    cols = cleared_sys_df.columns
    workload_idx = list(cols).index("workload")
    nfps = cols[workload_idx + 1:]
    chosen_nfp = st.sidebar.selectbox("NFP", nfps, index=0)

    # wl_filter = "tpcc"
    cleared_sys_df[["workload-name", "workload-scale"]] = cleared_sys_df["workload"].str.split("-", expand=True)
    workloads = list(cleared_sys_df["workload-name"].unique())
    chosen_wl = st.sidebar.selectbox("Workload Type", workloads, index=0)
    tpcc_df = cleared_sys_df[cleared_sys_df["workload-name"].str.contains(chosen_wl)]
    train_size_fraq = st.sidebar.slider("Training set size ratio", 0.05, 0.95, 0.35, step=0.05)
    print(tpcc_df.head())
    tpcc_df = tpcc_df.drop(columns=["workload", "workload-name", "config"])
    wl_levels = tpcc_df["workload-scale"].unique()

    nfp = chosen_nfp
    train_df, test_df = sklearn.model_selection.train_test_split(tpcc_df, train_size=train_size_fraq)
    y_train = train_df[nfp].to_numpy()
    y_test = test_df[nfp].to_numpy()
    train_df_without_nfp = train_df.drop(columns=[nfp])
    x_train = train_df_without_nfp.astype(int).to_numpy()
    x_test = test_df.drop(columns=[nfp]).astype(int).to_numpy()

    ## Models to fit
    st.sidebar.write("# Models")
    models=["WL-Ignorant MCMC", "Linear MCMC", "RelWL MCMC Model"]
    chosen_models = st.sidebar.multiselect("MCMC Models (expensive)", options=models, default=models)

    ## MCMC samples
    st.sidebar.write("# Training Parameters")
    mcmc_tune = st.sidebar.slider("Tuning MCMC samples", 250, 4000, value=500, step=250)
    mcmc_samples = st.sidebar.slider("Posterior MCMC samples", 250, 4000, value=250, step=250)
    num_chains = st.sidebar.slider("Parallel MCMC Chains", 1, 6, value=3, step=1)

    ## Scaling
    scaler = MinMaxScaler()
    x_train[:,-1] = scaler.fit_transform(np.atleast_2d(x_train[:,-1]).T)[:,-1].ravel()
    x_test[:,-1] = scaler.transform(np.atleast_2d(x_test[:,-1]).T).ravel()

    x_train_no_wl = x_train[:, :-1]
    x_test_no_wl = x_test[:, :-1]
    features = list(list(train_df_without_nfp.columns))
    st.title(f"{sws} on workload {chosen_wl}")
    st.write("Workload Levels")
    st.write(wl_levels)
    # st.write("## ")
    with st.expander("Training Data"):
        st.write(train_df)
    if st.sidebar.button("Let's gooooooo") or no_streamlit:
        container_own_models = st.container()
        container_reference_models = st.container()

        with container_own_models:
            # c1, c2, c3 = st.columns(3)
            # with c1:
            # num_chains = 3
            # mcmc_samples = 500
            # mcmc_tune = 750
            if "WL-Ignorant MCMC" in chosen_models:
                st.write("## WL-Ignorant MCMC")
                st.write("This model does not get the workload feature.")
                with st.spinner("Fitting MCMC"):
                    mcmc_reg = PyroMCMCWorkloadRegressor(mcmc_samples=mcmc_samples, mcmc_tune=mcmc_tune)
                    mcmc_reg.fit(x_train_no_wl, y_train, num_chains)
                plot_mcmc_scores(features[:-1], mcmc_reg, num_chains, x_test_no_wl, y_test)
            if "Linear MCMC" in chosen_models:
                st.write("## Linear MCMC")
                st.write("This model gets the workload feature but treats it as a regular option.")
                with st.spinner("Fitting MCMC Model"):
                    mcmc_reg = PyroMCMCWorkloadRegressor(mcmc_samples=mcmc_samples, mcmc_tune=mcmc_tune)
                    mcmc_reg.fit(x_train, y_train, num_chains)
                plot_mcmc_scores(features, mcmc_reg, num_chains, x_test, y_test)
            if "RelWL MCMC Model" in chosen_models:
                st.write("## RelWL MCMC Model")
                st.write("This model gets the workload feature and learns a relative transfer of computed performance values between workloads. "
                         "Hence, the model does not differenciate between different option influence.")
                with st.spinner("Fitting RelWL MCMC Model"):
                    rel_wl_mcmc_reg = RelativeScalingWorkloadRegressor(mcmc_samples=mcmc_samples, mcmc_tune=mcmc_tune)
                    rel_wl_mcmc_reg.fit(x_train, y_train, num_chains)
                plot_mcmc_scores(features, rel_wl_mcmc_reg, num_chains, x_test, y_test)

        with container_reference_models:
            # c1, c2, c3 = st.columns(3)
            # with c1:

            st.header("Baseline Regs with workload feature")
            st.write("## Linear Reg")
            with st.spinner("Fitting Linear Regression"):
                lin_reg = LinearRegression()
                lin_reg.fit(x_train, y_train)
                st_create_scores(lin_reg, x_test, y_test)
            # with c2:
            st.write("## RF")
            with st.spinner("Fitting RF"):
                rf_reg = RandomForestRegressor()
                rf_reg.fit(x_train, y_train)
                st_create_scores(rf_reg, x_test, y_test)

            st.header("Baseline Regs WITHOUT WL FEATURE!")
            st.write("## Dummy Mean Reg")
            with st.spinner("Fitting Dummy Regression"):
                dummy_regr = DummyRegressor(strategy="mean")
                dummy_regr.fit(x_train, y_train)
                st_create_scores(dummy_regr, x_test, y_test)
            st.write("## Linear Reg")
            with st.spinner("Fitting Linear Regression"):
                lin_reg = LinearRegression()
                lin_reg.fit(x_train_no_wl, y_train)
                st_create_scores(lin_reg, x_test_no_wl, y_test)
            st.write("## RF ")
            with st.spinner("Fitting RF"):
                rf_reg = RandomForestRegressor()
                rf_reg.fit(x_train_no_wl, y_train)
                st_create_scores(rf_reg, x_test_no_wl, y_test)
            # with c3:
            # st.header("ANN for reference")
            # st.write("## ANN")
            # with st.spinner("Fitting MLP"):
            #     mlp_reg = MLPRegressor(max_iter=2500)
            #     mlp_reg.fit(x_train, y_train)
            #     st_create_scores(mlp_reg, x_test, y_test)
        st.balloons()


def plot_mcmc_scores(features, mcmc_reg, num_chains, x_test, y_test):
    graph = numpyro.render_model(mcmc_reg.conditionable_model, model_args=(x_test,), filename="model.pdf",
                                 render_params=True, render_distributions=True)
    st.graphviz_chart(graph)
    with st.spinner("Plotting"):
        coords = {"features": features}
        dims = {"coefs": ["features"]}
        idata_kwargs = {
            "dims": dims,
            "coords": coords,
            # "constant_data": {"x": xdata}
        }
        az_data = az.from_numpyro(mcmc_reg.mcmc_fitted, num_chains=num_chains,  **idata_kwargs)
        az.plot_trace(az_data, )#compact=True, var_names=("base", "coefs"))
        fig = plt.gcf()
        plt.tight_layout()
        st.pyplot(fig)

        # az.plot_posterior(az_data, hdi_prob=0.95, point_estimate="mode",multimodal=True, )
        # fig = plt.gcf()
        # plt.tight_layout()
        # st.pyplot(fig)


        az.plot_forest(az_data)
        fig = plt.gcf()
        plt.tight_layout()
        st.pyplot(fig)
    with st.spinner("Scoring"):
        divergences = mcmc_reg.mcmc_fitted.get_extra_fields()["diverging"].sum()
        if divergences:
            st.error(f"Encountered {divergences} divergences! This means that the sampling did not find the posterior reliably and we should consider a different model structure.")
        st_create_scores(mcmc_reg, x_test, y_test)


def capture_output(q):
    st.write("Inside Process")
    with io.StringIO() as buf, redirect_stdout(buf):
        q.put(buf)

        "Inside captured environment"
        print('redirected')
        output = buf.getvalue()


def pandas_to_tensor(df):
    return torch.tensor(np.array(df)).float()


if __name__ == '__main__':
    main()
