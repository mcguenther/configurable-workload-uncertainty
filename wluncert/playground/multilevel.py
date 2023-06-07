import sys
import time
from functools import partial

import jax
from matplotlib import pyplot as plt
import torch
import pyro
import pyro.util
import pyro.optim
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
from jax import random, jit
from numpyro.handlers import condition as npcondition, seed as npseed, substitute as npsubstitute, trace as nptrace
import arviz as az
import numpyro
# from eda4uncert.grammar import BaseGrammar
import math
import os
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV, Lasso, Ridge, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import streamlit as st
import io
from contextlib import redirect_stdout

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from scipy.stats import gamma
from sklearn.pipeline import make_pipeline
from absl import logging

from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

# logging.set_verbosity(4)


numpyro.set_host_device_count(3)
import copy

st.set_page_config(page_title='Workload Uncertainty', page_icon="🪬")


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
    def __init__(self, ys, mcmc_samples: int = 500, mcmc_tune: int = 200, ):
        super().__init__(mcmc_samples, mcmc_tune)
        self.problem_y = ys

    @st.cache
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
        # self.prior_coef_means, self.prior_coef_stdvs, self.prior_root_mean, \
        # self.prior_root_std, err_mean, err_std, weighted_errs_per_sample, weighted_rel_errs_per_sample = self.get_prior_weighted_normal(
        #     X, y, gamma=5, stddev_multiplier=5)
        # gamma_prior = gamma.fit(weighted_errs_per_sample, floc=0)
        # gamma_shape, gamma_loc, gamma_scale = gamma_prior
        # gamma_k = gamma_shape
        # gamma_theta = gamma_scale
        # self.abs_err_gamma_alpha = gamma_k
        # self.abs_err_gamma_beta = 1 / gamma_theta
        # rel_gamma_prior = gamma.fit(weighted_rel_errs_per_sample, floc=0)
        # rel_gamma_shape, rel_gamma_loc, rel_gamma_scale = rel_gamma_prior
        # rel_gamma_k = rel_gamma_shape
        # rel_gamma_theta = rel_gamma_scale
        # self.rel_err_gamma_alpha = rel_gamma_k
        # self.rel_err_gamma_beta = 1 / rel_gamma_theta
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
        y_order_of_magnitude = np.mean(self.problem_y) * 2
        joint_coef_stdev = y_order_of_magnitude
        with numpyro.plate("coefs_vectorized", num_opts):
            rnd_influences = numpyro.sample("coefs", npdist.Normal(0, joint_coef_stdev), )
        mat_infl = rnd_influences.reshape(-1, 1)
        product = jnp.matmul(data, mat_infl).reshape(-1)
        base = numpyro.sample("base", npdist.HalfNormal(y_order_of_magnitude))
        result = product + base
        # error_var = numpyro.sample("error", npdist.Gamma(self.abs_err_gamma_alpha/5, self.abs_err_gamma_beta))
        # error_var = numpyro.sample("error", npdist.Gamma(1.0, 2.9))
        error_var = numpyro.sample("error", npdist.Exponential(1.5))
        # relative_error_var = numpyro.sample("error_rel", npdist.Gamma(self.rel_err_gamma_alpha/5, self.rel_err_gamma_beta))

        relative_error_var = numpyro.sample("error_rel", npdist.Laplace(0, 0.01))
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
        num_opts = data.shape[1]
        y_order_of_magnitude = np.mean(self.problem_y) * 2
        joint_coef_stdev = y_order_of_magnitude
        num_opts = data.shape[1]
        # wl_scale = numpyro.sample("workload-scaling", npdist.Normal(0, joint_coef_stdev))
        # wl_scale = numpyro.sample("workload-scaling", npdist.Normal(1.0, 50))
        lower, upper = az.hdi(self.problem_y, hdi_prob=0.5)
        robust_range = upper / lower
        st.write(f"Using scaling of {robust_range}")
        wl_scale = numpyro.sample("workload-scaling", npdist.HalfNormal(robust_range))
        with numpyro.plate("coefs_vectorized", num_opts):
            rnd_influences = numpyro.sample("coefs", npdist.Normal(0, joint_coef_stdev), )
        mat_infl = rnd_influences.reshape(-1, 1)
        product = jnp.matmul(data, mat_infl).reshape(-1)
        base = numpyro.sample("base", npdist.HalfNormal(y_order_of_magnitude))
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
    def __init__(self, *args, **kwargs):
        super(HierarchicalWorkloadRegressor, self).__init__(*args, **kwargs)
        self.workloads = []

    def fit(self, X, workloads, y, num_chains=3):
        workloads = tuple(workloads)
        obs_conditioned_model = self.condition(y)
        # obs_conditioned_model = jax.jit(obs_conditioned_model, static_argnums=1)

        reparam_config = {
            "coefs": LocScaleReparam(0),
            "base": LocScaleReparam(0),
            # "error": LocScaleReparam(0),
            "hyper_base_mean": LocScaleReparam(0),
            # "hyper_base_stddev": LocScaleReparam(0),
            "hyper_coef_means": LocScaleReparam(0),
            # "hyper_coef_stddevs": LocScaleReparam(0),
        }
        reparam_model = reparam(obs_conditioned_model, config=reparam_config)

        nuts_kernel = npNUTS(reparam_model, target_accept_prob=0.9)
        progress_bar = False
        mcmc = npMCMC(nuts_kernel, num_samples=self.mcmc_samples,
                      num_warmup=self.mcmc_tune, progress_bar=progress_bar, num_chains=num_chains, )
        rng_key = random.PRNGKey(10)
        n_workloads = len(np.unique(workloads))
        mcmc.run(rng_key, X, workloads, n_workloads, y)
        mcmc.print_summary()
        self.mcmc_fitted = mcmc
        self.samples = mcmc.get_samples()

    def conditionable_model2(self, data, workloads):
        workloads = np.array(workloads)
        y_order_of_magnitude = np.mean(self.problem_y)
        joint_coef_stdev = 2 * y_order_of_magnitude
        num_opts = data.shape[1]
        unique_workloads = np.unique(workloads)
        n_workloads = len(unique_workloads)
        hyper_mean_1 = numpyro.sample("hyper_coef_means1", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_2 = numpyro.sample("hyper_coef_means2", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_3 = numpyro.sample("hyper_coef_means3", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_4 = numpyro.sample("hyper_coef_means4", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_5 = numpyro.sample("hyper_coef_means5", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_6 = numpyro.sample("hyper_coef_means6", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_7 = numpyro.sample("hyper_coef_means7", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_8 = numpyro.sample("hyper_coef_means8", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_9 = numpyro.sample("hyper_coef_means9", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_10 = numpyro.sample("hyper_coef_means10", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_11 = numpyro.sample("hyper_coef_means11", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_12 = numpyro.sample("hyper_coef_means12", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_13 = numpyro.sample("hyper_coef_means13", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_14 = numpyro.sample("hyper_coef_means14", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_15 = numpyro.sample("hyper_coef_means15", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_16 = numpyro.sample("hyper_coef_means16", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_17 = numpyro.sample("hyper_coef_means17", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_18 = numpyro.sample("hyper_coef_means18", npdist.Normal(0, joint_coef_stdev), )
        hyper_mean_19 = numpyro.sample("hyper_coef_means19", npdist.Normal(0, joint_coef_stdev), )

        hyper_stddev_1 = numpyro.sample("hyper_coef_stddevs1", npdist.Exponential(0.1), )
        hyper_stddev_2 = numpyro.sample("hyper_coef_stddevs2", npdist.Exponential(0.1), )
        hyper_stddev_3 = numpyro.sample("hyper_coef_stddevs3", npdist.Exponential(0.1), )
        hyper_stddev_4 = numpyro.sample("hyper_coef_stddevs4", npdist.Exponential(0.1), )
        hyper_stddev_5 = numpyro.sample("hyper_coef_stddevs5", npdist.Exponential(0.1), )
        hyper_stddev_6 = numpyro.sample("hyper_coef_stddevs6", npdist.Exponential(0.1), )
        hyper_stddev_7 = numpyro.sample("hyper_coef_stddevs7", npdist.Exponential(0.1), )
        hyper_stddev_8 = numpyro.sample("hyper_coef_stddevs8", npdist.Exponential(0.1), )
        hyper_stddev_9 = numpyro.sample("hyper_coef_stddevs9", npdist.Exponential(0.1), )
        hyper_stddev_10 = numpyro.sample("hyper_coef_stddevs10", npdist.Exponential(0.1), )
        hyper_stddev_11 = numpyro.sample("hyper_coef_stddevs11", npdist.Exponential(0.1), )
        hyper_stddev_12 = numpyro.sample("hyper_coef_stddevs12", npdist.Exponential(0.1), )
        hyper_stddev_13 = numpyro.sample("hyper_coef_stddevs13", npdist.Exponential(0.1), )
        hyper_stddev_14 = numpyro.sample("hyper_coef_stddevs14", npdist.Exponential(0.1), )
        hyper_stddev_15 = numpyro.sample("hyper_coef_stddevs15", npdist.Exponential(0.1), )
        hyper_stddev_16 = numpyro.sample("hyper_coef_stddevs16", npdist.Exponential(0.1), )
        hyper_stddev_17 = numpyro.sample("hyper_coef_stddevs17", npdist.Exponential(0.1), )
        hyper_stddev_18 = numpyro.sample("hyper_coef_stddevs18", npdist.Exponential(0.1), )
        hyper_stddev_19 = numpyro.sample("hyper_coef_stddevs19", npdist.Exponential(0.1), )

        hyper_base_mean = numpyro.sample("hyper_base_mean", npdist.Normal(0, joint_coef_stdev), )
        hyper_base_stddev = numpyro.sample("hyper_base_stddev", npdist.Exponential(0.1), )

        with numpyro.plate("workload_plate_coefs", n_workloads):
            rnd_influences1 = hyper_mean_1 + numpyro.sample("coefs1", npdist.Normal(0, hyper_stddev_1), )
            rnd_influences2 = hyper_mean_2 + numpyro.sample("coefs2", npdist.Normal(0, hyper_stddev_2), )
            rnd_influences3 = hyper_mean_3 + numpyro.sample("coefs3", npdist.Normal(0, hyper_stddev_3), )
            rnd_influences4 = hyper_mean_4 + numpyro.sample("coefs4", npdist.Normal(0, hyper_stddev_4), )
            rnd_influences5 = hyper_mean_5 + numpyro.sample("coefs5", npdist.Normal(0, hyper_stddev_5), )
            rnd_influences6 = hyper_mean_6 + numpyro.sample("coefs6", npdist.Normal(0, hyper_stddev_6), )
            rnd_influences7 = hyper_mean_7 + numpyro.sample("coefs7", npdist.Normal(0, hyper_stddev_7), )
            rnd_influences8 = hyper_mean_8 + numpyro.sample("coefs8", npdist.Normal(0, hyper_stddev_8), )
            rnd_influences9 = hyper_mean_9 + numpyro.sample("coefs9", npdist.Normal(0, hyper_stddev_9), )
            rnd_influences10 = hyper_mean_10 + numpyro.sample("coefs10", npdist.Normal(0, hyper_stddev_10), )
            rnd_influences11 = hyper_mean_11 + numpyro.sample("coefs11", npdist.Normal(0, hyper_stddev_11), )
            rnd_influences12 = hyper_mean_12 + numpyro.sample("coefs12", npdist.Normal(0, hyper_stddev_12), )
            rnd_influences13 = hyper_mean_13 + numpyro.sample("coefs13", npdist.Normal(0, hyper_stddev_13), )
            rnd_influences14 = hyper_mean_14 + numpyro.sample("coefs14", npdist.Normal(0, hyper_stddev_14), )
            rnd_influences15 = hyper_mean_15 + numpyro.sample("coefs15", npdist.Normal(0, hyper_stddev_15), )
            rnd_influences16 = hyper_mean_16 + numpyro.sample("coefs16", npdist.Normal(0, hyper_stddev_16), )
            rnd_influences17 = hyper_mean_17 + numpyro.sample("coefs17", npdist.Normal(0, hyper_stddev_17), )
            rnd_influences18 = hyper_mean_18 + numpyro.sample("coefs18", npdist.Normal(0, hyper_stddev_18), )
            rnd_influences19 = hyper_mean_19 + numpyro.sample("coefs19", npdist.Normal(0, hyper_stddev_19), )

        with numpyro.plate("workload_plate_bases", n_workloads):
            bases = hyper_base_mean + numpyro.sample("base", npdist.Normal(0, hyper_base_stddev))

        respective_influence1 = rnd_influences1[workloads]
        respective_influence2 = rnd_influences2[workloads]
        respective_influence3 = rnd_influences3[workloads]
        respective_influence4 = rnd_influences4[workloads]
        respective_influence5 = rnd_influences5[workloads]
        respective_influence6 = rnd_influences6[workloads]
        respective_influence7 = rnd_influences7[workloads]
        respective_influence8 = rnd_influences8[workloads]
        respective_influence9 = rnd_influences9[workloads]
        respective_influence10 = rnd_influences10[workloads]
        respective_influence11 = rnd_influences11[workloads]
        respective_influence12 = rnd_influences12[workloads]
        respective_influence13 = rnd_influences13[workloads]
        respective_influence14 = rnd_influences14[workloads]
        respective_influence15 = rnd_influences15[workloads]
        respective_influence16 = rnd_influences16[workloads]
        respective_influence17 = rnd_influences17[workloads]
        respective_influence18 = rnd_influences18[workloads]
        respective_influence19 = rnd_influences19[workloads]
        respective_bases = bases[workloads]
        result_arr = respective_bases + data[:, 0] * respective_influence1 + data[:, 1] * respective_influence2 + \
                     data[:, 2] * respective_influence3 + data[:, 3] * respective_influence4 + \
                     data[:, 4] * respective_influence5 + data[:, 5] * respective_influence6 + \
                     data[:, 6] * respective_influence7 + data[:, 7] * respective_influence8 + \
                     data[:, 8] * respective_influence9 + data[:, 9] * respective_influence10 + \
                     data[:, 10] * respective_influence11 + data[:, 11] * respective_influence12 + \
                     data[:, 12] * respective_influence13 + data[:, 13] * respective_influence14 + \
                     data[:, 14] * respective_influence15 + data[:, 15] * respective_influence16 + \
                     data[:, 16] * respective_influence17 + data[:, 17] * respective_influence18 + \
                     data[:, 18] * respective_influence19
        error_var = numpyro.sample("error", npdist.Exponential(.10))
        with numpyro.plate("data_vectorized", result_arr.shape[0]) as ind:
            obs = numpyro.sample("measurements", npdist.Normal(result_arr, error_var))

    def conditionable_model(self, data, workloads, n_workloads, reference_y):
        workloads = jnp.array(workloads)
        y_order_of_magnitude = jnp.mean(reference_y)
        joint_coef_stdev = 0.5 * y_order_of_magnitude
        num_opts = data.shape[1]

        exponential_prior = 1.0
        with numpyro.plate("hypers_vectorized", num_opts):
            hyper_coef_means = numpyro.sample("hyper_coef_means", npdist.Normal(0, joint_coef_stdev), )
            hyper_coef_stddevs = numpyro.sample("hyper_coef_stddevs", npdist.Exponential(exponential_prior), )

        hyper_base_mean = numpyro.sample("hyper_base_mean", npdist.Normal(0, joint_coef_stdev), )
        hyper_base_stddev = numpyro.sample("hyper_base_stddev", npdist.Exponential(exponential_prior), )

        with numpyro.plate("coefs_vectorized", num_opts):
            with numpyro.plate("workload_plate_coefs", n_workloads):
                rnd_influences = numpyro.sample("coefs", npdist.Normal(hyper_coef_means, hyper_coef_stddevs), )

        with numpyro.plate("workload_plate_bases", n_workloads):
            bases = numpyro.sample("base", npdist.Normal(hyper_base_mean, hyper_base_stddev))

        respective_influences = rnd_influences[workloads]
        respective_bases = bases[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel() + respective_bases
        error_var = numpyro.sample("error", npdist.Exponential(exponential_prior))

        with numpyro.plate("data_vectorized", result_arr.shape[0]):
            obs = numpyro.sample("measurements", npdist.Normal(result_arr, error_var), obs=reference_y)
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
    return mape


def update_metric(st_empty, title, my_mape, ref_mape=None, ):
    st_empty.empty()
    st_empty.write(f"{title}")
    mape_rounded = round(my_mape, 1)
    diff_kw = {}
    if ref_mape:
        diff = round(my_mape - ref_mape)
        diff_kw = {
            "delta": f"{diff} p.p.",
            "delta_color": "inverse"
        }
    st_empty.metric(f"{title}", f"{mape_rounded} %", **diff_kw)


def hash_pandas(df):
    hash(tuple(df.columns))


@st.cache
def get_xz_df(cleared_sys_df, cluster_partition=5):
    uiq_df = cleared_sys_df.loc[cleared_sys_df["workload"].str.contains("uiq")]
    # uiq_df["workload-scale"] = None
    uiq_df.loc[uiq_df["workload"] == "uiq2-4.bin", "workload-scale"] = 4
    uiq_df.loc[uiq_df["workload"] == "uiq2-16.bin", "workload-scale"] = 16
    uiq_df.loc[uiq_df["workload"] == "uiq-32.bin", "workload-scale"] = 32
    uiq_df.loc[uiq_df["workload"] == "uiq2-8.bin", "workload-scale"] = 8
    uiq_df["workload-name"] = "uiq2"
    uiq_df["workload"] = "uiq2"
    # changing column order to *OPTIONS, workload, workload-scale, *NFPS
    cols = list(uiq_df.columns)
    config_col_idx = cols.index("config_id")
    nfp_cols = cols[:config_col_idx]
    nfp_cols.append("ratio")
    wl_cols = {"workload", "workload-scale"}
    non_nfp_cols = set(cols) - set(nfp_cols) - wl_cols
    uiq_df = uiq_df[[*non_nfp_cols, *wl_cols, *nfp_cols]]
    uiq_df = uiq_df[uiq_df["partition"] == cluster_partition]
    uiq_df = uiq_df.drop(columns=["config_id", "partition"], errors='ignore')
    return uiq_df


@st.cache
def get_jump3r_df(cleared_sys_df):
    mono_stereo_df = cleared_sys_df[cleared_sys_df["workload"].isin(["dual-channel.wav", "single-channel.wav"])]
    mono_stereo_df["workload-scale"] = mono_stereo_df["workload"] == "dual-channel.wav"
    mono_stereo_df["workload-name"] = "mono-stereo"
    cleared_sys_df = mono_stereo_df
    nfps = ["time", "max-resident-size"]
    all_cols = cleared_sys_df.columns
    middle_cols = ["workload", "workload-scale", "workload-name", "config"]
    options = set(all_cols) - {*nfps, *middle_cols}
    cleared_sys_df = cleared_sys_df[[*options, *middle_cols, *nfps]]
    # changing column order to *OPTIONS, workload, workload-scale, *NFPS
    return cleared_sys_df


@st.cache
def get_jump3r_df_keep_workloads(cleared_sys_df):
    cleared_sys_df = copy.deepcopy(cleared_sys_df)
    nfps = ["time", "max-resident-size"]
    all_cols = cleared_sys_df.columns
    middle_cols = ["workload", "config"]
    options = set(all_cols) - {*nfps, *middle_cols}
    cleared_sys_df = cleared_sys_df[[*options, *middle_cols, *nfps]]
    # changing column order to *OPTIONS, workload, workload-scale, *NFPS
    return cleared_sys_df


@st.cache
def remove_multicollinearity(df):
    return util.remove_multicollinearity(df)


def main():
    no_streamlit = len(sys.argv) > 1
    st.sidebar.write("# 🗄️ Training Data")
    sub_folder_name = "training-data/"

    dirname = os.path.dirname(__file__)
    data_path_parent = os.path.join(dirname, sub_folder_name)
    avail_sws = [s.replace(".csv", "") for s in list(os.listdir(data_path_parent))]
    only_show_working_experiments = st.sidebar.checkbox("Only show (mostly) working experiments", value=True)
    supported_experiments = {
        "h2": None,
        "jump3r": None,
        "measurements_xz-5.2.0": None
    }
    if only_show_working_experiments:
        avail_sws = list(supported_experiments)
    sws = st.sidebar.selectbox("Software System", avail_sws, index=1)

    data_path = os.path.join(data_path_parent, f"{sws}.csv")
    sys_df = pd.read_csv(data_path)

    df_no_multicollinearity = remove_multicollinearity(sys_df)
    cleared_sys_df = copy.deepcopy(df_no_multicollinearity)
    # wl_filter = "tpcc"
    if sws == "h2":
        cleared_sys_df[["workload-name", "workload-scale"]] = cleared_sys_df["workload"].str.split("-", expand=True)
    elif sws == "jump3r":
        cleared_sys_df = get_jump3r_df(df_no_multicollinearity)
        all_workloads_df = get_jump3r_df_keep_workloads(df_no_multicollinearity)
    elif sws == "measurements_xz-5.2.0":
        cleared_sys_df = get_xz_df(cleared_sys_df)

    # read workloads and NFPs
    cols = all_workloads_df.columns
    workload_idx = list(cols).index("workload")
    nfp_candidates = cols[workload_idx + 1:]
    typical_nfsp = ["throughput", "time", "max-resident-size", 'kernel-time', 'user-time', 'max-resident-set-size',
                    'ratio']
    nfps = [nfp for nfp in nfp_candidates if nfp in typical_nfsp]
    nfp = st.sidebar.selectbox("NFP", nfps, index=0)

    train_size_fraq = st.sidebar.slider("Training set size ratio", 0.05, 0.95, 0.02, step=0.05)
    print(all_workloads_df.head(20))
    data_df = all_workloads_df.drop(columns=["config", ], errors='ignore')
    data_df["workload"] = data_df["workload"].astype('category')
    data_df["workload"] = data_df["workload"].cat.codes
    workloads = data_df["workload"].unique()

    train_df, test_df = sklearn.model_selection.train_test_split(data_df, train_size=train_size_fraq)
    y_train = np.atleast_2d(train_df[nfp].to_numpy()).T
    y_test = np.atleast_2d(test_df[nfp].to_numpy()).T
    train_df_without_nfp = train_df.drop(columns=nfps)
    # x_train = train_df_without_nfp.astype(int).to_numpy()
    # x_test = test_df.drop(columns=nfps).astype(int).to_numpy()
    x_train = train_df_without_nfp.to_numpy()
    x_test = test_df.drop(columns=nfps).to_numpy()
    workloads_train = np.array(list(x_train[:, -1]))
    workloads_test = np.array(list(x_test[:, -1]))
    x_scaler = MinMaxScaler()
    y_scaler = StandardScaler()
    x_train = jnp.atleast_2d(x_scaler.fit_transform(x_train[:, :-1]))
    x_test = jnp.atleast_2d(x_scaler.transform(x_test[:, :-1]))
    y_train = jnp.atleast_2d(y_scaler.fit_transform(y_train))
    y_test = jnp.atleast_2d(y_scaler.transform(y_test))
    
    ## Models to fit
    st.sidebar.write("# Models")
    models = ["📊 Hyper MCMC", "📊 WL-Agnostic MCMC", "📊 Linear MCMC", "📊 RelWL MCMC Model", "Baselines"]
    chosen_models = st.sidebar.multiselect("MCMC Models (expensive)", options=models, default=models[:-1])

    ## MCMC samples
    st.sidebar.write("# Training Parameters")
    mcmc_tune = st.sidebar.slider("Tuning MCMC samples", 250, 4000, value=250, step=250)
    mcmc_samples = st.sidebar.slider("Posterior MCMC samples", 250, 4000, value=250, step=250)
    num_chains = st.sidebar.slider("Parallel MCMC Chains", 1, 6, value=3, step=1)

    ## Scaling
    # scaler = MinMaxScaler()
    # x_train[:, -1] = scaler.fit_transform(np.atleast_2d(x_train[:, -1]).T)[:, -1].ravel()
    # x_test[:, -1] = scaler.transform(np.atleast_2d(x_test[:, -1]).T).ravel()
    # we keep the raw workload names!

    x_train_no_wl = x_train[:, :-1]
    x_test_no_wl = x_test[:, :-1]
    features = list(list(train_df_without_nfp.columns))
    n_workloads = len(workloads)

    st.write("## Partially Pooled Hyper MCMC")
    st.write(
        "This model gets the workload feature. It learns an influence for each option and each workload. "
        "In addition, it learns a hyper prior aka hyperior of option influences across workloads. "
        "This way, we reduce overfitting as we reduce the effective number of parameters compared to "
        "a workload-aware model without hyper priors.")
    with st.spinner("Fitting MCMC"):
        mcmc_reg = HierarchicalWorkloadRegressor(y_train, mcmc_samples=mcmc_samples, mcmc_tune=mcmc_tune)
        mcmc_reg.fit(x_train, workloads_train, y_train, num_chains)
    mape_mcmc_agnostic = plot_mcmc_scores(features[:-1], mcmc_reg, num_chains, x_test, workloads_test,
                                          y_test)
    # update_metric(metric_no_wl_MCMC, "📊 MCMC Agn", mape_mcmc_agnostic, mape_lin_reg_no_workload_ft)


def plot_mcmc_scores(features, mcmc_reg, num_chains, x_test, workloads, y_test):
    graph = numpyro.render_model(mcmc_reg.conditionable_model, model_args=(x_test, workloads),
                                 # filename="model.pdf",
                                 render_params=True, render_distributions=True)
    sys_id = hash("-".join(features))
    dir_path = "./results"
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path, )
    st.graphviz_chart(graph)
    unique_workloads = np.unique(workloads)
    with st.spinner("Plotting"):
        coords = {
            "features": features, "workloads": unique_workloads}
        dims = {"coefs": ["workloads", "features"], "base": ["workloads"]}
        idata_kwargs = {
            "dims": dims,
            "coords": coords,
        }
        az_data = az.from_numpyro(mcmc_reg.mcmc_fitted, num_chains=num_chains, **idata_kwargs)
        az.plot_trace(az_data, )  # compact=True, var_names=("base", "coefs"))
        fig = plt.gcf()
        plt.tight_layout()
        plt.savefig(os.path.join(dir_path, f"trace-sys-{sys_id}.pdf"))
        st.pyplot(fig)

        # az.plot_posterior(az_data, hdi_prob=0.95, point_estimate="mode",multimodal=True, )
        # fig = plt.gcf()
        # plt.tight_layout()
        # st.pyplot(fig)

        az.plot_forest(az_data)
        fig = plt.gcf()
        plt.tight_layout()
        plt.savefig(os.path.join(dir_path, f"forest-sys-{sys_id}.pdf"))
        st.pyplot(fig)
    with st.spinner("Scoring"):
        divergences = mcmc_reg.mcmc_fitted.get_extra_fields()["diverging"].sum()
        if divergences:
            st.error(
                f"Encountered {divergences} divergences! This means that the sampling did not find the posterior reliably and we should consider a different model structure.")
        return st_create_scores(mcmc_reg, x_test, y_test)


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
