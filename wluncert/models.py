import sys
import time
from typing import List

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from abc import ABC, abstractmethod
import sklearn
import pprint
import seaborn as sns
import jax.numpy as jnp
import numpyro.distributions as npdist
from numpyro.infer import HMCECS as npHMCECS, MCMC as npMCMC, NUTS as npNUTS, HMC as npHMC, BarkerMH, \
    Predictive as npPredictive
from jax import random, jit
from numpyro.handlers import condition as npcondition, seed as npseed, substitute as npsubstitute, trace as nptrace
import arviz as az
import numpyro

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import math
import os

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV, Lasso, Ridge, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures

import numpy as np
import pandas as pd

from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

from wluncert.data import SingleEnvData

numpyro.set_host_device_count(4)
import copy


class NumPyroRegressor(ABC, BaseEstimator, RegressorMixin):
    def __init__(self):
        self.coef_ = None
        self.samples = None
        self.mcmc = None

    @staticmethod
    def conditionable_model(data):
        pass

    def condition(self, y):
        return npcondition(self.conditionable_model, data={"measurements": y})

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
    def predict(self, X, env_ids, n_samples: int = 1000, ci: float = None, yield_samples=True):
        y_pred_samples = self._predict_samples(X, env_ids, n_samples)
        y_pred_samples = np.array(y_pred_samples)
        if not ci:
            if not yield_samples:
                return self.get_mode_from_samples(y_pred_samples)
            else:
                return y_pred_samples
        else:
            ci_s = self.get_ci_from_samples(y_pred_samples, ci)
            if not yield_samples:
                return ci_s
            else:
                return ci_s, y_pred_samples

    def _predict_scalar(self, X, n_samples):
        pass

    def _predict_samples(self, X, env_ids, n_samples: int = 1000):
        pred = npPredictive(self.conditionable_model, num_samples=n_samples, parallel=True)
        x = jnp.atleast_2d(X)
        y_pred = pred(x, env_ids, None)["measurements"]
        return y_pred

    def get_ci_from_samples(self, samples, ci):
        ci_s = az.hdi(samples, hdi_prob=ci)
        return ci_s

    def get_mode_from_samples(self, samples):
        hdi = az.hdi(samples, hdi_prob=0.01)
        modes = np.mean(hdi, axis=1)
        return modes


class ExtraStandardizingEnvAgnosticModel(NumPyroRegressor):
    def coef_ci(self, ci: float):
        pass

    def get_tuples(self, feature_names):
        pass

    def fit(self, X, y):
        # reparam_config = {
        #     "influences": LocScaleReparam(0),
        #     "base": LocScaleReparam(0),
        # }
        X = jnp.atleast_2d(np.array(X.astype(float)))
        y = jnp.array(y)
        # reparam_model = reparam(self.model, config=reparam_config)
        reparam_model = self.model
        nuts_kernel = npNUTS(reparam_model)  # , target_accept_prob=0.9)
        n_chains = 3
        progress_bar = True
        self.mcmc = npMCMC(nuts_kernel, num_samples=1000,
                           num_warmup=1500, progress_bar=progress_bar,
                           num_chains=n_chains, chain_method="parallel")
        rng_key_ = random.PRNGKey(0)
        rng_key_, rng_key = random.split(rng_key_)
        self.mcmc.run(rng_key, X, y)
        self.save_plot()
        return self.mcmc

    def save_plot(self):
        az.plot_trace(az.from_numpyro(self.mcmc))
        plt.tight_layout()
        plt.savefig("./tmp/mcmc-agnostic.png")

    def _predict_samples(self, X, n_samples: int = 1_000):
        posterior_samples = self.mcmc.get_samples()
        pred = npPredictive(self.model, posterior_samples=posterior_samples, num_samples=n_samples, parallel=True)
        x = jnp.atleast_2d(np.array(X.astype(float)))
        rng_key_ = random.PRNGKey(0)
        y_pred = pred(rng_key_, x, None)["observations"]
        return y_pred

    def predict(self, X, n_samples: int = 1500, ci: float = None, yield_samples=False):
        y_pred_samples = self._predict_samples(X, n_samples)
        y_pred_samples = np.array(y_pred_samples)
        if not ci:
            if not yield_samples:
                return self.get_mode_from_samples(y_pred_samples)
            else:
                return y_pred_samples
        else:
            ci_s = self.get_ci_from_samples(y_pred_samples, ci)
            if not yield_samples:
                return ci_s
            else:
                return ci_s, y_pred_samples

    def model(self, data, reference_y):
        joint_coef_stdev = 0.25  # 0.25  # 2 * y_order_of_magnitude
        num_opts = data.shape[1]

        with numpyro.plate("options", num_opts):
            rnd_influences = numpyro.sample("influences", npdist.Laplace(0, joint_coef_stdev), )

        base = numpyro.sample("base", npdist.Laplace(0, joint_coef_stdev))
        result_arr = jnp.multiply(data, rnd_influences)
        result_arr = result_arr.sum(axis=1).ravel() + base
        error_var = numpyro.sample("error", npdist.Exponential(.01))

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
        return obs


class ExtraStandardizingSimpleModel(NumPyroRegressor):
    def coef_ci(self, ci: float):
        pass

    def get_tuples(self, feature_names):
        pass

    def fit(self, data: List[SingleEnvData]):
        X, envs, y = self.X_envids_y_from_data(data)
        n_envs = len(envs)
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
        }
        reparam_model = reparam(self.model, config=reparam_config)

        nuts_kernel = npNUTS(reparam_model)
        n_chains = 3
        progress_bar = True
        self.mcmc = npMCMC(nuts_kernel, num_samples=1000,
                           num_warmup=1500, progress_bar=progress_bar,
                           num_chains=n_chains, )
        rng_key_ = random.PRNGKey(0)
        rng_key_, rng_key = random.split(rng_key_)
        # mcmc.run(rng_key, X_agg[:, 0], X_agg[:, 1], X_agg[:, 2], given_obs=nfp_mean_agg, obs_stddev=nfp_stddev_agg)
        self.mcmc.run(rng_key, X, envs, n_envs, y)
        print("finished fitting multilevel model")
        print("plotting result")
        self.save_plot()
        print("finished plotting")
        return self.mcmc

    def X_envids_y_from_data(self, data):
        X_list = []
        y_list = []
        env_id_list = []
        for single_env_data in data:
            normalized_data = single_env_data.normalize()
            env_id = normalized_data.env_id
            X = normalized_data.get_X()
            y = normalized_data.get_y()
            data_len = len(y)
            new_env_id_sublist = [env_id] * data_len
            X_list.append(X)
            y_list.append(y)
            env_id_list.append(new_env_id_sublist)
        X_df = pd.concat(X_list)
        X_df = X_df.astype(float)
        X = jnp.array(X_df)
        y = jnp.concatenate(y_list)
        envs = jnp.array(env_id_list).ravel()
        return X, envs, y

    def predict(self, data: List[SingleEnvData]):
        preds = []

        X, envs, y = self.X_envids_y_from_data(data)
        for single_env_data in data:
            env_id = single_env_data.env_id
            reg, train_data = self.get_env_model(env_id)
            X = single_env_data.get_X()
            pred = reg.predict(X)
            un_normalized_pred = train_data.un_normalize(pred)
            preds.append(un_normalized_pred)
        return preds

    def model(self, data, workloads, n_workloads, reference_y):
        workloads = jnp.array(workloads)
        y_order_of_magnitude = jnp.std(reference_y)
        joint_coef_stdev = 0.25  # 2 * y_order_of_magnitude
        num_opts = data.shape[1]
        stddev_exp_prior = 0.250
        with numpyro.plate("options", num_opts):
            hyper_coef_means = numpyro.sample("means-hyper", npdist.Laplace(0, joint_coef_stdev), )
            hyper_coef_stddevs = numpyro.sample("stddevs-hyper", npdist.Exponential(stddev_exp_prior), )

        hyper_base_mean = numpyro.sample("base mean hyperior", npdist.Laplace(0, joint_coef_stdev), )
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
        error_var = numpyro.sample("error", npdist.Exponential(.01))

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
        return obs

    def save_plot(self):
        az.plot_trace(az.from_numpyro(self.mcmc))
        plt.tight_layout()
        plt.savefig("./tmp/mcmc-multilevel.png")


class NoPoolingEnvModel:
    def __init__(self, model_prototype: RegressorMixin):
        self.model_prototype = model_prototype
        self.existing_models = {}

    def get_env_model(self, env_id, train_data=None):
        if env_id not in self.existing_models:
            new_model = copy.deepcopy(self.model_prototype)
            self.existing_models[env_id] = new_model, train_data

        return self.existing_models[env_id]

    def fit(self, data: List[SingleEnvData]):
        for single_env_data in data:
            normalized_data = single_env_data.normalize()
            env_id = normalized_data.env_id
            reg, _ = self.get_env_model(env_id, normalized_data)
            X = normalized_data.get_X()
            y = normalized_data.get_y()
            reg.fit(X, y)

    def predict(self, data: List[SingleEnvData]):
        preds = []
        for single_env_data in data:
            env_id = single_env_data.env_id
            reg, train_data = self.get_env_model(env_id)
            normalized_predict_data = single_env_data.normalize(train_data)
            X = normalized_predict_data.get_X()
            pred = reg.predict(X)
            un_normalized_pred = train_data.un_normalize_y(pred)
            preds.append(un_normalized_pred)
        return preds


def get_pairwise_lasso_reg(lasso_alpha=0.0001):
    pairwise_mapper = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    lin_reg = Lasso(alpha=lasso_alpha, max_iter=5000)
    # lin_reg = LinearRegression()
    # feature_selector = SelectFromModel(lin_reg, threshold=None)

    pipeline = Pipeline([
        ('pairwise_mapper', pairwise_mapper),
        # ('feature_selector', feature_selector),
        ('lin_reg', lin_reg)
    ])

    return pipeline
