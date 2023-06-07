import sys
import time
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from abc import ABC, abstractmethod
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

numpyro.set_host_device_count(8)
import copy


class NumPyroRegressor(ABC, BaseEstimator, RegressorMixin):
    def __init__(self):
        self.coef_ = None
        self.samples = None

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
    def predict(self, X, n_samples: int = 1000, ci: float = None, yield_samples=True):
        y_pred_samples = self._predict_samples(X, n_samples)
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

    def _predict_samples(self, X, n_samples: int = 1000):
        pred = npPredictive(self.conditionable_model, num_samples=n_samples, parallel=True)
        x = jnp.atleast_2d(X)
        y_pred = pred(x, None)["measurements"]
        return y_pred

    def get_ci_from_samples(self, samples, ci):
        ci_s = az.hdi(samples, hdi_prob=ci)
        return ci_s

    def get_mode_from_samples(self, samples):
        modes = np.mean(az.hdi(samples, hdi_prob=0.01), axis=1)
        return modes


class UnstandardizingSimpleModel(NumPyroRegressor):
    def model(data, workloads, n_workloads, reference_y):
        workloads = jnp.array(workloads)
        y_order_of_magnitude = jnp.std(reference_y)
        joint_coef_stdev = 0.25  # 2 * y_order_of_magnitude
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
