import time

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

import os
from sklearn.ensemble import RandomForestRegressor

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


class PyroMCMCStructureRegressor(PyroMCMCRegressor):
    """
    Class to do MCMC with struture finding without categorical variables
    """

    def fit(self, X, y):
        # y = jnp.atleast_1d(y.cpu().detach().numpy())
        y = jnp.atleast_1d(y)
        # X = jnp.atleast_2d(X.cpu().detach().numpy())
        X = jnp.atleast_2d(X)
        obs_conditioned_model = self.condition(y)
        nuts_kernel = npNUTS(obs_conditioned_model, adapt_step_size=True,
                             dense_mass=False, find_heuristic_step_size=False)
        # berker_kernel = BarkerMH(obs_conditioned_model)
        # hmc_kernel = npHMC(obs_conditioned_model, adapt_step_size=True)
        # hmces_kernel = npHMCECS(nuts_kernel,)
        progress_bar = True
        mcmc = npMCMC(nuts_kernel, num_samples=self.mcmc_samples,
                      num_warmup=self.mcmc_tune, progress_bar=progress_bar)
        rng_key = random.PRNGKey(42)
        mcmc.run(rng_key, X)
        mcmc.print_summary()
        self.samples = mcmc.get_samples()

    @staticmethod
    def conditionable_model(data):
        num_opts = data.shape[1]
        with numpyro.plate("coefs_vectorized", num_opts):
            rnd_influences = numpyro.sample("coefs", npdist.Cauchy(0.0, 10.0), )
        mat_infl = rnd_influences.reshape(-1, 1)
        product = jnp.matmul(data, mat_infl).reshape(-1)
        base = numpyro.sample("base", npdist.Gamma(1.10, 30.0))
        result = product + base
        error_var = numpyro.sample("error", npdist.Gamma(1.0, 0.9))
        relative_error_var = numpyro.sample("error_rel", npdist.Gamma(1.0, 0.01))
        result = result + (relative_error_var * result)
        # error_var = numpyro.sample("error", npdist.HalfNormal(0.01) )
        with numpyro.plate("data_vectorized", len(data)) as ind:
            obs = numpyro.sample("measurements", npdist.Normal(result, error_var))
            return obs

    def condition(self, y):
        return npcondition(self.conditionable_model, data={"measurements": y})

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


def main():
    data_path = "/home/jdorn/code/workload-performance/resources/performance/h2.csv"
    sys_df = pd.read_csv(data_path)
    cleared_sys_df = util.remove_multicollinearity(sys_df)
    wl_filter = "tpcc"
    cleared_sys_df[["workload-name", "workload-scale"]] = cleared_sys_df["workload"].str.split("-", expand=True)
    tpcc_df = cleared_sys_df[cleared_sys_df["workload-name"].str.contains(wl_filter)]
    print(tpcc_df.head())
    tpcc_df = tpcc_df.drop(columns=["workload", "workload-name", "config"])
    nfp = "throughput"
    train_df, test_df = sklearn.model_selection.train_test_split(tpcc_df, train_size=0.5)
    y_train = train_df[nfp].to_numpy()
    y_test = test_df[nfp].to_numpy()
    x_train = train_df.drop(columns=[nfp]).astype(float).to_numpy()
    x_test = test_df.drop(columns=[nfp]).astype(float).to_numpy()
    reg = PyroMCMCStructureRegressor()
    reg.fit(x_train, y_train)
    reg.predict(x_test)
    score = reg.score(x_test, y_test)
    print(score)


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
