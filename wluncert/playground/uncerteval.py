import copy
import itertools
import seaborn as sns
import numpyro
import numpyro.distributions as npdist

import arviz as az
import jax.numpy as jnp
import sklearn.preprocessing

from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam
import numpy
from numpyro.infer import Predictive, MCMC as npMCMC, NUTS as npNUTS, BarkerMH as npBMH, HMC as npHMC, SA, \
    log_likelihood
import os
from jax import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpyro.infer.reparam import LocScaleReparam
from pycosa.util import remove_multicollinearity
from sklearn.model_selection import train_test_split
from abc import ABC
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
# from wluncert.multilevel import get_jump3r_df
# from wluncert.wlbench import get_train_df, wl_model_reparam
from multilevel import get_jump3r_df
from wlbench import get_train_df, wl_model_reparam
import pickle
import pprint

numpyro.set_host_device_count(3)


class DataProvider(ABC):
    def __init__(self):
        self.train_data = None
        self.test_data = None

    def get_train_test_data(self):
        return self.train_data, self.test_data

    def get_train_data(self):
        train, test = self.get_train_test_data()
        return test

    def get_test_data(self):
        train, test = self.get_train_test_data()
        return test

    def get_label(self):
        pass

    def plot_train_data(self):
        df = self.get_train_data()
        self.show_nfp_distribution_for_df(df)

    def plot_test_data(self):
        df = self.get_test_data()
        self.show_nfp_distribution_for_df(df)

    def show_nfp_distribution_for_df(self, df):
        cols = list(df.columns[:])
        nfp_name = cols[-1]
        wl_name = cols[-2]
        sns.displot(data=df.set_index(cols), hue=wl_name, x=nfp_name, kind="kde")
        plt.show()


class DfDataProvider(DataProvider):
    def __init__(self, label: str, df: pd.DataFrame, train_ratio=None, train_number_abs=None, rnd=0,
                 stratify_by_wl=False):
        super(DfDataProvider, self).__init__()
        self.len_whole_pop = len(df)
        self.df: pd.DataFrame = df
        self.stratify_by_wl = stratify_by_wl
        self.train_ratio = train_ratio
        self.train_number_abs = train_number_abs
        self.rnd = rnd
        self.label = label
        self.input_df = df
        self.train_ratio = train_ratio
        self.train_number_abs = train_number_abs

        df_no_multicollinearity = remove_multicollinearity(self.df.iloc[:, :-2])
        cleared_sys_df = copy.deepcopy(df_no_multicollinearity)
        cleared_sys_df[self.df.columns[-2:]] = self.df.iloc[:, -2:]
        self.df = cleared_sys_df

    def get_train_test_data(self):
        if self.train_data is None:
            if not self.train_ratio and not self.train_number_abs:
                raise Exception("Must specify either a ratio between 0 and 1 or an absolute number")
            if self.train_ratio is not None:
                if self.train_ratio < 0 or self.train_ratio > 1:
                    raise Exception("ratio must be between 0 and 1 ")
            if self.train_number_abs is not None:
                if self.train_number_abs > self.len_whole_pop or self.train_number_abs < 0:
                    raise Exception(
                        f"Abs train samples number must be > 0 an smaller than size of provided df, which is {self.len_whole_pop}")

            if self.stratify_by_wl:
                stratify = self.df.iloc[:, -2]
            else:
                stratify = None
            if self.train_ratio:
                self.train_data, self.test_data = train_test_split(self.df, train_size=self.train_ratio,
                                                                   random_state=self.rnd, stratify=stratify)
            else:
                self.train_data, self.test_data = train_test_split(self.df, train_size=self.train_number_abs,
                                                                   random_state=self.rnd, stratify=stratify)
        return self.train_data, self.test_data

    def get_label(self):
        return self.label


class StandardizerByTrainSet(DataProvider):
    def __init__(self, data: DataProvider):
        super().__init__()
        self.data = data
        self.train_data, self.test_data = copy.deepcopy(data.get_train_test_data())

        wl_name = self.train_data.columns[-2]
        nfp_name = self.train_data.columns[-1]
        nfp_grouped_by_wl = self.train_data.iloc[:, -2:].groupby([wl_name])
        self.wl_means = nfp_grouped_by_wl[nfp_name].mean().to_dict()
        self.wl_stds = nfp_grouped_by_wl[nfp_name].std().to_dict()
        # standardized_nfps_by_wl = nfp_grouped_by_wl.transform(lambda x: (x - x.mean()) / x.std())
        # self.train_data.iloc[:, -1] = standardized_nfps_by_wl

        for group_id in self.wl_means:
            g_mean = self.wl_means[group_id]
            g_std = self.wl_stds[group_id]

            self.test_data.loc[self.test_data[wl_name] == group_id, nfp_name] -= g_mean
            self.test_data.loc[self.test_data[wl_name] == group_id, nfp_name] /= g_std

            self.train_data.loc[self.train_data[wl_name] == group_id, nfp_name] -= g_mean
            self.train_data.loc[self.train_data[wl_name] == group_id, nfp_name] /= g_std

        self.data = pd.concat([self.train_data, self.test_data])
        self.train_nfp = self.train_data.iloc[:, -1]
        self.test_nfp = self.test_data.iloc[:, -1]

        # train_nfp_scaler = StandardScaler()
        # self.train_data.iloc[:, -1] = train_nfp_scaler.fit_transform(np.atleast_2d(train_nfp.to_numpy()).T).T.squeeze()
        # self.test_data.iloc[:, -1] = train_nfp_scaler.fit_transform(np.atleast_2d(test_nfp.to_numpy()).T).T.squeeze()

        self.train_features = self.train_data.iloc[:, :-2]
        self.test_features = self.test_data.iloc[:, :-2]
        # train_feature_scaler = StandardScaler()
        # self.train_data.iloc[:, :-2] = train_feature_scaler.fit_transform(train_features)
        # self.test_data.iloc[:, :-2] = train_feature_scaler.transform(test_features)

    def get_train_test_data(self):
        return self.train_data, self.test_data

    def get_label(self):
        return self.data.get_label()

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data


class Jump3rDataProvider(DfDataProvider):
    def __init__(self, df, *args, **kwargs):
        self.df = self.get_jump3r_df_keep_workloads(df)
        super(Jump3rDataProvider, self).__init__("Jump3r", self.df, *args, **kwargs)

    def get_jump3r_df(self, df):
        mono_stereo_df = df[df["workload"].isin(["dual-channel.wav", "single-channel.wav"])]
        mono_stereo_df["workload-scale"] = mono_stereo_df["workload"] == "dual-channel.wav"
        mono_stereo_df["workload-name"] = "mono-stereo"
        df = mono_stereo_df
        nfps = ["time", "max-resident-size"]
        all_cols = df.columns
        middle_cols = ["workload", "workload-scale", "workload-name", "config"]
        options = set(all_cols) - {*nfps, *middle_cols}
        df = df[[*options, *middle_cols, *nfps]]
        # changing column order to *OPTIONS, workload, workload-scale, *NFPS
        return df

    def get_jump3r_df_keep_workloads(self, df):
        df = copy.deepcopy(df)
        nfps = ["time", "max-resident-size"]
        all_cols = df.columns
        middle_cols = ["workload", "config"]
        options = set(all_cols) - {*nfps, *middle_cols}
        df = df[[*options, "workload", nfps[0]]]
        # changing column order to *OPTIONS, workload, workload-scale, *NFPS
        return df


class PyroModel(ABC):
    def __init__(self):
        pass

    def get_model(self):
        pass

    def get_arviz_dims(self):
        pass

    def get_label(self):
        pass


class NoPoolingModel(PyroModel):
    label = "no-pooling"

    def __init__(self, prior_method_call=None):
        super().__init__()
        self.prior_method_call = prior_method_call if prior_method_call is not None else npdist.Normal

    def get_model(self):
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
        }
        reparam_model = reparam(self.model_raw, config=reparam_config)
        return reparam_model

    def model_raw(self, data, workloads, n_workloads, reference_y=None):
        workloads = jnp.array(workloads)
        if reference_y is not None:
            mean_nfp = jnp.mean(reference_y)
            stddev_nfp = jnp.std(reference_y)
        else:
            y_order_of_magnitude = 1
            stddev_nfp = 1
        joint_coef_stdev = stddev_nfp  # 1
        num_opts = data.shape[1]
        stddev_exp_prior = stddev_nfp  # 1

        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample("influences", self.prior_method_call(0, joint_coef_stdev), )

        with numpyro.plate("workloads", n_workloads):
            bases = numpyro.sample("base", self.prior_method_call(0, joint_coef_stdev))

        respective_influences = rnd_influences[workloads]
        respective_bases = bases[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel() + respective_bases
        prior_error_expt = 1.0
        error_var = numpyro.sample("error", npdist.Exponential(prior_error_expt))

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
        return obs

    def get_arviz_dims(self):
        dims = {
            "influences": ["workloads", "features"],
            "influences_decentered": ["workloads", "features"],
            "base": ["workloads"],
            "base_decentered": ["workloads"],
        }
        return dims

    def get_label(self):
        return self.__class__.label


class CompletePoolingModel(PyroModel):
    label = "complete-pooling"

    def __init__(self, prior_method_call=None):
        super().__init__()
        self.prior_method_call = prior_method_call if prior_method_call is not None else npdist.Normal

    def get_model(self):
        return self.model_raw

    def model_raw(self, data, workloads, n_workloads, reference_y=None):
        workloads = jnp.array(workloads)
        if reference_y is not None:
            mean_nfp = jnp.mean(reference_y)
            stddev_nfp = jnp.std(reference_y)
        else:
            y_order_of_magnitude = 1
            stddev_nfp = 1
        joint_coef_stdev = stddev_nfp  # 1
        num_opts = data.shape[1]
        stddev_exp_prior = stddev_nfp  # 1

        with numpyro.plate("options", num_opts):
            rnd_influences = numpyro.sample("influences", self.prior_method_call(0, joint_coef_stdev), )

        base = numpyro.sample("base", self.prior_method_call(0, joint_coef_stdev))
        result_arr = jnp.multiply(data, rnd_influences)
        result_arr = result_arr.sum(axis=1).ravel() + base
        prior_error_expt = 1.0
        error_var = numpyro.sample("error", npdist.Exponential(prior_error_expt))

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
        return obs

    def get_arviz_dims(self):
        dims = {
            "influences": ["features"],
        }
        return dims

    def get_label(self):
        return self.__class__.label


class MultilevelLassoModel(PyroModel):
    label = "Hyperiors-Lasso"

    def get_model(self):
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
            # "abs-means-hyper": LocScaleReparam(0),
        }
        reparam_model = reparam(self.model_raw, config=reparam_config)
        return reparam_model

    def model_raw(self, data, workloads, n_workloads, reference_y=None):
        workloads = jnp.array(workloads)
        if reference_y is not None:
            mean_nfp = jnp.mean(reference_y)
            stddev_nfp = jnp.std(reference_y)
        else:
            y_order_of_magnitude = 1
            stddev_nfp = 1
        joint_coef_stdev = stddev_nfp  # 1
        num_opts = data.shape[1]
        stddev_exp_prior = stddev_nfp  # 1

        with numpyro.plate("options", num_opts):
            hyper_coef_means = numpyro.sample("options-means-hyper",
                                              npdist.Laplace(0, joint_coef_stdev), )
            hyper_coef_stddevs = numpyro.sample("options-stddevs-hyper", npdist.Exponential(stddev_exp_prior), )

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
        prior_error_expt = 1.0
        error_var = numpyro.sample("error", npdist.Exponential(prior_error_expt))

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
        return obs

    def get_arviz_dims(self):
        dims = {
            "influences": ["workloads", "features"],
            "influences_decentered": ["workloads", "features"],
            "base": ["workloads"],
            "base_decentered": ["workloads"],
            "options-means-hyper": ["features"],
            "options-stddevs-hyper": ["features"],
        }
        return dims


class MetaMetaWLModelLasso(PyroModel):
    label = "Meta-Hyperiors-Lasso"

    def get_model(self):
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
            "abs-means-hyper": LocScaleReparam(0),
        }
        reparam_model = reparam(self.model_raw, config=reparam_config)
        return reparam_model

    def model_raw(self, data, workloads, n_workloads, reference_y=None):
        workloads = jnp.array(workloads)
        if reference_y is not None:
            mean_nfp = jnp.mean(reference_y)
            stddev_nfp = jnp.std(reference_y)
        else:
            y_order_of_magnitude = 1
            stddev_nfp = 1
        joint_coef_stdev = stddev_nfp  # 1
        num_opts = data.shape[1]
        stddev_exp_prior = stddev_nfp  # 1

        hyper_option_mean = numpyro.sample("meta-means-hyper", npdist.Exponential(stddev_exp_prior), )

        with numpyro.plate("options", num_opts):
            hyper_coef_means = numpyro.sample("abs-means-hyper",
                                              npdist.Laplace(0, hyper_option_mean), )
            hyper_coef_stddevs = numpyro.sample("abs-stddevs-hyper", npdist.Exponential(stddev_exp_prior), )

        hyper_base_mean = numpyro.sample("base mean hyperior", npdist.Laplace(0, joint_coef_stdev), )
        hyper_base_stddev = numpyro.sample("base stddevs hyperior", npdist.Exponential(stddev_exp_prior), )

        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample("influences", npdist.Normal(hyper_coef_means, hyper_coef_stddevs), )

        with numpyro.plate("workloads", n_workloads):
            bases = numpyro.sample("base", npdist.Laplace(hyper_base_mean, hyper_base_stddev))

        respective_influences = rnd_influences[workloads]
        respective_bases = bases[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel() + respective_bases
        prior_error_expt = 1.0
        error_var = numpyro.sample("error", npdist.Exponential(prior_error_expt))

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
        return obs

    def get_arviz_dims(self):
        dims = {
            "influences": ["workloads", "features"],
            "influences_decentered": ["workloads", "features"],
            "base": ["workloads"],
            "base_decentered": ["workloads"],
            "abs-means-hyper": ["features"],
            "abs-stddevs-hyper": ["features"],
        }
        return dims


class AbsoluteInfMetaWLModelRidge(PyroModel):
    label = "Hyperiors-Ridge"

    def get_model(self):
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
        }
        reparam_model = reparam(self.model_raw, config=reparam_config)
        return reparam_model

    def model_raw(self, data, workloads, n_workloads, reference_y=None):
        workloads = jnp.array(workloads)
        if reference_y is not None:
            mean_nfp = jnp.mean(reference_y)
            stddev_nfp = jnp.std(reference_y)
        else:
            y_order_of_magnitude = 1
            stddev_nfp = 1
        joint_coef_stdev = stddev_nfp  # 1
        num_opts = data.shape[1]
        stddev_exp_prior = stddev_nfp  # 1

        with numpyro.plate("options", num_opts):
            hyper_coef_means = numpyro.sample("abs-means-hyper",
                                              npdist.Normal(0, joint_coef_stdev), )
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
        prior_error_expt = 1.0
        error_var = numpyro.sample("error", npdist.Exponential(prior_error_expt))

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
        return obs

    def get_arviz_dims(self):
        dims = {
            "influences": ["workloads", "features"],
            "influences_decentered": ["workloads", "features"],
            "base": ["workloads"],
            "base_decentered": ["workloads"],
            "abs-means-hyper": ["features"],
            "abs-stddevs-hyper": ["features"],
        }
        return dims


class HorseshoeModel(PyroModel):
    label = "Horseshoe"

    def get_model(self):
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
        }
        reparam_model = reparam(self.model_raw, config=reparam_config)
        return reparam_model

    def model_raw(self, data, workloads, n_workloads, reference_y=None):
        workloads = jnp.array(workloads)
        if reference_y is not None:
            mean_nfp = jnp.mean(reference_y)
            stddev_nfp = jnp.std(reference_y)
        else:
            y_order_of_magnitude = 1
            stddev_nfp = 1
        joint_coef_stdev = stddev_nfp  # 1
        num_opts = data.shape[1]
        stddev_exp_prior = stddev_nfp  # 1

        # tau: global scale
        tau = numpyro.sample("tau-global", npdist.HalfCauchy(1))
        with numpyro.plate("options", num_opts):
            hyper_coef_means = numpyro.sample("abs-means-hyper",
                                              npdist.Normal(0, stddev_exp_prior), )
            hyper_coef_stddevs = numpyro.sample("abs-stddevs-hyper", npdist.Exponential(stddev_exp_prior), )

        hyper_base_mean = numpyro.sample("base mean hyperior", npdist.Normal(0, joint_coef_stdev), )
        hyper_base_stddev = numpyro.sample("base stddevs hyperior", npdist.Exponential(stddev_exp_prior), )

        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                # horseshoe
                activision = numpyro.sample("lambdas-activ", npdist.HalfCauchy(1))
                horseshoe = numpyro.deterministic("horseshoe", activision * tau)
                rnd_influences = numpyro.sample("influences",
                                                npdist.Normal(hyper_coef_means, hyper_coef_stddevs * horseshoe), )

        with numpyro.plate("workloads", n_workloads):
            bases = numpyro.sample("base", npdist.Normal(hyper_base_mean, hyper_base_stddev))

        respective_influences = rnd_influences[workloads]
        respective_bases = bases[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel() + respective_bases
        prior_error_expt = 1.0
        error_var = numpyro.sample("error", npdist.Exponential(prior_error_expt))

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
        return obs

    def get_arviz_dims(self):
        dims = {
            "influences": ["workloads", "features"],
            "influences_decentered": ["workloads", "features"],
            "base": ["workloads"],
            "base_decentered": ["workloads"],
            "abs-means-hyper": ["features"],
            "abs-stddevs-hyper": ["features"],
        }
        return dims

    def get_label(self):
        return self.label


class ModelFitter:
    def __init__(self, model: PyroModel, rnd=1, num_samples=1500, num_warmup=1000):
        self.nfp_name = None
        self.workload_characteristic = None
        self.feature_names = None
        self.fitting_history = None
        self.model = model
        self.mcmc = None
        self.n_chains = 3
        self.rnd = rnd
        self.num_samples = num_samples
        self.num_warmup = num_warmup

    def fit(self, df, rnd=None):
        rnd = self.rnd if rnd is None else rnd
        cols = df.columns
        self.feature_names = list(cols[:-2])

        model = self.model.get_model()
        nuts_kernel = npNUTS(model, target_accept_prob=0.9)

        progress_bar = True
        self.mcmc = npMCMC(nuts_kernel, num_samples=self.num_samples,
                           num_warmup=self.num_warmup, progress_bar=progress_bar,
                           num_chains=self.n_chains, )
        rng_key_ = random.PRNGKey(rnd)
        rng_key_, rng_key = random.split(rng_key_)

        X, workload_per_df_row, n_workloads, nfp = self.get_model_params_from_df(df)
        self.mcmc.run(rng_key, X, workload_per_df_row, n_workloads, nfp)

    def refit(self, df, rnd=None):
        rnd = self.rnd if rnd is None else rnd
        rng_key = random.PRNGKey(rnd)
        X, workload_per_df_row, n_workloads, nfp = self.get_model_params_from_df(df)
        self.mcmc.run(rng_key, X, workload_per_df_row, n_workloads, nfp)



    def get_model_params_from_df(self, df):
        cols = df.columns
        feature_names = list(cols[:-2])
        self.workload_characteristic = cols[-2]
        self.nfp_name = cols[-1]
        workload_per_df_row = df[self.workload_characteristic].to_numpy()

        nfp_per_df_row = df[self.nfp_name]
        self.workload_levels, workload_ids_per_row = np.unique(workload_per_df_row, return_inverse=True)
        n_workloads = len(self.workload_levels)
        nfp = jnp.array(nfp_per_df_row.to_numpy())
        X = df[feature_names].to_numpy()
        X = jnp.array(X, dtype=int)
        return X, workload_ids_per_row, n_workloads, nfp

    def get_arviz_data(self, posterior_predictive_df=None):
        coords = {
            "features": self.feature_names,
            "workloads": list(self.workload_levels)
        }
        dims = self.model.get_arviz_dims()
        idata_kwargs = {
            "dims": dims,
            "coords": coords,
        }
        if posterior_predictive_df is None:
            az_data = az.from_numpyro(self.mcmc, num_chains=self.n_chains, **idata_kwargs)
        else:
            predictions = self.predict(posterior_predictive_df)
            n_samples = 100
            predictions = self._predict_samples(posterior_predictive_df, n_samples=n_samples)
            ground_truth_to_pred_map = dict(zip(list(posterior_predictive_df["nfp"]), predictions))
            posterior_inference_data = az.convert_to_inference_data(ground_truth_to_pred_map)
            az_data = az.from_numpyro(self.mcmc, num_chains=self.n_chains, posterior_predictive=predictions,
                                      **idata_kwargs)
        return az_data

    def _predict_samples(self, df, n_samples: int, rnd_key=0):
        # Predictive(model_fast, guide=guide, num_samples=100,
        # return_sites=("measurements",))

        model = self.model.get_model()
        prngkey = random.PRNGKey(rnd_key)
        pred = Predictive(model, num_samples=n_samples)

        X, workload_per_df_row, n_workloads, nfp = self.get_model_params_from_df(df)
        posterior_samples = pred(prngkey, X, workload_per_df_row, n_workloads, None)
        y_pred = posterior_samples["observations"]
        return y_pred

    def predict(self, df, n_samples: int = None, ci: float = None):
        """
        Performs a prediction conforming to the sklearn interface.

        Parameters
        ----------
        X : Array-like data
        n_samples : number of posterior predictive samples to return for each prediction
        ci : value between 0 and 1 representing the desired confidence of returned confidence intervals. E.g., ci= 0.8 will generate 80%-confidence intervals

        Returns
        -------
         - a scalar if only x is specified
         - a set of posterior predictive samples of size n_samples if is given and n_samples > 0
         - a set of pairs, representing lower and upper bounds of confidence intervals for each prediction if ci is given

        """
        if not n_samples:
            n_samples = 500
            y_samples = self._predict_samples(df, n_samples=n_samples)
            y_samples = np.array(y_samples)
            y_pred = np.mean(az.hdi(y_samples, hdi_prob=0.01), axis=1)
        else:
            y_samples = np.array(self._predict_samples(df, n_samples=n_samples))
            if ci:
                y_pred = az.hdi(y_samples, hdi_prob=ci)
            else:
                y_pred = y_samples
        return y_pred


class UncertEvaluator:
    def __init__(self, data_providers=None, rnd=0):
        self.data_providers = data_providers
        self.rnd = rnd

    def eval(self, predictor: PyroModel, do_plots=True):
        results = {}
        for provider in self.data_providers:
            label = provider.get_label()
            result = self.eval_single_dataset(provider, predictor, do_plots=do_plots)
            results[label] = result
        return results




    def eval_single_dataset(self, provider: DataProvider, predictor: PyroModel, do_plots=True):
        data_lbl = provider.get_label()
        model_lbl = predictor.get_label()
        experiment_label = f"{model_lbl}-on-{data_lbl}"
        train_data = provider.get_train_data()
        test_data = provider.get_test_data()

        model_fitter = ModelFitter(predictor)
        model_fitter.fit(train_data, self.rnd)
        # az_data = model_fitter.get_arviz_data(posterior_predictive_df=test_data)
        az_data = model_fitter.get_arviz_data()
        print(az_data)

        summary = az.summary(az_data)
        print(summary)

        waic_data = az.waic(az_data)
        effective_number_of_parameters = waic_data.p_waic
        elpd_waic = waic_data.waic
        print(waic_data)
        print(waic_data.waic_i)

        psis = az.loo(az_data)
        elpd_psis = psis.loo
        penalty_psis = psis.p_loo

        result_dict = {
            "effective_number_of_parameters_waic": effective_number_of_parameters,
            "elpd_waic": elpd_psis,
            "penalty_psis": penalty_psis,
            "elpd_psis": elpd_psis,

            "arviz_data": az_data,
            "mcmc": model_fitter.mcmc
        }

        if do_plots:
            self.do_plots(az_data, experiment_label)

        return result_dict

    def do_plots(self, az_data, experiment_label):
        az.plot_posterior(az_data)
        os.makedirs("results", exist_ok=True)
        self.store_and_show_plot(experiment_label)
        # az.plot_ppc(az_data, legend=True);

        # az.plot_trace(az_data, legend=True)
        # self.store_and_show_plot(experiment_label)

        az.plot_trace(az_data, legend=False, )
        self.store_and_show_plot(experiment_label)

    def store_and_show_plot(self, experiment_label, show=False):
        plt.suptitle(str(experiment_label))
        plt.tight_layout()
        plt.savefig(f"results/{experiment_label}-posterior.pdf")
        plt.savefig(f"results/{experiment_label}-posterior.png")
        if show:
            plt.show()


def main():
    influence_option_constant_time_modifyer = 0.00
    influence_option_relative_ratio = 0.20
    influence_option_static_influence_time_in_s = 10
    df = get_train_df(influence_option_constant_time_modifyer, influence_option_relative_ratio,
                      influence_option_static_influence_time_in_s)
    dp_diag_log_stereo = DfDataProvider("ArtifDiagLogStereo", df, train_ratio=0.5)

    data_path = os.path.join("training-data/jump3r.csv")
    jump3rdf = pd.read_csv(data_path)
    train_size_abs = 500
    dp_jump3r = Jump3rDataProvider("Jump3r", jump3rdf, train_number_abs=train_size_abs)
    scaled_jump3r = StandardizerByTrainSet(dp_jump3r)

    providers = [scaled_jump3r, dp_diag_log_stereo]
    evaluator = UncertEvaluator(providers)
    wl_model_simple = AbsoluteInfWLModel()
    results = evaluator.eval(wl_model_simple)
    print(results)


def main_grid_artif():
    model_constructors = {
        # "no-pooling-gauss": NoPoolingModel,
        # "no-pooling-laplace": lambda: NoPoolingModel(prior_method_call=npdist.Laplace),
        # "MultiLevelGaussRidge": AbsoluteInfMetaWLModelRidge,
        # "MultiLevelLaplaceLasso": AbsoluteInfMetaWLModelLasso,
        # "MultiLevelHorseshoe": HorseshoeModel,
        "MultiLevelMetaLasso": MetaMetaWLModelLasso,
    }
    all_results = {}
    currentDateAndTime = datetime.now()
    timestamp = currentDateAndTime.strftime("%Y-%m-%d %H-%M-%S")
    os.makedirs(f"results/{timestamp}")
    for label, model_constructor in model_constructors.items():
        random_seed = 10
        n_non_infnluential = 10
        const = [2, 1, 0.5, *([10 ** -1] * n_non_infnluential)]
        rel = [30 * 10 ** -2, 5 * 10 ** -2, 1 * 10 ** -2, ]
        lengths = [1, 20, 100, 300]
        train_number_abs = 100
        raw_provider = get_train_df_provider_configurable(const, rel, lengths, train_number_abs=train_number_abs)
        std_provider = StandardizerByTrainSet(raw_provider)
        providers = [std_provider]
        evaluator = UncertEvaluator(providers, rnd=random_seed)
        model = model_constructor()
        results = evaluator.eval(model, do_plots=True)
        dump_file = f"results/lastrun-{label}.p"
        print(f"dumping results to {dump_file}")
        pickle.dump(results, open(dump_file, "wb"))

        for datakey, r in results.items():
            fname = f"results-{label}-{datakey}"
            new_r = {**r}
            new_r.pop('arviz_data', None)
            new_r.pop('mcmc', None)
            pretty_dict_str = pprint.pformat(new_r)
            with open(f"results/{timestamp}/{fname}.txt", "w") as text_file:
                text_file.write(pretty_dict_str)
            all_results[label] = new_r

        pretty_dict_str = pprint.pformat(all_results)
        models_concat = "-".join(model_constructors)
        fname = f"modelcomp-{models_concat}"
        with open(f"results/{timestamp}/{fname}.txt", "w") as text_file:
            text_file.write(pretty_dict_str)
        print(results)


def get_3_vars_data():
    linear_scaling_for_whole_runtime_opt_infs = np.linspace(0, 1, 3)
    relative_influences_for_workload_dependent_opt = np.linspace(0, 1, 3)
    infl_workload_independent_opt = np.linspace(10, 100, 3)
    lengths = [5, 30, 100, ]
    providers = []
    grid = list(
        itertools.product(linear_scaling_for_whole_runtime_opt_infs, relative_influences_for_workload_dependent_opt,
                          infl_workload_independent_opt))
    print(f"Running experiments for grid of size {len(grid)}")
    for constant_time_bias, rel_ratio_opt_influence, static_option_influence in grid:
        df = get_train_df(constant_time_bias, rel_ratio_opt_influence, static_option_influence, lengths=lengths)

        dp_diag_log_stereo = DfDataProvider(
            f"ArtifDiagLogStereo-{constant_time_bias}-{rel_ratio_opt_influence}-{static_option_influence}", df,
            train_ratio=0.5)
        providers.append(dp_diag_log_stereo)
    return providers


def get_train_df_provider_configurable(influence_option_constant_time_modifyers, influence_option_relative_ratios,
                                       lengths, rel_noise=0.01, train_number_abs=50):
    feature_constant_time = "const"
    f_names_const = get_ft_names_for_vals(feature_constant_time, influence_option_constant_time_modifyers)
    feature_relative = "relative"
    f_names_rel = get_ft_names_for_vals(feature_relative, influence_option_relative_ratios)
    feature_ids = [*f_names_const, *f_names_rel]
    n_features = len(feature_ids)
    configs = np.array(list(itertools.product([True, False], repeat=n_features)))
    config_cols_constant_infs = configs[:, :len(f_names_const)]
    const_influences = np.matmul(config_cols_constant_infs, np.array(influence_option_constant_time_modifyers))

    dfs = []
    providers = []
    for l in lengths:
        perfs = l + const_influences
        abs_infs_from_rel_infs = np.array(influence_option_relative_ratios) * l
        config_cols_rel_infs = configs[:, len(f_names_const):]
        aggregated_rel_infs = np.matmul(config_cols_rel_infs, abs_infs_from_rel_infs)
        perfs += aggregated_rel_infs
        new_df = pd.DataFrame(configs, columns=feature_ids)
        new_df["wl-base-length"] = l
        new_df["nfp"] = perfs
        if rel_noise:
            noises = np.abs(np.random.normal(0, new_df["nfp"] * rel_noise))
            new_df["nfp"] += noises

        dfs.append(new_df)
        print()
    df = pd.concat(dfs)

    provider = DfDataProvider(
        f"ArtificialRelAbsWL", df,
        train_number_abs=train_number_abs)
    providers.append(provider)

    return provider


def get_plot_method(model, df):
    fitter = ModelFitter(model)
    X, workload_per_df_row, n_workloads, nfp = fitter.get_model_params_from_df(df)
    return lambda: numpyro.render_model(model.model_raw, model_args=[X, workload_per_df_row, n_workloads, nfp])


def get_ft_names_for_vals(feature_constant_time, influence_option_constant_time_modifyers):
    return [f"{feature_constant_time}{i + 1}" for i in range(len(influence_option_constant_time_modifyers))]


if __name__ == '__main__':
    # main()
    main_grid_artif()
