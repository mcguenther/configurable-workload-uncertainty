from typing import List

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from abc import ABC, abstractmethod
import numpyro.distributions as npdist
from numpyro.infer import MCMC as npMCMC, NUTS as npNUTS, Predictive as npPredictive
from jax import random, numpy as jnp
from numpyro.handlers import condition as npcondition
import arviz as az
import numpyro

from sklearn.pipeline import Pipeline

from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd

from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

from wluncert.data import SingleEnvData

numpyro.set_host_device_count(4)
import copy


class NumPyroRegressor(ABC, BaseEstimator, RegressorMixin):
    def __init__(self, num_samples=1000, num_warmup=1500, num_chains=3, progress_bar=True, plot=False,
                 feature_names=None, ):
        self.coef_ = None
        self.samples = None
        self.mcmc = None
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.num_chains = num_chains
        self.progress_bar = progress_bar
        self.plot = plot
        self.feature_names = feature_names

    @staticmethod
    def model(data):
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
        pass

    def _predict_samples(self, model_args: tuple, n_samples: int = 1_000):
        posterior_samples = self.mcmc.get_samples()
        # pred = npPredictive(self.model, posterior_samples=posterior_samples, num_samples=n_samples, parallel=True)
        # numpyro currently ignores num_samples if different from number of posterior samples
        pred = npPredictive(self.model, posterior_samples=posterior_samples,parallel=True)
        rng_key_ = random.PRNGKey(0)
        y_pred = pred(rng_key_, *model_args, None)["observations"]
        return y_pred

    def get_jnp_array(self, X):
        x = jnp.atleast_2d(np.array(X.astype(float)))
        return x

    def _internal_predict(self, model_args, n_samples: int = 1500, ci: float = None, yield_samples=False):
        y_pred_samples = self._predict_samples(model_args, n_samples)
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

    def get_ci_from_samples(self, samples, ci):
        ci_s = az.hdi(samples, hdi_prob=ci)
        return ci_s

    def get_mode_from_samples(self, samples):
        hdi = az.hdi(samples, hdi_prob=0.01)
        modes = np.mean(hdi, axis=1)
        return modes


class ExtraStandardizingEnvAgnosticModel(NumPyroRegressor):

    def __init__(self, *args, **kwargs):
        NumPyroRegressor.__init__(self, *args, **kwargs)
        self.env_lbl = None

    def set_env_lbl(self, env_lbl):
        self.env_lbl = env_lbl

    def coef_ci(self, ci: float):
        pass

    def get_tuples(self, feature_names):
        pass

    def fit(self, X, y):
        X = self.get_jnp_array(X)
        y = jnp.array(y)
        nuts_kernel = npNUTS(self.model)
        self.mcmc = npMCMC(nuts_kernel, num_samples=self.num_samples,
                           num_warmup=self.num_warmup, progress_bar=self.progress_bar,
                           num_chains=self.num_chains, chain_method="parallel")
        rng_key_ = random.PRNGKey(0)
        rng_key_, rng_key = random.split(rng_key_)
        self.mcmc.run(rng_key, X, y)
        if self.plot:
            self.save_plot()
        return self.mcmc

    def save_plot(self):
        arviz_data = self.get_arviz_data()
        az.plot_trace(arviz_data,legend=True)
        plt.tight_layout()
        plt.suptitle(self.env_lbl)
        plt.savefig("./tmp/mcmc-agnostic.png")
        plt.show()

        # print(az.summary(arviz_data))
        print("ELPD/LOO")
        waic_data = az.waic(arviz_data)
        print(waic_data)

    def predict(self, X, n_samples: int = 1500, ci: float = None, yield_samples=False):
        x = jnp.atleast_2d(np.array(X.astype(float)))
        result = self._internal_predict([x])
        return result

    def model(self, data, reference_y):
        joint_coef_stdev = 0.25  # 0.25  # 2 * y_order_of_magnitude
        num_opts = data.shape[1]

        with numpyro.plate("options", num_opts):
            rnd_influences = numpyro.sample("influences", npdist.Normal(0, joint_coef_stdev), )

        # base = numpyro.sample("base", npdist.Laplace(0, joint_coef_stdev))
        result_arr = jnp.multiply(data, rnd_influences)
        result_arr = result_arr.sum(axis=1).ravel()  # + base
        error_var = numpyro.sample("error", npdist.Exponential(.01))

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample("observations", npdist.Normal(result_arr, error_var), obs=reference_y)
        return obs

    def get_arviz_data(self):
        features_lbl = "features"
        coords = {
            features_lbl: self.feature_names,
        }
        dims = {
            "influences": [features_lbl],
        }
        kwargs = {
            "dims": dims,
            "coords": coords,
        }
        numpyro_data = az.from_numpyro(self.mcmc, **kwargs)
        return numpyro_data


class StandardizingModel:
    def __init__(self):
        self.data_map = {}

    def get_standardize_data(self, data: SingleEnvData):
        env_id = data.env_id
        return self.get_standardize_data_by_id(env_id)

    def get_standardize_data_by_id(self, env_id):
        env_id = int(env_id)
        return self.data_map[env_id]

    def X_envids_y_from_data_for_prediction(self, data):
        return self._internal_data_splitting(data, override_standardizers=False)

    def X_envids_y_from_data_for_fitting(self, data):
        return self._internal_data_splitting(data, override_standardizers=True)

    def _internal_data_splitting(self, data, override_standardizers=False):
        X_list = []
        y_list = []
        env_id_list = []
        for single_env_data in data:
            env_id = single_env_data.env_id
            if override_standardizers:
                normalized_data = single_env_data.normalize()
            else:
                normalized_train_data = self.get_standardize_data_by_id(env_id)
                normalized_data = single_env_data.normalize(normalized_train_data)
            X = normalized_data.get_X()
            y = normalized_data.get_y()
            data_len = len(y)
            new_env_id_sublist = [env_id] * data_len
            X_list.append(X)
            y_list.append(y)
            env_id_list.append(new_env_id_sublist)
            if override_standardizers:
                self.data_map[env_id] = normalized_data
        X_df = pd.concat(X_list)
        X_df = X_df.astype(float)
        X = jnp.array(X_df)
        y = jnp.concatenate(y_list)
        envs = jnp.array(env_id_list).ravel()
        return X, envs, y


class ExtraStandardizingSimpleModel(NumPyroRegressor, StandardizingModel):

    def __init__(self, *args, env_names=None, **kwargs):
        NumPyroRegressor.__init__(self, *args, **kwargs)
        StandardizingModel.__init__(self)
        self.env_names = env_names

    def coef_ci(self, ci: float):
        pass

    def get_tuples(self, feature_names):
        pass

    def fit(self, data: List[SingleEnvData]):
        X, envs, y = self.X_envids_y_from_data_for_fitting(data)
        n_envs = len(data)
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
        }
        reparam_model = reparam(self.model, config=reparam_config)
        nuts_kernel = npNUTS(reparam_model)
        self.mcmc = npMCMC(nuts_kernel, num_samples=self.num_samples,
                           num_warmup=self.num_warmup, progress_bar=self.progress_bar,
                           num_chains=self.num_chains, )
        rng_key_ = random.PRNGKey(0)
        rng_key_, rng_key = random.split(rng_key_)
        # mcmc.run(rng_key, X_agg[:, 0], X_agg[:, 1], X_agg[:, 2], given_obs=nfp_mean_agg, obs_stddev=nfp_stddev_agg)
        self.mcmc.run(rng_key, X, envs, n_envs, y)
        print("finished fitting multilevel model")
        if self.plot:
            reparam_config = {
                "influences": LocScaleReparam(0),
                "base": LocScaleReparam(0),
            }
            print("plotting prior trace")

            reparam_model = reparam(self.model, config=reparam_config)
            nuts_kernel = npNUTS(reparam_model)
            prior_mcmc = npMCMC(nuts_kernel, num_samples=500,
                                num_warmup=200, progress_bar=self.progress_bar,
                                num_chains=self.num_chains, )
            rng_key_ = random.PRNGKey(0)
            rng_key_, rng_key = random.split(rng_key_)
            # mcmc.run(rng_key, X_agg[:, 0], X_agg[:, 1], X_agg[:, 2], given_obs=nfp_mean_agg, obs_stddev=nfp_stddev_agg)
            prior_mcmc.run(rng_key, X[:10,:], envs[:10], n_envs, None)
            # arviz_data = self.get_arviz_data(prior_mcmc)
            self.save_plot(prior_mcmc, loo=False)
            self.save_plot()
        return self.mcmc

    def predict(self, data: List[SingleEnvData]):
        X, envs, y = self.X_envids_y_from_data_for_prediction(data)
        n_workloads = len(data)
        model_args = X, envs, n_workloads
        envs_int = np.array(envs).astype(int)
        preds = self._internal_predict(model_args)
        unstandardized_preds = []
        unstandardized_preds_dict = {int(env_id): [] for env_id in np.unique(envs_int)}
        for pred, env_id in zip(preds, envs_int):
            normalized_train_data = self.get_standardize_data_by_id(env_id)
            unstandardized_pred = normalized_train_data.un_normalize_y(pred)[0]
            unstandardized_preds.append(unstandardized_pred)
            unstandardized_preds_dict[env_id].append(unstandardized_pred)
        return_list = [unstandardized_preds_dict[d.env_id] for d in data]
        return return_list

    def model(self, data, workloads, n_workloads, reference_y):
        err_expectation = 0.5
        err_hyperior_expectation = 1 /err_expectation
        err_exponential_pdf_rate = 1/ err_hyperior_expectation
        joint_coef_stdev = 0.5 #0.5  # 2 * y_order_of_magnitude
        num_opts = data.shape[1]
        coefs_expected_stddev_change_over_envs = 0.1 #0.5
        coefs_hyperior_expected = 1/coefs_expected_stddev_change_over_envs
        coefs_stds_prior_exp_rate = 1 / coefs_hyperior_expected

        with numpyro.plate("options", num_opts):
            hyper_coef_means = numpyro.sample("influences-mean-hyperior", npdist.Normal(0, joint_coef_stdev), )
            hyper_coef_stddevs = numpyro.sample("influences-stddevs-hyperior", npdist.Exponential(coefs_hyperior_expected), )

        # hyper_base_mean = numpyro.sample("base-mean-hyperior", npdist.Normal(0, joint_coef_stdev), )
        # hyper_base_stddev = numpyro.sample("base-stddevs-hyperior", npdist.Exponential(stddev_exp_prior), )

        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample("influences", npdist.Normal(hyper_coef_means, hyper_coef_stddevs), )

        # with numpyro.plate("workloads", n_workloads):
        #     bases = numpyro.sample("base", npdist.Normal(hyper_base_mean, hyper_base_stddev))

        # error_hyperior = numpyro.sample("error-hyperior", npdist.Exponential(err_exponential_pdf_rate))
        # error_hyperior = numpyro.sample("error-hyperior", npdist.Gamma(2,0.1))
        # error_hyperior = numpyro.sample("error-hyperior", npdist.HalfNormal(1))
        with numpyro.plate("workloads", n_workloads):
            error_var = numpyro.sample("error", npdist.Exponential(1/err_expectation))

        right_error = error_var[workloads]
        respective_influences = rnd_influences[workloads]
        # respective_bases = bases[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel()  # + respective_bases

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample("observations", npdist.Normal(result_arr, right_error), obs=reference_y)
        return obs

    def save_plot(self, mcmc=None, loo=True):
        print("plotting result")
        # self.mcmc.print_summary()
        arviz_data = self.get_arviz_data(mcmc)
        print(az.summary(arviz_data))
        if loo:
            print("ELPD/LOO")
            waic_data = az.waic(arviz_data)
            print(waic_data)

        plot_vars = [v for v in list(arviz_data.posterior) if "decentered" not in v]
        file_template = "./tmp/mcmc-multilevel-{}.png"
        plot_path = file_template.format("trace")  # "./tmp/mcmc-multilevel-trace.png"
        legend = False
        az.plot_trace(arviz_data,
                      var_names=plot_vars, legend=legend, compact=True,
                      combined=True, chain_prop={"ls": "-"})
        plt.tight_layout()
        print("storing plot to", plot_path)
        plt.savefig(plot_path)
        plt.show()

        # plot_path = file_template.format("forest")  # "./tmp/mcmc-multilevel-trace.png"
        # az.plot_forest(arviz_data, var_names=plot_vars)
        # plt.tight_layout()
        # plt.savefig(plot_path)
        # plt.show()
        print("finished plotting")

    def get_arviz_data(self, mcmc=None):
        mcmc = self.mcmc if not mcmc else mcmc
        features_lbl = "features"
        env_lbl = "envs"
        coords = {
            features_lbl: self.feature_names,
            # "features": ["A", "B"],
            env_lbl: self.env_names
        }
        dims = {
            "influences": [env_lbl, features_lbl],
            "influences_decentered": [env_lbl, features_lbl],
            # "base": [env_lbl],
            # "base_decentered": [env_lbl],
            "influences-mean-hyperior": [features_lbl],
            "influences-stddevs-hyperior": [features_lbl],
        }
        kwargs = {
            "dims": dims,
            "coords": coords,
            # "constant_data": {"x": xdata}
        }
        numpyro_data = az.from_numpyro(mcmc, **kwargs)
        return numpyro_data


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


class CompletePoolingEnvModel(StandardizingModel):
    def __init__(self, model_prototype: RegressorMixin):
        super().__init__()
        self.model_prototype = model_prototype
        self.pooled_model = copy.deepcopy(self.model_prototype)

    def fit(self, data: List[SingleEnvData]):
        X, envs, y = self.X_envids_y_from_data_for_fitting(data)
        self.pooled_model.fit(X, y)
        # for single_env_data in data:
        #     normalized_data = single_env_data.normalize()
        #     env_id = normalized_data.env_id
        #     X = normalized_data.get_X()
        #     y = normalized_data.get_y()
        #     self.pooled_model.fit(X, y)

    def predict(self, data: List[SingleEnvData]):
        preds = []
        for single_env_data in data:
            env_id = single_env_data.env_id
            train_data = self.get_standardize_data_by_id(env_id)
            normalized_predict_data = single_env_data.normalize(train_data)
            X = normalized_predict_data.get_X()
            pred = self.pooled_model.predict(X)
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
