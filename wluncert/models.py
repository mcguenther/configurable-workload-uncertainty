import hashlib
import os
import warnings
import uuid

warnings.simplefilter(action="ignore", category=FutureWarning)
import time
import numpyro

# numpyro.enable_x64()
numpyro.set_host_device_count(50)

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import numpyro.distributions as npdist
from numpyro.infer import MCMC as npMCMC, NUTS as npNUTS, Predictive as npPredictive
from jax import random
from numpyro.handlers import condition as npcondition
import arviz as az

from sklearn.pipeline import Pipeline

from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures

import numpy as np

from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam
import seaborn as sns
import copy
from typing import List

import pandas as pd
from jax import numpy as jnp

from data import SingleEnvData, WorkloadTrainingDataSet, Preprocessing

# from wluncert.analysis import ModelEvaluation
import localflow as mlflow
from numpyro.distributions import constraints

NO_POOLING = "NO_POOLING"
COMPLETE_POOLING = "COMPLETE_POOLING"
PARTIAL_POOLING = "PARTIAL_POOLING"


def mlflow_log_artifact(*args, **kwargs):
    time.sleep(0.5)
    return mlflow.log_artifact(*args, **kwargs)


def mlflow_log_text(*args, **kwargs):
    time.sleep(0.5)
    return mlflow.log_text(*args, **kwargs)


class ExperimentationModelBase(ABC, BaseEstimator):
    def __init__(self, preprocessings: List[Preprocessing] = None):
        self.fitting_time = None
        self.prediction_times = []
        self.env_lbls = []
        self.feature_names = []
        self.preprocessings = preprocessings or []
        self.last_fitting_processing_time = 0
        self.last_prediction_processing_time = 0

    def set_envs(self, envs_data: WorkloadTrainingDataSet):
        self.feature_names = envs_data.get_feature_names()
        self.env_lbls = envs_data.get_env_lables()

    def fit(self, data: List[SingleEnvData], *args, **kwargs):
        if self.preprocessings:
            t_start = time.time()
            for preprocessing in self.preprocessings:
                data = preprocessing.fit_transform(data)

            self.feature_names = data[0].get_feature_names()
            t_cost = time.time() - t_start
            self.last_fitting_processing_time = t_cost

        t_start = time.time()
        self._fit(data, *args, **kwargs)
        t_cost = time.time() - t_start
        self.fitting_time = t_cost

    def predict(self, data: List[SingleEnvData], *args, **kwargs):
        if self.preprocessings:
            t_start = time.time()
            for preprocessing in self.preprocessings:
                data = preprocessing.transform(data)
            t_cost_pre = time.time() - t_start
            self.last_prediction_processing_time = t_cost_pre
        t_start = time.time()
        y_pred = self._predict(data, *args, **kwargs)
        t_cost = time.time() - t_start

        if self.preprocessings:
            t_start = time.time()
            for preprocessing in self.preprocessings:
                y_pred = preprocessing.inverse_transform_pred(y_pred, data)
            self.last_prediction_processing_time += time.time() - t_start
        self.prediction_times.append(t_cost)
        return y_pred

    def _internal_data_splitting(self, data):
        X_list = []
        y_list = []
        env_id_list = []
        for single_env_data in data:
            env_id = single_env_data.env_id
            X = single_env_data.get_X()
            y = single_env_data.get_y()
            data_len = len(y)
            new_env_id_sublist = [env_id] * data_len
            X_list.append(X)
            y_list.append(y)
            env_id_list.append(new_env_id_sublist)

        X_df = pd.concat(X_list)
        X_df = X_df.astype(float)
        X = jnp.array(X_df)
        y = jnp.concatenate(y_list)
        envs = [
            item for sublist in env_id_list for item in sublist
        ]  # jnp.array(env_id_list).ravel()
        jnp_envs = jnp.array(envs)
        return X, jnp_envs, y

    def _internal_data_to_list(self, envs, ys):
        # env_ys = [[] for _ in np.unique(envs)]
        env_ys = {env_id: [] for env_id in np.unique(envs)}

        for env, y in zip(envs, ys):
            env_ys[int(env)].append(y)
        return_list = list(env_ys.values())  # sum(env_ys.values(), [])
        return return_list

    def get_cost_dict(self):
        return {
            "pred_time_cost": self.get_total_prediction_time(),
            "fitting_time_cost": self.get_fitting_time(),
            "preproc-fittin_time_cost": self.last_fitting_processing_time,
            "preproc-pred-_time_cost": self.last_prediction_processing_time,
        }

    def get_total_prediction_time(self):
        return float(np.sum(self.prediction_times).ravel())

    def get_fitting_time(self):
        return self.fitting_time

    @abstractmethod
    def _fit(self, param, param1):
        pass

    @abstractmethod
    def _predict(self, param, param1):
        pass


class NumPyroRegressor(ExperimentationModelBase):
    def __init__(
        self,
        num_samples=500,
        num_warmup=1000,
        num_chains=3,
        progress_bar=False,
        plot=False,
        return_samples_by_default=False,
        preprocessings=None,
        persist_arviz=False,
    ):
        super().__init__(preprocessings)
        self.coef_ = None
        self.samples = None
        self.mcmc = None
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.num_chains = num_chains
        self.progress_bar = progress_bar
        self.plot = plot
        self.return_samples_by_default = return_samples_by_default
        self.persist_arviz = persist_arviz

    @staticmethod
    def model(data):
        pass

    def condition(self, y):
        return npcondition(self.model, data={"measurements": y})

    # @abstractmethod
    # def fit(self, X, y):
    #     pass

    @abstractmethod
    def get_tuples(self, feature_names):
        pass

    @abstractmethod
    def coef_ci(self, ci: float):
        pass

    # @abstractmethod
    # def predict(self, X, env_ids, n_samples: int = 1000, ci: float = None, yield_samples=True):
    #     pass

    def _predict_samples(self, model_args: tuple, n_samples: int = 1_000):
        posterior_samples = self.mcmc.get_samples()
        # pred = npPredictive(self.model, posterior_samples=posterior_samples, num_samples=n_samples, parallel=True)
        # numpyro currently ignores num_samples if different from number of posterior samples
        pred = npPredictive(
            self.model, posterior_samples=posterior_samples, parallel=True
        )
        rng_key_ = random.PRNGKey(0)
        y_pred = pred(rng_key_, *model_args, None)["observations"]
        return y_pred

    def get_jnp_array(self, X):
        x = jnp.atleast_2d(np.array(X.astype(float)))
        return x

    def _internal_predict(
        self, model_args, n_samples: int = 1500, ci: float = None, yield_samples=None
    ):
        yield_samples = yield_samples or self.return_samples_by_default
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

    def get_bayes_eval_dict(self, test_list):
        arviz_data = self.get_arviz_data()

        ######## debugging
        arviz_data_posterior = arviz_data.posterior.copy()
        arviz_data_posterior = arviz_data_posterior.where(
            ~np.isinf(arviz_data_posterior), np.nan
        )
        arviz_data.posterior = arviz_data_posterior

        #### debugging

        loo_data = az.loo(arviz_data, pointwise=False)
        log_likelihood = self.get_log_likelihood(test_list)
        post_vars = arviz_data.posterior.variables.mapping
        # n_features = len(post_vars["features"])
        rv_names = [
            k
            for k in post_vars
            if ("influences" in k or "base" in k)
            and "decentered" not in k
            and "_dim_" not in k
            # and "relative-influences" not in k
        ]
        n_rvs = 0
        for rv_name in rv_names:
            shape = post_vars[rv_name].shape
            if len(shape) == 3:  # dim 0: chains, dim 1: samples, dim 2: features
                n_rvs += shape[-1]
            elif len(shape) == 4:
                # dim 0: chains, dim 1: samples, dim 2: env, dim 3 features
                n_envs = shape[-2]
                n_features = shape[-1]
                n_rvs += n_envs * n_features
            elif len(shape) == 2:
                # single RV
                n_rvs += 1
            else:
                raise AssertionError(
                    "Error in computing RV count for RV with name",
                    rv_name,
                    "and shape",
                    shape,
                )
        original_total_influences = len(self.env_lbls) * len(self.feature_names)
        p_loo = loo_data.p_loo
        relative_DOF = p_loo / original_total_influences
        DOF_shrinkage = (original_total_influences - p_loo) / original_total_influences
        d = {
            "p_loo": p_loo,
            "n_rvs": n_rvs,
            "original_total_influences": original_total_influences,
            "relative_DOF": relative_DOF,
            "DOF_shrinkage": DOF_shrinkage,
            "elpd_loo_se": loo_data.se,
            "elpd_loo": loo_data.elpd_loo,
            "test_set_log-likelihood": float(log_likelihood),
            "warning": loo_data.warning,
            "log_scale": loo_data.scale == "log",
        }
        return d

    def evaluate(self, eval, test_list=None):
        eval.prepare_sample_modes()
        eval.add_mape()
        eval.add_R2()
        eval.add_mape_CI()
        cost_df = self.get_cost_dict()
        test_list = test_list or eval.get_test_list()
        bayesian_metrics = self.get_bayes_eval_dict(test_list)
        model_metrics = {**bayesian_metrics, **cost_df}
        eval.add_custom_model_dict(model_metrics)
        if self.persist_arviz:
            self.persist_arviz_data()
        if self.plot:
            eval.add_custom_pred_eval(self.debug_predictions)
        return eval

    def debug_predictions(self, y_true, y_pred):
        print("predictions")

        y_trues = []
        comparisons = []
        envs = []
        kinds = []

        for env_id, env_lbl in zip(range(len(y_true)), self.env_lbls):
            y_pred_env = y_pred[env_id]
            modes = az.hdi(y_pred_env.T, 0.05).mean(axis=1)
            y_true_env = y_true[env_id]
            diff_abs = np.abs(y_true_env - modes)
            diff_rel = diff_abs / y_true_env * 100
            n_data_points = len(y_true_env)

            y_trues.extend(list(y_true_env))
            comparisons.extend(list(modes))
            envs.extend([env_lbl] * n_data_points)
            kinds.extend(["pred"] * n_data_points)

            y_trues.extend(list(y_true_env))
            comparisons.extend(list(diff_rel))
            envs.extend([env_lbl] * n_data_points)
            kinds.extend(["ape"] * n_data_points)

            # plt.scatter(modes, y_true_env)
            # ax = plt.gca()
            # ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

        df = pd.DataFrame(
            data={
                "true": y_trues,
                "comparison": comparisons,
                "env": envs,
                "kind": kinds,
            }
        )
        mapes_per_env = (
            df.loc[df["kind"] == "ape"]
            .groupby(by="env")[["comparison", "true"]]
            .mean()
            .sort_values("true")
            .rename(columns={"comparison": "MAPE", "true": "mean perf"})
        )
        print(mapes_per_env)
        print()
        g = sns.FacetGrid(df, col="env", row="kind", sharex=False, sharey=False)
        g.map(
            sns.scatterplot,
            "true",
            "comparison",
            alpha=0.1,
        )
        # Adjust each axis to include 0
        for ax in g.axes.flat:
            # Get current x-axis and y-axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Adjust x-axis and y-axis limits to include 0
            new_xlim = (min(xlim[0], 0), max(xlim[1], 0))
            new_ylim = (min(ylim[0], 0), max(ylim[1], 0))
            if "pred" in ax.get_title().lower():
                # Adjust x-axis and y-axis limits to include 0
                new_xlim = (min(xlim[0], new_ylim[0]), max(xlim[1], new_ylim[1]))
                new_ylim = new_xlim
                # Calculate the points for the grey dashed line
                # Use the broader of the x and y limits to ensure the line spans the whole plot
                line_min = min(xlim[0], ylim[0])
                line_max = max(xlim[1], ylim[1])

                # Draw the grey dashed line
                ax.plot(
                    [line_min, line_max],
                    [line_min, line_max],
                    linestyle="--",
                    color="grey",
                )
            # Set new limits
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
        plt.suptitle("Predictions")
        plt.tight_layout()
        plt.show()
        print("predictions done")
        # sns.scatterplot(data=df, x="true", y="pred")
        # print(y_true)
        # print(y_pred)

    @classmethod
    def get_mode_from_samples(self, samples):
        hdi = az.hdi(samples, hdi_prob=0.1)
        modes = np.mean(hdi, axis=1)
        return modes

    def persist_arviz_data(self):
        az_data = self.get_arviz_data()
        tmp_file = f"tmp/arviz_data-{uuid.uuid4()}.netcdf"
        az_data.to_netcdf(filename=tmp_file)
        mlflow_log_artifact(tmp_file)
        os.remove(tmp_file)

    def get_log_likelihood(self, test_list):
        merged_y_true = []

        # i = 0
        # test_set = test_list[0]
        # y_obs = test_set.get_y()
        # X = self.get_jnp_array(
        #     test_set.get_X()
        # )  # jnp.array(test_set.get_X().to_numpy())
        # merged_y_true.extend(y_obs)
        # # model = self.condition(y_obs)
        # model = self.model
        # env_id = test_set.env_id
        # id_list = [env_id] * len(y_obs)
        # n_envs = 1
        # ll = numpyro.infer.util.log_likelihood(
        #     self.model,
        #     self.mcmc.get_samples(),
        #     X,
        #     jnp.array(id_list),
        #     n_envs,
        #     jnp.array(y_obs),
        #     batch_ndims=1,
        # )
        # # TODO clearify why we get a ll per posterior sample; in theory, all the posterior samples should form a distribution which yields the ll; so, wy have to average? Is this merely a technical detail?
        # # thought: we plug each sample into the model and compute the ll given the prior shapes.
        # ll_mean = float(ll["observations"].mean())

        data_transformed = test_list
        for preprocessing in self.preprocessings:
            data_transformed = preprocessing.transform(data_transformed)

        lls_for_wl = []

        n_envs = len(data_transformed)
        for test_set in data_transformed:
            y_obs = test_set.get_y()
            X = self.get_jnp_array(test_set.get_X())
            merged_y_true.extend(y_obs)
            model = self.model
            env_id = test_set.env_id
            id_list = [env_id] * len(y_obs)

            ll_samples = numpyro.infer.util.log_likelihood(
                model,
                self.mcmc.get_samples(),
                X,
                jnp.array(id_list),
                n_envs,
                jnp.array(y_obs),
                batch_ndims=1,
            )
            lls = ll_samples["observations"].mean(axis=0)
            lls_for_wl.extend(lls)
        # total likelihood of the data given the model and posteriors
        # TODO arviz, following vehtari et al, report the mean so we do the same for now to avoid vanishing probs
        ll = np.mean(lls_for_wl)

        return ll

    def array_to_seed(self, arr):
        # Ensure the array is in a consistent, hashable form
        arr_bytes = arr.ravel().view(np.uint8)

        # Create a hash of the bytes, using SHA-256 for example
        hash_obj = hashlib.sha256(arr_bytes)
        hash_digest = hash_obj.digest()

        # Convert the hash (bytes) to an integer
        # We use int.from_bytes and take a portion of the hash to fit into an integer
        seed = int.from_bytes(
            hash_digest[:4], "little"
        )  # Using the first 4 bytes for a 32-bit integer

        # Ensure the seed is non-negative
        seed = seed % (
            2**31
        )  # Make sure it fits in a 31-bit space, which is positive for a signed int

        return seed


class MCMCMultilevelPartial(NumPyroRegressor):
    pooling_cat = PARTIAL_POOLING

    def __init__(self, *args, **kwargs):
        NumPyroRegressor.__init__(self, *args, **kwargs)
        self.model_id = "mcmc-partial_pooling"

        self.joint_coef_stdev = 1.0  # 0.62  # 0.5  # 0.5  # 2 * y_order_of_magnitude
        self.coefs_expected_stddev_change_over_envs = 1.0  # 0.2  # s0.5  # 1.
        self.error_expectation = 1.0
        self.base_hyper_prior = -0.5, 1.0
        self.base_hyper_exp_std = 1.0

    def model(self, data, workloads, n_workloads, reference_y):
        err_expectation = 0.2
        err_hyperior_expectation = 1 / err_expectation
        err_exponential_pdf_rate = 1 / err_hyperior_expectation
        joint_coef_stdev = 0.2  # 0.5  # 2 * y_order_of_magnitude
        num_opts = data.shape[1]
        coefs_expected_stddev_change_over_envs = 0.2  # 0.5
        coefs_hyperior_expected = 1 / coefs_expected_stddev_change_over_envs
        coefs_stds_prior_exp_rate = 1 / coefs_hyperior_expected

        with numpyro.plate("options", num_opts):
            hyper_coef_means = numpyro.sample(
                "influences-mean-hyperior",
                npdist.Normal(0, joint_coef_stdev),
            )
            hyper_coef_stddevs = numpyro.sample(
                "influences-stddevs-hyperior",
                npdist.Exponential(coefs_hyperior_expected),
            )
        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample(
                    "influences",
                    npdist.Normal(hyper_coef_means, hyper_coef_stddevs),
                )
        # hyper_base_means = numpyro.sample("base-mean-hyperior",
        #                                   npdist.Normal(0, joint_coef_stdev), )
        # hyper_base_stddevs = numpyro.sample("base-stddevs-hyperior",
        #                                     npdist.Exponential(coefs_hyperior_expected), )
        with numpyro.plate("workloads", n_workloads):
            error_var = numpyro.sample("error", npdist.Exponential(1 / err_expectation))
            # base = numpyro.sample("base", npdist.Normal(hyper_base_means, hyper_base_stddevs))
        # error_var = numpyro.sample("error", npdist.Exponential(1 / error_var))

        right_error = error_var[workloads]
        respective_influences = rnd_influences[workloads]
        # respective_bases = base[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel()  # + respective_bases

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample(
                "observations", npdist.Normal(result_arr, right_error), obs=reference_y
            )
        return obs

    def coef_ci(self, ci: float):
        pass

    def get_tuples(self, feature_names):
        pass

    def _fit(self, data: List[SingleEnvData]):
        X, envs, y = self._internal_data_splitting(data)
        n_envs = len(data)
        reparam_config = self.get_reparam_dict()
        reparam_model = reparam(self.model, config=reparam_config)
        nuts_kernel = npNUTS(
            reparam_model,
            # max_tree_depth=15,
            target_accept_prob=0.9,
            init_strategy=numpyro.infer.init_to_median,
            # init_strategy=numpyro.infer.init_to_sample,
        )
        self.mcmc = npMCMC(
            nuts_kernel,
            num_samples=self.num_samples,
            num_warmup=self.num_warmup,
            progress_bar=self.progress_bar,
            num_chains=self.num_chains,
        )
        rnd_key_seed = self.array_to_seed(y)
        rng_key_ = random.PRNGKey(rnd_key_seed)
        rng_key_, rng_key = random.split(rng_key_)
        # mcmc.run(rng_key, X_agg[:, 0], X_agg[:, 1], X_agg[:, 2], given_obs=nfp_mean_agg, obs_stddev=nfp_stddev_agg)

        self.mcmc.run(rng_key, X, envs, n_envs, y)

        if self.plot:
            self.save_plot()
            # self.plot_options()

        if self.plot and False:
            self.plot_prior_dists(X, envs, n_envs, y)
        return self.mcmc

    def plot_options(self, model_names=None, var_names=None):
        var_names = (
            var_names
            if var_names is not None
            else [
                "influences-mean-hyperior",
                "influences",
            ]
        )
        az_data = self.get_arviz_data()
        num_plots = len(self.feature_names)
        n_cols = 4
        num_rows = (num_plots - 1) // n_cols + 1
        num_cols = min(num_plots, n_cols)
        print(
            "n_plots",
            num_plots,
            "ncols",
            n_cols,
            "num_rows",
            num_rows,
            "num_cols",
            num_cols,
        )

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6)
        )

        for i, feature_name in enumerate(self.feature_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]
            coords = {"features": [feature_name]}
            az.plot_forest(
                az_data,
                combined=True,
                var_names=var_names,
                model_names=model_names,
                kind="ridgeplot",
                hdi_prob=0.999,
                ridgeplot_overlap=3,
                linewidth=3,
                coords=coords,
                ax=ax,
            )

            ax.set_xlim(-1.5, 1.5)
            ax.set_title(f"Option influence {feature_name}")

        plt.suptitle("Hyper Prior vs. Influences")
        plt.tight_layout()
        time.sleep(0.1)
        plt.show()

    def plot_prior_dists(self, X, envs, n_envs, y):
        reparam_config = self.get_reparam_dict()
        print("plotting prior trace")
        reparam_model = reparam(self.model, config=reparam_config)
        nuts_kernel = npNUTS(reparam_model)
        prior_mcmc = npMCMC(
            nuts_kernel,
            num_samples=750,
            num_warmup=500,
            progress_bar=self.progress_bar,
            num_chains=self.num_chains,
        )
        rng_key_ = random.PRNGKey(0)
        rng_key_, rng_key = random.split(rng_key_)
        # mcmc.run(rng_key, X_agg[:, 0], X_agg[:, 1], X_agg[:, 2], given_obs=nfp_mean_agg, obs_stddev=nfp_stddev_agg)
        print("starting prior plot!")
        # prior_mcmc.run(rng_key, X[:20, :], envs[:20], n_envs, None)
        n = min(len(envs), 500)
        prior_mcmc.run(rng_key, X[:n], envs[:n], n_envs, y[:n])
        # arviz_data = self.get_arviz_data(prior_mcmc)
        self.save_plot(prior_mcmc, loo=False, lbl="prior")

    def get_reparam_dict(self):
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
        }
        return reparam_config

    def _predict(self, data: List[SingleEnvData]):
        X, envs, y = self._internal_data_splitting(data)
        n_workloads = len(data)
        model_args = X, envs, n_workloads
        envs_int = np.array(envs).astype(int)
        preds = self._internal_predict(model_args)
        # unstandardized_preds = []
        unstandardized_preds_dict = {int(env_id): [] for env_id in np.unique(envs_int)}
        for i, env_id in enumerate(envs_int):
            pred = preds[:, i]
            # unstandardized_preds.append(pred)
            # single_env_data = data[env_id]
            unstandardized_preds_dict[env_id].append(pred)
        return_list = [np.array(unstandardized_preds_dict[d.env_id]) for d in data]
        return return_list

    def save_plot(self, mcmc=None, loo=True, lbl=None):
        lbl = lbl or "trace"
        print("plotting result")
        # self.mcmc.print_summary()
        arviz_data = self.get_arviz_data(mcmc)
        print(az.summary(arviz_data))
        if loo:
            print("ELPD/LOO")
            waic_data = az.waic(arviz_data)
            print(waic_data)
        plot_vars = [v for v in list(arviz_data.posterior) if "decentered" not in v]
        file_template = "./tmp/mcmc-" + self.model_id + "-{}.png"
        plot_path = file_template.format(lbl)  # "./tmp/mcmc-multilevel-trace.png"
        legend = False
        az.plot_trace(
            arviz_data,
            var_names=plot_vars,
            legend=legend,
            compact=True,
            combined=True,
            chain_prop={"ls": "-"},
        )
        if lbl is not None:
            plt.suptitle(lbl)
        plt.tight_layout()
        # print("storing plot to", plot_path)
        # plt.savefig(plot_path)
        plt.show()
        print("finished plotting")

    def get_arviz_data(self, mcmc=None):
        mcmc = self.mcmc if not mcmc else mcmc
        features_lbl = "features"
        env_lbl = "envs"
        coords = {
            features_lbl: self.feature_names,
            # "features": ["A", "B"],
            env_lbl: self.env_lbls,
        }
        dims = {
            "influences": [env_lbl, features_lbl],
            "influences_decentered": [env_lbl, features_lbl],
            "influences-mean-hyperior": [features_lbl],
            "influences-stddevs-hyperior": [features_lbl],
            "base": [env_lbl],
            # "base_decentered": [env_lbl],
        }
        kwargs = {
            "dims": dims,
            "coords": coords,
            # "constant_data": {"x": xdata}
        }
        numpyro_data = az.from_numpyro(mcmc, **kwargs)
        return numpyro_data

    def get_pooling_cat(self):
        return self.pooling_cat


class MCMCPartialRobustLasso(MCMCMultilevelPartial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-partial-pooling-robust"
        # print("2")

    def model(self, data, workloads, n_workloads, reference_y):
        num_opts = data.shape[1]
        coefs_hyperior_expected = 1 / self.coefs_expected_stddev_change_over_envs
        scale_laplace = self.joint_coef_stdev / np.sqrt(2)

        with numpyro.plate("options", num_opts):
            # in laplace, the spread is not the stddev as in normal dist, but it is the variance
            # Scale for Laplace to match standard deviation
            hyper_coef_means = numpyro.sample(
                "influences-mean-hyperior",
                npdist.Laplace(0, scale_laplace),
            )
            hyper_coef_stddevs = numpyro.sample(
                "influences-stddevs-hyperior",
                npdist.Exponential(coefs_hyperior_expected),
            )

        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample(
                    "influences",
                    npdist.Normal(hyper_coef_means, hyper_coef_stddevs),
                )
        with numpyro.plate("workloads", n_workloads):
            error_var = numpyro.sample(
                "error", npdist.Exponential(1 / self.error_expectation)
            )

        base_hyper = numpyro.sample("base-hyper", npdist.Normal(*self.base_hyper_prior))
        base_hyper_std = numpyro.sample(
            "base-hyper-std", npdist.Exponential(1 / self.base_hyper_exp_std)
        )

        with numpyro.plate("workloads", n_workloads):
            base = numpyro.sample("base", npdist.Normal(base_hyper, base_hyper_std))

        # with numpyro.plate("workloads", n_workloads):
        #     base = numpyro.sample("base", npdist.Normal(-0.50, 2.0))

        right_error = error_var[workloads]
        respective_influences = rnd_influences[workloads]
        respective_bases = base[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel() + respective_bases

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample(
                "observations", npdist.Normal(result_arr, right_error), obs=reference_y
            )
        return obs


class MCMCPartialRobustLassoAdaptiveShrinkage(MCMCPartialRobustLasso):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-partial-pooling-robust-shrinkage-hyper"

    def get_reparam_dict(self):
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
            "influences-mean-hyperior": LocScaleReparam(0),
            # "influences-stddevs-hyperior": LocScaleReparam(0),
        }
        return reparam_config

    def model(self, data, workloads, n_workloads, reference_y):
        num_opts = data.shape[1]
        coefs_hyperior_expected = 1 / self.coefs_expected_stddev_change_over_envs
        scale_laplace = self.joint_coef_stdev / np.sqrt(2)
        # hyper_hyper_regulariation_strength_exp = numpyro.deterministic(
        #     "influences-regularization-expected", 1 / hyper_hyper_regulariation_strength
        # )

        # hyper_hyper_influence_variation = numpyro.sample(
        #     "influence-variation-hyperior",
        #     npdist.Exponential(coefs_hyperior_expected),
        # )

        base_hyper_prior = numpyro.sample(
            "base-hyper", npdist.Normal(*self.base_hyper_prior)
        )
        base_hyper_prior_var = numpyro.sample(
            "base-hyper_var", npdist.Exponential(1 / self.base_hyper_exp_std)
        )

        with numpyro.plate("workloads", n_workloads):
            base = numpyro.sample(
                "base", npdist.Laplace(base_hyper_prior, base_hyper_prior_var)
            )

        hyper_hyper_regulariation_strength = numpyro.sample(
            "influences-regularization-shrinkage",
            # npdist.Exponential(1 / self.joint_coef_stdev),  # / scale_laplace),
            npdist.Exponential(1 / self.joint_coef_stdev),  # / scale_laplace),
        )

        # hyper_hyper_coef_stddevs = numpyro.sample(
        #     "influences-stddevs-hyperior_prior",
        #     npdist.Exponential(1 / self.coefs_expected_stddev_change_over_envs),
        # )

        # hyper_hyper_coef_stddevs_expected = numpyro.deterministic(
        #     "influences-stddevs-hyperior_prior",
        #     1 / hyper_hyper_coef_stddevs,
        # )
        with numpyro.plate("options", num_opts):
            # in laplace, the spread is not the stddev as in normal dist, but it is the variance
            # Scale for Laplace to match standard deviation

            hyper_coef_means = numpyro.sample(
                "influences-mean-hyperior",
                npdist.Laplace(0, hyper_hyper_regulariation_strength),
            )
            # hyper_coef_stddevs_decentered = numpyro.sample(
            #     "influences-stddevs-hyperior-decentered",
            #     npdist.HalfNormal(1.0),
            # )
            # hyper_coef_stddevs = numpyro.deterministic(
            #     "influences-stddevs-hyperior",
            #     hyper_coef_stddevs_decentered * hyper_hyper_coef_stddevs,
            # )

            hyper_coef_stddevs = numpyro.sample(
                "influences-stddevs-hyperior",
                npdist.Exponential(coefs_hyperior_expected),
            )
        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample(
                    "influences",
                    npdist.Normal(hyper_coef_means, hyper_coef_stddevs),
                )

        # error_hyper_prior = numpyro.sample(
        #     "error-hyper-rescaled",
        #     npdist.Exponential(1 / self.error_expectation),
        # )

        error_hyper = numpyro.sample(
            "error-hyper", npdist.Exponential(1 / self.error_expectation)
        )
        #### ERROR
        with numpyro.plate("workloads", n_workloads):
            error_var_decent = numpyro.sample(
                "error_decentered", npdist.HalfNormal(1.0)
            )
            error_var = numpyro.deterministic("error", error_var_decent * error_hyper)

        # with numpyro.plate("workloads", n_workloads):
        #     error_var = numpyro.sample(
        #         "error", npdist.HalfNormal(self.error_expectation)
        #     )

        right_error = error_var[workloads]
        respective_influences = rnd_influences[workloads]
        respective_bases = base[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel() + respective_bases

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample(
                "observations", npdist.Normal(result_arr, right_error), obs=reference_y
            )
        return obs


class MCMCPartialHorseshoe(MCMCMultilevelPartial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-partial-pooling-horseshoe"

    def get_reparam_dict(self):
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
            "influences-mean-hyperior": LocScaleReparam(0),
        }
        return reparam_config

    def model(self, data, workloads, n_workloads, reference_y):
        err_expectation = 1.0
        num_opts = data.shape[1]
        coefs_expected_stddev_change_over_envs = 1.0
        coefs_hyperior_expected = 1 / coefs_expected_stddev_change_over_envs
        student_prior = npdist.LeftTruncatedDistribution(
            npdist.StudentT(5, scale=1.0), low=0.0
        )
        base_hyper = numpyro.sample("base-hyper", npdist.Normal(-0.5, 1.0))
        base_hyper_std = numpyro.sample("base-hyper-std", npdist.Exponential(1 / 0.5))

        with numpyro.plate("workloads", n_workloads):
            bases = numpyro.sample("base", npdist.Normal(base_hyper, base_hyper_std))

        tau = numpyro.sample("tau", student_prior)

        with numpyro.plate("options", num_opts):
            lambdas = numpyro.sample("lambdas", student_prior)
            horseshoe_sigma = tau**2 * lambdas**2
            hyper_coef_means = numpyro.sample(
                "influences-mean-hyperior",
                npdist.Normal(0, horseshoe_sigma),
            )
            hyper_coef_stddevs = numpyro.sample(
                "influences-stddevs-hyperior",
                npdist.Exponential(1 / coefs_expected_stddev_change_over_envs),
            )

        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample(
                    "influences",
                    npdist.Normal(hyper_coef_means, hyper_coef_stddevs),
                )

        with numpyro.plate("workloads", n_workloads):
            error_var = numpyro.sample("error", npdist.Exponential(1 / err_expectation))

        right_error = error_var[workloads]
        respective_influences = rnd_influences[workloads]
        respective_bases = bases[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel() + respective_bases

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample(
                "observations", npdist.Normal(result_arr, right_error), obs=reference_y
            )
        return obs


def half_student_t(scale):
    return npdist.LeftTruncatedDistribution(npdist.StudentT(3, scale=scale), low=0.0)


class MCMCRHS(MCMCPartialHorseshoe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-partial-pooling-RHS"
        self.influential_opt_share = 0.2

    def get_reparam_dict(self):
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
            "influences-mean-hyperior": LocScaleReparam(0),
        }
        return reparam_config

    def model(self, data, workloads, n_workloads, reference_y):
        ##### MAGIC NUMBERS ######
        num_train_samples = len(workloads)
        num_opts = data.shape[1]
        #### BASE ####
        base_hyper = numpyro.sample("base-hyper", npdist.Normal(*self.base_hyper_prior))
        base_hyper_std = numpyro.sample(
            "base-hyper-std", npdist.Exponential(1 / self.base_hyper_exp_std)
        )

        with numpyro.plate("workloads", n_workloads):
            bases = numpyro.sample("base", npdist.Normal(base_hyper, base_hyper_std))

        error_hyper = numpyro.sample(
            "error-hyper", npdist.Exponential(1 / self.error_expectation)
        )
        #### ERROR
        with numpyro.plate("workloads", n_workloads):
            error_var_decent = numpyro.sample(
                "error_decentered", npdist.HalfNormal(1.0)
            )
            error_var = numpyro.deterministic("error", error_var_decent * error_hyper)

            # error_var = numpyro.sample(
            #     "error", npdist.Exponential(1 / self.error_expectation)
            # )

        #### HYPERIOR HORSESHOE
        D = num_opts
        D0 = self.influential_opt_share * num_opts
        v_DOF = 5
        s2 = jnp.var(data[:, 0])
        # OLD:  tau = numpyro.sample("tau", student_prior)
        sigma = error_var.mean()  # 1.0
        # Step 1: Define priors for global and local shrinkage parameters
        num_inluential = numpyro.sample("num_influential", npdist.HalfNormal(D0))
        share_infl = numpyro.deterministic(
            "share_influential", num_inluential / num_opts
        )
        tau_decentered = numpyro.sample(
            "tau_decentered",
            npdist.LeftTruncatedDistribution(
                npdist.StudentT(v_DOF, scale=1.0), low=0.0
            ),
        )

        tau = numpyro.deterministic(
            "tau",
            tau_decentered
            * (
                num_inluential
                / (num_opts - num_inluential)
                * sigma
                / jnp.sqrt(num_train_samples)
            ),
        )

        with numpyro.plate("options", num_opts):
            lambdas = numpyro.sample(
                "lambdas",
                npdist.LeftTruncatedDistribution(
                    npdist.StudentT(v_DOF, scale=D), low=0.0
                ),
            )

        # Step 2: Specify prior for the auxiliary variable c^2 (Gaussian slab)
        c2 = numpyro.sample("c2", npdist.InverseGamma(v_DOF / 2, v_DOF * s2 / 2))

        # Step 3: Define tilde{lambda}
        # lambda_tilde = jnp.sqrt(c2) / (tau + (c2 / lambda_) ** 2)
        lambda_tilde = lambdas * jnp.sqrt(c2 / (c2 + tau**2 * lambdas**2))

        # # Step 4: Reparameterization for half-Cauchy
        # z = numpyro.sample("z", npdist.Normal(0.0, 1.0))
        # beta = numpyro.deterministic("beta", z * tau * lambda_tilde)
        hyper_hyper_coef_stddevs = numpyro.sample(
            "influences-stddevs-hyperior_prior",
            npdist.Exponential(1 / self.coefs_expected_stddev_change_over_envs),
        )

        # hyper_hyper_coef_stddevs_expected = numpyro.deterministic(
        #     "influences-stddevs-hyperior_prior",
        #     1 / hyper_hyper_coef_stddevs,
        # )

        ##### HYPERIOR
        with numpyro.plate("options", num_opts):
            horseshoe_sigma = tau * lambda_tilde
            hyper_coef_means = numpyro.sample(
                "influences-mean-hyperior",
                npdist.Normal(0, horseshoe_sigma),
            )
            hyper_coef_stddevs_decentered = numpyro.sample(
                "influences-stddevs-hyperior-decentered",
                npdist.HalfNormal(1.0),
            )
            hyper_coef_stddevs = numpyro.deterministic(
                "influences-stddevs-hyperior",
                hyper_coef_stddevs_decentered * hyper_hyper_coef_stddevs,
            )

        ###### INFLUENCES
        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample(
                    "influences",
                    npdist.Normal(hyper_coef_means, hyper_coef_stddevs),
                )

        ##### MISC

        right_error = error_var[workloads]
        respective_influences = rnd_influences[workloads]
        respective_bases = bases[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel() + respective_bases

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample(
                "observations", npdist.Normal(result_arr, right_error), obs=reference_y
            )
        return obs


class MCMCPartialSelfStandardizing(MCMCMultilevelPartial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-selfstd"
        self.printed_priors = False

    def get_reparam_dict(self):
        reparam_config = {
            "relative-influences": LocScaleReparam(0),
            # "base": LocScaleReparam(0),
            # "influences-mean-hyperior": LocScaleReparam(0),
            # "influences-stddevs-hyperior": LocScaleReparam(0),
            # "influences": LocScaleReparam(0),
        }
        return reparam_config

    def model(self, data, workloads, n_workloads, reference_y):
        err_expectation = 1.0
        err_hyperior_expectation = 1 / err_expectation
        err_exponential_pdf_rate = 1 / err_hyperior_expectation
        joint_coef_stdev = 0.5  # 0.5  # 2 * y_order_of_magnitude
        num_opts = data.shape[1]

        if reference_y is None:
            y_means = jnp.array([1] * n_workloads)
            y_stds = smallest_vals_dists = relative_variances = y_mins = y_means

            y_means_np = y_stds_np = relative_variances_np = y_mins_np = np.array(
                [1] * n_workloads
            )
            coefs_exp_stddev_change_over_envs = 1
        else:
            y_df = pd.DataFrame({"w": workloads, "y": reference_y})
            workload_groups = y_df.groupby(["w"])
            y_means_np = np.atleast_1d(workload_groups.mean().values.squeeze())
            y_means = jnp.array(y_means_np)
            y_mins_np = np.atleast_1d(workload_groups.min().values.squeeze())
            y_mins = jnp.array(y_mins_np)
            y_stds_np = np.atleast_1d(workload_groups.std().values.squeeze())
            if np.any(np.isnan(y_stds_np)):
                # only one sample 0_O
                y_stds_np = y_mins_np

            y_stds = jnp.array(y_stds_np)
            smallest_vals_dists = workload_groups["y"].apply(smallest_distance).values
            default_rel_std = 0.1
            alternative_base_std = y_mins_np * default_rel_std
            zero_maks = smallest_vals_dists == 0.0
            smallest_vals_dists[zero_maks] = alternative_base_std[zero_maks]
            relative_variances_np = y_stds_np / y_means_np

            relative_variances = y_stds_np / y_means_np
        ratio_of_influential_options = 1.0
        coefs_exp_rel_infl_scaled = jnp.mean(
            ratio_of_influential_options * relative_variances
        )
        coefs_exp_rel_infl_scaled_np = np.mean(
            ratio_of_influential_options * relative_variances
        )

        coefs_exp_stddev_change_over_envs = jnp.std(relative_variances)
        # coefs_exp_stddev_change_over_envs_np = np.std(relative_variances)
        # coefs_hyperior_expected = 1 / coefs_exp_stddev_change_over_envs
        # coefs_hyperior_expected_np = 1 / coefs_exp_stddev_change_over_envs_np

        #### BASE ####
        if not self.printed_priors and self.plot:
            print("base TEST")
            print("mean_y per env", y_means_np)
            print("min_y per env", y_mins_np)
            print("dist between two smallest y vals", smallest_vals_dists)
            print("y_std", y_stds_np)
            print("num opts", num_opts)
            df = pd.DataFrame({"y": reference_y, "workload": workloads})
            g = sns.FacetGrid(df, col="workload", sharex=False, col_wrap=3)
            g.map(sns.histplot, "y", kde=True)
            plt.show()
        with numpyro.plate("workloads", n_workloads):
            base = numpyro.sample(
                "base",
                npdist.TruncatedNormal(y_means_np - y_stds_np, y_stds_np, low=0.0),
                # npdist.Exponential(y_mins_np),
            )
        with numpyro.plate("workloads", n_workloads):
            variance = numpyro.sample(
                "variance",
                # npdist.HalfCauchy(
                #     1 / y_stds_np,
                # ),
                # npdist.Exponential(
                # 1 / y_stds_np,
                # ),
                npdist.TruncatedNormal(y_stds_np, y_stds_np / 50, low=0.0),
                # npdist.TruncatedNormal(y_stds_np, 1, low=0.0),
            )

        # std_std = numpyro.sample("standardization_std", npdist.Exponential(base_order_of_magnitude_std))

        #### HYPERIORS ####
        # if not self.printed_priors:
        #     print("influences-mean-hyperior", coefs_exp_rel_infl_scaled_np)
        #     print("influences-stddevs-hyperior", coefs_hyperior_expected_np)
        with numpyro.plate("options", num_opts):
            hyper_coef_means = numpyro.sample(
                "influences-mean-hyperior",
                npdist.Laplace(0, 1 / np.sqrt(2)),  # coefs_exp_rel_infl_scaled_np),
                # npdist.Exponential(0.8),
            )
            hyper_coef_stddevs = numpyro.sample(
                "influences-stddevs-hyperior",
                npdist.Exponential(1.0),  # coefs_hyperior_expected_np),
            )

        ### RELATIVE INFLUENCES ####
        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample(
                    "relative-influences",
                    npdist.Normal(hyper_coef_means, hyper_coef_stddevs),
                )

        rnd_influences_absolute = numpyro.deterministic(
            "influences",
            rnd_influences * variance[:, jnp.newaxis],
        )

        # with numpyro.plate("options", num_opts):
        #     hyper_coef_means = numpyro.sample(
        #         "influences-mean-hyperior",
        #         npdist.Laplace(0, y_stds_np * 2),  # coefs_exp_rel_infl_scaled_np),
        #         # npdist.Exponential(0.8),
        #     )
        #     hyper_coef_stddevs = numpyro.sample(
        #         "influences-stddevs-hyperior",
        #         npdist.Exponential(
        #             1 / np.std(y_stds_np)
        #         ),  # coefs_hyperior_expected_np),
        #     )

        ### EXTRA ABS INFLUENCES ###
        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                extra_abs_influences = numpyro.sample(
                    "influences-const",
                    npdist.Laplace(
                        0,
                        jnp.repeat(
                            y_stds_np[:, jnp.newaxis] / np.sqrt(2), num_opts, axis=1
                        ),
                    ),
                )

        # multiplication checks out, try with 5 options and 4 WL: np.arange(5*4).reshape(4,5)* np.arange(1,5)[:, np.newaxis]
        with numpyro.plate("workloads", n_workloads):
            error_var = numpyro.sample(
                "error",
                npdist.Exponential(1 / y_stds_np),
            )

        scaled_error = error_var  # * variance
        right_error = scaled_error[workloads]
        respective_influences = extra_abs_influences[workloads]
        respective_abs_influences = rnd_influences_absolute[workloads]
        respective_bases = base[workloads]
        # result_arr = jnp.multiply(data, respective_influences)
        result_arr_extra_absolute = jnp.multiply(data, respective_abs_influences)
        result_arr_const = jnp.multiply(data, respective_influences)
        result_arr = (
            # result_arr.sum(axis=1).ravel() +
            +result_arr_extra_absolute.sum(axis=1).ravel()
            + result_arr_const.sum(axis=1).ravel()
            + respective_bases
        )

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample(
                "observations", npdist.Normal(result_arr, right_error), obs=reference_y
            )

        self.printed_priors = True
        return obs


class MCMCPartialSelfStandardizingConstInfl(MCMCPartialSelfStandardizing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-selfstd-const-hyper"

    def get_reparam_dict(self):
        reparam_config = {
            "relative-influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
        }
        return reparam_config

    def model(self, data, workloads, n_workloads, reference_y):
        num_opts = data.shape[1]

        if reference_y is None:
            y_means = jnp.array([1] * n_workloads)
            y_means_np = y_stds_np = relative_variances_np = y_mins_np = np.array(
                [1] * n_workloads
            )
        else:
            y_df = pd.DataFrame({"w": workloads, "y": reference_y})
            workload_groups = y_df.groupby(["w"])
            y_means_np = np.atleast_1d(workload_groups.mean().values.squeeze())
            y_mins_np = np.atleast_1d(workload_groups.min().values.squeeze())
            y_stds_np = np.atleast_1d(workload_groups.std().values.squeeze())
            if np.any(np.isnan(y_stds_np)):
                # only one sample 0_O
                y_stds_np = y_mins_np
            smallest_vals_dists = workload_groups["y"].apply(smallest_distance).values
            default_rel_std = 0.1
            alternative_base_std = y_mins_np * default_rel_std
            zero_maks = smallest_vals_dists == 0.0
            smallest_vals_dists[zero_maks] = alternative_base_std[zero_maks]

        #### BASE ####
        if not self.printed_priors and self.plot:
            print("base TEST")
            print("mean_y per env", y_means_np)
            print("min_y per env", y_mins_np)
            print("y_std", y_stds_np)
            print("num opts", num_opts)
            df = pd.DataFrame({"y": reference_y, "workload": workloads})
            # g = sns.FacetGrid(df, col="workload", sharex=False, col_wrap=3)
            # g.map(sns.histplot, "y", kde=True)
            # plt.show()

        base_hyper = numpyro.sample("base-hyper", npdist.Normal(*self.base_hyper_prior))
        base_hyper_std = numpyro.sample(
            "base-hyper-std", npdist.Exponential(1 / self.base_hyper_exp_std)
        )

        with numpyro.plate("workloads", n_workloads):
            base = numpyro.sample(
                "base",
                npdist.Normal(
                    y_means_np + base_hyper * y_means_np,
                    y_stds_np * base_hyper_std,
                    # low=0.0,
                ),
            )
        with numpyro.plate("workloads", n_workloads):
            variance = numpyro.sample(
                "variance",
                npdist.TruncatedNormal(y_stds_np, y_stds_np / 2.0, low=0.0),
            )
        #### HYPERIORS ####
        # hyper_hyper_regulariation_strength = numpyro.sample(
        #     "influences-regularization-strength",
        #     npdist.Exponential(1),  # / scale_laplace),
        # )

        with numpyro.plate("options", num_opts):
            hyper_coef_means = numpyro.sample(
                "influences-mean-hyperior",
                npdist.Laplace(0, self.joint_coef_stdev / np.sqrt(2)),  # / np.sqrt(2)),
            )
            hyper_coef_stddevs = numpyro.sample(
                "influences-stddevs-hyperior",
                npdist.Exponential(1 / self.coefs_expected_stddev_change_over_envs),
            )

        ### RELATIVE INFLUENCES ####
        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample(
                    "relative-influences",
                    npdist.Normal(hyper_coef_means, hyper_coef_stddevs),
                )

        rnd_influences_absolute = numpyro.deterministic(
            "influences",
            rnd_influences * variance[:, jnp.newaxis],
        )

        with numpyro.plate("options", num_opts):
            extra_abs_influences = numpyro.sample(
                "influences-const",
                npdist.Laplace(0, np.min(y_stds_np)),  # / np.sqrt(2),
            )

        ##### ERROR
        error_hyper = numpyro.sample(
            "error-hyper", npdist.Exponential(1 / self.error_expectation)
        )
        # multiplication checks out, try with 5 options and 4 WL: np.arange(5*4).reshape(4,5)* np.arange(1,5)[:, np.newaxis]
        with numpyro.plate("workloads", n_workloads):
            error_exp_value = y_stds_np
            # error_var = numpyro.sample(
            #     "error",
            #     npdist.Exponential(1 / error_exp_value),
            # )
            error_var_decent = numpyro.sample(
                "error-decentered",
                npdist.HalfNormal(1.0),
            )
            error_var = numpyro.deterministic(
                "error", error_var_decent * error_hyper * error_exp_value
            )

        scaled_error = error_var  # * variance
        right_error = scaled_error[workloads]
        respective_influences = extra_abs_influences  # [workloads]
        respective_abs_influences = rnd_influences_absolute[workloads]
        respective_bases = base[workloads]
        # result_arr = jnp.multiply(data, respective_influences)
        result_arr_extra_absolute = jnp.multiply(data, respective_abs_influences)
        result_arr_const = jnp.multiply(data, respective_influences)
        result_arr = (
            # result_arr.sum(axis=1).ravel() +
            +result_arr_extra_absolute.sum(axis=1).ravel()
            + result_arr_const.sum(axis=1).ravel()
            + respective_bases
        )

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample(
                "observations", npdist.Normal(result_arr, right_error), obs=reference_y
            )

        self.printed_priors = True
        return obs


def smallest_distance(group):
    smallest_values = group.nsmallest(2)
    if len(smallest_values) < 2:
        return None  # or 0, if you prefer to consider single-value groups as having zero distance
    return smallest_values.iloc[1] - smallest_values.iloc[0]


class MCMCCombinedNoPooling(MCMCMultilevelPartial):
    pooling_cat = NO_POOLING

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-no-pooling-combined"

    def model(self, data, workloads, n_workloads, reference_y):
        err_expectation = 1.0
        num_opts = data.shape[1]
        expected_coef_stdev = 0.2
        scale_laplace = expected_coef_stdev / np.sqrt(2)
        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample(
                    "influences",
                    npdist.Laplace(0, scale_laplace),
                )
        with numpyro.plate("workloads", n_workloads):
            error_var = numpyro.sample("error", npdist.Exponential(1 / err_expectation))
        with numpyro.plate("workloads", n_workloads):
            base = numpyro.sample("base", npdist.Normal(-1, 0.5))

        right_error = error_var[workloads]
        respective_influences = rnd_influences[workloads]
        respective_bases = base[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel() + respective_bases

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample(
                "observations", npdist.Normal(result_arr, right_error), obs=reference_y
            )
        return obs

    def plot_options(self, model_names=None, var_names=None):
        var_names = (
            var_names
            if var_names is not None
            else [
                "influences",
            ]
        )
        az_data = self.get_arviz_data()
        num_plots = len(self.feature_names)
        n_cols = 4
        num_rows = (num_plots - 1) // n_cols + 1
        num_cols = min(num_plots, n_cols)
        print(
            "n_plots",
            num_plots,
            "ncols",
            n_cols,
            "num_rows",
            num_rows,
            "num_cols",
            num_cols,
        )

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6)
        )

        for i, feature_name in enumerate(self.feature_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]
            coords = {"features": [feature_name]}
            az.plot_forest(
                az_data,
                combined=True,
                var_names=var_names,
                model_names=model_names,
                kind="ridgeplot",
                hdi_prob=0.999,
                ridgeplot_overlap=3,
                linewidth=3,
                coords=coords,
                ax=ax,
            )

            ax.set_xlim(-1.5, 1.5)
            ax.set_title(f"Option influence {feature_name}")

        plt.suptitle("Influences across Envs")
        plt.tight_layout()
        time.sleep(0.1)
        plt.show()


class MCMCCombinedNoPoolingAdaptiveShrinkage(MCMCCombinedNoPooling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-no-pooling-combined-adaptive-shrinkage"

    def model(self, data, workloads, n_workloads, reference_y):
        err_expectation = 1.0
        num_opts = data.shape[1]
        expected_coef_stdev = 1.0
        scale_laplace = expected_coef_stdev / np.sqrt(2)
        hyper_hyper_regulariation_strength = numpyro.sample(
            "influences-regularization-strength",
            npdist.Exponential(1 / scale_laplace),
        )
        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample(
                    "influences",
                    npdist.Laplace(0, hyper_hyper_regulariation_strength),
                )
        with numpyro.plate("workloads", n_workloads):
            error_var = numpyro.sample("error", npdist.Exponential(1 / err_expectation))
        with numpyro.plate("workloads", n_workloads):
            base = numpyro.sample("base", npdist.Normal(-1, 0.5))

        right_error = error_var[workloads]
        respective_influences = rnd_influences[workloads]
        respective_bases = base[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel() + respective_bases

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample(
                "observations", npdist.Normal(result_arr, right_error), obs=reference_y
            )
        return obs


class MCMCCombinedCompletePooling(MCMCCombinedNoPooling):
    pooling_cat = COMPLETE_POOLING

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "mcmc-complete-pooling-combined"

    def model(self, data, workloads, n_workloads, reference_y):
        joint_coef_stdev = 0.2  # 0.25  # 2 * y_order_of_magnitude
        num_opts = data.shape[1]

        with numpyro.plate("options", num_opts):
            rnd_influences = numpyro.sample(
                "influences",
                npdist.Laplace(0, joint_coef_stdev / np.sqrt(2)),
            )

        base = numpyro.sample("base", npdist.Normal(-1, 1.0))
        result_arr = jnp.multiply(data, rnd_influences)
        result_arr = result_arr.sum(axis=1).ravel() + base
        error_var = numpyro.sample("error", npdist.Exponential(1 / 0.5))

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample(
                "observations", npdist.Normal(result_arr, error_var), obs=reference_y
            )
        return obs

    def get_arviz_data(self, mcmc=None):
        mcmc = mcmc or self.mcmc
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
        numpyro_data = az.from_numpyro(mcmc, **kwargs)
        return numpyro_data


class ExtraStandardizingSimpleHyperHyper(MCMCMultilevelPartial):
    pooling_cat = PARTIAL_POOLING

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.model_id = "mcmc-partial_pooling_with_hyper"

    def get_reparam_dict(self):
        reparam_config = {
            "influences": LocScaleReparam(0),
            "base": LocScaleReparam(0),
            "influences-mean-hyperior": LocScaleReparam(0),
        }
        return reparam_config

    def model(self, data, workloads, n_workloads, reference_y):
        err_expectation = 0.3
        err_hyperior_expectation = 1 / err_expectation
        err_exponential_pdf_rate = 1 / err_hyperior_expectation
        joint_coef_stdev = 0.5  # 0.5  # 2 * y_order_of_magnitude
        num_opts = data.shape[1]
        coefs_expected_stddev_change_over_envs = 0.1  # 0.5
        coefs_hyperior_expected = 1 / coefs_expected_stddev_change_over_envs
        coefs_stds_prior_exp_rate = 1 / coefs_hyperior_expected

        hyper_hyper_coef_locations = numpyro.sample(
            "influences-mean-hyper-hyperior",
            npdist.Cauchy(0, 1),
        )

        with numpyro.plate("options", num_opts):
            # hyper_coef_means = numpyro.sample("influences-mean-hyperior", npdist.Normal(0, joint_coef_stdev), )
            hyper_coef_means = numpyro.sample(
                "influences-mean-hyperior",
                npdist.Normal(-1, hyper_hyper_coef_locations),
            )
            hyper_coef_stddevs = numpyro.sample(
                "influences-stddevs-hyperior",
                npdist.Exponential(coefs_hyperior_expected),
            )

        # hyper_base_mean = numpyro.sample("base-mean-hyperior", npdist.Normal(0, joint_coef_stdev), )
        # hyper_base_stddev = numpyro.sample("base-stddevs-hyperior", npdist.Exponential(stddev_exp_prior), )

        with numpyro.plate("options", num_opts):
            with numpyro.plate("workloads", n_workloads):
                rnd_influences = numpyro.sample(
                    "influences",
                    npdist.Normal(hyper_coef_means, hyper_coef_stddevs),
                )

        # with numpyro.plate("workloads", n_workloads):
        #     bases = numpyro.sample("base", npdist.Normal(hyper_base_mean, hyper_base_stddev))

        # error_hyperior = numpyro.sample("error-hyperior", npdist.Exponential(err_exponential_pdf_rate))
        # error_hyperior = numpyro.sample("error-hyperior", npdist.Gamma(2,0.1))
        # error_hyperior = numpyro.sample("error-hyperior", npdist.HalfNormal(1))
        with numpyro.plate("workloads", n_workloads):
            error_var = numpyro.sample("error", npdist.Exponential(1 / err_expectation))

        right_error = error_var[workloads]
        respective_influences = rnd_influences[workloads]
        # respective_bases = bases[workloads]
        result_arr = jnp.multiply(data, respective_influences)
        result_arr = result_arr.sum(axis=1).ravel()  # + respective_bases

        with numpyro.plate("data", result_arr.shape[0]):
            obs = numpyro.sample(
                "observations", npdist.Normal(result_arr, right_error), obs=reference_y
            )
        return obs


def get_pairwise_lasso_reg(lasso_alpha=0.0001):
    pairwise_mapper = PolynomialFeatures(
        degree=2, include_bias=False, interaction_only=True
    )
    lin_reg = Lasso(alpha=lasso_alpha, max_iter=5000)
    # lin_reg = LinearRegression()
    # feature_selector = SelectFromModel(lin_reg, threshold=None)

    pipeline = Pipeline(
        [
            ("pairwise_mapper", pairwise_mapper),
            # ('feature_selector', feature_selector),
            ("lin_reg", lin_reg),
        ]
    )

    return pipeline


class NoPoolingEnvModel(ExperimentationModelBase):
    pooling_cat = NO_POOLING

    def __init__(self, model_prototype, preprocessings=None):
        super().__init__(preprocessings)
        self.model_prototype = model_prototype
        self.existing_models = {}

    def get_env_model(self, env_id, train_data=None):
        if env_id not in self.existing_models:
            new_model = copy.deepcopy(self.model_prototype)
            self.existing_models[env_id] = new_model, train_data

        return self.existing_models[env_id]

    def _fit(self, data: List[SingleEnvData], *args, **kwargs):
        for single_env_data in data:
            env_id = single_env_data.env_id
            reg, _ = self.get_env_model(env_id, single_env_data)
            X = single_env_data.get_X()
            y = single_env_data.get_y()
            reg.fit(X, y)

    def _predict(self, data: List[SingleEnvData], *args, **kwargs):
        preds = []
        for single_env_data in data:
            env_id = single_env_data.env_id
            reg, train_data = self.get_env_model(env_id)
            X = single_env_data.get_X()
            pred = reg.predict(X)
            preds.append(pred)
        return preds

    def get_pooling_cat(self):
        return self.pooling_cat

    def evaluate(self, eval):
        eval.add_mape()
        eval.add_R2()
        cost_df = self.get_cost_dict()
        eval.add_custom_model_dict(cost_df)
        return eval


class CompletePoolingEnvModel(ExperimentationModelBase):
    pooling_cat = COMPLETE_POOLING

    def __init__(self, model_prototype, preprocessings=None):
        super().__init__(preprocessings)
        self.model_prototype = model_prototype
        self.pooled_model = copy.deepcopy(self.model_prototype)

    def _fit(self, data: List[SingleEnvData], *args, **kwargs):
        X, envs, y = self._internal_data_splitting(data)
        feature_names = data[0].get_feature_names()
        df = pd.DataFrame(X, columns=feature_names)
        self.pooled_model.fit(df, y)

    def _predict(self, data: List[SingleEnvData], *args, **kwargs):
        X, envs, y = self._internal_data_splitting(data)
        feature_names = data[0].get_feature_names()
        df = pd.DataFrame(X, columns=feature_names)
        preds = self.pooled_model.predict(df)
        return_y_list = self._internal_data_to_list(envs, preds)
        return return_y_list

    def get_pooling_cat(self):
        return self.pooling_cat

    def evaluate(self, eval):
        eval.add_mape()
        eval.add_R2()
        cost_df = self.get_cost_dict()
        eval.add_custom_model_dict(cost_df)
        return eval
