from typing import List, Dict
import random
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from wluncert.data import DataLoaderStandard, DataAdapterJump3r, WorkloadTrainingDataSet, SingleEnvData
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from wluncert.models import NoPoolingEnvModel, get_pairwise_lasso_reg, ExtraStandardizingSimpleModel, \
    ExtraStandardizingEnvAgnosticModel
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from tqdm import tqdm

import numpyro
numpyro.set_host_device_count(6)

class ExperimentMultitask:
    def __init__(self, model_lbl, model, envs_lbl, environments_data: WorkloadTrainingDataSet, split_args: Dict,
                 exp_id=None, rnd=0):
        self.model = model
        self.rnd = rnd
        self.environments_data = environments_data
        self.split_args_dict = split_args
        list_of_single_env_data = self.environments_data.get_workloads_data()
        self.train_list = []
        self.test_list = []
        self.predictions = []
        self.model_lbl = model_lbl
        self.envs_lbl = envs_lbl
        self.exp_id = exp_id
        for env_data in list_of_single_env_data:
            split = env_data.get_split(rnd=self.rnd, **self.split_args_dict)
            # train_data, test_data, _, _ = split.normalize()
            train_data = split.train_data
            test_data = split.test_data
            self.train_list.append(train_data)
            self.test_list.append(test_data)

    def run(self):
        # for train_data in self.train_list:
        self.model.fit(self.train_list)
        self.predictions = self.model.predict(self.test_list)
        return self.predictions

    def get_id(self):
        deterministic_id = f"{self.model_lbl} on {self.envs_lbl} -{self.exp_id}-"
        return deterministic_id

    def eval(self):
        eval = Evaluation()
        errs_df = eval.scalar_accuracy(self.test_list, self.predictions)
        errs_df["model"] = self.model_lbl
        errs_df["setting"] = self.envs_lbl
        errs_df["exp_id"] = self.exp_id
        return errs_df


class Evaluation:
    def __init__(self):
        pass

    def scalar_accuracy(self, test_sets: List[SingleEnvData], predictions):
        merged_y_true = []
        merged_predictions = []
        col_names = "err_type", "env", "err"
        tups = []
        err_type_mape = "mape"
        err_type_r2 = "R2"
        for test_set, y_pred in zip(test_sets, predictions):
            y_true = test_set.get_y()
            merged_y_true.append(y_true)
            merged_predictions.append(y_pred)
            mape = Evaluation.mape_100(y_true, y_pred)
            r2 = Evaluation.R2(y_true, y_pred)
            env_id = test_set.env_id
            tups.append((err_type_mape, env_id, mape))
            tups.append((err_type_r2, env_id, r2))
        merged_err_mape = Evaluation.mape_100(merged_y_true, merged_predictions)
        merged_err_R2 = Evaluation.R2(merged_y_true, merged_predictions)
        overall_tup_mape = err_type_mape, "overall", merged_err_mape
        overall_tup_R2 = err_type_r2, "overall", merged_err_R2
        tups.append(overall_tup_mape)
        tups.append(overall_tup_R2)
        df = pd.DataFrame(tups, columns=col_names)
        return df

    @classmethod
    def mape_100(cls, y_true, predictions):
        return mean_absolute_percentage_error(y_true, predictions) * 100

    @classmethod
    def R2(cls, y_true, predictions):
        return r2_score(y_true, predictions)


class Replication:
    def __init__(self, models: Dict, data_providers: Dict, train_sizes_relative_to_option_number, rnds=None):
        self.models = models
        self.data_providers = data_providers
        self.train_sizes_relative_to_option_number = train_sizes_relative_to_option_number
        self.rnds = rnds if rnds is not None else [0]

    def run(self):
        tasks = []
        for model_lbl, model_proto in self.models.items():
            for data_lbl, data_set in self.data_providers.items():
                for train_size in self.train_sizes_relative_to_option_number:
                    for rnd in self.rnds:
                        task = ExperimentMultitask(model_lbl, model_proto, data_lbl, data_set,
                                                   split_args={"n_train_samples_rel_opt_num": train_size},
                                                   exp_id=train_size,
                                                   rnd=rnd)
                        tasks.append(task)

        print("provisioned experiments", flush=True)
        eval_dfs = []
        progress_bar = tqdm(total=len(tasks), desc="Running tasks", unit="task")
        random.seed(self.rnds[0])
        random.shuffle(tasks)
        for task in tasks:
            task.run()
            errs = task.eval()
            eval_dfs.append(errs)
            progress_bar.update(1)
        progress_bar.close()
        merged_df = pd.concat(eval_dfs)
        return merged_df


def main():
    print("loading data")
    path_jump3r = "./training-data/jump3r.csv"
    jump3r_data_raw = DataLoaderStandard(path_jump3r)
    data_jump3r = DataAdapterJump3r(jump3r_data_raw)
    wl_data: WorkloadTrainingDataSet = data_jump3r.get_wl_data()

    # known_env_data, target_data = wl_data.get_loo_wl_data()
    # example_split = target_data[0].get_split(20)
    # example_split.normalize()
    data_providers = {"jump3r": wl_data}

    print("loaded data")

    rf_proto = RandomForestRegressor()
    model_rf = NoPoolingEnvModel(rf_proto)

    pairwise_reg_proto = get_pairwise_lasso_reg()
    model_pairwise_reg = NoPoolingEnvModel(pairwise_reg_proto)

    lin_reg_proto = LinearRegression()
    model_lin_reg = NoPoolingEnvModel(lin_reg_proto)

    dummy_proto = DummyRegressor()
    model_dummy = NoPoolingEnvModel(dummy_proto)

    mcmc_no_pooling_proto = ExtraStandardizingEnvAgnosticModel()
    model_mcmc_no_pooling = NoPoolingEnvModel(mcmc_no_pooling_proto)

    model_partial_extra_standardization = ExtraStandardizingSimpleModel()
    models = {
        "partial-pooling-mcmc-extra": model_partial_extra_standardization,
        # "mcmc-no-pooling": model_mcmc_no_pooling,
        # "no-pooling-rf": model_rf,
        # "no-pooling-lin": model_lin_reg,
        # "no-pooling-dummy": model_dummy,
        #
        # # # "no-pooling-pairwise": model_pairwise_reg,
    }


    print("created models")
    # train_sizes = 1, 2, 5#3, 5,
    train_sizes = 1, 2, 4
    rnds = list(range(3))

    rep = Replication(models, data_providers, train_sizes, rnds)
    errs = rep.run()
    # exp = ExperimentMultitask("no-pooling-rf", model, "jump3r", wl_data, split_args=20)
    # exp.run()
    # # eval = Evaluation()
    # errs = exp.eval()
    # print(errs.head())
    errs.to_csv("./results/last_experiment.csv")


if __name__ == "__main__":
    main()
