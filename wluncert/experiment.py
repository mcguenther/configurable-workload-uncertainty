import copy
import itertools
import os.path
import random
import time
import uuid
from typing import List, Dict

import localflow as mlflow
import numpy as np

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import uuid
from analysis import ModelEvaluation
from data import SingleEnvData, Standardizer
from utils import get_date_time_uuid
import warnings
from joblib import (
    parallel_backend,
    register_parallel_backend,
    Parallel,
    delayed,
    parallel_backend,
)
from tqdm import tqdm

# MLFLOW_URI = "http://172.26.92.43:5000"
MLFLOW_USR = "jdorn"
MLFLOW_PWD = "xWpS3yWZKxDXEqFWBVh5SAz84d9uyEWuFEUs"
MLFLOW_URI = f"https://{MLFLOW_USR}:{MLFLOW_PWD}@mlflow.server.jdbrothers.de"
# jdorn:xWpS3yWZKxDXEqFWBVh5SAz84d9uyEWuFEUs@mlflow.server.jdbrothers.de
# MLFLOW_URI = "https://mlflow.sws.informatik.uni-leipzig.de"
# EXPERIMENT_NAME = "jdorn-multilevel"
EXPERIMENT_NAME = "jdorn-artif"


SLEEP_TIME_BASE_MAX = 0.17


def get_rnd_sleep_time():
    return np.random.uniform(0.15, SLEEP_TIME_BASE_MAX)


def mlflow_log_params(*args, **kwargs):
    time.sleep(get_rnd_sleep_time())
    # print("logging parameters")
    return mlflow.log_params(*args, **kwargs)


def mlflow_log_metrics(*args, **kwargs):
    time.sleep(get_rnd_sleep_time())
    # print("logging metrics")
    return mlflow.log_metrics(*args, **kwargs)


def mlflow_log_dict(*args, **kwargs):
    time.sleep(get_rnd_sleep_time())
    # print("logging dict")
    return mlflow.log_dict(*args, **kwargs)


class ExperimentTask:
    def __init__(
        self,
        model_lbl,
        model,
        envs_lbl,
        train_list,
        test_list,
        train_size: int,
        rel_train_size=None,
        exp_id=None,
        rnd=0,
        pooling_cat=None,
    ):
        self.model = model
        self.rnd = rnd
        self.loo_wise_predictions = {}
        self.model_lbl = model_lbl
        self.envs_lbl = envs_lbl
        self.rel_train_size = rel_train_size
        self.exp_id = exp_id
        self.train_size = train_size
        self.train_list: List[SingleEnvData] = train_list
        self.test_list: List[SingleEnvData] = test_list
        self.pooling_cat = pooling_cat
        self.training_features = self.train_list[0].get_feature_names()

    def get_metadata_dict(
        self,
    ):
        d = {
            "rnd": self.rnd,
            "model": self.model_lbl,
            "train_size": self.train_size,
            "subject_system": self.envs_lbl,
            "relative_train_size": self.rel_train_size,
            "pooling_cat": self.pooling_cat,
            "exp_id": self.exp_id,
            "n_train_features": len(self.training_features),
            # "training-feature-names": self.training_features,
        }
        # artifact_file = f"tmp/feature_names-{uuid.uuid4()}.txt"
        artifact_file = f"feature_names.txt"
        mlflow_log_dict(
            {"training_feature_names": self.training_features}, artifact_file
        )
        # os.remove(artifact_file)
        return d


class ExperimentTransfer(ExperimentTask):
    label = "transfer"

    def __init__(self, *args, transfer_budgets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfer_sample_budgets = [
            3,
        ]
        self.relative_transfer_sample_budges = (0.1, 0.2)
        # we have as low as 6 workloads per system. hence, to stay comparable, we use up to 5 workloads as source workloads
        self.transfer_known_workloads = [
            1,
            3,
            5,
        ]
        self.eval_permutation_num_cap = 5

    def run(self, return_predictions=False):
        base_log_dict = self.get_metadata_dict()
        type_dict = {"experiment-type": self.label}
        any_train_data = self.train_list[0]
        n_train_samples = len(any_train_data)
        calculated_abs_transfer_budgets = {
            rel: max(int(rel * n_train_samples), 3)
            for rel in self.relative_transfer_sample_budges
        }
        log_dict = {
            "absolute_transfer_budgets": self.transfer_sample_budgets,
            "relative_to_absolute_mapped_budgets": calculated_abs_transfer_budgets,
            **type_dict,
        }
        log_dict = {**log_dict, **base_log_dict}
        mlflow_log_params(log_dict)

        loo_wise_predictions = {}
        loo_wise_standardized_test_sets = {}
        n_envs = len(self.train_list)
        env_ids = list(range(n_envs))
        joint_abs_train_samples = sorted(
            list(
                set(
                    [
                        *self.transfer_sample_budgets,
                        *calculated_abs_transfer_budgets.values(),
                    ]
                )
            )
        )
        for n_transfer_samples in joint_abs_train_samples:
            if n_train_samples <= n_transfer_samples:
                # print(
                #     f"skipping transfer budget {n_transfer_samples} because there are only {n_train_samples} total train samples!"
                # )
                continue
            name = f"n_transfer_samples-{n_transfer_samples}"
            with mlflow.start_run(nested=True, run_name=name):
                mlflow_log_params(
                    {
                        "n_transfer_samples": n_transfer_samples,
                    }
                )
                for loo_source_envs_budget in self.transfer_known_workloads:
                    training_env_sets = list(
                        itertools.combinations(env_ids, loo_source_envs_budget)
                    )
                    run_name = f"train-with-{loo_source_envs_budget}-source"
                    with mlflow.start_run(nested=True, run_name=run_name):
                        mlflow_log_params(
                            {
                                "number_of_source_envs": loo_source_envs_budget,
                            }
                        )
                        if (
                            self.eval_permutation_num_cap
                            and len(training_env_sets) > self.eval_permutation_num_cap
                        ):
                            unique_seed = hash(
                                (self.rnd, n_transfer_samples, loo_source_envs_budget)
                            )
                            rng = random.Random(unique_seed)
                            training_env_sets = rng.sample(
                                training_env_sets,
                                self.eval_permutation_num_cap,
                            )
                        for env_set in training_env_sets:
                            rem_envs = [e for e in env_ids if e not in env_set]
                            train_envs_str = "+".join(str(i) for i in env_set)
                            name = f"train-{train_envs_str}"
                            # print(f"Training on {train_envs_str}")
                            with mlflow.start_run(nested=True, run_name=name):
                                mlflow_log_params(
                                    {
                                        "source_envs": env_set,
                                    }
                                )
                                new_loo_train_list = []
                                new_loo_test_list = []
                                for i, (
                                    single_env_train_data,
                                    single_env_test_data,
                                ) in enumerate(zip(self.train_list, self.test_list)):
                                    # Determine train and test data based on whether the current environment is in the env_set
                                    if i in env_set:
                                        # Use the entire dataset for training in environments included in env_set
                                        env_train = single_env_train_data
                                        env_test = None
                                    else:
                                        # For environments not in env_set, split the data into training and testing subsets
                                        # use env id as rnd seed to get different transfer samples per env
                                        split_data = single_env_train_data.get_split(
                                            n_transfer_samples, rnd=i
                                        )
                                        env_train = split_data.train_data
                                        env_test = single_env_test_data

                                    # Append the determined train and test data to their respective lists
                                    new_loo_train_list.append(env_train)
                                    new_loo_test_list.append(env_test)
                                loo_model = copy.deepcopy(self.model)
                                loo_model.fit(new_loo_train_list)
                                predictions = loo_model.predict(new_loo_test_list)
                                self.eval_single_model(
                                    loo_model,
                                    predictions,
                                    new_loo_test_list,
                                )

    def eval_single_model(self, model, predictions, test_list):
        eval = ModelEvaluation(
            predictions,
            test_list,
        )
        eval = model.evaluate(eval)
        df: pd.DataFrame = eval.get_scores()
        return df

    def get_metadata_df(
        self,
    ):
        df = pd.DataFrame(
            [[self.model_lbl, self.train_size, self.rnd, self.envs_lbl, self.exp_id]],
            columns=["model", "budget_abs", "rnd", "subject_system", "exp_id"],
        )
        return df

    def get_id(self):
        deterministic_id = f"transfer model {self.model_lbl} on {self.envs_lbl} with {self.rel_train_size}N initial train budget"
        return deterministic_id

    def get_transfer_train_set_sizes(
        self,
        single_env_train_data: SingleEnvData,
        min_train_size_abs: int,
        relative_steps: List[float] = None,
        absolute_steps: List[float] = None,
    ):
        max_size = single_env_train_data.get_len()
        absolute_steps = absolute_steps or [int(i * max_size) for i in relative_steps]
        final_sizes = []
        for step in absolute_steps:
            if step < min_train_size_abs:  # and min_train_size_abs not in final_sizes:
                size_candidate = min_train_size_abs
            else:
                size_candidate = step

            if size_candidate not in final_sizes:
                final_sizes.append(size_candidate)
        return final_sizes

    def get_transfer_train_sets(
        self, single_env_train_data, single_env_test_data, abs_sizes
    ):
        train_sets = []
        test_sets = []
        for t_size in abs_sizes:
            if t_size >= len(single_env_train_data):
                train_set = single_env_train_data
            else:
                train_set = single_env_train_data.get_split(
                    t_size,
                ).train_data
            # s = Standardizer()
            # train_data = s.fit_transform([train_set])[0]
            # test_data = s.transform([single_env_test_data])[0]
            train_sets.append(train_set)
            test_sets.append(single_env_test_data)
        return train_sets, test_sets


class ExperimentMultitask(ExperimentTask):
    label = "multitask"

    def run(self, return_predictions=False):
        mlflow.log_params(
            {
                "rnd": self.rnd,
                "model": self.model_lbl,
                "software-system": self.envs_lbl,
                "exp_id": self.exp_id,
                "train_size": self.train_size,
                "relative_train_size": self.rel_train_size,
                "pooling_cat": self.pooling_cat,
                "experiment-type": self.label,
            }
        )
        self.model.fit(self.train_list)
        self.predictions = self.model.predict(self.test_list)
        return self.eval()

    def get_id(self):
        deterministic_id = (
            f"multitask-{self.model_lbl} on {self.envs_lbl}-trainx{self.rel_train_size}"
        )
        return deterministic_id  # self.exp_id

    def eval(self):
        meta_dict = self.get_metadata_dict()
        eval = ModelEvaluation(
            self.predictions,
            self.test_list,
        )
        eval = self.model.evaluate(eval)
        df: pd.DataFrame = eval.get_scores()

        myuuid = uuid.uuid4()

        scores_csv = os.path.abspath(
            f"tmp/multitask_scores-{self.get_id()}-{str(myuuid)}.csv"
        )
        df.to_csv(scores_csv)
        mlflow.log_artifact(scores_csv)
        os.remove(scores_csv)
        df_annotated = df.assign(**meta_dict)
        model_meta_data = eval.get_metadata()
        model_meta_data_annotated = model_meta_data.assign(**meta_dict)
        return df_annotated, model_meta_data_annotated


class Replication:
    def __init__(
        self,
        experiment_classes,
        models: Dict,
        data_providers: Dict,
        train_sizes_relative_to_option_number,
        rnds=None,
        n_jobs=False,
        replication_lbl="last-experiment",
        plot=False,
        do_transfer_task=False,
        max_test_samples_abs=None,
    ):
        self.replication_lbl = get_date_time_uuid() + "-" + replication_lbl
        self.progress_bar = None
        self.plot = plot
        self.models = models
        self.experiment_classes = experiment_classes
        self.n_jobs = n_jobs
        self.data_providers = data_providers
        self.train_sizes_relative_to_option_number = (
            train_sizes_relative_to_option_number
        )
        self.rnds = rnds if rnds is not None else [0]
        self.result = None
        self.do_transfer_task = do_transfer_task
        self.max_test_samples_abs = max_test_samples_abs
        self.parent_run_id = None
        self.experiment_name = f"uncertainty-learning-{self.replication_lbl}"

    def provision_experiment(self, args):
        model_lbl, model_proto, data_lbl, data_set, train_size, rnd = args
        print(
            f"provisioning model={model_lbl} data={data_lbl} train_size={train_size} rnd={rnd}",
            flush=True,
        )
        max_train_size = max(self.train_sizes_relative_to_option_number)
        data_per_env: List[SingleEnvData] = data_set.get_workloads_data()
        train_list = []
        test_list = []

        rng = np.random.default_rng(rnd)
        seeds = [
            rng.integers(0, 2**30, dtype=np.uint32) for _ in range(len(data_per_env))
        ]

        for i, env_data in zip(seeds, data_per_env):
            new_seed_for_env = i
            split = env_data.get_split(
                rnd=new_seed_for_env,
                n_train_samples_rel_opt_num=train_size,
                max_train_samples_rel_opt_num=max_train_size,
                max_test_samples_abs=self.max_test_samples_abs,
            )
            train_data = split.train_data
            train_list.append(train_data)
            test_list.append(split.test_data)

        abs_train_size = len(train_list[0])
        tasks = []

        for task_class in self.experiment_classes:
            model_proto_for_env = copy.deepcopy(model_proto)
            model_proto_for_env.set_envs(data_set)
            pooling_cat = model_proto_for_env.get_pooling_cat()
            new_task = task_class(
                model_lbl,
                model_proto_for_env,
                data_lbl,
                train_list,
                test_list,
                train_size=abs_train_size,
                pooling_cat=pooling_cat,
                rel_train_size=train_size,
                exp_id=self.experiment_name,
                rnd=rnd,
            )
            tasks.append(new_task)

        print(
            f"finished provisioning {len(tasks)} tasks for model={model_lbl} data={data_lbl}",
            flush=True,
        )

        return tasks

    def run(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
        run_name = self.experiment_name.replace(" ", "")
        with mlflow.start_run(run_name=run_name) as run:
            self.parent_run_id = run.info.run_id
            print(self.parent_run_id)

        tasks = {key: [] for key in self.experiment_classes}

        # Prepare arguments for parallel execution
        args_list = [
            (model_lbl, model_proto, data_lbl, data_set, train_size, rnd)
            for model_lbl, model_proto in self.models.items()
            for data_lbl, data_set in self.data_providers.items()
            for train_size in self.train_sizes_relative_to_option_number
            for rnd in self.rnds
        ]
        print(f"Starting to provision {len(args_list)} experiments", flush=True)

        # Use joblib for parallelization

        with parallel_backend("multiprocessing", n_jobs=-4):
            results = Parallel(
                # n_jobs=-4,
                verbose=1,
                batch_size=150,
                # return_as="generator_unordered",
            )(delayed(self.provision_experiment)(args) for args in args_list)

        # Flatten the results and organize tasks
        for task_list in results:
            for task in task_list:
                tasks[type(task)].append(task)

        print("Provisioned experiments", flush=True)

        random.seed(self.rnds[0])
        for task_type in tasks:
            random.shuffle(tasks[task_type])
            print(f"Planning {self.n_jobs} jobs")

            # for task in tqdm(tasks[task_type]):
            #     self.handle_task(task)

            # Use joblib for task execution
            Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(self.handle_task)(task) for task in tqdm(tasks[task_type])
            )

        print(self.parent_run_id)
        return self.parent_run_id

    def handle_task(self, task: ExperimentTask):
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
        run_name = task.get_id()
        print(f"starting task {run_name}", flush=True)
        with mlflow.start_run(run_id=self.parent_run_id):
            with mlflow.start_run(run_name=run_name, nested=True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    task.run()
        print(f"finished task {run_name}", flush=True)
        del task.model
        del task
