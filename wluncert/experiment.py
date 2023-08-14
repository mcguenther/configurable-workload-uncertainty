import copy
import os.path
import random
from typing import List, Dict

import mlflow
from mlflow.models import ModelSignature, infer_signature
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from analysis import ModelEvaluation
from data import SingleEnvData, Standardizer
from utils import get_date_time_uuid


class ExperimentTask:
    pass


class ExperimentTransfer(ExperimentTask):
    label = "transfer"

    def __init__(
        self,
        model_lbl,
        model,
        envs_lbl,
        train_list,
        test_list,
        train_size: int,
        exp_id=None,
        rnd=0,
        pooling_cat=None,
    ):
        self.loo_wise_predictions = None
        self.model = model
        self.rnd = rnd
        self.predictions = []
        self.model_lbl = model_lbl
        self.envs_lbl = envs_lbl
        self.exp_id = exp_id
        self.train_size = train_size
        self.train_list: List[SingleEnvData] = train_list
        self.test_list: List[SingleEnvData] = test_list
        self.pooling_cat = pooling_cat

    def run(self, return_predictions=False):
        loo_wise_predictions = {}
        for loo_idx in range(len(self.train_list)):

            single_env_test_data = self.test_list[loo_idx]
            single_env_train_data = self.train_list[loo_idx]
            abs_train_sizes = self.get_transfer_train_set_sizes(
                single_env_train_data,
                min_train_size_abs=3,
                relative_steps=[0.0, 0.25, 0.5, 0.75, 1.0],
            )
            sub_train_sets, standardized_test_sets = self.get_transfer_train_sets(
                single_env_train_data, single_env_test_data, abs_sizes=abs_train_sizes
            )
            for train_subset, standardized_test_set in zip(
                sub_train_sets, standardized_test_sets
            ):
                new_loo_train_list = []
                new_loo_test_list = []
                for i, (single_env_train_data, single_env_test_data) in enumerate(
                    zip(self.train_list, self.test_list)
                ):
                    train_data = single_env_train_data if i != loo_idx else train_subset
                    new_loo_train_list.append(train_data)
                    # test_data = single_env_test_data if i != loo_idx else standardized_test_set
                    # new_loo_test_list.append(test_data)
                self.model.fit(new_loo_train_list)
                predictions = self.model.predict([standardized_test_set])
                loo_wise_predictions[loo_idx] = predictions
        self.loo_wise_predictions = [
            loo_wise_predictions[i] for i in sorted(loo_wise_predictions.keys())
        ]

    def get_metadata_df(
        self,
    ):
        df = pd.DataFrame(
            [[self.model_lbl, self.train_size, self.rnd, self.envs_lbl, self.exp_id]],
            columns=["model", "budget_abs", "rnd", "subject_system", "exp_id"],
        )
        return df

    def get_id(self):
        deterministic_id = f"{self.model_lbl} on {self.envs_lbl} -{self.exp_id}-"
        return deterministic_id

    def eval(self):
        meta_df = self.get_metadata_df()
        values = meta_df.iloc[0].values
        columns = meta_df.columns

        eval = ModelEvaluation(
            self.loo_wise_predictions, self.test_list, self.train_list, meta_df
        )
        eval = self.model.evaluate(eval)
        df = eval.get_scores()
        df_annotated = df.assign(**dict(zip(columns, values)))
        model_meta_data = eval.get_metadata()
        model_meta_data_annotated = model_meta_data.assign(**dict(zip(columns, values)))
        # for env_predictions, test_data, train_data in zip(self.predictions, self.test_list, self.train_list):
        #
        # errs_df = eval.scalar_accuracy(self.test_list, self.predictions)
        # errs_df["model"] = self.model_lbl
        # errs_df["setting"] = self.envs_lbl
        # errs_df["exp_id"] = self.exp_id
        return df_annotated, model_meta_data_annotated

    def get_transfer_train_set_sizes(
        self,
        single_env_train_data: SingleEnvData,
        min_train_size_abs: int,
        relative_steps: List[float],
    ):
        max_size = single_env_train_data.get_len()
        absolute_steps = [int(i * max_size) for i in relative_steps]
        final_sizes = []
        for step in absolute_steps:
            if step < min_train_size_abs and min_train_size_abs not in final_sizes:
                final_sizes.append(min_train_size_abs)
            else:
                final_sizes.append(step)
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
                train_set = single_env_train_data.get_split(t_size).train_data
            # s = Standardizer()
            # train_data = s.fit_transform([train_set])[0]
            # test_data = s.transform([single_env_test_data])[0]
            train_sets.append(train_set)
            test_sets.append(single_env_test_data)
        return train_sets, test_sets


class ExperimentMultitask(ExperimentTask):
    label = "multitask"

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

    def get_metadata_df(
        self,
    ):
        df = pd.DataFrame(
            [[self.model_lbl, self.train_size, self.rnd, self.envs_lbl, self.exp_id]],
            columns=["model", "budget_abs", "rnd", "subject_system", "exp_id"],
        )
        return df

    def get_id(self):
        deterministic_id = (
            f"multitask-{self.model_lbl} on {self.envs_lbl}-trainx{self.rel_train_size}"
        )
        return deterministic_id  # self.exp_id

    def eval(self):
        meta_df = self.get_metadata_df()
        # Get the values from the first row of df1
        values = meta_df.iloc[0].values
        # Get the column names of df1
        columns = meta_df.columns

        eval = ModelEvaluation(
            self.predictions, self.test_list, self.train_list, meta_df
        )
        eval = self.model.evaluate(eval)
        df: pd.DataFrame = eval.get_scores()
        scores_csv = os.path.abspath("tmp/multitask_scores.csv")
        df.to_csv(scores_csv)
        mlflow.log_artifact(scores_csv)
        df_annotated = df.assign(**dict(zip(columns, values)))
        model_meta_data = eval.get_metadata()
        model_meta_data_annotated = model_meta_data.assign(**dict(zip(columns, values)))
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
    ):
        self.replication_lbl = get_date_time_uuid() + "-" + replication_lbl
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

        self.experiment_name = f"uncertainty-learning-{self.replication_lbl}"

    def run(self):
        # if not mlflow.create_experiment(experiment_name):
        # mlflow.set_tracking_uri(
        #     "file:///home/jdorn/tmp/workload-uncertainty-learning/wluncert/mlflow/"
        # )
        mlflow.set_experiment(experiment_name=self.experiment_name)

        tasks = {key: [] for key in self.experiment_classes}
        transfer_tasks = []
        for model_lbl, model_proto in self.models.items():
            for data_lbl, data_set in self.data_providers.items():
                model_proto_for_env = copy.deepcopy(model_proto)
                model_proto_for_env.set_envs(data_set)
                for train_size in self.train_sizes_relative_to_option_number:
                    for rnd in self.rnds:

                        data_per_env: List[
                            SingleEnvData
                        ] = data_set.get_workloads_data()
                        train_list = []
                        test_list = []
                        for env_data in data_per_env:
                            split = env_data.get_split(
                                rnd=rnd, n_train_samples_rel_opt_num=train_size
                            )
                            train_data = split.train_data
                            train_list.append(train_data)
                            test_list.append(env_data)
                        abs_train_size = len(train_list[0])
                        pooling_cat = model_proto_for_env.get_pooling_cat()
                        for task_class in self.experiment_classes:
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
                            tasks[task_class].append(new_task)

        print("provisioned experiments", flush=True)

        random.seed(self.rnds[0])
        results = {key: [] for key in tasks}
        scores_list = []
        metas_list = []
        result_dict = {}
        for task_type in tasks:
            random.shuffle(tasks[task_type])
            if self.n_jobs:
                new_result = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.handle_task)(task) for task in tqdm(tasks[task_type])
                )
                results[task_type] = new_result
            else:
                progress_bar = tqdm(
                    total=len(tasks),
                    desc="Running multitask learning tasks",
                    unit="task",
                )
                for type_wise_tasks in tasks[task_type]:
                    new_result = self.handle_task(type_wise_tasks)
                    results[task_type].append(new_result)
                    progress_bar.update(1)
                progress_bar.close()

            eval_scores = [scores for scores, metas in results[task_type]]
            eval_metas = [metas for scores, metas in results[task_type]]
            merged_df_scores = pd.concat(eval_scores)
            merged_df_scores["experiment"] = task_type.label
            merged_df_metas = pd.concat(eval_metas)
            merged_df_metas["experiment"] = task_type.label
            scores_list.append(merged_df_scores)
            metas_list.append(merged_df_metas)
            task_id = task_type.label
            result_dict[task_id] = {"scores": merged_df_scores, "meta": merged_df_metas}
        # merged_scores = pd.concat(scores_list)
        # merged_metas = pd.concat(metas_list)
        self.result = result_dict
        return result_dict

    def store(self):
        experiment_base_path = f"./results/{self.replication_lbl}"
        os.makedirs(experiment_base_path, exist_ok=True)
        for exp_type, result_dict in self.result.items():
            merged_df_scores, merged_df_metas = (
                result_dict["scores"],
                result_dict["meta"],
            )
            print("starting storing")
            # preview_large = merged_df_scores.sample(10_000)
            # preview_small = merged_df_scores.head()
            # if self.plot:
            #     print(preview_small)
            # preview_large.to_csv(f"{experiment_base_path}_preview.csv", index=False)
            scores_path = os.path.join(experiment_base_path, f"scores-{exp_type}.csv")
            experiment_idx = ["model", "env", "budget_abs", "rnd", "subject_system"]
            # merged_df_scores["env"] = merged_df_scores["env"].map(lambda x: x if x != "overall" else -1)
            merged_df_scores.to_csv(scores_path, index=False)
            # merged_df_scores.to_parquet(scores_path, index=False, partition_cols=experiment_idx)
            # dd_df.to_parquet(scores_path, partition_on=experiment_idx,
            #                  write_index=False,
            #                  name_function=lambda i: f"{self.replication_lbl}.part.{i}.parquet")

            model_meta_path = os.path.join(
                experiment_base_path, f"model-meta-{exp_type}.csv"
            )
            # merged_df_metas["env"] = merged_df_metas["env"].map(lambda x: x if x != "overall" else -1)
            merged_df_metas.to_csv(model_meta_path, index=False)
            # merged_df_metas.to_parquet(model_meta_path, index=False, partition_cols=experiment_idx)

            # dd_df = dd.from_pandas(merged_df_metas, chunksize=20_000_000)
            # dd_df.to_parquet(model_meta_path,
            #                  write_index=False,
            #                  name_function=lambda i: f"{self.replication_lbl}.part.{i}.parquet")

            # merged_df.to_parquet(f"{experiment_base_path}.parquet", index=False, compression="brotli")
        return experiment_base_path

    # def handle_task(self, progress_bar, task):
    def handle_task(self, task: ExperimentTask):
        mlflow.set_tracking_uri("http://172.26.92.43:5000")
        mlflow.set_experiment(experiment_name=self.experiment_name)
        run_name = task.get_id()
        with mlflow.start_run(run_name=run_name):
            predictions = task.run()
            # eval = ModelEvaluation()
            # eval.scalar_accuracy_on_dask()
            errs, meta = task.eval()

            #
        return errs, meta
