import copy
import os.path
import random
from typing import List, Dict

import dask.dataframe as dd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from wluncert.analysis import ModelEvaluation
from wluncert.data import SingleEnvData
from wluncert.utils import get_date_time_uuid


class ExperimentMultitask:
    def __init__(self, model_lbl, model, envs_lbl, train_list, test_list, train_size: int,
                 exp_id=None, rnd=0, pooling_cat=None):
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
        # for train_data in self.train_list:
        self.model.fit(self.train_list)
        self.predictions = self.model.predict(self.test_list)
        if not return_predictions:
            return
        else:
            dfs = []
            for env_predictions, test_data, train_data in zip(self.predictions, self.test_list, self.train_list):
                y_true = test_data.get_y()
                pred_n_dims = len(env_predictions.shape)
                prediction_has_samples = pred_n_dims > 1
                y_true_lbl = "y_true"
                y_pred_lbl = "y_pred"
                col_names = [y_true_lbl, y_pred_lbl]

                sample_idx_lbl = "testing_sample_idx"
                if not prediction_has_samples:
                    comparison_array = np.concatenate([np.atleast_2d(y_true).T, np.atleast_2d(env_predictions).T],
                                                      axis=1)
                    comparison_dataframe = pd.DataFrame(comparison_array, columns=col_names)
                    comparison_dataframe = comparison_dataframe.reset_index().rename(columns={"index": sample_idx_lbl})
                else:
                    true_and_samples_array = np.concatenate([np.atleast_2d(y_true).T, np.atleast_2d(env_predictions)],
                                                            axis=1)
                    n_samples = env_predictions.shape[1]
                    pred_col_names = [f"{str(i)}" for i in range(n_samples)]
                    all_columns = [y_true_lbl, *pred_col_names]
                    pred_df = pd.DataFrame(true_and_samples_array, columns=all_columns)
                    pred_df = pred_df.reset_index().rename(columns={"index": sample_idx_lbl})
                    comparison_dataframe = pred_df.melt(id_vars=[y_true_lbl],
                                                        value_vars=pred_col_names,
                                                        var_name=sample_idx_lbl,
                                                        value_name=y_pred_lbl,
                                                        )
                    comparison_dataframe[sample_idx_lbl] = comparison_dataframe[sample_idx_lbl].astype(int)

                self.add_task_metadata_to_df(comparison_dataframe, train_data)
                dfs.append(comparison_dataframe)
            result_df = pd.concat(dfs)
            return result_df

    def get_metadata_df(self, ):
        df = pd.DataFrame([[self.model_lbl, self.train_size, self.rnd, self.envs_lbl, self.exp_id]],
                          columns=["model", "budget_abs", "rnd", "subject_system", "exp_id"])
        # df["model"] = self.model_lbl
        # df["budget_abs"] = self.train_size
        # df["rnd"] = self.rnd
        # df["subject_system"] = self.envs_lbl
        # df["exp_id"] = self.exp_id
        return df

    def get_id(self):
        deterministic_id = f"{self.model_lbl} on {self.envs_lbl} -{self.exp_id}-"
        return deterministic_id

    def eval(self):
        meta_df = self.get_metadata_df()
        # Get the values from the first row of df1
        values = meta_df.iloc[0].values

        # Get the column names of df1
        columns = meta_df.columns

        eval = ModelEvaluation(self.predictions, self.test_list, self.train_list, meta_df)
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


class Replication:
    def __init__(self, models: Dict, data_providers: Dict, train_sizes_relative_to_option_number, rnds=None,
                 n_jobs=False, replication_lbl="last-experiment", plot=False):
        self.replication_lbl = get_date_time_uuid() + "-" + replication_lbl
        self.plot = plot
        self.models = models
        self.n_jobs = n_jobs
        self.data_providers = data_providers
        self.train_sizes_relative_to_option_number = train_sizes_relative_to_option_number
        self.rnds = rnds if rnds is not None else [0]
        self.result = None

    def run(self):
        tasks = []
        for model_lbl, model_proto in self.models.items():
            for data_lbl, data_set in self.data_providers.items():
                model_proto_for_env = copy.deepcopy(model_proto)
                model_proto_for_env.set_envs(data_set)
                for train_size in self.train_sizes_relative_to_option_number:
                    for rnd in self.rnds:
                        data_per_env: List[SingleEnvData] = data_set.get_workloads_data()
                        train_list = []
                        test_list = []
                        for env_data in data_per_env:
                            split = env_data.get_split(rnd=rnd, n_train_samples_rel_opt_num=train_size)
                            train_data = split.train_data
                            train_list.append(train_data)
                            test_list.append(env_data)

                        pooling_cat = model_proto_for_env.get_pooling_cat()
                        task = ExperimentMultitask(model_lbl, model_proto_for_env, data_lbl, train_list, test_list,
                                                   # exp_id=f"{train_size}-rnd-{rnd}",
                                                   train_size=train_size,
                                                   pooling_cat=pooling_cat,
                                                   exp_id=train_size,
                                                   rnd=rnd)
                        tasks.append(task)

        print("provisioned experiments", flush=True)

        random.seed(self.rnds[0])
        random.shuffle(tasks)
        if self.n_jobs:
            eval_scores_and_metas = Parallel(n_jobs=self.n_jobs)(
                delayed(self.handle_task)(task) for task in tqdm(tasks))
        else:
            eval_scores_and_metas = []
            progress_bar = tqdm(total=len(tasks), desc="Running tasks", unit="task")
            for task in tasks:
                new_result = self.handle_task(task)
                eval_scores_and_metas.append(new_result)
                progress_bar.update(1)
            progress_bar.close()

        eval_scores = [scores for scores, metas in eval_scores_and_metas]
        eval_metas = [metas for scores, metas in eval_scores_and_metas]
        merged_df_scores = pd.concat(eval_scores)
        merged_df_metas = pd.concat(eval_metas)
        self.result = merged_df_scores, merged_df_metas
        return merged_df_scores, merged_df_metas

    def store(self, merged_df=None):
        merged_df = merged_df or self.result
        merged_df_scores, merged_df_metas = merged_df
        print("starting storing")
        # preview_large = merged_df_scores.sample(10_000)
        # preview_small = merged_df_scores.head()
        # if self.plot:
        #     print(preview_small)
        experiment_base_path = f"./results/{self.replication_lbl}"
        os.makedirs(experiment_base_path, exist_ok=True)
        # preview_large.to_csv(f"{experiment_base_path}_preview.csv", index=False)
        scores_path = os.path.join(experiment_base_path, "scores.csv")
        dd_df = dd.from_pandas(merged_df_scores, chunksize=20_000_000)
        experiment_idx = ["model", "env", "budget_abs", "rnd", "subject_system"]
        # merged_df_scores["env"] = merged_df_scores["env"].map(lambda x: x if x != "overall" else -1)
        merged_df_scores.to_csv(scores_path, index=False)
        # merged_df_scores.to_parquet(scores_path, index=False, partition_cols=experiment_idx)
        # dd_df.to_parquet(scores_path, partition_on=experiment_idx,
        #                  write_index=False,
        #                  name_function=lambda i: f"{self.replication_lbl}.part.{i}.parquet")

        model_meta_path = os.path.join(experiment_base_path, "model-meta.csv")
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
    def handle_task(self, task: ExperimentMultitask):
        predictions = task.run()
        # eval = ModelEvaluation()
        # eval.scalar_accuracy_on_dask()
        errs, meta = task.eval()
        #
        return errs, meta
