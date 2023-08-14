import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from analysis import Analysis
from wluncert.experiment import Replication
from wluncert.main import get_all_datasets, get_all_models


class ExperimentBugTest(unittest.TestCase):
    def test_pairwise_mapping(self):
        n_jobs = 12
        print("pwd", os.getcwd())
        data_providers = get_all_datasets(train_data_folder="../training-data")
        selected_data = "jump3r",
        data_providers = {data_providers[key]: data_providers[key] for key in selected_data}
        models = get_all_models(False, n_jobs, True)
        model_lbls = ["partial-pooling-mcmc-extra", "partial-pooling-mcmc-extra-pw"]
        chosen_models = {lbl: models[lbl] for lbl in model_lbls}
        # robust_mcmc = models["partial-pooling-mcmc-extra"]
        # robust_mcmc_pw = models["partial-pooling-mcmc-extra-pw"]
        train_sizes = 2,
        rnds = 42,

        rep = Replication(chosen_models, data_providers, train_sizes, rnds, n_jobs=n_jobs,
                          replication_lbl="optionw-vs-pairw")
        rep.run()
        experiment_base_path = rep.store()
        al = Analysis(experiment_base_path)
        al.run()

    def test_all_models_quick(self):
        n_jobs = 12
        print("pwd", os.getcwd())
        data_providers = get_all_datasets(train_data_folder="../training-data")
        selected_data = "jump3r",
        data_providers = {data_providers[key]: data_providers[key] for key in selected_data}
        models = get_all_models(False, n_jobs, False)

        print("created models")

        for model_name in models:
            train_sizes = 2,
            rnds = 42,
            chosen_models = {model_name: models[model_name]}
            rep_lbl = f"testing-model-{model_name}"
            print(f"Testing model {model_name}")
            rep = Replication(chosen_models, data_providers, train_sizes, rnds, n_jobs=n_jobs, replication_lbl=rep_lbl)
            merged_df_scores, merged_df_metas = rep.run()
            experiment_base_path = rep.store()
            al = Analysis(experiment_base_path)
            al.run()
            print(f"model {model_name} ok")


if __name__ == '__main__':
    unittest.main()
