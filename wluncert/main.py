from wluncert.data import DataLoaderStandard, DataAdapterJump3r, TrainTestSplitForWorkloads, WorkloadTrainingDataSet


class Expperiment:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def run(self):
        pass


def main():
    path_jump3r = "./training-data/jump3r.csv"
    jump3r_data_raw = DataLoaderStandard(path_jump3r)
    data_jump3r = DataAdapterJump3r(jump3r_data_raw)
    wl_data: WorkloadTrainingDataSet = data_jump3r.get_wl_data()

    known_env_data, target_data = wl_data.get_loo_wl_data()
    known_env_data_split =
    print(train_loo)


if __name__ == "__main__":
    main()
