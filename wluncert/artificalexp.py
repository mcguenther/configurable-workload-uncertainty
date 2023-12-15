import itertools

import numpy as np
import pandas as pd
import platform
import main as mainfile
import seaborn as sns
from matplotlib import pyplot as plt
import argparse
from scipy import stats as sps
import main as main_experiment_module
import mlflow


class Generator:
    def __init__(self, rel_noise_std=0.05, n_reps=1):
        self.rel_noise_std = rel_noise_std
        self.n_reps = n_reps

    def generate(self):
        n_workloads = 3
        n_options = 8
        combinations = list(
            itertools.product(list(range(n_workloads)), *([[0, 1]] * n_options))
        )
        tups = []
        for conf in combinations:
            # print(conf)
            perfet_perf = self.oracle(*conf)
            scale = self.rel_noise_std * perfet_perf
            simulated_measurements = sps.norm.rvs(
                loc=perfet_perf, scale=scale, random_state=0, size=self.n_reps
            )
            selected_measurement = np.median(simulated_measurements)
            options = conf[1:]
            wl_id = conf[0]
            tup = *options, wl_id, selected_measurement
            tups.append(tup)
        df = pd.DataFrame(
            tups,
            columns=[
                "abs1",
                "abs2",
                "rel1",
                "rel2",
                "flipAbs",
                "flipRel",
                "null1",
                "null2",
                "workload",
                "time",
            ],
        )
        df["workload"] = df["workload"].replace(
            {0: "CCTV.mp4", 1: "vlog.mp4", 2: "movie.mkv"}
        )
        print(df)
        df.to_csv("training-data/artificial/artificial_data.csv", index=False)
        return df

        # wl_id, abs, rel, flipAbs, flipRel, null

    def oracle(
        self,
        wl_id,
        abs1,
        abs2,
        rel1,
        rel2,
        flipAbs,
        flipRel,
        null1,
        null2,
    ):
        # fmt: off
        influences_list = [
            [10, 5, 5, 3, 3, 0, 3, 0, 0, ],
            [20, 5, 5, 6, 6, 5, 6, 0, 0, ],
            [30, 5, 5, 9, 9, 5, 0, 0, 0, ],
        ]
        # fmt: on
        infl = np.array(influences_list)
        applicable_inflnuences = infl[wl_id, :]
        feature_vector = [
            1,
            abs1,
            abs2,
            rel1,
            rel2,
            flipAbs,
            flipRel,
            null1,
            null2,
        ]
        result = applicable_inflnuences @ feature_vector
        return result


def main():
    hostname = platform.node()
    print("Hostname:", hostname)

    parser = argparse.ArgumentParser()
    # Add your other arguments here

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Unleash the generator!",
    )



    args = parser.parse_args()
    do_generate = args.generate

    # if do_generate:
    gen = Generator()
    gen.generate()


if __name__ == "__main__":
    main()
