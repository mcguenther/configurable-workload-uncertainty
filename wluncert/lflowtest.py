from matplotlib import pyplot as plt

import localflow as mlflow
import time

mlflow.set_experiment("testing")

with mlflow.start_run(run_name="lvl1") as run:
    print("entered lvl1")
    mlflow.log_params({"lvl": 1})
    sleep_time = 0.25
    time.sleep(sleep_time)
    with mlflow.start_run(run_name="lvl2") as run2:
        print("entered lvl2")
        mlflow.log_params({"lvl": 2})
        time.sleep(sleep_time)
        with mlflow.start_run(run_name="lvl2") as run2:
            print("entered lvl3")
            mlflow.log_params({"lvl": 3})
            plt.hist([1,2,3,2,3,4,5,6,7,8,8,9,8,7,8,9,0,])
            filename = "tmp.pdf"
            plt.savefig(filename)
            mlflow.log_artifact(filename)
            time.sleep(sleep_time)
    print("back in lvl1")

    mlflow.log_metric("score", 9003)
    time.sleep(1)
