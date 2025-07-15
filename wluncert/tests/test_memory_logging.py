import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wluncert import localflow as mlflow


def test_memory_metric_logged(tmp_path):
    mlflow.RESULT_ROOT = str(tmp_path) + "/"
    experiment_name = "memtest"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="run") as run:
        data = [0] * 10000
        del data
    exp_folder = mlflow.get_experiment_folder(experiment_name)
    run_id = os.listdir(exp_folder)[0]
    metrics_path = os.path.join(exp_folder, run_id, mlflow.METRICS_FILE)
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert "metrics.max_memory_mb" in metrics
    assert metrics["metrics.max_memory_mb"] > 0
