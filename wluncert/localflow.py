from __future__ import annotations
from typing import List

import os
import re
import shutil
import time
from datetime import datetime
from contextlib import ContextDecorator
from contextlib import contextmanager
import json
import traceback
import shortid
import shortuuid

RESULT_ROOT = "/home/jdorn/results/localflow/"
META_FILE = "meta.json"
METRICS_FILE = "metrics.json"
PARAMS_FILE = "params.json"
FOLDER_CHAR_LIMIT = 25
import arviz as az


class TaskTracker:
    def __init__(self):
        self.experiment = None
        self.tasks = []
        self.current_task: LocalTask = None
        self.finished = False

    def log_params(self, d):
        self.current_task.log_params(d)

    def log_metrics(self, d):
        self.current_task.log_metrics(d)

    def log_dict(self, d, name):
        self.current_task.log_dict(d, name)

    def log_metric(self, key, val):
        self.current_task.log_metrics({key: val})

    def start_run(self, run_name=None, nested=True, run_id=None):
        new_task = LocalTask(name=run_name, path_id=run_id, experiment_id=self.experiment)
        if self.current_task:
            new_task.set_parent(self.current_task)
        self.current_task = new_task
        new_id = self.current_task.get_path_id()
        self.finished = False
        return new_task

    def end_task(self):
        self.current_task.finish()
        if self.current_task.parent:
            self.current_task = self.current_task.parent
        else:
            self.current_task = None
            self.finished = True

    def log_artifact(self, path):
        self.current_task.log_artifact(path)

    def set_tracking_uri(self, *args):
        pass

    def set_experiment(self, experiment_name):
        self.experiment = get_short_valid_id(experiment_name)
        return self.experiment

    def get_tasks(self):
        return self.tasks

    def log_error(self, e):
        self.current_task.log_error(e)


def get_experiment_folder(experiment_name):
    experiment_name_clean = get_short_valid_id(experiment_name)
    return os.path.join(RESULT_ROOT, experiment_name_clean)


def get_experiment_run_names(experiment_name):
    exp_folder = get_experiment_folder(experiment_name)
    names = os.listdir(exp_folder)
    return names


class ExperimentResult:
    def __init__(self, experiment_name):
        self.experiment_id = get_short_valid_id(experiment_name)
        self.folder = get_experiment_folder(self.experiment_id)
        self.runs = self.__get_runs()

    def __get_runs(self):
        exp_folder = get_experiment_folder(self.experiment_id)
        names = os.listdir(exp_folder)
        paths = [os.path.join(self.folder, name) for name in names]
        runs = [RunResult(self, path) for path in paths]
        runs_d = {run.run_id: run for run in runs}
        return runs_d

    def get_run_by_id(self, run_id) -> RunResult:
        return self.runs.get(run_id, None)


class RunResult:
    def __init__(self, experiment: ExperimentResult, path, parent_run: RunResult = None):
        self.experimen = experiment
        self.path = path
        self.parent = parent_run
        self.run_id = os.path.basename(self.path)

    def has_sub_runs(self) -> bool:
        for fname in os.listdir(self.path):
            return os.path.isdir(os.path.join(self.path, fname))

    def get_sub_runs(self) -> List[RunResult]:
        sub_runs = []
        for fname in os.listdir(self.path):
            child_abs_path = os.path.join(self.path, fname)
            if os.path.isdir(child_abs_path):
                new_run = RunResult(self.experimen, child_abs_path, self)
                sub_runs.append(new_run)
        return sub_runs

    def get_params(self):
        return self.get_dict(PARAMS_FILE)

    def get_metrics(self):
        return self.get_dict(METRICS_FILE)

    def get_meta(self):
        return self.get_dict(META_FILE)

    def get_arviz_data(self):
        for fname in os.listdir(self.path):
            if fname.endswith("netcdf"):
                filke_path_abs = os.path.join(self.path, fname)
                az_data = az.from_netcdf(filke_path_abs)
                return az_data
        return None

    def get_artifact_path(self, artifact_name):
        return os.path.join(self.path, artifact_name)

    def get_dict(self, dict_name):
        d_path = os.path.join(self.path, f"{dict_name.replace('.json', '')}.json")
        with open(d_path, 'r') as file:
            # print(d_path)
            try:
                d = json.load(file)
            except Exception as e:
                print(e)
                print(d_path)
                with open(d_path) as f:
                    data = f.read()
                    print(data)
                raise e
            return d


class LocalTask:
    def __init__(self, name=None, path_id=None, experiment_id=None):
        self.time_cost = None
        self.end_time = None
        self.experiment_id = experiment_id
        self.finished = None
        if name:
            self.path_id = create_unique_filename(name)
        else:
            self.path_id = path_id

        self.params = {}
        self.metrics = {}
        self.artifact_paths = []
        self.dicts = {}
        self.parent = None
        self.error = None
        self.start_time = time.time()
        self.info = TaskInfo(self.path_id)

    def get_task_folder(self, ):
        if self.parent is None:
            experiment_folder = get_experiment_folder(self.experiment_id)
            parent_folder = experiment_folder
        else:
            parent_folder = self.parent.get_task_folder()

        whole_path = os.path.join(parent_folder, self.path_id)
        os.makedirs(whole_path, exist_ok=True)
        return whole_path

    def set_parent(self, parent: LocalTask):
        self.parent = parent

    def log_params(self, d):
        self.params = {**self.params, **d}

    def log_metrics(self, d):
        self.metrics = {**self.metrics, **d}

    def log_dict(self, d, name):
        self.dicts = {name: {**self.params, **d}}

    def log_artifact(self, artifact_path):
        out_path = self.get_task_folder()
        shutil.copy2(artifact_path, out_path)
        # self.artifact_paths.append(artifact_path)

    def get_path_id(self):
        return self.path_id

    def persist_pretty_dict(self, file_name, d):
        out_path = os.path.join(self.get_task_folder(), file_name)
        try:
            with open(out_path, 'w') as file:
                json.dump(d, file, indent=4)
        except Exception as e:
            return f"An error occurred: {e}"

    def finish(self):
        self.finished = True
        self.end_time = time.time()
        self.time_cost = self.end_time - self.start_time
        self.persist_metas()
        self.persist_misc_dicts()
        persist_metrics = {f"metrics.{key}": val for key, val in self.metrics.items()}
        self.persist_pretty_dict(METRICS_FILE, persist_metrics)
        persist_params = {f"params.{key}": val for key, val in self.params.items()}
        self.persist_pretty_dict(PARAMS_FILE, persist_params)
        self.persist_artefacts()

        joined_dict = {**persist_metrics, **persist_params}

        # print(joined_dict)

    def log_error(self, e):
        self.error = e

    def persist_metas(self):
        error_str = None if self.error is None else str(self.error)
        meta_d = {
            "finished": self.finished,
            "time_cost": self.time_cost,
            "start": self.start_time,
            "end": self.end_time,
            "error": error_str,
        }
        self.persist_pretty_dict(META_FILE, meta_d)

    def persist_artefacts(self):
        out_path = self.get_task_folder()
        for a in self.artifact_paths:
            shutil.copy2(a, out_path)

    def persist_misc_dicts(self):
        for dict_name, d in self.dicts.items():
            self.persist_pretty_dict(dict_name, d)


class TaskInfo:
    def __init__(self, run_id):
        self.run_id = run_id


def get_short_valid_id(input_str):
    # Ersetzen von Nicht-ASCII-Zeichen durch Unterstriche
    cleaned_str = re.sub(r'[^\x00-\x7F]', '-', input_str)

    # Kürzen des Strings auf die ersten 10 Zeichen
    shortened_str = cleaned_str[:FOLDER_CHAR_LIMIT]
    return shortened_str


def create_unique_filename(input_str):
    shortened_str = get_short_valid_id(input_str)
    # Erstellen eines Datums-Strings im Format YYYYMMDD_HHMMSS
    date_str = datetime.now().strftime("%y%m%d-%H-%M-%S")
    uuid = shortuuid.uuid()[:10]
    # Kombinieren des Datums-Strings mit dem gekürzten String
    unique_filename = f"{date_str}-{shortened_str}-{uuid}"

    return unique_filename


TRACKER = TaskTracker()


class TaskContext(ContextDecorator):
    def __init__(self, run_id, run_name, *args, **kwargs):
        self.run_name = run_name
        self.run_id = run_id

    def __enter__(self):
        TRACKER.start_run(run_name=self.run_name, run_id=self.run_id)
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            # something went wrong
            TRACKER.log_error(type, value, traceback)
        TRACKER.end_task()
        return True


@contextmanager
def start_run(run_id=None, run_name=None, *args, **kwargs):
    run_obj = TRACKER.start_run(run_name=run_name, run_id=run_id)
    try:
        yield run_obj
    except Exception as e:
        TRACKER.log_error(e)
        tb = traceback.format_exc()
        print("[LFlow]", e)
        print(tb)
    finally:
        TRACKER.end_task()


def log_params(d):
    TRACKER.log_params(d)


def log_metrics(d):
    TRACKER.log_metrics(d)


def log_dict(d, name):
    TRACKER.log_dict(d, name)


def log_metric(key, val):
    TRACKER.log_metric(key, val)


def log_artifact(path):
    TRACKER.log_artifact(path)


def set_tracking_uri(ui):
    pass


def set_experiment(experiment_name):
    TRACKER.set_experiment(experiment_name)
