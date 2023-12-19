from __future__ import annotations

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
            self.current_task.add_child(new_task)
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
        self.children = []
        self.error = None
        self.start_time = time.time()
        self.info = TaskInfo(self.path_id)

    def get_task_folder(self, ):
        if self.parent is None:
            experiment_folder = os.path.join(RESULT_ROOT, self.experiment_id)
            parent_folder = experiment_folder
        else:
            parent_folder = self.parent.get_task_folder()

        whole_path = os.path.join(parent_folder, self.path_id)
        os.makedirs(whole_path, exist_ok=True)
        return whole_path

    def set_parent(self, parent: LocalTask):
        self.parent = parent

    def add_child(self, child: LocalTask):
        self.children.append(child)

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
