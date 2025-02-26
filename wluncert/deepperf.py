import os
import os
import time
import random
import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator
import multiprocessing
from functools import partial

from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator
import multiprocessing

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class DeepPerfModel(BaseEstimator):
    def __init__(self, random_seed=0, verbose=False, n_jobs=-1, batch_size=32):
        self.n_layers = None
        self.learning_rate = None
        self.regularization_parameter = None
        self.model = None
        self.random_seed = random_seed
        self.verbose = verbose
        self.n_jobs = n_jobs if n_jobs > 0 else 1
        self.batch_size = batch_size

        tf.random.set_seed(self.random_seed)

        # Enable mixed precision
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

        if self.verbose:
            print("")
            print("DeepPerf:")
            print("Initializing DeepPerfModel...")
            print(f"TensorFlow version: {tf.__version__}")
            print(f"Number of CPUs to be used: {self.n_jobs}")
            print(f"Batch size: {self.batch_size}")

            # Print TensorFlow device information
            print("TensorFlow Devices:")
            devices = tf.config.list_physical_devices()
            for device in devices:
                print(f"  {device.device_type}: {device.name}")

            # Determine and print the device TensorFlow is using
            device_name = tf.test.gpu_device_name()
            if device_name:
                print(f"TensorFlow is using GPU: {device_name}")
            else:
                print("TensorFlow is using CPU")

            # Print more detailed device information
            print("\nDetailed Device Information:")
            logical_devices = tf.config.list_logical_devices()
            for device in logical_devices:
                print(f"  {device.device_type}: {device.name}")

            # Check if TensorFlow can access GPU
            if tf.test.is_built_with_cuda():
                print("\nTensorFlow is built with CUDA support.")
            else:
                print("\nTensorFlow is not built with CUDA support.")

            if tf.config.list_physical_devices("GPU"):
                print("TensorFlow can access GPU(s).")
            else:
                print("TensorFlow cannot access any GPUs.")
            print("")

        # Enable TensorFlow to use multiple threads
        # tf.config.threading.set_intra_op_parallelism_threads(self.n_jobs)
        # tf.config.threading.set_inter_op_parallelism_threads(self.n_jobs)

    def _create_model(self, n_layers, regularization_parameter):
        if self.verbose:
            print(
                f"Creating model with {n_layers} layers and regularization parameter {regularization_parameter}"
            )

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    128,
                    activation="relu",
                    activity_regularizer=tf.keras.regularizers.L1(
                        regularization_parameter
                    ),
                    kernel_initializer="glorot_normal",
                    dtype="float32",
                )
            ]
            + [
                tf.keras.layers.Dense(
                    128,
                    activation="relu",
                    kernel_initializer="glorot_normal",
                    dtype="float32",
                )
                for _ in range(n_layers)
            ]
            + [
                tf.keras.layers.Dense(
                    1,
                    activation="linear",
                    kernel_initializer="glorot_normal",
                    dtype="float32",
                )
            ]
        )
        return model

    def _get_error(self, y_pred, y_true):
        error = tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))
        if self.verbose:
            print(f"Calculated error: {error}")
        return error.numpy()

    def _train_and_evaluate(self, params, train_dataset, eval_dataset, y_val):
        lr, n_layers, reg_param = params
        model = self._create_model(n_layers, reg_param)
        # lr = tf.cast(lr, dtype=tf.float32)
        # lr = np.float32(lr)
        lr = float(lr)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        model.fit(
            train_dataset,
            epochs=2000,
            verbose=0,
        )
        y_pred = model.predict(
            eval_dataset,
            verbose=self.verbose,
        ).ravel()
        error = self._get_error(y_pred, y_val)
        return (params, error)

    # def _parallel_search(self, param_grid, X_train, y_train, X_val, y_val):
    #     results = []
    #     for params in param_grid:
    #         # params = [tf.cast(x, dtype=tf.float32) for x in params]
    #         result = self._train_and_evaluate(params, X_train, y_train, X_val, y_val)
    #         results.append(result)
    #     return min(results, key=lambda x: x[1])

    def _parallel_search(self, param_grid, train_dataset, eval_dataset, y_val):
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            train_evaluate_partial = partial(
                self._train_and_evaluate,
                train_dataset,
                eval_dataset,
                y_val,
            )
            results = pool.map(train_evaluate_partial, param_grid)
        return min(results, key=lambda x: x[1])

    def _sequential_search(self, param_grid, train_dataset, eval_dataset, y_val):
        best_error = float("inf")
        best_params = None

        if self.verbose:
            print("Starting sequential hyperparameter search...")

        for params in param_grid:
            if self.verbose:
                print(f"Evaluating parameters: {params}")

            if self.verbose:
                print("Training and evaluating model")
                start_train = time.time()
            result = self._train_and_evaluate(
                params, train_dataset, eval_dataset, y_val
            )

            if self.verbose:
                print(f"training took {time.time() - start_train:0.2f}s")
            current_params, current_error = result

            if current_error < best_error:
                best_error = current_error
                best_params = current_params

                if self.verbose:
                    print(f"New best error: {best_error}")
                    print(f"New best parameters: {best_params}")

        if self.verbose:
            print("Sequential hyperparameter search completed.")
            print(f"Best parameters found: {best_params}")
            print(f"Best error: {best_error}")

        return best_params, best_error

    def _find_hyperparameters(self, X, y):
        if self.verbose:
            print("Starting hyperparameter search...")

        X_train, X_val, y_train, y_val = train_test_split(
            X.values, y, test_size=0.33, random_state=self.random_seed
        )
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_val = X_val.reshape((X_val.shape[0], -1))
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val),
        ).batch(self.batch_size)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(
            self.batch_size
        )

        steps_learning_rate = 2  # 5
        lr_range = np.logspace(np.log10(1e-10), np.log10(0.1), steps_learning_rate)
        layer_range = range(3, 21)
        layer_range = range(3, 21, 4)
        steps_regularization = 2  # 5
        reg_range = np.logspace(np.log10(1e-10), np.log10(10), steps_regularization)

        param_grid = [
            (lr, n, rp) for lr in lr_range for n in layer_range for rp in reg_range
        ]
        n = 4
        param_grid = random.sample(param_grid, n)

        best_params, best_error = self._sequential_search(
            param_grid, train_dataset, val_dataset, y_val
        )

        self.learning_rate, self.n_layers, self.regularization_parameter = best_params

        if self.verbose:
            print("Hyperparameter search completed.")
            print(
                f"Optimal parameters - Learning rate: {self.learning_rate}, Layers: {self.n_layers}, Regularization: {self.regularization_parameter}"
            )
        return True

    def fit(self, X, y):
        # with tf.device("/GPU:0"):
        if self.verbose:
            print("Starting model fitting process...")
            physical_devices = tf.config.list_physical_devices("GPU")
            print("Using GPUS:", physical_devices)
        # X = tf.cast(X, dtype=tf.float32)
        # y = tf.cast(y, dtype=tf.float32)
        self._find_hyperparameters(X, y)

        self.model = self._create_model(self.n_layers, self.regularization_parameter)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        X = X.values
        X_train = X.reshape((X.shape[0], -1))
        y = np.array(y)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y)).batch(
            self.batch_size
        )

        self.model.fit(train_dataset, epochs=2000, verbose=self.verbose)
        if self.verbose:
            print("Model fitting completed.")
        return self

    def predict(self, X):
        # with tf.device("/GPU:0"):
        if self.verbose:
            print("Making predictions...")

        X = X.values
        X_pred = X.reshape((X.shape[0], -1))

        predictions = self.model.predict(X_pred, verbose=self.verbose).ravel()
        if self.verbose:
            print("Predictions completed.")
        return predictions

    def score(self, X, y):
        if self.verbose:
            print("Calculating score...")
        y_pred = self.predict(X)
        error = self._get_error(y_pred, y)
        if self.verbose:
            print(f"Score (negative error): {-error}")
        return -error  # Return negative error as score (higher is better)

    def get_params(self, deep=True):
        return {
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            "n_layers": self.n_layers,
            "learning_rate": self.learning_rate,
            "regularization_parameter": self.regularization_parameter,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        if "random_seed" in parameters:
            tf.random.set_seed(self.random_seed)
        if "n_jobs" in parameters:
            tf.config.threading.set_intra_op_parallelism_threads(self.n_jobs)
            tf.config.threading.set_inter_op_parallelism_threads(self.n_jobs)
        return self


# Usage example:
# deepperf_model = DeepPerfModel(verbose=True, n_jobs=-1)  # Use all available CPUs
# no_pooling_model = NoPoolingEnvModel(model_prototype=deepperf_model)
# complete_pooling_model = CompletePoolingEnvModel(model_prototype=deepperf_model)
