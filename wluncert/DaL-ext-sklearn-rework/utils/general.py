import os
import math
import numpy as np
from sklearn.svm import SVR
from numpy import genfromtxt
from sklearn.tree import _tree
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")


def get_sample_sizes(subject_system):
    if subject_system == 'Apache_AllNumeric':
        sample_sizes = [9,18,27,36,45]
    elif subject_system == 'BDBC_AllNumeric':
        sample_sizes = [18,36,54,72,90]
    elif subject_system == 'BDBJ_AllNumeric':
        sample_sizes = [26,52,78,104,130]
    elif subject_system == 'Dune_AllNumeric':
        sample_sizes = [224,692,1000,1365,1612]
    elif subject_system == 'Lrzip':
        sample_sizes = [127,295,386,485,907]
    elif subject_system == 'VP8':
        sample_sizes = [121,273,356,467,830]
    elif subject_system == 'hipacc_AllNumeric':
        sample_sizes = [261,528,736,1281,2631]
    elif subject_system == 'hsmgp_AllNumeric':
        sample_sizes = [77,173,384,480,864]
    elif subject_system == 'kanzi':
        sample_sizes = [31,62,93,124,155]
    elif subject_system == 'nginx':
        sample_sizes = [228,468,814,1012,1352]
    elif subject_system == 'sqlite':
        sample_sizes = [14,28,42,56,70]
    elif subject_system == 'x264_AllNumeric':
        sample_sizes = [16,32,48,64,80]
    return sample_sizes


def load_data(dir_data):
    whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
    # encode numerical features
    def is_col_cat(mat, i):
        return len(np.unique(mat[:, i])) > 2
    def get_categorical_cols(mat):
        return [i for i in range(mat.shape[-1]) if is_col_cat(mat, i)]
    features = whole_data[:, 0:-1]
    categorical_cols = get_categorical_cols(features)
    # non_categorical_cols = [i for i in range(features.shape[-1]) if i not in categorical_cols]
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder()
        one_hot_features = encoder.fit_transform(features).toarray()
        whole_data = np.concatenate((one_hot_features, whole_data[:, -1][:, np.newaxis]), axis=1)
    return whole_data


def get_non_zero_indexes(whole_data='', total_tasks=1):
    (N, n) = whole_data.shape
    n = n - 1
    delete_index = set()
    temp_index = list(range(N))
    for i in range(total_tasks):
        temp_Y = whole_data[:, n - i]
        for j in range(len(temp_Y)):
            if temp_Y[j] == 0:
                delete_index.add(j)
    non_zero_indexes = np.setdiff1d(temp_index, list(delete_index))
    return non_zero_indexes


def process_training_data(whole_data, training_index, N_features, n, main_task):
    temp_X = whole_data[training_index, 0:N_features]
    # scale x
    temp_max_X = np.amax(temp_X, axis=0)
    if 0 in temp_max_X:
        temp_max_X[temp_max_X == 0] = 1
    temp_X = np.divide(temp_X, temp_max_X)
    X_train = np.array(temp_X)

    # Split train data into 2 parts (67-33)
    N_cross = int(np.ceil(len(temp_X) * 2 / 3))
    X_train1 = (temp_X[0:N_cross, :])
    X_train2 = (temp_X[N_cross:len(temp_X), :])

    ### process y
    temp_Y = whole_data[training_index, n - main_task][:, np.newaxis]
    # scale y
    temp_max_Y = np.max(temp_Y) / 100
    if temp_max_Y == 0:
        temp_max_Y = 1
    temp_Y = np.divide(temp_Y, temp_max_Y)
    Y_train = np.array(temp_Y)

    # Split train data into 2 parts (67-33)
    Y_train1 = (temp_Y[0:N_cross, :])
    Y_train2 = (temp_Y[N_cross:len(temp_Y), :])

    return temp_max_X, X_train, X_train1, X_train2, temp_max_Y, Y_train, Y_train1, Y_train2


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def build_model(regression_mod='RF', test_mode=True, training_X=[], training_Y=[]):
    """
    to build the specified regression model, given the training data
    :param regression_mod: the regression model to build
    :param test_mode: won't tune the hyper-parameters if test_mode == False
    :param training_X: the array of training features
    :param training_Y: the array of training label
    :return: the trained model
    """
    model = None
    if regression_mod == 'RF':
        model = RandomForestRegressor(random_state=0)
        param = {'n_estimators': np.arange(10, 100, 10),
                 'criterion': ['squared_error']}
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = RandomForestRegressor(**gridS.best_params_, random_state=0)

    elif regression_mod == 'DT':
        model = DecisionTreeRegressor(random_state=0)
        param = {'criterion': ['squared_error'],
                 'splitter': ['best'],
                 # 'min_samples_leaf': [1, 2, 3]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = DecisionTreeRegressor(**gridS.best_params_, random_state=0)

    elif regression_mod == 'KNN':
        min = 2
        max = 3
        if len(training_X) > 30:
            max = 16
            min = 5
        model = KNeighborsRegressor(n_neighbors=min)
        param = {'n_neighbors': np.arange(2, max, 2),
                 'weights': ('uniform', 'distance'),
                 'algorithm': ['auto'],  # 'ball_tree','kd_tree'),
                 'leaf_size': [10, 30, 50, 70, 90],
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = KNeighborsRegressor(**gridS.best_params_)

    elif regression_mod == 'SVR':
        model = SVR()
        param = {'kernel': ('linear', 'rbf'),
                 'degree': [2, 3, 4, 5],
                 'gamma': ('scale', 'auto'),
                 'coef0': [0, 2, 4, 6, 8, 10],
                 'epsilon': [0.01, 1]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = SVR(**gridS.best_params_)

    elif regression_mod == 'LR':
        model = LinearRegression()
        param = {'fit_intercept': (True, False),
                 'n_jobs': [1, -1]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = LinearRegression(**gridS.best_params_)

    elif regression_mod == 'KR':
        x1 = np.arange(0.1, 5, 0.5)
        model = KernelRidge()
        param = {'alpha': x1,
                 'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                 'coef0': [1, 2, 3, 4, 5]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = KernelRidge(**gridS.best_params_)

    return model

# the function to extract the dividing conditions recursively, and divide the training data into clusters (divisions)
def recursive_dividing(node, depth, tree_, X, samples=[], max_depth=1, min_samples=2, cluster_indexes_all=None):
    indent = "  " * depth
    if cluster_indexes_all is None:
        cluster_indexes_all = []
    if depth <= max_depth:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:  # if it's not the leaf node
            left_samples = []
            right_samples = []
            # get the node and the dividing threshold
            name = tree_.feature[node]
            threshold = tree_.threshold[node]
            # split the samples according to the threshold
            for i_sample in range(0, len(samples)):
                if X[samples[i_sample], name] <= threshold:
                    left_samples.append(samples[i_sample])
                else:
                    right_samples.append(samples[i_sample])
            # check if the minimum number of samples is statisfied
            if (len(left_samples) <= min_samples or len(right_samples) <= min_samples):
                # print('{}Not enough samples to cluster with {} and {} samples'.format(indent, len(left_samples), len(right_samples)))
                cluster_indexes_all.append(samples)
            else:
                # print("{}{} samples with feature {} <= {}:".format(indent, len(left_samples), name, threshold))
                cluster_indexes_all = recursive_dividing(tree_.children_left[node], depth + 1, tree_, X, left_samples, max_depth, min_samples, cluster_indexes_all)
                # print("{}{} samples with feature {} > {}:".format(indent, len(right_samples), name, threshold))
                cluster_indexes_all = recursive_dividing(tree_.children_right[node], depth + 1, tree_, X, right_samples, max_depth, min_samples, cluster_indexes_all)
        else:
            cluster_indexes_all.append(samples)
    # the base case: add the samples to the cluster
    elif depth == max_depth + 1:
        cluster_indexes_all.append(samples)
    return cluster_indexes_all


def init_dir(dir_name):
    """Creates directory if it does not exists"""
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)