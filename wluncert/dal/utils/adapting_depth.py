import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import genfromtxt
from imblearn.over_sampling import SMOTE
import os
from collections import Counter
from doepy import read_write
from random import sample
from .general import build_model
from .hyperparameter_tuning import nn_l1_val, hyperparameter_tuning
from sklearn import tree
from .mlp_sparse_model_tf2 import MLPSparseModel
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from .general import get_non_zero_indexes, process_training_data
import warnings
warnings.filterwarnings('ignore')


def is_dominated(point, points):
    return any(all(p[i] <= point[i] for i in range(len(point))) for p in points)


def get_intersection_point(p1, p2, p3):
    u = ((p1 - p2).dot(p3 - p2)) / np.linalg.norm(p3 - p2) ** 2
    return p2 + u * (p3 - p2)


def is_above(p1, p2, p3):
    return ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) < 0
    # return False


def is_in(point, points):
    for temp_point in points:
        if temp_point[0] == point[0] and temp_point[1] == point[1]:
            return True
    return False


def draw_scatter_plot(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    if normalization_mode == 'norm':
        plt.scatter(x, y, label='d={}, HV={:.2f}'.format(temp_depth, hypervolume),
                    color=colors[temp_depth - 1])
    else:
        plt.scatter(x, y, label='d={}, Log HV={:.0f}'.format(temp_depth, hypervolume),
                    color=colors[temp_depth - 1])
    # for i in range(len(points) - 1):
    #     x_values = [points[i][0], points[i + 1][0]]
    #     y_values = [points[i][1], points[i + 1][1]]
    #     plt.plot(x_values, y_values, color=colors[max_depth-1], linestyle='dotted')

def mean_squared_loss(numbers):
    mean = np.mean(numbers)
    return np.mean((numbers - mean) ** 2)

def sum_squared_loss(numbers):
    mean = np.mean(numbers)
    return np.sum((numbers - mean) ** 2)

def get_depth_AvgHV(dir_data, N_train, ne, seed, min_samples):
    whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
    (N, n) = whole_data.shape
    N_features = n - 1
    non_zero_indexes = get_non_zero_indexes(whole_data)
    subject_system = dir_data.split('/')[1].split('.')[0]
    lower_bound = 1
    upper_bound = 'max'
    ref_point_rate = 1.1
    max_dist_difference = 0

    N_test = (len(non_zero_indexes) - N_train)

    random.seed(ne * seed)

    testing_index = sample(list(non_zero_indexes), N_test)
    training_index = np.setdiff1d(non_zero_indexes, testing_index)
    Y = whole_data[training_index, -1][:, np.newaxis]
    X = whole_data[training_index, 0:N_features]

    # build and train a CART to extract the dividing conditions
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV
    DT = DecisionTreeRegressor(criterion='squared_error', splitter='best', random_state=seed)
    DT.fit(X, Y)
    tree_ = DT.tree_  # get the tree structure
    # selected_depthes = range(tree_.max_depth)
    if upper_bound == 'max':
        selected_depthes = range(lower_bound, tree_.max_depth+1)
    elif tree_.max_depth+1 < upper_bound:
        selected_depthes = range(lower_bound, tree_.max_depth+1)
    else:
        selected_depthes = range(lower_bound, upper_bound)
    # print('Selected depths: {}'.format(selected_depthes))

    points_divisions = []
    points_all = []
    for max_depth in selected_depthes:
        X_train = []
        Y_train = []
        cluster_indexes = []

        # the function to extract the dividing conditions recursively,
        # and divide the training data into clusters (divisions)
        from sklearn.tree import _tree
        def recurse(node, depth, samples=[]):
            indent = "  " * depth
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
                        # print('{}Not enough samples to cluster with {} and {} samples'.format(indent,len(left_samples),len(right_samples)))
                        cluster_indexes.append(samples)
                    else:
                        # print("{}{} samples with feature {} <= {}:".format(indent, len(left_samples), name,
                        #                                                    threshold))
                        recurse(tree_.children_left[node], depth + 1, left_samples)
                        # print("{}{} samples with feature {} > {}:".format(indent, len(right_samples), name,
                        #                                                   threshold))
                        recurse(tree_.children_right[node], depth + 1, right_samples)
                else:
                    cluster_indexes.append(samples)
            # the base case: add the samples to the cluster
            elif depth == max_depth + 1:
                cluster_indexes.append(samples)
        # run the defined recursive function above
        recurse(node=0, depth=1, samples=training_index)

        k = len(cluster_indexes)  # the number of divided subsets

        # generate the samples and labels for classification
        total_index = cluster_indexes[0]  # samples in the first cluster
        clusters = np.zeros(int(len(cluster_indexes[0])))  # labels for the first cluster
        for i in range(k):
            if i > 0:  # the samples and labels for each cluster
                total_index = total_index + cluster_indexes[i]
                clusters = np.hstack((clusters, np.ones(int(len(cluster_indexes[i]))) * i))

        # get max_X and max_Y for scaling
        max_X = np.amax(whole_data[total_index, 0:N_features], axis=0)  # scale X to 0-1
        if 0 in max_X:
            max_X[max_X == 0] = 1
        max_Y = np.max(whole_data[total_index, -1]) / 100  # scale Y to 0-100
        if max_Y == 0:
            max_Y = 1

        # get the loss and size for each division
        points = []
        for i in range(k):  # for each division
            temp_X = whole_data[cluster_indexes[i], 0:N_features]
            temp_Y = whole_data[cluster_indexes[i], -1][:, np.newaxis]
            # Scale X and Y
            X_train.append(np.divide(temp_X, max_X))
            Y_train.append(np.divide(temp_Y, max_Y))


            SqLoss_all = mean_squared_loss(whole_data[cluster_indexes[i], -1])

            points.append([SqLoss_all, -len(cluster_indexes[i])])
        # not average, just take each division as a point
        for temp_point in points:
            if temp_point not in points_all:
                points_all.append(temp_point)
        points_divisions.append(points)

    max_loss = 0
    min_size = -len(non_zero_indexes)
    for i_depth, temp_depth in enumerate(selected_depthes):
        if len(points_divisions[i_depth]) > 0:
            temp_max_loss = np.max(np.array(points_divisions[i_depth])[:, 0:1].ravel())
            temp_min_size = np.max(np.array(points_divisions[i_depth])[:, 1:2].ravel())
            # print(temp_min_size, np.array(points_divisions[i_depth])[:, 1:2].ravel())
            if temp_max_loss > max_loss:
                max_loss = temp_max_loss
            if temp_min_size > min_size:
                min_size = temp_min_size

    ref_point = np.array(
        [max_loss * ref_point_rate, min_size * (1 - (ref_point_rate - 1))])
    # print('ref_point: ', ref_point)
    from pymoo.indicators.hv import HV
    indicator = HV(ref_point=ref_point)

    best_division_HV = 0
    highest_HY = 0
    # print('\nComputing Hypervolumes...')
    for i_depth, temp_depth in enumerate(selected_depthes):
        # Calculate the hypervolume
        hypervolume = []
        for temp_point in points_divisions[i_depth]:
            hypervolume.append(indicator(np.array(temp_point)))
        # print(hypervolume)
        hypervolume = np.mean(hypervolume)
        # overall_hypervolume = indicator(np.array(points_divisions[i_depth]))
        # print('--{} Run{} d={} avg_Hypervolume: {}'.format(subject_system, ne + 1, temp_depth, hypervolume, overall_hypervolume))

        if  hypervolume > highest_HY and np.abs(hypervolume - highest_HY)/hypervolume > max_dist_difference:
            highest_HY = hypervolume
            best_division_HV = temp_depth
    # print('Best division: {}'.format(best_division_HV))

    return  best_division_HV