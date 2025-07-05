import argparse
import numpy as np
import time
import os
import random
from random import sample
from numpy import genfromtxt
import pandas as pd
from collections import Counter
from utils.general import get_non_zero_indexes, process_training_data
from utils.HINNPerf_data_preproc import DataPreproc
from utils.HINNPerf_args import list_of_param_dicts
from utils.HINNPerf_models import MLPHierarchicalModel
from utils.HINNPerf_model_runner import ModelRunner
import warnings
warnings.filterwarnings('ignore')

verbose = False

def get_HINNPerf_MRE(args=[[],[],[],[],[]]):
    whole_data = args[0]
    training_index = args[1]
    testing_index = args[2]
    test_mode = args[3]
    config = args[4]

    data_gen = DataPreproc(whole_data, training_index, testing_index)
    runner = ModelRunner(data_gen, MLPHierarchicalModel)

    if test_mode == True:
        config = dict(
            input_dim=[data_gen.config_num],
            num_neuron=[128],
            num_block=[2],
            num_layer_pb=[2],
            lamda=[0.001],
            linear=[False],
            gnorm=[True],
            lr=[0.001],
            decay=[None],
            verbose=[verbose]
        )
    elif config == []:
        config = dict(
            input_dim=[data_gen.config_num],
            num_neuron=[128],
            num_block=[2,3,4],
            num_layer_pb=[2,3,4],
            lamda=[0.001, 0.1, 10.,100],
            linear=[False],
            gnorm=[True, False],
            lr=[0.0001, 0.001, 0.01],
            decay=[None],
            verbose=[verbose]
        )
    config_list = list_of_param_dicts(config)

    abs_error_val_min = float('inf')
    best_config = config_list[0]

    for con in config_list:
        abs_error_train, abs_error_val = runner.train(con)
        if abs_error_val_min > abs_error_val:
            abs_error_val_min = abs_error_val
            best_config = con

    Y_pred_test, rel_error = runner.test(best_config)
    # print('Best_config: {}'.format(best_config))
    return rel_error

def get_HINNPerf_MRE_and_predictions(args=[[],[],[],[],[]]):
    whole_data = args[0]
    training_index = args[1]
    testing_index = args[2]
    test_mode = args[3]
    config = args[4]

    data_gen = DataPreproc(whole_data, training_index, testing_index)
    runner = ModelRunner(data_gen, MLPHierarchicalModel)

    if test_mode == True:
        config = dict(
            input_dim=[data_gen.config_num],
            num_neuron=[128],
            num_block=[2],
            num_layer_pb=[2],
            lamda=[0.001],
            linear=[False],
            gnorm=[True],
            lr=[0.001],
            decay=[None],
            verbose=[verbose]
        )
    elif config == []:
        config = dict(
            input_dim=[data_gen.config_num],
            num_neuron=[128],
            num_block=[2,3,4],
            num_layer_pb=[2,3,4],
            lamda=[0.001, 0.1, 10.,100],
            linear=[False],
            gnorm=[True, False],
            lr=[0.0001, 0.001, 0.01],
            decay=[None],
            verbose=[verbose]
        )
    config_list = list_of_param_dicts(config)

    abs_error_val_min = float('inf')
    best_config = config_list[0]

    for con in config_list:
        abs_error_train, abs_error_val = runner.train(con)
        if abs_error_val_min > abs_error_val:
            abs_error_val_min = abs_error_val
            best_config = con

    Y_pred_test, rel_error = runner.test(best_config)
    # print('Best_config: {}'.format(best_config))
    return rel_error, Y_pred_test

def get_HINNPerf_best_config(args=[[],[],[],[],[]]):
    whole_data = args[0]
    training_index = args[1]
    testing_index = args[2]
    test_mode = args[3]
    config = args[4]

    data_gen = DataPreproc(whole_data, training_index, testing_index)
    runner = ModelRunner(data_gen, MLPHierarchicalModel)

    if test_mode == True:
        config = dict(
            input_dim=[data_gen.config_num],
            num_neuron=[128],
            num_block=[2],
            num_layer_pb=[2],
            lamda=[0.001],
            linear=[False],
            gnorm=[True],
            lr=[0.001],
            decay=[None],
            verbose=[verbose]
        )
    elif config == []:
        config = dict(
            input_dim=[data_gen.config_num],
            num_neuron=[128],
            num_block=[2,3,4],
            num_layer_pb=[2,3,4],
            lamda=[0.001, 0.1, 10.,100],
            linear=[False],
            gnorm=[True, False],
            lr=[0.0001, 0.001, 0.01],
            decay=[None],
            verbose=[verbose]
        )
    config_list = list_of_param_dicts(config)

    abs_error_val_min = float('inf')
    best_config = config_list[0]

    for con in config_list:
        abs_error_train, abs_error_val = runner.train(con)
        if abs_error_val_min > abs_error_val:
            abs_error_val_min = abs_error_val
            best_config = con
    # print('Best_config: {}'.format(best_config))
    return best_config



