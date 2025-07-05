import os
import time
import random
import numpy as np
import pandas as pd
from random import sample
from utils.general import *
from imblearn.over_sampling import SMOTE
from utils.adapting_depth import get_depth_AvgHV
from utils.HINNPerf_model_runner import ModelRunner
from utils.HINNPerf_data_preproc import DataPreproc
from sklearn.ensemble import RandomForestClassifier
from utils.HINNPerf_models import MLPHierarchicalModel
from sklearn.feature_selection import mutual_info_regression
from utils.runHINNPerf import get_HINNPerf_MRE, get_HINNPerf_best_config
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    learning_model = 'DaL'
    selected_datasets = [4] # menu: 0 - Apache_AllNumeric, 1 - BDBC_AllNumeric, 2 - BDBJ_AllNumeric, 3 - Dune_AllNumeric, 4 - Lrzip, 5 - VP8, 6 - hipacc_AllNumeric, 7 - hsmgp_AllNumeric, 8 - kanzi, 9 - nginx, 10 - sqlite, 11 - x264_AllNumeric
    selected_sizes = [0, 1, 2, 3, 4] # choose from [0, 1, 2, 3, 4]
    save_results = True # save the results
    test_mode = True # if True, disable hyperparameter tuning for a quick test run
    min_samples_division = 4 # the minimum number of samples per division
    end_run = 30 # N_experiments = end_run-start_run
    start_run = 0 # start from 0
    min_depth = 0 # the minimum depth to select
    max_epoch = 2000 # max_epoch for training the local model
    N_experiments = end_run-start_run
    depth_selection_mode = ['AvgHV'] # 'AvgHV' / 'fixed-1', 'fixed-2', 'fixed-3', 'fixed-4'

    # get all available datasets
    file_names = []
    for home, dirs, files in os.walk('data/'.format()):
        for filename in files:
            file_names.append(filename)
    file_names.sort()
    dir_datas = ['data/{}'.format(file_name) for file_name in file_names]
    # for temp, temp_file in enumerate(file_names):
    #     print('{}-{} '.format(temp, temp_file))
    print('\nRuning {}, save_results: {}, test_mode: {}, run{}-{}, depth_selection_mode: {}, selected_sizes: {}, selected_datasets: {}...'.format(learning_model, save_results, test_mode, start_run, end_run, depth_selection_mode, selected_sizes, selected_datasets))

    for mode in depth_selection_mode:
        for dir_data in [dir_datas[temp] for temp in selected_datasets]: # for each dataset
            whole_data = load_data(dir_data)
            (N, n) = whole_data.shape
            N_features = n - 1
            non_zero_indexes = get_non_zero_indexes(whole_data) # delete the zero-performance samples
            subject_system = dir_data.split('/')[1].split('.')[0]
            sample_sizes = get_sample_sizes(subject_system)
            print('Dataset: {}, Total sample size: {}, N_features: {}, Sample sizes: {}, N_expriments: {}'.format(dir_data, len(non_zero_indexes), N_features, sample_sizes, N_experiments))
            saving_folder = '{}/results/{}'.format(os.getcwd(), subject_system)
            if save_results and not os.path.exists(saving_folder):
                print('Creating folder: {}'.format(saving_folder))
                os.makedirs(saving_folder)

            for i_size in selected_sizes:
                N_train = sample_sizes[i_size]
                non_zero_indexes = get_non_zero_indexes(whole_data)
                if N_train > int(len(non_zero_indexes) * 8 / 10):
                    N_train = int(len(non_zero_indexes) * 8 / 10)
                N_test = (len(non_zero_indexes) - N_train)
                seed = 2

                for ne in range(start_run, end_run):
                    print('\n---{}, Run {}, Size {}, Training size: {}, Testing size: {}---'.format(subject_system, ne+1, i_size+1, N_train, N_test))
                    start_time = time.time() # Start measure time

                    print('> Adapting d...'.format(mode))
                    if mode == 'AvgHV':
                        max_depth = get_depth_AvgHV(dir_data, N_train, ne, seed, min_samples_division)
                    elif mode.startswith('fixed') and len(mode.split('-')) > 1:
                        max_depth = int(mode.split('-')[1])
                    else:
                        max_depth = 1
                    time_adapting_depth = (time.time() - start_time) / 60
                    print('\t>> Selected depth: {}'.format(max_depth))
                    print('\t>> Adapting time cost (minutes): {}'.format(time_adapting_depth))


                    if max_depth <= min_depth:
                        print('Error: d={} is samller than the min_depth {}\n'.format(max_depth, min_depth))
                    else:
                        file_start = '{}_{}_d{}_{}-{}_{}'.format(learning_model, subject_system, max_depth, N_train, N_test,
                                                                 seed)
                        saving_dir = '{}/{}.csv'.format(saving_folder, file_start)
                        if not os.path.exists('{}/{}.csv'.format(saving_folder, file_start)):
                            saving_table = {'Run': [], 'MRE': [], 'Time': [], 'Time_adapting':[], 'Time_dividing':[], 'Time_training':[], 'Time_predicting':[], 'num_block': [], 'num_layer_pb':[], 'lamda': [], 'gnorm': [], 'lr': [], 'max_epoch': [max_epoch]}
                            for temp_run in range(30):
                                saving_table['Run'].append(temp_run + 1)
                                saving_table['MRE'].append('None')
                                saving_table['Time'].append('None')
                                saving_table['Time_adapting'].append('None')
                                saving_table['Time_dividing'].append('None')
                                saving_table['Time_training'].append('None')
                                saving_table['Time_predicting'].append('None')
                                saving_table['num_block'].append([])
                                saving_table['num_layer_pb'].append([])
                                saving_table['lamda'].append([])
                                saving_table['gnorm'].append([])
                                saving_table['lr'].append([])
                                if temp_run != 0:
                                    saving_table['max_epoch'].append(' ')

                            if save_results:
                                pd.DataFrame(saving_table).to_csv(saving_dir, index=False)
                                print('Creating {}...'.format(saving_dir))
                        elif os.path.exists('{}/{}.csv'.format(saving_folder, file_start)):
                            saving_table = pd.read_csv(saving_dir).to_dict('list')

                        # check if the run has been finished
                        finished = False
                        if (isinstance(saving_table['MRE'][ne], float) and isinstance(saving_table['Time'][ne], float)) or (isinstance(saving_table['MRE'][ne], str) and isinstance(saving_table['Time'][ne], str)):
                            if len('{}'.format(saving_table['MRE'][ne]).split('.')) > 1 and len('{}'.format(saving_table['Time'][ne]).split('.')) > 1:
                                print('{} Run {} has finished, MRE: {}, time: {}'.format(file_start, ne + 1, saving_table['MRE'][ne], saving_table['Time'][ne]))
                                finished = True
                        if not finished:
                            random.seed(ne * seed)
                            start_time_dividing = time.time()
                            # generate the training and testing indexes
                            non_zero_indexes = get_non_zero_indexes(whole_data)
                            testing_index = sample(list(non_zero_indexes), N_test)
                            non_zero_indexes = np.setdiff1d(non_zero_indexes, testing_index)
                            training_index = sample(list(non_zero_indexes), N_train)

                            # compute the weights of each feature using Mutual Information, for eliminating insignificant features
                            weights = []
                            feature_weights = mutual_info_regression(whole_data[training_index, 0:N_features], whole_data[training_index, -1], random_state=0)
                            for i in range(N_features):
                                weight = feature_weights[i]
                                # print('Feature {} weight: {}'.format(i, weight))
                                weights.append(weight)

                            # initialize variables
                            max_X = []
                            config = []
                            rel_errors = []
                            cluster_indexes_all = []

                            # generate clustering labels based on the dividing conditions of DT
                            print('> Dividing...')
                            # get the training X and Y for clustering
                            Y = whole_data[non_zero_indexes, -1][:, np.newaxis]
                            X = whole_data[non_zero_indexes, 0:N_features]

                            # build and train a CART to extract the dividing conditions
                            DT = DecisionTreeRegressor(random_state=seed, criterion='squared_error', splitter='best')
                            DT.fit(X, Y)
                            tree_ = DT.tree_  # get the tree structure
                            # recursively divide samples
                            cluster_indexes_all = recursive_dividing(0, 1, tree_, X, non_zero_indexes, max_depth,
                                                                     min_samples_division, cluster_indexes_all)
                            k = len(cluster_indexes_all)  # the number of divided subsets
                            print('\t>> Number of divisions: {}'.format(k))
                            time_dividing = ((time.time() - start_time_dividing) / 60)
                            print('\t>> Dividing time cost (minutes): {}'.format(time_dividing))

                            lamdas = [0.001, 0.01, 0.1, 1]  # the list of l2 regularization parameters for hyperparameter tuning
                            gnorms = [True, False]  # gnorm parameters for hyperparameter tuning
                            lrs = [0.0001, 0.001, 0.01]  # the list of learning rates for hyperparameter tuning
                            init_config = dict(
                                num_neuron=[128],
                                num_block=[2, 3, 4],
                                num_layer_pb=[2, 3, 4],
                                lamda=lamdas,
                                linear=[False],
                                gnorm=gnorms,
                                lr=lrs,
                                decay=[None],
                                verbose=[False]
                            )
                            if k <= 1: # if there is only one cluster, DaL can not be used
                                start_time_training = time.time()
                                print('> Training HINNPerf...')
                                rel_error = get_HINNPerf_MRE([whole_data, training_index, testing_index, test_mode, init_config])
                                print('\t>> HINNPerf MRE: {}'.format(rel_error))

                                time_training = ((time.time() - start_time_training) / 60)
                                print('\t>> Training time (min): {}'.format(time_training))

                                # End measuring time
                                total_time = ((time.time() - start_time) / 60)
                                time_predicting = total_time - time_training - time_dividing
                                print('\t>> Predicting time (min) : {}'.format(time_predicting))
                                print('Total time (min) : {}'.format(total_time))

                                if save_results:
                                    saving_dir = '{}/{}.csv'.format(saving_folder, file_start)
                                    saving_table = pd.read_csv(saving_dir).to_dict('list')
                                    saving_table['Run'][ne] = ne + 1
                                    saving_table['MRE'][ne] = rel_error
                                    saving_table['Time'][ne] = total_time
                                    saving_table['Time_adapting'][ne] = time_adapting_depth
                                    saving_table['Time_dividing'][ne] = time_dividing
                                    saving_table['Time_training'][ne] = time_training
                                    saving_table['Time_predicting'][ne] = time_predicting
                                    pd.DataFrame(saving_table).to_csv(saving_dir, index=False)
                                    print('Saving to {}...'.format(saving_dir))
                            else:
                                # generate the training indexes for each cluster
                                N_trains = []  # the number of training samples for each cluster
                                cluster_indexes = []
                                for i in range(k):
                                    if int(N_train) > len(cluster_indexes_all[i]):  # if N_train is too big
                                        N_trains.append(int(len(cluster_indexes_all[i])))
                                    else:
                                        N_trains.append(int(N_train))
                                    cluster_indexes.append(random.sample(cluster_indexes_all[i], N_trains[i])) # sample N_train samples from the cluster
                                # generate the indexes and labels for classification
                                total_index = cluster_indexes[0]  # samples in the first cluster
                                clusters = np.zeros(int(len(cluster_indexes[0])))  # labels for the first cluster
                                for i in range(k):
                                    if i > 0:  # the samples and labels for each cluster
                                        total_index = total_index + cluster_indexes[i]
                                        clusters = np.hstack((clusters, np.ones(int(len(cluster_indexes[i]))) * i))
                                # get max_X for scaling, where total_index contains samples in all clusters
                                max_X = np.amax(whole_data[total_index, 0:N_features], axis=0)  # total_index contains samples in all clusters
                                if 0 in max_X:
                                    max_X[max_X == 0] = 1

                                print('> Training RF classifier...')
                                # process the sample to train a classification model
                                X_smo = np.divide(whole_data[total_index, 0:N_features], max_X)
                                y_smo = clusters
                                for j in range(N_features):
                                    X_smo[:, j] = X_smo[:, j] * weights[j]  # assign the weight for each feature
                                enough_data = True
                                for i in range(0, k):
                                    if len(cluster_indexes[i]) < 5:
                                        enough_data = False
                                if enough_data:
                                    smo = SMOTE(random_state=1, k_neighbors=3) # SMOTE is an oversampling algorithm when the sample size is too small
                                    X_smo, y_smo = smo.fit_resample(X_smo, y_smo)
                                # build a random forest classifier to classify testing samples
                                forest = RandomForestClassifier(random_state=seed, criterion='gini')
                                if (not test_mode) and enough_data: # tune the hyperparameters if not in test mode
                                    param = {'n_estimators': np.arange(10, 100, 10)}
                                    gridS = GridSearchCV(forest, param)
                                    gridS.fit(X_smo, y_smo)
                                    print(gridS.best_params_)
                                    forest = RandomForestClassifier(**gridS.best_params_, random_state=seed, criterion='gini')
                                forest.fit(X_smo, y_smo)  # training
    
                                # classify the testing samples
                                print('\t>> Classifying testing samples')
                                testing_clusters = []  # classification labels for the testing samples
                                for i in range(0, k):
                                    testing_clusters.append([])
                                for temp_index in testing_index:
                                    temp_X = np.divide(whole_data[temp_index, 0:N_features], max_X)[np.newaxis, :]
                                    for j in range(N_features):
                                        temp_X[:, j] = temp_X[:, j] * weights[j]  # assign the weight for each feature
                                    temp_cluster = forest.predict(temp_X.reshape(1, -1))  # predict the dedicated local DNN using RF
                                    testing_clusters[int(temp_cluster)].append(temp_index)

                                ### Train DNN_DaL
                                print('> Training Local models...')
                                start_time_training = time.time()
                                ## tune DNN for each cluster (division) with multi-thread
                                from concurrent.futures import ThreadPoolExecutor
                                with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool: # create a multi-thread pool
                                    args = []  # prepare arguments for hyperparameter tuning
                                    for i in range(k):  # for each division
                                        args.append([whole_data, cluster_indexes[i], testing_clusters[i], test_mode, init_config])
                                    for i, best_config in enumerate(pool.map(get_HINNPerf_best_config, args)):
                                        print('\t>> Tuning division {}... ({} samples)'.format(i + 1, len(cluster_indexes[i])))
                                        config.append(best_config)
    
                                for i in range(k):
                                    print('\t>> Learning division {}... ({} samples)'.format(i + 1, len(cluster_indexes[i])))
                                    # train a local DNN model using the optimal hyperparameters
                                    data_gen = DataPreproc(whole_data, cluster_indexes[i], testing_clusters[i])
                                    # the error rates for all testing samples of a division
                                    runner = ModelRunner(data_gen, MLPHierarchicalModel, max_epoch=max_epoch)
                                    # store all the error rates in a list
                                    rel_error = runner.get_rel_error(config[i])
                                    rel_errors += list(rel_error)
                                rel_errors = np.mean(rel_errors) * 100
                                # compute the MRE (MAPE) using the testing samples
                                print('> Testing...')
                                print('\t>> {} MRE: {}'.format(learning_model, round(rel_errors, 2)))

                                # End measuring time
                                time_training = ((time.time() - start_time_training) / 60)
                                print('\t>> Training time (min): {}'.format(time_training))
                                total_time = ((time.time() - start_time) / 60)
                                time_predicting = total_time - time_training - time_dividing
                                print('\t>> Predicting time (min) : {}'.format(time_predicting))
                                print('Total time (min) : {}'.format(total_time))
    
                                if save_results:
                                    saving_dir = '{}/{}.csv'.format(saving_folder, file_start)
                                    saving_table = pd.read_csv(saving_dir).to_dict('list')
                                    saving_table['num_block'][ne] = []
                                    saving_table['num_layer_pb'][ne] = []
                                    saving_table['lamda'][ne] = []
                                    saving_table['gnorm'][ne] = []
                                    saving_table['lr'][ne] = []
                                    for i in range(k):
                                        saving_table['num_block'][ne].append(config[i]['num_block'])
                                        saving_table['num_layer_pb'][ne].append(config[i]['num_layer_pb'])
                                        saving_table['lamda'][ne].append(config[i]['lamda'])
                                        saving_table['gnorm'][ne].append(config[i]['gnorm'])
                                        saving_table['lr'][ne].append(config[i]['lr'])
                                    saving_table['Run'][ne] = ne + 1
                                    saving_table['MRE'][ne] = rel_errors
                                    saving_table['Time'][ne] = total_time
                                    saving_table['Time_adapting'][ne] = time_adapting_depth
                                    saving_table['Time_dividing'][ne] = time_dividing
                                    saving_table['Time_training'][ne] = time_training
                                    saving_table['Time_predicting'][ne] = time_predicting
                                    pd.DataFrame(saving_table).to_csv(saving_dir, index=False)
                                    print('Saving to {}...'.format(saving_dir))
