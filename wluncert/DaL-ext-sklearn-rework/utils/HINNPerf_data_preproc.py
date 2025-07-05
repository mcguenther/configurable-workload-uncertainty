import numpy as np
from numpy import genfromtxt


class DataPreproc():
    """ Generic class for data preprocessing """

    def __init__(self, whole_data, training_index, testing_index):
        """
        Args:
            sys_name: [string] the name of the dataset
        """
        # self.sys_name = sys_name
        # self.data_dir = 'datasets/' + sys_name + '_AllNumeric.csv'
        self.whole_data = whole_data
        self.__read_whole_data()
        self.training_index = training_index
        self.testing_index = testing_index


    def __read_whole_data(self):
        # print('Read whole dataset ' + self.sys_name + ' from csv file ...')
        # self.whole_data = genfromtxt(self.data_dir, delimiter=',', skip_header=1)
        (self.all_sample_num, config_num) = self.whole_data.shape
        self.config_num = config_num - 1

        self.X_all = self.whole_data[:, 0:self.config_num]
        self.Y_all = self.whole_data[:, self.config_num][:, np.newaxis]


    def __normalize(self, X, Y):
        """
        Normalize the data and labels
        Args:
            X: [sample_size, config_size] features
            Y: [sample_size, 1] labels
        """
        max_X = np.amax(X, axis=0)              # [sample_size, config_size] --> [config_size]
        if 0 in max_X: max_X[max_X == 0] = 1
        X_sample = np.divide(X, max_X)

        max_Y = np.max(Y)/100
        if max_Y == 0: max_Y = 1
        Y_sample = np.divide(Y, max_Y)

        return X_sample, Y_sample, max_X, np.array([max_Y])


    def __normalize_gaussian(self, X, Y):
        """
        Normalize the data and labels
        Args:
            X: [sample_size, config_size] features
            Y: [sample_size, 1] labels
        """
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_sample = (X - mean_X) / (std_X + (std_X == 0) * .001)

        mean_Y = np.mean(Y, axis=0)
        std_Y = np.std(Y, axis=0)
        Y_sample = (Y - mean_Y) / (std_Y + (std_Y == 0) * .001)

        return X_sample, Y_sample, mean_X, std_X, mean_Y, std_Y


    def get_train_valid_samples(self, gnorm=False):
        """
        Args:
            sample_size: [int] the total number of training samples
        """
        # np.random.seed(seed)
        # permutation = np.random.permutation(self.all_sample_num)
        # training_index = permutation[0:sample_size]
        sample_size = len(self.training_index)
        X_sample = self.X_all[self.training_index, :]
        Y_sample = self.Y_all[self.training_index, :]
        sample_cross = int(np.ceil(sample_size * 2 / 3))
        X_train = X_sample[0:sample_cross, :]
        Y_train = Y_sample[0:sample_cross, :]
        X_valid = X_sample[sample_cross:sample_size, :]
        Y_valid = Y_sample[sample_cross:sample_size, :]

        if gnorm:
            X_train_norm, Y_train_norm, mean_X, std_X, mean_Y, std_Y = self.__normalize_gaussian(X_train, Y_train)
            X_valid_norm = (X_valid - mean_X) / (std_X + (std_X == 0) * .001)
            Y_valid_norm = (Y_valid - mean_Y) / (std_Y + (std_Y == 0) * .001)

            return X_train_norm, Y_train_norm, X_valid_norm, Y_valid_norm, mean_Y, std_Y
        else:
            # X_train and Y_train contain the training samples
            X_train_norm, Y_train_norm, max_X, max_Y = self.__normalize(X_train, Y_train)
            X_valid_norm = np.divide(X_valid, max_X)
            Y_valid_norm = np.divide(Y_valid, max_Y)

            return X_train_norm, Y_train_norm, X_valid_norm, Y_valid_norm, max_Y


    def get_train_test_samples(self, gnorm=False):
        # np.random.seed(seed)
        # permutation = np.random.permutation(self.all_sample_num)
        # training_index = permutation[0:sample_size]
        X_train = self.X_all[self.training_index, :]
        Y_train = self.Y_all[self.training_index, :]
        # testing_index = np.setdiff1d(np.array(range(self.all_sample_num)), training_index)
        X_test = self.X_all[self.testing_index, :]
        Y_test = self.Y_all[self.testing_index, :]

        if gnorm:
            X_train_norm, Y_train_norm, mean_X, std_X, mean_Y, std_Y = self.__normalize_gaussian(X_train, Y_train)
            X_test_norm = (X_test - mean_X) / (std_X + (std_X == 0) * .001)

            return X_train_norm, Y_train_norm, X_test_norm, Y_test, mean_Y, std_Y
        else:
            X_train_norm, Y_train_norm, max_X, max_Y = self.__normalize(X_train, Y_train)
            X_test_norm = np.divide(X_test, max_X)

            return X_train_norm, Y_train_norm, X_test_norm, Y_test, max_Y
