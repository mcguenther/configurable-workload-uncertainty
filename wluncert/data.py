from abc import ABC, abstractmethod
import copy
from typing import List

import pandas as pd
from pycosa import util
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SingleEnvData:
    def __init__(self, df: pd.DataFrame, environment_col_name, nfps):
        self.df_raw = df
        self.df = copy.deepcopy(self.df_raw)
        self.env_col_name = environment_col_name
        self.env_id = int(list(self.df[environment_col_name].unique())[0])
        self.df = self.df.drop(columns=[environment_col_name])
        self.nfps = list(nfps)
        self.std = None
        self.mean = None

    def get_selected_nfp_name(self):
        return self.nfps[0]

    def get_option_names(self):
        x = self.get_X()
        col_names = list(x.columns)
        return col_names

    def get_n_options(self) -> int:
        return len(self.get_option_names())

    def get_X(self):
        new_cols = self.get_feature_names()
        x = self.df[new_cols]
        return x

    def get_feature_names(self):
        cols = list(copy.deepcopy(self.df.columns))
        # cols.remove([self.environment_col_name, *self.nfps])
        new_cols = [c for c in cols if c not in self.nfps]
        return new_cols

    def get_y(self, nfp_name=None):
        nfp_name = self.default_to_selected_nfp(nfp_name)
        y = np.array(self.df[nfp_name])
        return y

    def default_to_selected_nfp(self, nfp_name):
        if nfp_name is None:
            nfp_name = self.get_selected_nfp_name()
        return nfp_name

    def get_split(self, n_train_samples_abs=None, n_train_samples_rel_opt_num=None, rnd=0):
        n_opts = self.get_n_options()
        absolute_train_size = n_train_samples_abs if n_train_samples_rel_opt_num is None else n_train_samples_rel_opt_num * n_opts
        absolute_train_size = int(absolute_train_size)
        print("Splitting train set with abs samples of", absolute_train_size)
        df_train, df_test = train_test_split(self.df_raw, train_size=absolute_train_size, random_state=rnd)
        train_data = SingleEnvData(df_train, self.env_col_name, self.nfps)
        test_data = SingleEnvData(df_test, self.env_col_name, self.nfps)
        return SingleEnvDataTrainTestSplit(train_data, test_data)

    def normalize(self, other_normalized=None):
        """
        normalizes the nfp. estimates mean and std from data. if params mean and std given, uses those instead.
        :param mean:
        :param std:
        :return:
        """
        new_data = SingleEnvDataNormalized(self.df_raw, self.env_col_name, self.nfps, other_normalized=other_normalized)
        return new_data


class SingleEnvDataNormalized(SingleEnvData):
    def __init__(self, df: pd.DataFrame, environment_col_name, nfps, other_normalized=None):
        super().__init__(df, environment_col_name, nfps)
        if other_normalized:
            self.scaler_X = other_normalized.scaler_X
            self.scaler_y = other_normalized.scaler_y
            self.normalize_X(self.scaler_X)
            self.normalize_y(self.scaler_y)
        else:
            self.scaler_X = None
            self.scaler_y = {}
            self.normalize_X()
            self.normalize_y()

    def fit_scaler_X(self):
        X = self.get_X()
        self.scaler_X = StandardScaler()
        self.scaler_X = self.scaler_X.fit(X)
        return self.scaler_X

    def normalize_X(self, scaler=None):
        X = self.get_X()
        self.scaler_X = scaler if scaler else self.fit_scaler_X()
        X_scaled = self.scaler_X.transform(X)
        features = self.get_feature_names()
        self.df[features] = X_scaled

    def normalize_y(self, scaler=None):
        for nfp_name in self.nfps:
            y = np.atleast_2d(self.get_y(nfp_name)).T
            if scaler:
                y_scaled = scaler[nfp_name].transform(y)
            y_scaler = StandardScaler()
            y_scaled = y_scaler.fit_transform(y)
            self.df[nfp_name] = y_scaled.ravel()
            self.scaler_y[nfp_name] = y_scaler

    def un_normalize_y(self, y, nfp_name=None):
        y = np.atleast_2d(y).T
        nfp_name = self.default_to_selected_nfp(nfp_name)
        scaler = self.scaler_y[nfp_name]
        inverse_y = scaler.inverse_transform(y).ravel()
        return inverse_y


class SingleEnvDataTrainTestSplit:
    def __init__(self, train_data: SingleEnvData, test_data: SingleEnvData):
        self.train_data = train_data
        self.test_data = test_data

    def normalize(self):
        self.train_data, = self.train_data.normalize()
        self.test_data = self.test_data.normalize(other_normalized=self.train_data)
        return self.train_data, self.test_data


class WorkloadTrainingDataSet:
    def __init__(self, df, environment_col_name, environment_lables, nfps):
        self.df = df
        self.environment_col_name = environment_col_name
        self.environment_lables = list(environment_lables)
        self.nfps = list(nfps)

    def get_df(self):
        return self.df

    def get_workloads_data(self, workload_lables=None) -> list[SingleEnvData]:
        if workload_lables is None:
            workload_lables = self.environment_lables
        for wl_lable in workload_lables:
            if wl_lable not in self.environment_lables:
                raise ValueError(
                    f"Invalid environment lable: {wl_lable}. Available lables: {','.join(self.environment_lables)}")
        workload_ids = [self.environment_lables.index(lbl) for lbl in workload_lables]
        r_df = self.get_df()
        env_datas = []
        for wl_id in workload_ids:
            env_df = r_df[r_df[self.environment_col_name] == wl_id]
            env_data = SingleEnvData(env_df, self.environment_col_name, self.nfps)
            env_datas.append(env_data)
        # r_df = r_df[r_df[self.environment_col_name].isin(workload_ids)]
        return env_datas

    def get_loo_wl_data(self):
        env_data_list = []
        for env in self.environment_lables:
            remaining_envs = [x for x in self.environment_lables if x != env]
            loo_df = self.get_workloads_data(remaining_envs)
            env_data_list.append(loo_df)
        target_env_data_list = [self.get_workloads_data([lbl])[0] for lbl in self.environment_lables]
        return env_data_list, target_env_data_list

    def number_of_envs(self):
        df = self.get_df()
        env_col = df.columns[-1]
        n_unique = df[env_col].unique()
        return n_unique


class DataLoaderStandard():
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path

    def get_standard_CSV(self):
        sys_df = pd.read_csv(self.base_path)
        df_no_multicollinearity = remove_multicollinearity(sys_df)
        cleared_sys_df = copy.deepcopy(df_no_multicollinearity)
        return cleared_sys_df

    def get_df(self):
        df = self.get_standard_CSV()
        return df


class DataSource(ABC):
    def __init__(self):
        self.environment_col_name = "workload"

    @abstractmethod
    def get_wl_data(self) -> WorkloadTrainingDataSet:
        pass


class DataAdapter(DataSource, ABC):
    def __init__(self, data_loader: DataLoaderStandard):
        super().__init__()
        self.data_source = data_loader
        cleared_df = self.data_source.get_df()
        self.transformed_df = self.get_transformed_df(cleared_df)

    @abstractmethod
    def get_environment_col_name(self):
        pass

    @abstractmethod
    def get_environment_lables(self):
        pass

    # @abstractmethod
    # def get_feature_names(self):
    #     pass



    @abstractmethod
    def get_nfps(self):
        pass

    @abstractmethod
    def get_transformed_df(self, cleared_df):
        pass

    def get_wl_data(self) -> WorkloadTrainingDataSet:
        train_data = WorkloadTrainingDataSet(self.transformed_df,
                                             self.get_environment_col_name(),
                                             self.get_environment_lables(), self.get_nfps())
        return train_data


class DataAdapterXZ(DataAdapter):
    def __init__(self, data_loader: DataLoaderStandard):
        self.environment_col_name = "workload_id"
        self.nfps = ["time", "max-resident-size"]
        self.environment_lables = None
        super().__init__(data_loader)

    def get_environment_col_name(self):
        return self.environment_col_name

    def get_environment_lables(self):
        return self.environment_lables

    def get_nfps(self):
        return self.nfps

    def get_transformed_df(self, cleared_sys_df, ):
        uiq_df = cleared_sys_df.loc[cleared_sys_df["workload"].str.contains("uiq")]
        # uiq_df["workload-scale"] = None
        uiq_df.loc[uiq_df["workload"] == "uiq2-4.bin", "workload-scale"] = 4
        uiq_df.loc[uiq_df["workload"] == "uiq2-16.bin", "workload-scale"] = 16
        uiq_df.loc[uiq_df["workload"] == "uiq-32.bin", "workload-scale"] = 32
        uiq_df.loc[uiq_df["workload"] == "uiq2-8.bin", "workload-scale"] = 8
        uiq_df["workload-name"] = "uiq2"
        uiq_df["workload"] = "uiq2"
        uiq_df[self.environment_col_name], self.environment_lables = uiq_df[
            "workload"].factorize()
        # changing column order to *OPTIONS, workload, workload-scale, *NFPS
        cols = list(uiq_df.columns)
        config_col_idx = cols.index("config_id")
        self.nfps = cols[:config_col_idx]
        self.nfps.append("ratio")
        wl_cols = {"workload", "workload-scale"}
        non_nfp_cols = set(cols) - set(self.nfps) - wl_cols
        uiq_df = uiq_df[[*non_nfp_cols, *wl_cols, *self.nfps]]
        # uiq_df = uiq_df[uiq_df["partition"] == cluster_partition]
        uiq_df = uiq_df.drop(columns=["config_id", "partition"], errors='ignore')
        return uiq_df


class DataAdapterJump3r(DataAdapter):

    def __init__(self, data_loader: DataLoaderStandard):
        self.environment_col_name = "workload_id"
        self.nfps = ["time", "max-resident-size"]
        self.environment_lables = None
        super().__init__(data_loader)

    def get_environment_col_name(self):
        return self.environment_col_name

    def get_environment_lables(self):
        return list(self.environment_lables)

    def get_nfps(self):
        return self.nfps

    def get_transformed_df(self, cleared_sys_df, ):
        # mono_stereo_df = cleared_sys_df[cleared_sys_df["workload"].isin(["dual-channel.wav", "single-channel.wav"])]
        # mono_stereo_df["workload-scale"] = mono_stereo_df["workload"] == "dual-channel.wav"
        cleared_sys_df[self.environment_col_name], self.environment_lables = cleared_sys_df[
            "workload"].factorize()  # == "dual-channel.wav"
        # mono_stereo_df["workload-name"] = "mono-stereo"
        all_cols = cleared_sys_df.columns
        middle_cols = [self.environment_col_name, "config"]
        options = set(all_cols) - {*self.nfps, *middle_cols}
        cleared_sys_df = cleared_sys_df[[*options, self.environment_col_name, *self.nfps]]
        # changing column order to *OPTIONS, workload, workload-scale, *NFPS
        return cleared_sys_df


def remove_multicollinearity(df):
    return util.remove_multicollinearity(df)
