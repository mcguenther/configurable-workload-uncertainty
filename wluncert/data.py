import os.path
from abc import ABC, abstractmethod
import copy
from typing import List

import pandas as pd
import scipy
from pycosa import util
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
    MinMaxScaler,
    MaxAbsScaler,
)

from jax import numpy as jnp


def has_multiple_columns(data):
    if isinstance(data, pd.DataFrame):
        return len(data.columns) > 1
    elif isinstance(data, np.ndarray):
        return data.ndim > 1 and data.shape[1] > 1
    else:
        return False


class SingleEnvData:
    def __init__(self, df: pd.DataFrame, environment_col_name, nfps):
        self.df = copy.deepcopy(df)
        self.env_id = int(list(self.df[environment_col_name].unique())[0])
        self.env_col_name = environment_col_name
        self.nfps = list(nfps)
        self.X_df = self.get_X()
        self.y_df = self.get_all_y()
        # self.df = self.df.drop(columns=[environment_col_name])

    def get_len(self):
        return len(list(self.get_y()))

    def __len__(self):
        return self.get_len()

    def get_env_id(self):
        return self.env_id

    def get_selected_nfp_name(self):
        return self.nfps[0]

    def get_n_options(self) -> int:
        return len(self.get_feature_names())

    def get_X(self):
        new_cols = self.get_feature_names()
        x = self.df.loc[:, new_cols]
        return x

    def set_X(self, df_X):
        # ys = self.get_y()
        # nfp_name = self.default_to_selected_nfp()
        # y_df = self.get_all_y() #pd.DataFrame(ys, columns=[nfp_name])
        self.update_df(X=df_X)

    def set_y(self, ys, nfp_name=None):
        if has_multiple_columns(ys):
            # set all_y
            for nfp_name in ys:
                col = ys[nfp_name]
                self.set_y(np.array(col), nfp_name=nfp_name)
            return

        nfp_name = nfp_name or self.default_to_selected_nfp()
        y_df = self.get_all_y()  # pd.DataFrame(ys, columns=[nfp_name])
        y_df[nfp_name] = ys
        # X = self.get_X()
        self.update_df(ys=y_df)

    def update_df(self, X=None, env_ids=None, ys=None):
        X = X if X is not None else self.get_X()
        ys = ys if ys is not None else self.get_all_y()
        env_ids = env_ids or [self.env_id] * len(ys)
        env_df = pd.DataFrame(env_ids, columns=[self.env_col_name])
        X_and_envs = pd.concat([X, env_df, ys], axis=1)
        self.df = X_and_envs
        return X_and_envs

    def get_feature_names(self):
        cols = list(copy.deepcopy(self.df.columns))
        # cols.remove([self.environment_col_name, *self.nfps])
        new_cols = [c for c in cols if c not in self.nfps and c != self.env_col_name]
        return new_cols

    def get_y(self, nfp_name=None):
        nfp_name = self.default_to_selected_nfp(nfp_name)
        y = self.df[nfp_name].to_numpy()
        return y

    def default_to_selected_nfp(self, nfp_name=None):
        nfp_name = nfp_name or self.get_selected_nfp_name()
        return nfp_name

    def get_split(
        self,
        n_train_samples_abs=None,
        n_train_samples_rel_opt_num=None,
        rnd=0,
        n_test_samples_rel_opt_num=None,
    ):
        n_opts = self.get_n_options()
        absolute_train_size = self.map_real_to_abs_number(
            n_opts, n_train_samples_abs, n_train_samples_rel_opt_num
        )
        # print("Splitting train set with abs samples of", absolute_train_size)
        # absolute_train_size = min(absolute_train_size, len(self.df))
        df_train, df_test = train_test_split(
            self.df, train_size=absolute_train_size, random_state=rnd
        )
        if n_test_samples_rel_opt_num:
            absolute_test_size = self.map_real_to_abs_number(
                n_opts, None, n_test_samples_rel_opt_num
            )
            _, df_test = train_test_split(
                self.df, train_size=absolute_test_size, random_state=rnd
            )
        train_data = SingleEnvData(df_train, self.env_col_name, self.nfps)
        test_data = SingleEnvData(df_test, self.env_col_name, self.nfps)

        return SingleEnvDataTrainTestSplit(train_data, test_data)

    def map_real_to_abs_number(
        self, n_opts, n_train_samples_abs, n_train_samples_rel_opt_num
    ):
        absolute_train_size = (
            n_train_samples_abs
            if n_train_samples_rel_opt_num is None
            else n_train_samples_rel_opt_num * n_opts
        )
        absolute_train_size = int(absolute_train_size)
        absolute_train_size = max(absolute_train_size, 1)
        return absolute_train_size

    # def normalize(self, other_normalized=None):
    #     """
    #     normalizes the nfp. estimates mean and std from data. if params mean and std given, uses those instead.
    #     :param mean:
    #     :param std:
    #     :return:
    #     """
    #     new_data = SingleEnvDataNormalized(self.df_raw, self.env_col_name, self.nfps, other_normalized=other_normalized)
    #     return new_data
    def get_all_y(self):
        return pd.DataFrame(
            np.array([self.get_y(nfp_name) for nfp_name in self.nfps]).T,
            columns=self.nfps,
        )


class SingleEnvDataNormalized(SingleEnvData):
    def __init__(
        self, df: pd.DataFrame, environment_col_name, nfps, other_normalized=None
    ):
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
        inverse_y = scaler.inverse_transform(y)
        return inverse_y


class SingleEnvDataTrainTestSplit:
    def __init__(self, train_data: SingleEnvData, test_data: SingleEnvData):
        self.train_data = train_data
        self.test_data = test_data

    def normalize(self):
        s = Standardizer()
        self.train_data = s.fit_transform([self.train_data])[0]
        self.test_data = s.transform([self.test_data])[0]
        return self.train_data, self.test_data


class WorkloadTrainingDataSet:
    def __init__(self, df, environment_col_name, environment_lables, nfps):
        self.df = df
        self.environment_col_name = environment_col_name
        self.environment_lables = list(environment_lables)
        self.nfps = list(nfps)
        self.wl_data = self._get_workloads_data()

    def get_df(self, use_env_lbls=False):
        df = self.df
        if use_env_lbls:
            replacement_dict = {
                index: label for index, label in enumerate(self.environment_lables)
            }
            # Replace values in the 'column_name' column using the dictionary
            df[self.environment_col_name] = df[self.environment_col_name].replace(
                replacement_dict
            )
        return self.df

    def get_env_lables(self):
        return self.environment_lables

    def get_feature_names(self):
        single_ens_data_list = self.get_workloads_data()
        my_env = single_ens_data_list[0]
        return my_env.get_feature_names()

    def get_workloads_data(self, workload_lables=None) -> list[SingleEnvData]:
        return self.wl_data

    def _get_workloads_data(self, workload_lables=None) -> list[SingleEnvData]:
        if workload_lables is None:
            workload_lables = self.environment_lables
        for wl_lable in workload_lables:
            if wl_lable not in self.environment_lables:
                raise ValueError(
                    f"Invalid environment lable: {wl_lable}. Available lables: {','.join(self.environment_lables)}"
                )
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
        target_env_data_list = [
            self.get_workloads_data([lbl])[0] for lbl in self.environment_lables
        ]
        return env_data_list, target_env_data_list

    def number_of_envs(self):
        df = self.get_df()
        env_col = self.get_env_col_name()
        n_unique = df[env_col].unique()
        return n_unique

    def get_env_col_name(self):
        df = self.get_df()
        env_col = df.columns[-1]
        return env_col


class DataLoaderStandard:
    def __init__(self, base_path, sep=None):
        super().__init__()
        self.base_path = base_path
        self.sep = sep

    def get_standard_CSV(self):
        sys_df = pd.read_csv(self.base_path, sep=self.sep)
        df_no_multicollinearity = remove_multicollinearity(sys_df)
        cleared_sys_df = copy.deepcopy(df_no_multicollinearity)
        return cleared_sys_df

    def get_df(self):
        df = self.get_standard_CSV()
        return df


class DataLoaderDashboardData:
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path
        self.nfps = []
        self.worload_col_name = "workload"
        self.cleared_sys_df = self._process_csv()

    def _process_csv(self):
        sample_path = os.path.join(self.base_path, "sample.csv")
        measurements_path = os.path.join(self.base_path, "measurements.csv")
        sample_df = pd.read_csv(sample_path)
        measurement_df = pd.read_csv(measurements_path)
        joined_df = pd.merge(sample_df, measurement_df, on="config_id")
        joined_df = joined_df.drop(columns=["partition", "config_id"], errors="ignore")
        # nfps = "time,kernel-time,user-time,max-resident-set-size,avg-resident-set-size,avg-mem-use".split(",")
        # "config_id,partition,workload"
        col_names_to_exclude = "partition", "config_id", self.worload_col_name
        self.nfps = [c for c in measurement_df.columns if c not in col_names_to_exclude]

        df_no_multicollinearity = remove_multicollinearity(joined_df)
        self.cleared_sys_df = copy.deepcopy(df_no_multicollinearity)
        return self.cleared_sys_df

    def get_standard_CSV(self):
        return self.cleared_sys_df

    def get_df(self):
        return self.cleared_sys_df


class DataSource(ABC):
    def __init__(self, environment_col_name=None):
        self.environment_col_name = environment_col_name or "workload"

    @abstractmethod
    def get_wl_data(self) -> WorkloadTrainingDataSet:
        pass


class DataAdapter(DataSource, ABC):
    def __init__(self, data_loader: DataLoaderStandard, environment_col_name=None):
        super().__init__(environment_col_name=environment_col_name)
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

    def factorize_workload_col(self, df):
        (
            df.loc[:, self.environment_col_name],
            self.environment_lables,
        ) = df[self.environment_col_name].factorize()
        return df

    def get_wl_data(self) -> WorkloadTrainingDataSet:
        train_data = WorkloadTrainingDataSet(
            self.transformed_df,
            self.get_environment_col_name(),
            self.get_environment_lables(),
            self.get_nfps(),
        )
        return train_data


class DashboardDatAdapter(DataAdapter):
    def get_environment_col_name(self):
        return self.environment_col_name

    def get_environment_lables(self):
        return self.environment_lables

    def get_nfps(self):
        return self.nfps

    def get_transformed_df(
        self,
        cleared_sys_df,
    ):
        return cleared_sys_df


class DataAdapterXZ(DashboardDatAdapter):
    def __init__(self, data_loader: DataLoaderStandard):
        # self.environment_col_name = "workload"
        # self.nfps = ["time", "max-resident-size"]
        super().__init__(data_loader)
        # self.environment_lables = None #list(cleared_df[data_loader.worload_col_name].unique())
        self.nfps = [
            nfp
            for nfp in data_loader.nfps
            if "resident" not in nfp and "mem" not in nfp
        ]  # "time,kernel-time,user-time,max-resident-set-size,avg-resident-set-size,avg-mem-use".split(",")
        # self.nfps =  [nfp for nfp in data_loader.nfps if "resident" not in nfp and "mem" not in nfp]   #"time,kernel-time,user-time,max-resident-set-size,avg-resident-set-size,avg-mem-use".split(",")

    def get_environment_col_name(self):
        return self.environment_col_name

    def get_environment_lables(self):
        return self.environment_lables

    def get_nfps(self):
        return self.nfps

    def get_transformed_df(
        self,
        cleared_sys_df,
    ):
        unwanted_wl = "artificl.tar"
        cleared_sys_df_wanted_wls = cleared_sys_df.loc[
            cleared_sys_df[self.environment_col_name] != unwanted_wl
        ]
        (
            cleared_sys_df_wanted_wls.loc[:, self.environment_col_name],
            self.environment_lables,
        ) = cleared_sys_df_wanted_wls[self.environment_col_name].factorize()
        return cleared_sys_df_wanted_wls


class DataAdapterNFPApache(DataAdapter):
    def __init__(self, data_loader: DataLoaderStandard):
        self.environment_col_name = "nfp-id"
        self.nfp_columns = ["performance", "cpu", "energy", "fixed-energy"]
        self.nfp_val_col = "y"
        super().__init__(data_loader, self.environment_col_name)

    def get_environment_col_name(self):
        return self.environment_col_name

    def get_environment_lables(self):
        return list(range(len(self.nfp_columns)))

    def get_nfps(self):
        return [self.nfp_val_col]

    def get_transformed_df(self, cleared_sys_df):
        # Use only latest version
        cleared_sys_df = cleared_sys_df.loc[
            cleared_sys_df["revision"] == cleared_sys_df["revision"].max()
        ]
        cleared_sys_df = cleared_sys_df.drop(columns=["revision"])

        # Melt the DataFrame
        df_melted = cleared_sys_df.melt(
            id_vars=[
                col for col in cleared_sys_df.columns if col not in self.nfp_columns
            ],
            value_vars=self.nfp_columns,
            var_name=self.environment_col_name,
            value_name=self.nfp_val_col,
        )

        # Create a dictionary to map NFP names to their index
        nfp_index_map = {nfp: idx for idx, nfp in enumerate(self.nfp_columns)}

        # Replace NFP names with their index
        df_melted[self.environment_col_name] = df_melted[self.environment_col_name].map(
            nfp_index_map
        )

        # Sort the DataFrame
        df_melted = df_melted.sort_values(
            by=[self.environment_col_name]
            + [col for col in cleared_sys_df.columns if col not in self.nfp_columns]
        )

        # Reset the index
        df_melted = df_melted.reset_index(drop=True)

        # Scale the data
        scaler = MaxAbsScaler()
        scaled_data = scaler.fit_transform(
            df_melted.drop(columns=[self.environment_col_name])
        )

        # Create a new DataFrame with scaled data
        scaled_df = pd.DataFrame(
            scaled_data,
            columns=[
                col for col in df_melted.columns if col != self.environment_col_name
            ],
        )
        scaled_df[self.environment_col_name] = df_melted[self.environment_col_name]

        # Reorder columns to match the expected format
        feature_cols = [
            col
            for col in scaled_df.columns
            if col not in [self.environment_col_name, self.nfp_val_col]
        ]
        final_df = scaled_df[
            feature_cols + [self.environment_col_name, self.nfp_val_col]
        ]

        return final_df

    def factorize_workload_col(self, df):
        # No need to factorize as we're using indices directly
        return df


class DataAdapterH2(DataAdapterXZ):
    def __init__(self, data_loader: DataLoaderStandard):
        # self.environment_col_name = "workload"
        # self.nfps = ["time", "max-resident-size"]
        super().__init__(data_loader)
        # self.environment_lables = None #list(cleared_df[data_loader.worload_col_name].unique())
        self.nfps = [
            nfp
            for nfp in data_loader.nfps
            if "resident" not in nfp and "mem" not in nfp
        ]  # "time,kernel-time,user-time,max-resident-set-size,avg-resident-set-size,avg-mem-use".split(",")
        # self.nfps =  [nfp for nfp in data_loader.nfps if "resident" not in nfp and "mem" not in nfp]   #"time,kernel-time,user-time,max-resident-set-size,avg-resident-set-size,avg-mem-use".split(",")

    def get_transformed_df(
        self,
        cleared_sys_df,
    ):
        unwanted_wl = "artificl.tar"
        cleared_sys_df_wanted_wls = cleared_sys_df.loc[
            cleared_sys_df[self.environment_col_name] != unwanted_wl
        ]
        self.factorize_workload_col(cleared_sys_df_wanted_wls)
        return cleared_sys_df_wanted_wls


class DataAdapterX264(DataAdapterXZ):
    def __init__(self, data_loader: DataLoaderStandard):
        super().__init__(data_loader)
        self.nfps = [
            nfp
            for nfp in data_loader.nfps
            if "resident" not in nfp and "mem" not in nfp
        ]

    def get_transformed_df(
        self,
        cleared_sys_df,
    ):
        return self.factorize_workload_col(cleared_sys_df)


class DataAdapterBatik(DataAdapterX264):
    pass


class DataAdapterDConvert(DataAdapterX264):
    pass


class DataAdapterKanzi(DataAdapterX264):
    pass


class DataAdapterLrzip(DataAdapterX264):
    pass


class DataAdapterZ3(DataAdapterX264):
    pass


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

    def get_transformed_df(
        self,
        cleared_sys_df,
    ):
        cleared_sys_df = self.factorize_workload_col(cleared_sys_df)
        all_cols = cleared_sys_df.columns
        middle_cols = [self.environment_col_name, "config"]
        options = set(all_cols) - {*self.nfps, *middle_cols}
        cleared_sys_df = cleared_sys_df[
            [*options, self.environment_col_name, *self.nfps]
        ]
        # changing column order to *OPTIONS, workload, workload-scale, *NFPS
        return cleared_sys_df


class DataAdapterFastdownward(DataAdapter):
    def __init__(self, data_loader: DataLoaderStandard):
        self.environment_col_name = "revisions"
        self.nfps = ["intperformance", "intmemory", "extperformance", "extmemory"]
        self.environment_lables = None
        super().__init__(data_loader, self.environment_col_name)

    def get_environment_col_name(self):
        return self.environment_col_name

    def get_environment_lables(self):
        return list(self.environment_lables)

    def get_nfps(self):
        return self.nfps

    def get_transformed_df(
        self,
        cleared_sys_df,
    ):
        cleared_sys_df = self.factorize_workload_col(cleared_sys_df)
        cleared_sys_df = cleared_sys_df.drop(
            columns=["random"],
        )
        all_cols = cleared_sys_df.columns
        middle_cols = [self.environment_col_name]
        options = set(all_cols) - {*self.nfps, *middle_cols}
        cleared_sys_df = cleared_sys_df[
            [*options, self.environment_col_name, *self.nfps]
        ]
        # changing column order to *OPTIONS, workload, workload-scale, *NFPS
        return cleared_sys_df


class DataAdapterVP9(DataAdapter):
    env_name = "workload"
    obsolete_col_names = ["id", "std-time"]
    nfp_lbls = [
        "median-time",
        "mean-time",
        "mean-compression-ratio",
    ]

    def __init__(self, data_loader: DataLoaderStandard):
        self.environment_col_name = self.get_environment_col_name()
        self.nfps = self.get_nfp_lbls()
        self.obsolete_columns = self.get_obsolete_column_names()
        self.environment_lables = None
        super().__init__(data_loader, self.environment_col_name)

    def get_obsolete_column_names(self):
        return DataAdapterVP9.obsolete_col_names

    def get_nfp_lbls(self):
        return DataAdapterVP9.nfp_lbls

    def get_environment_col_name(self):
        return DataAdapterVP9.env_name

    def get_environment_lables(self):
        return list(self.environment_lables)

    def get_nfps(self):
        return self.nfps

    def get_transformed_df(
        self,
        cleared_sys_df,
    ):
        cleared_sys_df = self.factorize_workload_col(cleared_sys_df)

        cleared_sys_df = cleared_sys_df.drop(
            columns=self.obsolete_columns,
        )
        all_cols = cleared_sys_df.columns
        middle_cols = [self.environment_col_name]
        options = set(all_cols) - {*self.nfps, *middle_cols}
        cleared_sys_df = cleared_sys_df[
            [*options, self.environment_col_name, *self.nfps]
        ]
        # changing column order to *OPTIONS, workload, workload-scale, *NFPS
        return cleared_sys_df


class DataAdapterx265(DataAdapterVP9):
    obsolete_col_names = ["id", "std-time", "std-compression-ratio"]
    nfp_lbls = [
        "median-time",
        "mean-time",
        "mean-compression-ratio",
    ]

    def get_obsolete_column_names(self):
        return DataAdapterx265.obsolete_col_names

    def get_nfp_lbls(self):
        return DataAdapterx265.nfp_lbls


class DataAdapterArtificial(DataAdapter):
    def __init__(self, data_loader: DataLoaderStandard, noise_std=None):
        self.environment_col_name = "workload"
        self.noise_std = noise_std
        self.nfps = ["time"]
        self.environment_lables = None
        super().__init__(data_loader, self.environment_col_name)

    def get_environment_col_name(self):
        return self.environment_col_name

    def get_environment_lables(self):
        return list(self.environment_lables)

    def get_nfps(self):
        return self.nfps

    def get_transformed_df(
        self,
        cleared_sys_df,
    ):
        cleared_sys_df = self.factorize_workload_col(cleared_sys_df)
        all_cols = cleared_sys_df.columns
        middle_cols = [self.environment_col_name]
        options = set(all_cols) - {*self.nfps, *middle_cols}
        cleared_sys_df = cleared_sys_df[
            [*options, self.environment_col_name, *self.nfps]
        ]
        cleared_sys_df = self.apply_noise(cleared_sys_df)
        # changing column order to *OPTIONS, workload, workload-scale, *NFPS
        return cleared_sys_df

    def apply_noise(self, cleared_sys_df):
        noisy_df = cleared_sys_df
        if self.noise_std:
            for nfp in self.get_nfps():
                noisy_df[nfp] = noisy_df[nfp] + np.random.normal(
                    0, self.noise_std, len(noisy_df)
                )
        return noisy_df


def remove_multicollinearity(df):
    return util.remove_multicollinearity(df)


class Preprocessing(ABC):
    @abstractmethod
    def transform(self, env_data: List[SingleEnvData]):
        pass

    @abstractmethod
    def fit_transform(self, env_data: List[SingleEnvData]):
        pass

    @abstractmethod
    def inverse_transform_pred(self, y_pred, data):
        pass


class Standardizer(Preprocessing):
    def __init__(self, standardize_y=True):
        self.standandizer_map = {}
        self.standardize_y = standardize_y

    def transform(self, env_data: List[SingleEnvData]):
        new_env_data_list = []
        for single_env_data in env_data:
            if single_env_data is None:
                new_env_data_list.append(single_env_data)
            else:
                new_data = self.apply_standardizer_to_env_data(single_env_data)
                new_env_data_list.append(new_data)
        return new_env_data_list

    def apply_standardizer_to_env_data(self, env_data: SingleEnvData):
        env_id = int(env_data.env_id)
        std_mapper_X, std_mappers_y = self.standandizer_map[env_id]
        X = env_data.get_X()
        std_X = std_mapper_X.transform(X)
        df_X = pd.DataFrame(std_X, columns=env_data.get_feature_names())
        y = copy.deepcopy(env_data.get_all_y())
        if self.standardize_y:
            for nfp_name, std_scaler in std_mappers_y.items():
                nfp_vals = pd.DataFrame(y[nfp_name], columns=[nfp_name])
                scaled_nfp_vals = std_scaler.transform(nfp_vals)
                y[nfp_name] = scaled_nfp_vals
        new_data = copy.deepcopy(env_data)
        new_data.set_X(df_X)
        new_data.set_y(y)
        return new_data

    def fit_transform(self, env_data: List[SingleEnvData]):
        new_env_data_list = []
        for single_env_data in env_data:
            env_id = int(single_env_data.env_id)
            X = single_env_data.get_X()
            std_mapper_X = MaxAbsScaler()
            std_mapper_X.fit(X)
            std_mappers_ys = {nfp: StandardScaler() for nfp in single_env_data.nfps}
            for nfp_name, scaler in std_mappers_ys.items():
                y_vals = single_env_data.get_y(nfp_name)
                y_df = pd.DataFrame(y_vals, columns=[nfp_name])
                scaler.fit(y_df)

            self.standandizer_map[env_id] = std_mapper_X, std_mappers_ys
            new_data = self.apply_standardizer_to_env_data(single_env_data)
            new_env_data_list.append(new_data)
        return new_env_data_list

    def inverse_transform_pred(self, y_pred, env_data):
        new_y_list = []
        for ys, single_env_data in zip(y_pred, env_data):
            if single_env_data is None:
                new_y_list.append([])
            else:
                env_id = single_env_data.env_id
                std_mapper_X, std_mappers_y = self.standandizer_map[env_id]
                nfp_name = single_env_data.get_selected_nfp_name()
                scaler = std_mappers_y[nfp_name]
                # y_df = pd.DataFrame(ys, columns=[nfp_name])
                if self.standardize_y:
                    if len(np.array(ys).shape) == 1:
                        new_ys = scaler.inverse_transform(np.atleast_2d(ys)).ravel()
                    else:
                        new_ys = scaler.inverse_transform(ys)
                    new_y_list.append(new_ys)
                else:
                    new_y_list.append(ys)
        return new_y_list


class PaiwiseOptionMapper(Preprocessing):
    def __init__(
        self,
    ):
        self.interactions = None
        self.poly = None
        self.interaction_idx = []

    @classmethod
    def get_cols_that_are_not_constant_nor_identical_cols(self, df):
        constant_cols = df.columns[df.nunique() == 1]
        non_constant_cols = [c for c in df.columns if c not in constant_cols]
        df_non_constant = df.loc[:, non_constant_cols]
        identical_cols = []
        non_identical_col_names = []
        for col_name in non_constant_cols:
            col = tuple(df_non_constant[col_name])
            if col not in identical_cols:
                identical_cols.append(col)
                non_identical_col_names.append(col_name)
        return non_identical_col_names

    def perform_polynomial_mapping(self, df):
        # Perform polynomial feature mapping with degree 2
        if not self.poly:
            self.poly = PolynomialFeatures(
                degree=2, include_bias=False, interaction_only=True
            )
            transformed_df = self.poly.fit_transform(df)
        else:
            transformed_df = self.poly.transform(df)
        df_poly = pd.DataFrame(
            transformed_df, columns=self.poly.get_feature_names_out(df.columns)
        )
        if len(df) < 2:
            # we just keep the training data as there are no influences to be learned either way
            df_result = df_poly
        else:
            new_cols = self.get_cols_that_are_not_constant_nor_identical_cols(df_poly)
            # Concatenate original DataFrame with polynomial features
            df_result = df_poly.loc[:, new_cols]
        self.store_final_interactions(df_result)
        return df_result

    def store_final_interactions(self, df):
        self.interactions = [str(c).split(" ") for c in df.columns]

    def map_with_stored_interactions(self, df_X):
        mapped_cols = []
        inter_names = [" ".join(inter) for inter in self.interactions]
        for interaction in self.interactions:
            cols = df_X[interaction]
            interaction_activison = np.prod(
                cols.values, axis=1
            )  # pd. cols.multiply(axis=1)
            mapped_cols.append(interaction_activison)
        if not mapped_cols:
            return None
        result_df = pd.DataFrame(np.array(mapped_cols).T, columns=inter_names)
        return result_df

    def transform(self, env_data: List[SingleEnvData]):
        new_env_data_list = []
        for single_env_data in env_data:
            X = single_env_data.get_X()
            mapped_X = self.map_with_stored_interactions(X)
            new_env = copy.deepcopy(single_env_data)
            new_env.set_X(mapped_X)
            new_env_data_list.append(new_env)
        return new_env_data_list

    def fit_transform(self, env_data: List[SingleEnvData]):
        new_env_data_list = []
        all_term_names = set()
        for single_env_data in env_data:
            X = single_env_data.get_X()
            new_X = self.perform_polynomial_mapping(X)
            poly_cols = set(new_X.columns)
            all_term_names = all_term_names.union(poly_cols)
        transformed_with_merged_interactions = self.transform(env_data)
        return transformed_with_merged_interactions

    def inverse_transform_pred(self, y_pred, env_data):
        return y_pred
