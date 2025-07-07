from sklearn.preprocessing import MaxAbsScaler

from data import DataLoaderStandard, DataAdapter
import pandas as pd

from abc import abstractmethod


class DataAdapterNFPBase(DataAdapter):
    def __init__(self, data_loader: DataLoaderStandard, nfp_columns, nfp_val_col="y"):
        self.environment_col_name = "nfp-id"
        self.nfp_columns = nfp_columns
        self.nfp_val_col = nfp_val_col
        super().__init__(data_loader, self.environment_col_name)

    def get_environment_col_name(self):
        return self.environment_col_name

    def get_environment_lables(self):
        return list(range(len(self.nfp_columns)))

    def get_nfps(self):
        return [self.nfp_val_col]

    def get_transformed_df(self, cleared_sys_df):
        # Call the preprocess method which can be overridden by child classes
        preprocessed_df = self.preprocess_df(cleared_sys_df)

        # Melt the DataFrame
        df_melted = preprocessed_df.melt(
            id_vars=[
                col for col in preprocessed_df.columns if col not in self.nfp_columns
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
            + [col for col in preprocessed_df.columns if col not in self.nfp_columns]
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

    @abstractmethod
    def preprocess_df(self, df):
        """
        Preprocess the DataFrame before transformation.
        This method should be implemented by child classes.
        """
        pass

    def factorize_workload_col(self, df):
        # No need to factorize as we're using indices directly
        return df


class DataAdapterNFPApache(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = ["performance", "cpu", "energy", "fixed-energy"]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        # Use only latest version
        if "revision" in df.columns:
            df = df.loc[df["revision"] == df["revision"].max()]
            df = df.drop(columns=["revision"])
        return df


class DataAdapterNFPbrotli(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = ["performance", "energy"]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        # Use only latest version
        if "revision" in df.columns:
            df = df.loc[df["revision"] == df["revision"].max()]
            df = df.drop(columns=["revision"])
        return df


class DataAdapterNFP7z(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = [
            "performance",
            "energy",
        ]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        return df


class DataAdapterNFPexastencil(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = [
            "performance",
            "energy",
        ]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        return df


class DataAdapterNFPHSQLDB(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = [
            "performance",
            "energy",
        ]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        return df


class DataAdapterNFPjump3r(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = ["performance", "energy"]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        # Use only latest version
        cat_var = "workload"
        if cat_var in df.columns:
            df = df.loc[df[cat_var] == df[cat_var].max()]
            df = df.drop(columns=[cat_var])
        return df


class DataAdapterNFPkanzi(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = ["performance", "energy"]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        # Use only latest version
        cat_var = "workload"
        if cat_var in df.columns:
            df = df.loc[df[cat_var] == df[cat_var].max()]
            df = df.drop(columns=[cat_var])
        return df


class DataAdapterNFPLLVM(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = [
            "performance",
            "energy",
        ]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        return df


class DataAdapterNFPlrzip(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = [
            "performance",
            "energy",
        ]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        return df


class DataAdapterNFPMongoDB(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = [
            "performance",
            "energy",
        ]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        return df


class DataAdapterNFPnginx(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = [
            "performance",
            "energy",
        ]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        return df


class DataAdapterNFPposgreSQL(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = ["performance", "energy"]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        # Use only latest version
        if "revision" in df.columns:
            df = df.loc[df["revision"] == df["revision"].max()]
            df = df.drop(columns=["revision"])
        return df


class DataAdapterNFPposgreVP8(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = ["performance", "energy"]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        # Use only latest version
        if "revision" in df.columns:
            df = df.loc[df["revision"] == df["revision"].max()]
            df = df.drop(columns=["revision"])
        return df


class DataAdapterNFPx264(DataAdapterNFPBase):
    def __init__(self, data_loader: DataLoaderStandard):
        nfp_columns = [
            "performance",
            "energy",
        ]
        super().__init__(data_loader, nfp_columns)

    def preprocess_df(self, df):
        return df
