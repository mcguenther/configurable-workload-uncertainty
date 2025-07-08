import sys

print(
    sys.version
)  # Full version string (e.g., '3.11.4 (main, Jun  7 2023, 12:33:22) [Clang 14.0.0]')

print(
    sys.version_info
)  # Version tuple (e.g., sys.version_info(major=3, minor=11, micro=4, releaselevel='final', serial=0))


import openml
import pandas as pd
import os

# Use parquet files for faster read/write
USE_PARQUET = True
FORCE_DOWNLOAD = True

# Dataset info: (name, openml_id, version_label)
datasets = [
    ("tuxkconfig_413", 46759, "v4.13"),
    ("tuxkconfig_415", 46739, "v4.15"),
    ("tuxkconfig_420", 46740, "v4.20"),
    ("tuxkconfig_500", 46741, "v5.00"),
    ("tuxkconfig_504", 46742, "v5.04"),
    ("tuxkconfig_507", 46743, "v5.07"),
    ("tuxkconfig_508", 46744, "v5.08"),
]

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "tuxkconfig_datasets")
merged_path = os.path.join(
    data_dir, f"tuxkconfig_merged.{ 'parquet' if USE_PARQUET else 'csv' }"
)
os.makedirs(data_dir, exist_ok=True)

merged_df = pd.DataFrame()

for name, dataset_id, version in datasets:
    extension = "parquet" if USE_PARQUET else "csv"
    file_path = os.path.join(data_dir, f"{name}.{extension}")

    if not FORCE_DOWNLOAD and os.path.exists(file_path):
        print(f"✅ {name}.{extension} found, using existing file.")
        if USE_PARQUET:
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
    else:
        print(f"⬇️ Downloading {name} (ID: {dataset_id})")
        dataset = openml.datasets.get_dataset(dataset_id)
        download = dataset.get_data()
        df, *_ = download
        print(list(df.columns[-25:][::-1]))

        y = df["vmlinux"]
        df = df.drop(df.columns[-21:], axis=1)
        df["binary-size"] = y
        print(list(df.columns[-25:][::-1]))

        if USE_PARQUET:
            df.to_parquet(file_path)
        else:
            df.to_csv(file_path, index=False)
        print(f"📝 Saved {name}.{extension}.")

    df["version"] = version
    merged_df = pd.concat([merged_df, df], ignore_index=True)

if USE_PARQUET:
    merged_df.to_parquet(merged_path)
else:
    merged_df.to_csv(merged_path, index=False)
print(f"📝 Merged dataset written to {merged_path}.")
