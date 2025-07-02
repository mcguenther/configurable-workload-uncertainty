import openml
import pandas as pd
import os

# Use parquet files for faster read/write
USE_PARQUET = True


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

    if os.path.exists(file_path):
        print(f"✅ {name}.{extension} found, using existing file.")
        if USE_PARQUET:
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
    else:
        try:
            print(f"⬇️ Downloading {name} (ID: {dataset_id})")
            dataset = openml.datasets.get_dataset(dataset_id)
            df, *_ = dataset.get_data()
            if USE_PARQUET:
                df.to_parquet(file_path)
            else:
                df.to_csv(file_path, index=False)
            print(f"📝 Saved {name}.{extension}.")
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")
            continue

    df["version"] = version
    merged_df = pd.concat([merged_df, df], ignore_index=True)

if USE_PARQUET:
    merged_df.to_parquet(merged_path)
else:
    merged_df.to_csv(merged_path, index=False)
print(f"📝 Merged dataset written to {merged_path}.")
