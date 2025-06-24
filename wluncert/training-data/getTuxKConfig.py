import openml
import pandas as pd
import os


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

data_dir = "tuxkconfig_datasets"
merged_path = "tuxkconfig_merged.csv"
os.makedirs(data_dir, exist_ok=True)

merged_df = pd.DataFrame()

for name, dataset_id, version in datasets:
    file_path = os.path.join(data_dir, f"{name}.csv")

    if os.path.exists(file_path):
        print(f"✅ {name}.csv found, using existing file.")
        df = pd.read_csv(file_path)
    else:
        try:
            print(f"⬇️ Downloading {name} (ID: {dataset_id})")
            dataset = openml.datasets.get_dataset(dataset_id)
            df, *_ = dataset.get_data()
            df.to_csv(file_path, index=False)
            print(f"📝 Saved {name}.csv.")
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")
            continue

    df["version"] = version
    merged_df = pd.concat([merged_df, df], ignore_index=True)

merged_df.to_csv(merged_path, index=False)
print(f"📝 Merged dataset written to {merged_path}.")
