import openml
import pandas as pd
import os
import hashlib
import json


def hash_file(path):
    """Return the MD5 hash of the file contents"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


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

# Directory and hashfile
data_dir = "tuxkconfig_datasets"
hash_path = os.path.join(data_dir, "hashes.json")
merged_path = "tuxkconfig_merged.csv"
os.makedirs(data_dir, exist_ok=True)

# Load saved hashes (if any)
if os.path.exists(hash_path):
    with open(hash_path, "r") as f:
        saved_hashes = json.load(f)
else:
    saved_hashes = {}

merged_df = pd.DataFrame()
all_valid = True

for name, dataset_id, version in datasets:
    file_path = os.path.join(data_dir, f"{name}.csv")
    file_hash = saved_hashes.get(name)
    needs_download = True

    if os.path.exists(file_path):
        current_hash = hash_file(file_path)
        if current_hash == file_hash:
            print(f"✅ {name}.csv is valid (hash matches).")
            df = pd.read_csv(file_path)
            needs_download = False
        else:
            print(f"⚠️ Hash mismatch for {name}.csv, will re-download.")
    else:
        print(f"📁 {name}.csv not found, will download.")

    if needs_download:
        try:
            print(f"⬇️ Downloading {name} (ID: {dataset_id})")
            dataset = openml.datasets.get_dataset(dataset_id)
            df, *_ = dataset.get_data()
            df.to_csv(file_path, index=False)

            # Recalculate and store hash
            saved_hashes[name] = hash_file(file_path)
            print(f"📝 Saved {name}.csv and updated hash.")
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")
            all_valid = False
            continue

    df["version"] = version
    merged_df = pd.concat([merged_df, df], ignore_index=True)

# Save updated hashes
with open(hash_path, "w") as f:
    json.dump(saved_hashes, f, indent=2)

# Merged file hash check
merged_ok = False
if os.path.exists(merged_path):
    merged_hash = hash_file(merged_path)
    expected_hash = saved_hashes.get("merged")
    if merged_hash == expected_hash:
        print(f"✅ Merged file is valid (hash matches).")
        merged_ok = True
    else:
        print(f"⚠️ Merged file hash mismatch. Will re-merge.")

if not merged_ok and all_valid:
    merged_df.to_csv(merged_path, index=False)
    saved_hashes["merged"] = hash_file(merged_path)
    with open(hash_path, "w") as f:
        json.dump(saved_hashes, f, indent=2)
    print(f"📝 Merged file written and hash saved.")
