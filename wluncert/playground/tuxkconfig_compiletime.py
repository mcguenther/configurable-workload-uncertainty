import os
import pandas as pd


def find_dataset_path() -> str:
    """Return path to the merged TuxKConfig dataset if it exists."""
    base = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "training-data",
        "tuxkconfig_datasets",
    )
    parquet_path = os.path.join(base, "tuxkconfig_merged.parquet")
    csv_path = os.path.join(base, "tuxkconfig_merged.csv")
    if os.path.exists(parquet_path):
        return parquet_path
    if os.path.exists(csv_path):
        return csv_path
    return ""


def main() -> None:
    dataset_path = find_dataset_path()
    if not dataset_path:
        print("TuxKConfig dataset not found. Did you run getTuxKConfig.py?")
        return

    if dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    time_cols = [c for c in df.columns if "time" in c.lower()]
    if not time_cols:
        print("Dataset does not contain a compilation time column.")
        return

    col = time_cols[0]
    if len(time_cols) > 1:
        print(f"Multiple time columns found: {time_cols}. Using '{col}'.")

    avg_time = df[col].mean()
    print(
        f"Average compilation time across all configurations: {avg_time:.2f} (column '{col}')"
    )


if __name__ == "__main__":
    main()
