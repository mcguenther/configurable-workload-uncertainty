import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("./results/last_experiment.csv")
    # sns.lineplot(df, x="exp_id", y="err", style="err_type", )
    selected_error_df = df[df["err_type"] == "mape"]
    sns.relplot(data=selected_error_df, x="exp_id", y="err",
                hue="model", col="env", row="setting", kind="line", )
    plt.yscale("log")
    plt.show()
    plt.savefig("./results/multitaks-result.png", )


if __name__ == "__main__":
    main()
