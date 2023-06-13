import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("./results/last_experiment.csv")
    # sns.lineplot(df, x="exp_id", y="err", style="err_type", )
    err_type = "mape"
    # err_type = "R2"
    selected_error_df = df[df["err_type"] == err_type]
    sns.relplot(data=selected_error_df, x="exp_id", y="err",
                hue="model", col="env", kind="line", col_wrap=4, )#row="setting", )
    # plt.yscale("log")
    plt.ylim((0, 50))
    plt.savefig("./results/multitask-result.png", )
    plt.show()


if __name__ == "__main__":
    main()
