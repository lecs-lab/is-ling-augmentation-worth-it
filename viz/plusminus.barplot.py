"""
Creates line plots for each task showing the performance impact of each
strategy by taking the mean difference of combinations with the strategy
versus those without.
"""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shared

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

metric = "chrF"


def create_plot(csv_path: pathlib.Path, output_dir: pathlib.Path):
    language, task = csv_path.stem.split(".")
    df = shared.create_filtered_dataframe(csv_path)
    df = shared.method_names(df)

    # Select individual strategies
    strategies = ["Ins-Noise"]
    if language == "usp":
        strategies += ["Ins-Conj", "Del-Any", "Del-Excl", "Upd-TAM", "Dup"]
    elif language == "arp":
        strategies += ["Ins-Intj", "Perm"]

    # Average the metric differential over all
    diffs = []
    for strategy in strategies:

        def remove_strategy(method: str):
            return " + ".join([s for s in method.split(" + ") if s != strategy])

        with_strategy = df[df["Method"].str.contains(strategy, regex=False)].rename(
            columns={f"test/{metric}": "score_with"}
        )
        with_strategy["other_strategies"] = with_strategy["Method"].apply(  # type:ignore
            remove_strategy
        )
        without_strategy = df[~df["Method"].str.contains(strategy, regex=False)].rename(
            columns={f"test/{metric}": "score_without", "Method": "other_strategies"}
        )
        paired = with_strategy[
            ["other_strategies", "training_size", "random-seed", "score_with"]
        ].merge(
            right=without_strategy[
                ["other_strategies", "training_size", "random-seed", "score_without"]
            ],
            on=["other_strategies", "training_size", "random-seed"],
            how="inner",
        )
        strategy_diffs = paired["score_with"] - paired["score_without"]
        diffs.append(
            {
                "Method": strategy,
                f"Δ {metric}": strategy_diffs.mean(),
                "std": strategy_diffs.std(ddof=1),
            }
        )
    diffs = pd.DataFrame(diffs)

    plot = sns.catplot(
        data=diffs,
        x=f"Δ {metric}",
        y="Method",
        kind="bar",
        hue="Method",
        order=strategies,
        palette=shared.method_colors,
        errorbar=None,
        legend=False,
        height=3,
        aspect=1,
        orient="h",
    )
    for ax in plot.axes.flat:
        ax.axvline(0, color="black", linestyle="-", alpha=1.0, zorder=0, linewidth=2)
        ax.set_ylabel("")
    plot.savefig(output_dir / f"{language}.{task}.{metric}.pdf", format="pdf")
    return diffs


output_folder = pathlib.Path(__file__).parent / "figures/strategy_plus_minus"
output_folder.mkdir(parents=True, exist_ok=True)
all_dfs = []
for file in (pathlib.Path(__file__).parent / "results").iterdir():
    if file.suffix != ".csv":
        continue
    df = create_plot(csv_path=file, output_dir=output_folder)
    df["task"] = file.stem
    all_dfs.append(df)

all_dfs = pd.concat(all_dfs)
methods = sorted(all_dfs["Method"].unique())
latex_table_string = "Task & " + " & ".join(methods)

for task, group in all_dfs.groupby("task"):
    scores = [group[group["Method"] == method] for method in methods]
    scores = [
        f"{score[f'Δ {metric}'].item():.2f} ({score['std'].item():.2f})"
        if len(score) == 1
        else " "
        for score in scores
    ]
    scores = " & ".join(scores)
    latex_table_string += f"\n{task} & {scores}"

print(latex_table_string)
