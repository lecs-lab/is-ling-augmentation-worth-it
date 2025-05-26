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

from utils import create_filtered_dataframe, method_names

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def create_plot(csv_path: pathlib.Path, output_dir: pathlib.Path, metric="chrF"):
    language, task = csv_path.stem.split(".")
    df = create_filtered_dataframe.create_filtered_dataframe(csv_path)
    df = method_names.method_names(df)

    # Select individual strategies
    strategies = ["Ins-Noise"]
    if language == "usp":
        strategies += ["Ins-Conj", "Del", "Del-Excl", "Upd-TAM", "Dup"]
    elif language == "arp":
        strategies += ["Ins-Intj", "Perm"]

    # Average the metric differential over all
    diffs = []
    for strategy in strategies:
        with_strategy = df[df["Method"].str.contains(strategy, regex=False)]
        without_strategy = df[~df["Method"].str.contains(strategy, regex=False)]
        diff = (
            with_strategy[f"test/{metric}"].mean()
            - without_strategy[f"test/{metric}"].mean()
        )
        diffs.append({"Method": strategy, f"Δ {metric}": diff})
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


output_folder = pathlib.Path(__file__).parent / "figures/strategy_plus_minus"
output_folder.mkdir(parents=True, exist_ok=True)
for file in (pathlib.Path(__file__).parent / "results").iterdir():
    if file.suffix != ".csv":
        continue
    create_plot(csv_path=file, output_dir=output_folder)
