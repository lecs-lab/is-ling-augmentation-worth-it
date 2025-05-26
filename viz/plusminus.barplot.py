"""
Creates line plots for each task showing the performance improvement
(from the baseline) of each augmentation strategy in isolation.
"""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shared
from matplotlib.lines import Line2D

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
        x="Method",
        y=f"Δ {metric}",
        kind="bar",
        hue="Method",
        order=strategies,
        palette=shared.method_colors,
        errorbar=None,
        legend=False,
        height=3,
        aspect=1,
    )
    for ax in plot.axes.flat:
        ax.axhline(0, color="black", linestyle="-", alpha=1.0, zorder=0, linewidth=2)
    plot.savefig(output_dir / f"{language}.{task}.{metric}.pdf", format="pdf")


def create_legend(output_dir: pathlib.Path):
    def _handle(m):
        ls = "-" if shared.method_dashes[m] == "" else ":"
        return Line2D(
            [0], [0], lw=4, color=shared.method_colors[m], linestyle=ls, label=m
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axis_off()
    ling_handles = [_handle(m) for m in shared.linguistic_strategies]
    nonl_handles = [_handle(m) for m in shared.non_linguistic_strategies]
    legend_kw = dict(frameon=False, loc="upper center", bbox_transform=fig.transFigure)
    leg_ling = plt.legend(
        handles=ling_handles,
        title=None,
        ncol=len(ling_handles),
        bbox_to_anchor=(0.01, 0.95),
        **legend_kw,
    )
    leg_nonling = plt.legend(
        handles=nonl_handles,
        title=None,
        ncol=len(ling_handles),
        bbox_to_anchor=(0.01, 0.98),
        **legend_kw,
    )
    fig.add_artist(leg_nonling)
    fig.add_artist(leg_ling)
    fig.delaxes(ax)
    fig.savefig(
        output_dir / "legend.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01
    )


output_folder = pathlib.Path(__file__).parent / "figures/strategy_plus_minus"
output_folder.mkdir(parents=True, exist_ok=True)
for file in (pathlib.Path(__file__).parent / "results").iterdir():
    if file.suffix != ".csv":
        continue
    create_plot(csv_path=file, output_dir=output_folder)
    create_legend(output_dir=output_folder)
