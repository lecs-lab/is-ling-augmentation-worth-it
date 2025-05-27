"""
Creates line plots for each task showing the performance improvement
(from the baseline) of each augmentation strategy in isolation.
"""

import pathlib
import typing

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shared
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def create_plot(csv_path: pathlib.Path, output_dir: pathlib.Path, metric="chrF"):
    language, task = csv_path.stem.split(".")
    df = shared.create_filtered_dataframe(csv_path)
    df = shared.method_names(df)

    baseline_df = df[df["Method"] == "Baseline"][
        ["training_size", f"test/{metric}"]
    ].copy()
    baseline_df = baseline_df.rename(columns={f"test/{metric}": f"baseline_{metric}"})  # type:ignore

    # Select individual strategies
    strategies = ["Ins-Noise"]
    if language == "usp":
        strategies += ["Del-Excl", "Del-Any", "Ins-Conj", "Upd-TAM", "Dup"]
    elif language == "arp":
        strategies += ["Ins-Intj", "Perm"]
    df = df[df["Method"].isin(strategies)]
    df = typing.cast(pd.DataFrame, df)
    df = pd.merge(df, baseline_df, on="training_size", how="left")
    df[f"{metric}_diff"] = df[f"test/{metric}"] - df[f"baseline_{metric}"]

    unique_training_sizes = sorted(df["training_size"].unique())

    def add_grid_lines(facetgrid):
        for ax in facetgrid.axes.flat:
            for tx in unique_training_sizes:
                ax.axvline(
                    tx, color="gray", linestyle="--", alpha=0.2, zorder=0, linewidth=0.5
                )
            ax.axhline(
                0, color="black", linestyle="-", alpha=0.5, zorder=0, linewidth=1
            )

    plot = sns.relplot(
        data=df,
        x="training_size",
        y=f"{metric}_diff",
        kind="line",
        hue="Method",
        style="Method",
        palette=shared.method_colors,
        dashes=shared.method_dashes,
        errorbar=None,
        legend=False,
        height=3,
        aspect=1,
        linewidth=2,
    )
    plot.set_axis_labels("Training Size", f"Δ {metric}")
    add_grid_lines(plot)
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


output_folder = pathlib.Path(__file__).parent / "figures/individual"
output_folder.mkdir(parents=True, exist_ok=True)
for file in (pathlib.Path(__file__).parent / "results").iterdir():
    if file.suffix != ".csv":
        continue
    create_plot(csv_path=file, output_dir=output_folder)
    create_legend(output_dir=output_folder)
