"""
Creates line plots for each task showing the performance improvement
(from the baseline) of the n best combined strategies, selected using
the eval scores averaged over training sizes.
"""

import pathlib
from typing import Any, Dict, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from utils import create_filtered_dataframe, method_names

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

STYLE_CONFIG: dict[str, Any] = {
    "height": 3,
    "aspect": 1,
    "linewidth": 2,
}


def get_methods_by_language(
    results_dir: pathlib.Path, metric="chrF", num_best=3
) -> Tuple[Set[str], Set[str]]:
    """Get methods separated by language."""
    arp_methods = set()
    usp_methods = set()

    for file in results_dir.iterdir():
        if file.suffix != ".csv":
            continue

        df = create_filtered_dataframe.create_filtered_dataframe(file)
        df = method_names.method_names(df)

        # Select best strategies
        df = df[df["Method"] != "Baseline"]
        averages_by_method = df.groupby(["Method"], as_index=False)[
            [f"eval/{metric}", f"test/{metric}"]
        ].mean()
        best_methods = averages_by_method.nlargest(num_best, f"eval/{metric}")

        # Determine language from filename
        if "arp" in file.stem:
            arp_methods.update(best_methods["Method"])
        elif "usp" in file.stem:
            usp_methods.update(best_methods["Method"])

    return arp_methods, usp_methods


def create_plot(
    csv_path: pathlib.Path,
    output_dir: pathlib.Path,
    color_palette: Dict[str, str],
    metric="chrF",
    num_best=3,
) -> Set[str]:
    """Create individual plot and return the methods used."""
    language, task = csv_path.stem.split(".")
    df = create_filtered_dataframe.create_filtered_dataframe(csv_path)
    df = method_names.method_names(df)

    baseline_df = df[df["Method"] == "Baseline"][
        ["training_size", f"test/{metric}"]
    ].copy()
    baseline_df = baseline_df.rename(columns={f"test/{metric}": f"baseline_{metric}"})

    # Select best strategies
    df = df[df["Method"] != "Baseline"]
    averages_by_method = df.groupby(["Method"], as_index=False)[
        [f"eval/{metric}", f"test/{metric}"]
    ].mean()
    best_methods = averages_by_method.nlargest(num_best, f"eval/{metric}")
    df = df[df["Method"].isin(best_methods["Method"])]
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

    # Create the plot using our color palette
    g = sns.relplot(
        data=df,
        x="training_size",
        y=f"{metric}_diff",
        kind="line",
        hue="Method",
        palette=color_palette,
        errorbar=None,
        legend=False,
        **STYLE_CONFIG,
    )

    g.set_axis_labels("Training Size", f"Δ {metric}")
    add_grid_lines(g)
    g.savefig(output_dir / f"{language}.{task}.{metric}.pdf", format="pdf")
    plt.close()

    return set(df["Method"].unique())


def create_language_legend(
    methods: Set[str],
    color_palette: Dict[str, str],
    output_dir: pathlib.Path,
    language: str,
):
    """Create a legend file for a specific language."""
    fig, ax = plt.subplots(figsize=(8, 0.5))
    handles = [
        Line2D(
            [],
            [],
            color=color_palette[method],
            linestyle="-",
            linewidth=STYLE_CONFIG["linewidth"],
            label=method,
        )
        for method in sorted(methods)
    ]

    legend = ax.legend(
        handles=handles,
        ncol=len(methods) if language == "arp" else len(methods) / 3,
        loc="center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.5),
    )

    ax.set_axis_off()

    # Save legend with tight layout
    fig.savefig(
        output_dir / f"legend_{language}.pdf",
        bbox_inches="tight",
        bbox_extra_artists=[legend],
        pad_inches=0.1,
        format="pdf",
    )
    plt.close(fig)


def main():
    output_folder = pathlib.Path(__file__).parent / "figures/best_combined"
    output_folder.mkdir(parents=True, exist_ok=True)

    results_dir = pathlib.Path(__file__).parent / "results"

    # Get methods separated by language
    arp_methods, usp_methods = get_methods_by_language(results_dir)

    # Create color palette for all methods
    all_methods = arp_methods | usp_methods  # union of both sets
    palette = sns.color_palette("deep", n_colors=len(all_methods))
    color_palette = dict(zip(sorted(all_methods), palette))

    # Create all plots using the shared color palette
    for file in results_dir.iterdir():
        if file.suffix != ".csv":
            continue
        create_plot(
            csv_path=file,
            output_dir=output_folder,
            color_palette=color_palette,
        )

    # Create separate legends for each language
    create_language_legend(
        methods=arp_methods,
        color_palette=color_palette,
        output_dir=output_folder,
        language="arp",
    )
    create_language_legend(
        methods=usp_methods,
        color_palette=color_palette,
        output_dir=output_folder,
        language="usp",
    )


if __name__ == "__main__":
    main()
