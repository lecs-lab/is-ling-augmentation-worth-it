# %%
import typing

import pandas as pd
import seaborn as sns
from viz_utils import create_filtered_dataframe, method_names

# %%
# Have user enter csv file name
csv_file = input("Enter CSV file name, including its relative path: ")

if "transl-transc" in csv_file:
    experiment_name = "transl-transc"
elif "transc-transl" in csv_file:
    experiment_name = "transc-transl"
elif "gloss" in csv_file:
    experiment_name = "gloss"
else:
    raise ValueError()


filtered_df = create_filtered_dataframe.create_filtered_dataframe(csv_file)
if "usp" in csv_file:
    language = "Uspanteko"
elif "arp" in csv_file:
    language = "Arapaho"
else:
    raise ValueError()

# %%
final_df = method_names.method_names(filtered_df)

# %%
# Isolate baseline and individual method runs
strategies = ["Ins-Noise"]

if language == "Uspanteko":
    strategies += [
        "Del-Excl",
        "Del",
        "Ins-Conj",
        "Upd-TAM",
        "Dup",
    ]
elif language == "Arapaho":
    strategies += ["Ins-Intj", "Perm"]

baseline_df = filtered_df[filtered_df["Method"] == "Baseline"][
    ["training_size", "test/BLEU", "test/chrF"]
].copy()
baseline_df = baseline_df.rename(columns={"test/BLEU": "baseline_BLEU", "test/chrF": "baseline_chrF"}) # type:ignore

df = filtered_df[filtered_df["Method"].isin(strategies)]
df = typing.cast(pd.DataFrame, df)

df = pd.merge(df, baseline_df, on="training_size", how="left")
df["BLEU_diff"] = df["test/BLEU"] - df["baseline_BLEU"]
df["chrF_diff"] = df["test/chrF"] - df["baseline_chrF"]


method_colors = {
    "Ins-Noise": "#254653",  # blue
    "Del-Excl": "#299D8F",  # teal
    "Del": "#F4A261",  # light orange
    "Ins-Conj": "#43E0D8",  # light blue
    "Upd-TAM": "#E76F51",  # dark orange
    "Dup": "#E9C46A",  # yellow
    "Ins-Intj": "#43E0D8",  # light blue
    "Perm": "#bbbbbb",  # gray
}

unique_training_sizes = sorted(df["training_size"].unique())


def add_grid_lines(facetgrid):
    for ax in facetgrid.axes.flat:
        for tx in unique_training_sizes:
            ax.axvline(
                tx, color="gray", linestyle="--", alpha=0.2, zorder=0, linewidth=0.5
            )
        ax.axhline(0, color="black", linestyle="-", alpha=1.0, zorder=0, linewidth=2)


# %%
# BLEU Score visualization
individual_bleu = sns.relplot(
    data=df,
    x="training_size",
    y="BLEU_diff",
    kind="line",
    hue="Method",
    palette=method_colors,
    errorbar=None,
    legend=False,
)
individual_bleu.set_axis_labels("Training Size", "Δ BLEU")
add_grid_lines(individual_bleu)

# Output to file
individual_bleu.savefig(
    f"{language}_{experiment_name}_individual_bleu.pdf", format="pdf"
)

# %%
# chrF Score visualization
individual_chrf = sns.relplot(
    data=df,
    x="training_size",
    y="chrF_diff",
    kind="line",
    hue="Method",
    palette=method_colors,
    errorbar=None,
    legend=False,
)
individual_chrf.set_axis_labels("Training Size", "Δ chrF")
add_grid_lines(individual_chrf)

# Output to file
individual_chrf.savefig(
    f"{language}_{experiment_name}_individual_chrf.pdf", format="pdf"
)


# # %%
# # Loss Curve Visualization
# individual_loss = sns.relplot(
#     data=df,
#     x="training_size",
#     y="test/loss",
#     kind="line",
#     hue="Method",
#     errorbar=None,
# )
# individual_loss.set_axis_labels("Training Size", "Loss")

# # Output to file
# individual_loss.savefig(f"{experiment_name}_individual_loss.pdf", format="pdf")


print("Done. Check data-viz folder for the outputted PNG files.")
