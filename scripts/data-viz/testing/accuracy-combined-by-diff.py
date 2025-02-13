# %%
import pandas as pd
import numpy as np
import seaborn as sns
import math
import typing
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
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
# Isolate baseline runs and create baseline BLEU column
baseline_df = filtered_df[filtered_df["Method"] == "Baseline"][
    ["training_size", "test/BLEU"]
].copy()
baseline_df = baseline_df.rename(columns={"test/BLEU": "baseline_BLEU"})

# %%
final_df = pd.merge(final_df, baseline_df, on="training_size", how="left")

# %%
# Create copy of combined dataframe to ID top runs from
df = final_df.copy()

# %%
df["BLEU_diff"] = df["eval/BLEU"] - df["baseline_BLEU"]

# %%
# Force unique combinations in the top 5
df = df.sort_values('BLEU_diff', ascending=False)
df.drop_duplicates(subset='Name', inplace=True)
df.drop_duplicates(subset='Method', inplace=True)

# %%
top_runs = df.nlargest(5, 'BLEU_diff')

# %%
# Isolate top 5 combination runs
num_one = top_runs.iloc[0]['Method']
aug_one = (final_df['Method'] == num_one)
first_df = final_df[aug_one]

num_two = top_runs.iloc[1]['Method']
aug_two = (final_df['Method'] == num_two)
second_df = final_df[aug_two]

num_three = top_runs.iloc[2]['Method']
aug_three = (final_df['Method'] == num_three)
third_df = final_df[aug_three]

num_four = top_runs.iloc[3]['Method']
aug_four = (final_df['Method'] == num_four)
fourth_df = final_df[aug_four]

num_five = top_runs.iloc[4]['Method']
aug_five = (final_df['Method'] == num_five)
fifth_df = final_df[aug_five]


# %%
result = pd.concat([first_df, second_df, third_df, fourth_df, fifth_df])

# %%
result["BLEU_diff"] = result["test/BLEU"] - result["baseline_BLEU"]

# %%
unique_training_sizes = sorted(result["training_size"].unique())


def add_grid_lines(facetgrid):
    for ax in facetgrid.axes.flat:
        for tx in unique_training_sizes:
            ax.axvline(
                tx, color="gray", linestyle="--", alpha=0.2, zorder=0, linewidth=0.5
            )
        ax.axhline(0, color="black", linestyle="-", alpha=1.0, zorder=0, linewidth=2)



# %%
# Create legend for combined methods plot
method_colors = {
    "Delete with exclusions,  Random delete,  Random duplicate,  Insert conjunction": "#254653", #dark blue 
    "Delete with exclusions,  Random delete,  Insert conjunction": "#299D8F",  # teal
    "Delete with exclusions,  Random duplicate,  Insert conjunction,  Insert noise": "#F4A261",  # light orange
    "Random duplicate,  Insert noise,  TAM update": "#43E0D8",  # light blue
    "Delete with exclusions,  Random duplicate,  Insert conjunction,  Insert noise,  TAM update": "#E76F51",  # dark orange
    "Random delete,  Insert conjunction": "#E9C46A",  # yellow
    "Random delete,  Random duplicate,  Insert conjunction,  TAM update": "#aaaaaa",  # gray
    "Delete with exclusions,  Random delete,  Insert conjunction,  TAM update": "#000000", # black
    "Insert conjunction,  TAM update": "#CC7722" #ochre
    
}

handles = [
    mlines.Line2D(
        [], [], color=color, marker="o", linestyle="None", markersize=8, label=label
    )
    for label, color in method_colors.items()
]
num_columns = math.ceil(len(handles) / 2)
fig_legend, ax_legend = plt.subplots(figsize=(8, 0.5))
legend = ax_legend.legend(
    handles=handles, title="Method", loc="center", ncol=num_columns, frameon=False
)
ax_legend.axis("off")
legend.set_title(None)

fig_legend.canvas.draw()
bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())

fig_legend.savefig(
    f"{language}_{experiment_name}_combined_diff_bleu_legend.pdf",
    format="pdf",
    bbox_inches=bbox,
    pad_inches=0,
)
plt.close(fig_legend)

# %%
# BLEU Score visualization
combined_bleu = sns.relplot(
    data=result,
    x="training_size",
    y="BLEU_diff",
    kind="line",
    palette=method_colors,
    hue="Method",
    errorbar=None,
    legend=False,
)
combined_bleu.set_axis_labels('Training Size', 'Δ BLEU')
add_grid_lines(combined_bleu)

# Output to file
combined_bleu.savefig(
    f"{language}_{experiment_name}_combined_diff_bleu.pdf", format="pdf"
)

# %%
# chrF Score visualization
# combined_chrf = sns.relplot(
#     data=result,
#     x="training_size", y="test/chrF", kind='line', hue='Method', errorbar=None
# )
# combined_chrf.set_axis_labels('Training Size', 'chrF Score')

# # Output to file
# combined_chrf.savefig('combined_chrf.png')

# %%
# # Loss Curve Visualization
# combined_loss = sns.relplot(
#     data=result,
#     x="training_size", y="test/loss", kind='line', hue='Method', errorbar=None
# )
# combined_loss.set_axis_labels('Training Size', 'Loss')

# # Output to file
# combined_loss.savefig('combined_bleu.png')


