# %%
import pandas as pd
import numpy as np
import seaborn as sns
import math
import typing
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from viz_utils import create_filtered_dataframe, method_names
import pandas.io.formats.style

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
# Average across runs for each method combo
final_bleu = final_df.groupby(['Method'], as_index=False)[['eval/BLEU', 'test/BLEU']].mean()

# %%
top_runs = final_bleu.nlargest(3, 'eval/BLEU')

# %%
# Isolate baseline and top 5 combination runs
baseline_df = filtered_df[filtered_df["Method"] == "Baseline"][
    ["training_size", "test/BLEU"]
].copy()
baseline_df = baseline_df.rename(columns={"test/BLEU": "baseline_BLEU"})

num_one = top_runs.iloc[0]['Method']
aug_one = (filtered_df['Method'] == num_one)
first_df = filtered_df[aug_one]

num_two = top_runs.iloc[1]['Method']
aug_two = (filtered_df['Method'] == num_two)
second_df = filtered_df[aug_two]

num_three = top_runs.iloc[2]['Method']
aug_three = (filtered_df['Method'] == num_three)
third_df = filtered_df[aug_three]


# %%
df = pd.concat([first_df, second_df, third_df])

# %%
result = pd.merge(df, baseline_df, on="training_size", how="left")

# %%
result["BLEU_diff"] = result["test/BLEU"] - result["baseline_BLEU"]

# %%
unique_training_sizes = sorted(df["training_size"].unique())


def add_grid_lines(facetgrid):
    for ax in facetgrid.axes.flat:
        for tx in unique_training_sizes:
            ax.axvline(
                tx, color="gray", linestyle="--", alpha=0.2, zorder=0, linewidth=0.5
            )
        ax.axhline(0, color="black", linestyle="-", alpha=1.0, zorder=0, linewidth=2)

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
    f"{language}_{experiment_name}_combined_by_avg_bleu.pdf", format="pdf"
)

# %%
# Get mean and std for each method/training size combo
result['mean'] = result.groupby(['Method', 'training_size'])['BLEU_diff'].transform('mean')
result['std'] = result.groupby(['Method', 'training_size'])['BLEU_diff'].transform('std')

# %%
# Remove duplicates of the same runs
result.sort_values(by='BLEU_diff', ascending=False, inplace=True)
result.drop_duplicates(subset=['Method', 'training_size'], inplace=True)

# %%
# Format output for Latex table 'mean (std)'
for index, row in result.iterrows():
    result.at[index,'final'] = f"{row['mean']:.2f} ({row['std']:.2f})"

# %%
# Clean up columns and reformat so the columns correspond to training sizes
output = result.melt(id_vars=['Method', 'training_size'], value_vars='final', value_name='BLEU')

output.drop(columns='variable', inplace=True)
output.reset_index(drop=True, inplace=True)

output = output.pivot(columns='training_size', index=['Method'])

# %%
s = pd.io.formats.style.Styler(output, precision=2)

latex = s.to_latex(
    column_format='p{5cm}| ccccc',
    clines= 'all;index',
    caption="BLEU score differential between baseline and test across the top three method combinations. Reported as the mean over three runs, with the format mean(std).",
    label='tab:accuracy_combined_scores'
)

# %%
print(latex)

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


