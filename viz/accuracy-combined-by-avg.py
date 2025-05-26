# %%
import pandas as pd
import pandas.io.formats.style
import seaborn as sns

from utils import create_filtered_dataframe, method_names

METRIC = "chrF"

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
final_scores = final_df.groupby(["Method"], as_index=False)[
    [f"eval/{METRIC}", f"test/{METRIC}"]
].mean()

if language == "Arapaho":
    index = final_scores[final_scores["Method"] == "Baseline"].index
    final_scores.drop(index, inplace=True)
# %%
top_runs = final_scores.nlargest(3, f"eval/{METRIC}")

# %%
# Isolate baseline and top 5 combination runs
baseline_df = filtered_df[filtered_df["Method"] == "Baseline"][
    ["training_size", f"test/{METRIC}"]
].copy()
baseline_df = baseline_df.rename(columns={f"test/{METRIC}": f"baseline_{METRIC}"})

num_one = top_runs.iloc[0]["Method"]
aug_one = filtered_df["Method"] == num_one
first_df = filtered_df[aug_one]

num_two = top_runs.iloc[1]["Method"]
aug_two = filtered_df["Method"] == num_two
second_df = filtered_df[aug_two]

num_three = top_runs.iloc[2]["Method"]
aug_three = filtered_df["Method"] == num_three
third_df = filtered_df[aug_three]


# %%
df = pd.concat([first_df, second_df, third_df])

# %%
result = pd.merge(df, baseline_df, on="training_size", how="left")

# %%
result[f"{METRIC}_diff"] = result[f"test/{METRIC}"] - result[f"baseline_{METRIC}"]

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
    "Del-Excl + Dup + Ins-Conj": "#254653",  # dark blue
    "Del-Excl + Dup + Ins-Conj + Ins-Noise": "#299D8F",  # teal
    "Dup + Ins-Conj + Ins-Noise": "#F4A261",  # light orange
    "Del + Dup + Ins-Conj + Upd-TAM": "#43E0D8",  # light blue
    "Ins-Conj + Upd-TAM": "#E76F51",  # dark orange
    "Del-Excl + Del + Ins-Conj + Upd-TAM": "#E9C46A",  # yellow
    "Del-Excl + Dup + Ins-Conj + Ins-Noise + Upd-TAM": "#aaaaaa",  # gray
    "Del + Dup + Ins-Conj + Ins-Noise": "#000000",  # black
    "Del-Excl + Del + Ins-Conj": "#CC7722",  # ochre
    # Arapaho
    "Ins-Intj": "#254653",  # dark blue
    "Ins-Intj + Ins-Noise": "#299D8F",  # teal
    "Ins-Noise": "#F4A261",  # light orange
    # "Insert interjection,  Insert noise": "#43E0D8",  # light blue
    # "Del-Excl": "#E76F51",  # dark orange
    # "Ins-Intj": "#E9C46A",  # yellow
}

# %%
# BLEU Score visualization
combined_results = sns.relplot(
    data=result,
    x="training_size",
    y=f"{METRIC}_diff",
    kind="line",
    palette=method_colors,
    hue="Method",
    errorbar=None,
    legend=False,
)
combined_results.set_axis_labels("Training Size", f"Δ {METRIC}")
add_grid_lines(combined_results)

# Output to file
combined_results.savefig(
    f"{language}_{experiment_name}_combined_by_avg_{METRIC}.pdf", format="pdf"
)

# %%
# Get mean and std for each method/training size combo
result["mean"] = result.groupby(["Method", "training_size"])[
    f"{METRIC}_diff"
].transform("mean")
result["std"] = result.groupby(["Method", "training_size"])[f"{METRIC}_diff"].transform(
    "std"
)

# %%
# Remove duplicates of the same runs
result.sort_values(by=f"{METRIC}_diff", ascending=False, inplace=True)
result.drop_duplicates(subset=["Method", "training_size"], inplace=True)

# %%
# Format output for Latex table 'mean (std)'
for index, row in result.iterrows():
    result.at[index, "final"] = f"{row['mean']:.2f} ({row['std']:.2f})"

# %%
# Clean up columns and reformat so the columns correspond to training sizes
output = result.melt(
    id_vars=["Method", "training_size"], value_vars="final", value_name=METRIC
)

output.drop(columns="variable", inplace=True)
output.reset_index(drop=True, inplace=True)

output = output.pivot(columns="training_size", index=["Method"])

# %%
s = pd.io.formats.style.Styler(output, precision=2)

latex = s.to_latex(
    column_format="p{5cm}| ccccc",
    clines="all;index",
    caption=f"{METRIC} score differential between baseline and test across the top three method combinations. Reported as the mean over three runs, with the format mean(std).",
    label="tab:accuracy_combined_scores",
)

# %%
print(latex)
