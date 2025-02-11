# %%
import typing

import pandas as pd
import seaborn as sns
from viz_utils import create_filtered_dataframe, method_names

# %%
# Have user enter csv file name
csv_file = input("Enter CSV file name, including its relative path: ")

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
strategies = ["Baseline", "Insert noise"]

if language == "Uspanteko":
    strategies += [
        "Delete with exclusions",
        "Random delete",
        "Insert conjunction",
        "TAM update",
        "Random duplicate",
    ]
elif language == "Arapaho":
    strategies += ["Insert noise", "Insert interjection", "Sentence permutations"]

df = filtered_df[filtered_df["Method"].isin(strategies)]
df = typing.cast(pd.DataFrame, df)

# %%
# BLEU Score visualization
individual_bleu = sns.relplot(
    data=df,
    x="training_size",
    y="test/BLEU",
    kind="line",
    hue="Method",
    errorbar=None,
)
individual_bleu.set_axis_labels("Training Size", "BLEU Score")

# Output to file
individual_bleu.savefig("individual_bleu.pdf", format="pdf")

# %%
# chrF Score visualization
individual_chrf = sns.relplot(
    data=df,
    x="training_size",
    y="test/chrF",
    kind="line",
    hue="Method",
    errorbar=None,
)
individual_chrf.set_axis_labels("Training Size", "chrF Score")

# Output to file
individual_chrf.savefig("individual_chrf.pdf", format="pdf")

# %%
# Loss Curve Visualization
individual_loss = sns.relplot(
    data=df,
    x="training_size",
    y="test/loss",
    kind="line",
    hue="Method",
    errorbar=None,
)
individual_loss.set_axis_labels("Training Size", "Loss")

# Output to file
individual_loss.savefig("individual_loss.pdf", format="pdf")

print("Done. Check data-viz folder for the outputted PNG files.")
