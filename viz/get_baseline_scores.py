"""Usage: python viz/get_baseline_scores.py <results_file>

Produces a Latex-formatted table for the baseline results (no augmentation)
"""

import argparse

from utils import create_filtered_dataframe, method_names

parser = argparse.ArgumentParser()
parser.add_argument("results_file")
args = parser.parse_args()

df = create_filtered_dataframe.create_filtered_dataframe(args.results_file)
df = method_names.method_names(df)

df = df[df["Method"] == "Baseline"]
result = df.groupby("training_size").agg({"test/chrF": ["mean", "std"]})
print(result)

formatted_results = []
for training_size in sorted(result.index):
    mean_val = result.loc[training_size, ("test/chrF", "mean")]
    std_val = result.loc[training_size, ("test/chrF", "std")]
    formatted_results.append(f"{mean_val:.1f} ({std_val:.1f})")

print(" & ".join(formatted_results))
