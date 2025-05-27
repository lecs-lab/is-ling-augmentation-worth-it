"""Usage: python viz/get_baseline_scores.py <results_file>

Produces a Latex-formatted table for the baseline results (no augmentation)
"""

import argparse

import shared

parser = argparse.ArgumentParser()
parser.add_argument("results_file")
args = parser.parse_args()

df = shared.create_filtered_dataframe(args.results_file)
df = shared.method_names(df)

df = df[df["Method"] == "Baseline"]
result = df.groupby("training_size").agg({"test/chrF": ["mean", "std"]})
print(result)

formatted_results = []
for training_size in sorted(result.index):
    mean_val = result.loc[training_size, ("test/chrF", "mean")]
    std_val = result.loc[training_size, ("test/chrF", "std")]
    formatted_results.append(f"{mean_val:.1f} ({std_val:.1f})")

print(" & ".join(formatted_results))
