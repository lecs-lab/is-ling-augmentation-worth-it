import argparse

from viz_utils import create_filtered_dataframe, method_names

parser = argparse.ArgumentParser()
parser.add_argument("results_file")
args = parser.parse_args()

df = create_filtered_dataframe.create_filtered_dataframe(args.results_file)
df = method_names.method_names(df)

df = df[df["Method"] == "Baseline"]
result = df.groupby("training_size").agg({"test/BLEU": ["mean", "std"]})
print(result)
