# %%
import pandas as pd
import seaborn as sns
from viz_utils import create_filtered_dataframe, method_names

# %%
# Have user enter csv file name
csv_file = input("Enter CSV file name, including its relative path: ")

filtered_df = create_filtered_dataframe.create_filtered_dataframe(csv_file)
if 'usp' in csv_file:
    language = 'Uspanteko'
elif 'arp' in csv_file:
    language = 'Arapaho'

# %%
final_df = method_names.method_names(filtered_df)

# %%
# Isolate baseline and top 5 combination runs
top_runs = final_df.nlargest(5, ['test/BLEU', 'test/chrF'])

aug_baseline = (filtered_df['Method'] == 'Baseline') 
baseline_df = filtered_df[aug_baseline]

num_one = top_runs.iloc[0]['Method']
aug_one = (filtered_df['Method'] == num_one)
first_df = filtered_df[aug_one]

num_two = top_runs.iloc[1]['Method']
aug_two = (filtered_df['Method'] == num_two)
second_df = filtered_df[aug_two]

num_three = top_runs.iloc[2]['Method']
aug_three = (filtered_df['Method'] == num_three)
third_df = filtered_df[aug_three]

num_four = top_runs.iloc[3]['Method']
aug_four = (filtered_df['Method'] == num_four)
fourth_df = filtered_df[aug_four]

num_five = top_runs.iloc[4]['Method']
aug_five = (filtered_df['Method'] == num_five)
fifth_df = filtered_df[aug_five]


# %%
result = pd.concat([baseline_df, first_df, second_df, third_df, fourth_df, fifth_df])

# %%
# BLEU Score visualization
combined_bleu = sns.relplot(
    data=result,
    x="training_size", y="test/BLEU", kind='line', hue='Method', errorbar=None
)
combined_bleu.set_axis_labels('Training Size', 'BLEU Score')

# Output to file
combined_bleu.savefig('combined_bleu.png')

# %%
# chrF Score visualization
combined_chrf = sns.relplot(
    data=result,
    x="training_size", y="test/chrF", kind='line', hue='Method', errorbar=None
)
combined_chrf.set_axis_labels('Training Size', 'chrF Score')

# Output to file
combined_chrf.savefig('combined_chrf.png')

# %%
# Loss Curve Visualization
combined_loss = sns.relplot(
    data=result,
    x="training_size", y="eval/loss", kind='line', hue='Method', errorbar=None
)
combined_loss.set_axis_labels('Training Size', 'Loss')

# Output to file
combined_loss.savefig('combined_loss.png')


