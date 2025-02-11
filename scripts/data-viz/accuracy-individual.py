# %%
import pandas as pd
import seaborn as sns
from viz_utils import create_filtered_dataframe, method_names

# %%
# Have user enter csv file name
csv_file = input("Enter CSV file name, including its relative path: ")

if 'transl-transc' in csv_file:
    experiment_name = 'transl-transc'
elif 'transc-transl' in csv_file:
    experiment_name = 'transc-transl'
elif 'gloss' in csv_file:
    experiment_name = 'gloss'


filtered_df = create_filtered_dataframe.create_filtered_dataframe(csv_file)
if 'usp' in csv_file:
    language = 'Uspanteko'
elif 'arp' in csv_file:
    language = 'Arapaho'

# %%
final_df = method_names.method_names(filtered_df)

# %%
# Isolate baseline and individual method runs
aug_baseline = (filtered_df['Method'] == 'Baseline') 
baseline_df = filtered_df[aug_baseline]

aug_noise = (filtered_df['Method'] == 'Insert noise')
noise_df = filtered_df[aug_noise]

if language == 'Uspanteko':
    aug_delete_excl = (filtered_df['Method'] == 'Delete with exclusions')
    delete_excl_df = filtered_df[aug_delete_excl]

    aug_delete = (filtered_df['Method'] == 'Random delete') 
    delete_df = filtered_df[aug_delete]

    aug_conj = (filtered_df['Method'] == 'Insert conjunction') 
    conj_df = filtered_df[aug_conj]

    aug_tam = (filtered_df['Method'] == 'TAM update')
    tam_df = filtered_df[aug_tam]

    aug_dup = (filtered_df['Method'] == 'Random duplicate') 
    dup_df = filtered_df[aug_dup]
    
    result = pd.concat([baseline_df, noise_df, delete_excl_df, delete_df, conj_df, tam_df, dup_df])


elif language == 'Arapaho':
    aug_interjection = (filtered_df['Method'] == 'Insert interjection') 
    interjection_df = filtered_df[aug_interjection]

    aug_permutations = (filtered_df['Method'] == 'Sentence permutations') 
    permutation_df = filtered_df[aug_permutations]

    result = pd.concat([baseline_df, noise_df, interjection_df, permutation_df])


# %%
# BLEU Score visualization
individual_bleu = sns.relplot(
    data=result,
    x="training_size", y="test/BLEU", kind='line', hue='Method', errorbar=None
)
individual_bleu.set_axis_labels('Training Size', 'BLEU Score')

# Output to file
individual_bleu.savefig(f'{experiment_name}_individual_bleu.png')

# %%
# chrF Score visualization
individual_chrf = sns.relplot(
    data=result,
    x="training_size", y="test/chrF", kind='line', hue='Method', errorbar=None
)
individual_chrf.set_axis_labels('Training Size', 'chrF Score')

# Output to file
individual_chrf.savefig(f'{experiment_name}_individual_chrf.png')

# %%
# Loss Curve Visualization
individual_loss = sns.relplot(
    data=result,
    x="training_size", y="test/loss", kind='line', hue='Method', errorbar=None
)
individual_loss.set_axis_labels('Training Size', 'Loss')

# Output to file
individual_loss.savefig(f'{experiment_name}_individual_loss.png')

print("Done. Check data-viz folder for the outputted PNG files.")


