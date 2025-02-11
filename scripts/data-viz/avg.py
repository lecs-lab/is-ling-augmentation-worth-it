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
filtered_df = method_names.method_names(filtered_df)

if 'usp' in csv_file:
    language = 'Uspanteko'
elif 'arp' in csv_file:
    language = 'Arapaho'

# %%
# Create two dataframes for each method: 
# (1) Contains all runs with a method 
# (2) Contains all runs without the method

# Insert noise -- shared method
with_insert_noise = filtered_df['Method'].str.contains('Insert noise')
with_insert_noise_df = filtered_df[with_insert_noise]
with_insert_noise_df['Filter'] = 'With'

without_insert_noise_df = filtered_df[~with_insert_noise]
without_insert_noise_df['Filter'] = 'Without'

insert_noise_df = pd.concat([with_insert_noise_df, without_insert_noise_df])    # Recombine after adding Filter column
insert_noise_df['Includes'] = 'Insert noise'

if language == 'Uspanteko':
    # Random Delete
    with_random_delete = filtered_df['Method'].str.contains('Random delete')
    with_random_delete_df = filtered_df[with_random_delete]
    with_random_delete_df['Filter'] = 'With'
    
    without_random_delete_df = filtered_df[~with_random_delete]
    without_random_delete_df['Filter'] = 'Without'

    random_delete_df = pd.concat([with_random_delete_df, without_random_delete_df])
    random_delete_df['Includes'] = 'Random delete'

    # Exclusion delete
    with_exclusion_delete = filtered_df['Method'].str.contains('Delete with exclusions')
    with_exclusion_delete_df = filtered_df[with_exclusion_delete]
    with_exclusion_delete_df['Filter'] = 'With'

    without_exclusion_delete_df = filtered_df[~with_exclusion_delete]
    without_exclusion_delete_df['Filter'] = 'Without'

    exclusion_delete_df = pd.concat([with_exclusion_delete_df, without_exclusion_delete_df])
    exclusion_delete_df['Includes'] = 'Delete with exclusions'

    # Random duplicate
    with_random_duplicate = filtered_df['Method'].str.contains('Random duplicate')
    with_random_duplicate_df = filtered_df[with_random_duplicate]
    with_random_duplicate_df['Filter'] = 'With'

    without_random_duplicate_df = filtered_df[~with_random_duplicate]
    without_random_duplicate_df['Filter'] = 'Without'

    random_duplicate_df = pd.concat([with_random_duplicate_df, without_random_duplicate_df])
    random_duplicate_df['Includes'] = 'Random duplicate'

    # Insert conjunction
    with_insert_conjunction = filtered_df['Method'].str.contains('Insert conjunction')
    with_insert_conjunction_df = filtered_df[with_insert_conjunction]
    with_insert_conjunction_df['Filter'] = 'With'

    without_insert_conjunction_df = filtered_df[~with_insert_conjunction]
    without_insert_conjunction_df['Filter'] = 'Without'

    insert_conjunction_df = pd.concat([with_insert_conjunction_df, without_insert_conjunction_df])
    insert_conjunction_df['Includes'] = 'Insert conjunction'

    # TAM update
    with_tam_update = filtered_df['Method'].str.contains('TAM update')
    with_tam_update_df = filtered_df[with_tam_update]
    with_tam_update_df['Filter'] = 'With'

    without_tam_update_df = filtered_df[~with_tam_update]
    without_tam_update_df['Filter'] = 'Without'

    tam_update_df = pd.concat([with_tam_update_df, without_tam_update_df])
    tam_update_df['Includes'] = 'TAM update'

    # Combine method dataframes
    results = pd.concat([insert_noise_df, random_delete_df, exclusion_delete_df, random_duplicate_df, insert_conjunction_df, tam_update_df])

elif language == 'Arapaho':
    # Insert interjection
    with_insert_interjection = filtered_df['Method'].str.contains('Insert interjection')
    with_insert_interjection_df = filtered_df[with_insert_interjection]
    with_insert_interjection_df['Filter'] = 'With'

    without_insert_interjection_df = filtered_df[~with_insert_interjection]
    without_insert_interjection_df['Filter'] = 'Without'

    insert_interjection_df = pd.concat([with_insert_interjection_df, without_insert_interjection_df])
    insert_interjection_df['Includes'] = 'Insert interjection'

    # Sentence permutations
    with_sentence_permutations = filtered_df['Method'].str.contains('Sentence permutations')
    with_sentence_permutations_df = filtered_df[with_sentence_permutations]
    with_sentence_permutations_df['Filter'] = 'With'

    without_sentence_permutations_df = filtered_df[~with_sentence_permutations]
    without_sentence_permutations_df['Filter'] = 'Without'

    sentence_permutations_df = pd.concat([with_sentence_permutations_df, without_sentence_permutations_df])
    sentence_permutations_df['Includes'] = 'Sentence Permutations'

    # Combine method dataframes
    results = pd.concat([insert_noise_df, insert_interjection_df, sentence_permutations_df])

# %%
results.dropna(inplace=True)

# %%
# Split into with and without 
with_final_df = results[results['Filter']== 'With']
without_final_df = results[results['Filter']== 'Without']

# %%
# Find average for each metric
with_final_bleu = with_final_df.groupby(['training_size','Includes'], as_index=False)['test/BLEU'].mean()
with_final_chrf = with_final_df.groupby(['training_size','Includes'], as_index=False)['test/chrF'].mean()
with_final_loss = with_final_df.groupby(['training_size','Includes'], as_index=False)['test/loss'].mean()

without_final_bleu = without_final_df.groupby(['training_size','Includes'], as_index=False)['test/BLEU'].mean()
without_final_chrf = without_final_df.groupby(['training_size','Includes'], as_index=False)['test/chrF'].mean()
without_final_loss = without_final_df.groupby(['training_size','Includes'], as_index=False)['test/loss'].mean()

# %%
# Merge metric dataframes into one
with_final_df = with_final_bleu.merge(with_final_chrf).merge(with_final_loss)
without_final_df = without_final_bleu.merge(without_final_chrf).merge(without_final_loss)

# %%
# Create plot_df dataframe
plot_df = with_final_df.copy().drop(['test/BLEU', 'test/loss', 'test/chrF'], axis=1)

# %%
# Find differences for each metric between runs with the method and runs without the method
plot_df['loss_difference'] = with_final_df['test/loss'] - without_final_df['test/loss']
plot_df['chrf_difference'] = with_final_df['test/chrF'] - without_final_df['test/chrF']
plot_df['bleu_difference'] = with_final_df['test/BLEU'] - without_final_df['test/BLEU']

# %%
# BLEU Score visualization
average_difference_bleu = sns.catplot(
    data=plot_df,
    x="training_size", y="bleu_difference", kind='bar', hue='Includes', errorbar=None
)
average_difference_bleu.set_axis_labels('Training Size', 'Difference in BLEU Score')
average_difference_bleu.legend.set_title('Method')

# Output to file
average_difference_bleu.savefig(f'{experiment_name}_average_difference_bleu.png')


# %%
# chrF Score visualization
average_difference_chrf = sns.catplot(
    data=plot_df,
    x="training_size", y="chrf_difference", kind='bar', hue='Includes', errorbar=None
)
average_difference_chrf.set_axis_labels('Training Size', 'Difference in chrF Score')
average_difference_chrf.legend.set_title('Method')
# Output to file
average_difference_chrf.savefig(f'{experiment_name}_average_difference_chrf.png')


# %%
# Loss visualization
average_difference_loss = sns.catplot(
    data=plot_df,
    x="training_size", y="loss_difference", kind='bar', hue='Includes', errorbar=None
)
average_difference_loss.set_axis_labels('Training Size', 'Difference in Loss')
average_difference_loss.legend.set_title('Method')
# Output to file
average_difference_loss.savefig(f'{experiment_name}_average_difference_loss.png')