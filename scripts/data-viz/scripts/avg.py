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

# Ins-Noise -- shared method
with_insert_noise = filtered_df['Method'].str.contains('Ins-Noise')
with_insert_noise_df = filtered_df[with_insert_noise]
with_insert_noise_df['Filter'] = 'With'

without_insert_noise_df = filtered_df[~with_insert_noise]
without_insert_noise_df['Filter'] = 'Without'

insert_noise_df = pd.concat([with_insert_noise_df, without_insert_noise_df])    # Recombine after adding Filter column
insert_noise_df['Includes'] = 'Ins-Noise'

if language == 'Uspanteko':
    # Del
    with_random_delete = filtered_df['Method'].str.contains('Del')
    with_random_delete_df = filtered_df[with_random_delete]
    with_random_delete_df['Filter'] = 'With'

    without_random_delete_df = filtered_df[~with_random_delete]
    without_random_delete_df['Filter'] = 'Without'

    random_delete_df = pd.concat([with_random_delete_df, without_random_delete_df])
    random_delete_df['Includes'] = 'Del'

    # Exclusion delete
    with_exclusion_delete = filtered_df['Method'].str.contains('Del-Excl')
    with_exclusion_delete_df = filtered_df[with_exclusion_delete]
    with_exclusion_delete_df['Filter'] = 'With'

    without_exclusion_delete_df = filtered_df[~with_exclusion_delete]
    without_exclusion_delete_df['Filter'] = 'Without'

    exclusion_delete_df = pd.concat([with_exclusion_delete_df, without_exclusion_delete_df])
    exclusion_delete_df['Includes'] = 'Del-Excl'

    # Dup
    with_random_duplicate = filtered_df['Method'].str.contains('Dup')
    with_random_duplicate_df = filtered_df[with_random_duplicate]
    with_random_duplicate_df['Filter'] = 'With'

    without_random_duplicate_df = filtered_df[~with_random_duplicate]
    without_random_duplicate_df['Filter'] = 'Without'

    random_duplicate_df = pd.concat([with_random_duplicate_df, without_random_duplicate_df])
    random_duplicate_df['Includes'] = 'Dup'

    # Ins-Conj
    with_insert_conjunction = filtered_df['Method'].str.contains('Ins-Conj')
    with_insert_conjunction_df = filtered_df[with_insert_conjunction]
    with_insert_conjunction_df['Filter'] = 'With'

    without_insert_conjunction_df = filtered_df[~with_insert_conjunction]
    without_insert_conjunction_df['Filter'] = 'Without'

    insert_conjunction_df = pd.concat([with_insert_conjunction_df, without_insert_conjunction_df])
    insert_conjunction_df['Includes'] = 'Ins-Conj'

    # Upd-TAM
    with_tam_update = filtered_df['Method'].str.contains('Upd-TAM')
    with_tam_update_df = filtered_df[with_tam_update]
    with_tam_update_df['Filter'] = 'With'

    without_tam_update_df = filtered_df[~with_tam_update]
    without_tam_update_df['Filter'] = 'Without'

    tam_update_df = pd.concat([with_tam_update_df, without_tam_update_df])
    tam_update_df['Includes'] = 'Upd-TAM'

    # Combine method dataframes
    results = pd.concat([insert_noise_df, random_delete_df, exclusion_delete_df, random_duplicate_df, insert_conjunction_df, tam_update_df])

elif language == 'Arapaho':
    # Ins-Intj
    with_insert_interjection = filtered_df['Method'].str.contains('Ins-Intj')
    with_insert_interjection_df = filtered_df[with_insert_interjection]
    with_insert_interjection_df['Filter'] = 'With'

    without_insert_interjection_df = filtered_df[~with_insert_interjection]
    without_insert_interjection_df['Filter'] = 'Without'

    insert_interjection_df = pd.concat([with_insert_interjection_df, without_insert_interjection_df])
    insert_interjection_df['Includes'] = 'Ins-Intj'

    # Perm
    with_sentence_permutations = filtered_df['Method'].str.contains('Perm')
    with_sentence_permutations_df = filtered_df[with_sentence_permutations]
    with_sentence_permutations_df['Filter'] = 'With'

    without_sentence_permutations_df = filtered_df[~with_sentence_permutations]
    without_sentence_permutations_df['Filter'] = 'Without'

    sentence_permutations_df = pd.concat([with_sentence_permutations_df, without_sentence_permutations_df])
    sentence_permutations_df['Includes'] = 'Perm'

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
with_final_bleu = with_final_df.groupby(['Includes'], as_index=False)['test/BLEU'].mean()
with_final_chrf = with_final_df.groupby(['Includes'], as_index=False)['test/chrF'].mean()
with_final_loss = with_final_df.groupby(['Includes'], as_index=False)['test/loss'].mean()

without_final_bleu = without_final_df.groupby(['Includes'], as_index=False)['test/BLEU'].mean()
without_final_chrf = without_final_df.groupby(['Includes'], as_index=False)['test/chrF'].mean()
without_final_loss = without_final_df.groupby(['Includes'], as_index=False)['test/loss'].mean()

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
method_colors = {
    "Ins-Noise": "#254653",  # blue
    "Del-Excl": "#299D8F",  # teal
    "Del": "#F4A261",  # light orange
    "Ins-Conj": "#43E0D8",  # light blue
    "Upd-TAM": "#E76F51",  # dark orange
    "Dup": "#E9C46A",  # yellow
    "Ins-Intj": "#43E0D8",  # light blue
    "Perm": "#bbbbbb",  # gray
}

# unique_training_sizes = sorted(plot_df["training_size"].unique())


def add_grid_lines(facetgrid):
    for ax in facetgrid.axes.flat:
        # for tx in unique_training_sizes:
        #     ax.axvline(
        #         tx, color="gray", linestyle="--", alpha=0.2, zorder=0, linewidth=0.5
        #     )
        ax.axhline(0, color="black", linestyle="-", alpha=1.0, zorder=0, linewidth=2)

# %%
# BLEU Score visualization

if language == "Uspanteko":
    method_order = ["Upd-TAM", "Ins-Conj", "Ins-Noise", "Del", "Del-Excl", "Dup"]
else:
    method_order = ["Ins-Intj", "Ins-Noise", "Perm"]

average_difference_bleu = sns.catplot(
    data=plot_df,
    x="Includes",
    y="bleu_difference",
    kind="bar",
    hue="Includes",
    order=method_order,
    palette=method_colors,
    errorbar=None,
    legend=False,
)
average_difference_bleu.set_axis_labels('Strategy', "Δ BLEU")
add_grid_lines(average_difference_bleu)

for ax in average_difference_bleu.axes.flat:
    ax.set_xticklabels([])

# Output to file
average_difference_bleu.savefig(
    f"{language}_{experiment_name}_average_difference_bleu.pdf", format="pdf"
)


# CHRF

average_difference_chrf = sns.catplot(
    data=plot_df,
    x="Includes",
    y="chrf_difference",
    kind="bar",
    hue="Includes",
    order=method_order,
    palette=method_colors,
    errorbar=None,
    legend=False,
)
average_difference_chrf.set_axis_labels('Strategy', "Δ chrF")
add_grid_lines(average_difference_chrf)

for ax in average_difference_chrf.axes.flat:
    ax.set_xticklabels([])

# Output to file
average_difference_chrf.savefig(
    f"{language}_{experiment_name}_average_difference_chrf.pdf", format="pdf"
)
