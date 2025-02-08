import pandas as pd

def create_filtered_dataframe(csv_file):
    ''' Creates a filtered dataframe from the CSV output of the experiment runs.

    Args:
        csv_file: The CSV file to process. Must include relative path if the CSV is not stored in the data-viz directory

    Returns:
        filtered_df: A dataframe with only the relevant columns for analysis.
    '''
    df = pd.read_csv(csv_file)

    # Exclude failed runs
    df = df[df['State'] == 'finished']

    filtered_df = df.filter(['Name', 
        'aug_run_delete_w_exclusions',
        'aug_run_insert_interjection',
        'aug_run_random_delete',
        'aug_run_random_duplicate',
        'aug_run_random_insert_conj', 
        'aug_run_random_insert_noise',
        'aug_run_sentence_permutations',
        'aug_run_tam_update',
        'direction',
        'random-seed',
        'training_size',
        'eval/BLEU',
        'eval/chrF',
        'eval/loss',
        'test/BLEU',
        'test/chrF',
        'test/loss',
        'train/loss'
    ]).copy()

    filtered_df['Method'] = ''
    
    for column in filtered_df:
        filtered_df[column] = filtered_df[column].fillna(0) # Fill any empty cells with 0

    for column in filtered_df:
        if column.startswith('aug'):    
            filtered_df[column] = filtered_df[column].astype(bool).astype(int)  # Convert boolean T/F values to 1/0
    
    # Put categories in ascending order
    filtered_df['training_size'] = pd.Categorical(filtered_df['training_size'], ordered=True, categories=['100', '500', '1000', '5000', 'full'])

    return filtered_df