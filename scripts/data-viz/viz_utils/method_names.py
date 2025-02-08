def method_names(df):
    ''' Creates a column that lists all of the methods used in each row.

    Args:
        df: The dataframe to update

    Returns:
        df: The updated dataframe
    '''

    for index, row in df.iterrows():
        
        methods = []
        for column in df:
            if column == 'aug_run_delete_w_exclusions' and row['aug_run_delete_w_exclusions'] == 1:
                methods.append('Delete with exclusions')
            elif column == 'aug_run_random_delete' and row['aug_run_random_delete'] ==1:
                methods.append('Random delete')
            elif column =='aug_run_insert_interjection' and row['aug_run_insert_interjection'] == 1:
                methods.append('Insert interjection')
            elif column == 'aug_run_random_duplicate' and row['aug_run_random_duplicate'] == 1:
                methods.append('Random duplicate')
            elif column == 'aug_run_random_insert_conj' and row['aug_run_random_insert_conj'] == 1:
                methods.append('Insert conjunction')
            elif column ==  'aug_run_random_insert_noise' and row['aug_run_random_insert_noise'] == 1:
                methods.append('Insert noise')
            elif column == 'aug_run_sentence_permutations' and row['aug_run_sentence_permutations'] == 1:
                methods.append('Sentence permutations')
            elif column == 'aug_run_tam_update' and row['aug_run_tam_update'] == 1:
                methods.append('TAM update')
        if not methods:
            methods.append('Baseline')
        df['Method'][index] = ',  '.join(methods)
    return df