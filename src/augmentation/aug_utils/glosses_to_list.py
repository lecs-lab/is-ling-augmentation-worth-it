def glosses_to_list(df):
    ''' Converts the IGT gloss dataframe into a list of lists for easier processing.

    Args:
        df: The dataframe. 

    Returns:
        glosses: The original IGT gloss converted into a list of lists. The lines of the gloss are split into 
            tokens and stored as lists. These are then stored inside the main IGT gloss list. 

    '''
    glosses = df.values.tolist() 

    for gloss in glosses:
        for idx, line in enumerate(gloss):
            gloss[idx] = line.split()

    return glosses