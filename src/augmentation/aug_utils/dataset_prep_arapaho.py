def dataset_prep(gloss):
    ''' Prepares glosses for dataset creation.

    Args:
        gloss: A processed IGT gloss.

    Returns:
        None

    '''
    if gloss is not None:
        gloss[0] = ' '.join(gloss[0])
        gloss[1] = ' '.join(gloss[1])
        gloss[2] = ' '.join(gloss[2])

        aug_row = {
            'transcription': gloss[0],
            'glosses': gloss[1],
            'translation': gloss[2]
        }
        return aug_row
