def dataset_prep(gloss, is_segmented):
    ''' Prepares glosses for dataset creation.

    Args:
        gloss: A processed IGT gloss.
        is_segmented: A bool value to indicate whether the data is segmented

    Returns:
        None

    '''
   
    if is_segmented:
        if gloss is not None:
            gloss[0] = ' '.join(gloss[0])
            gloss[1] = ' '.join(gloss[1])
            gloss[2] = ' '.join(gloss[2])
            gloss[3] = ' '.join(gloss[3])

            aug_row = {
                'transcription': gloss[0],
                'segmentation': gloss[1],
                'glosses': gloss[2],
                'translation': gloss[3]
            }
            return aug_row
        
    elif not is_segmented:
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

