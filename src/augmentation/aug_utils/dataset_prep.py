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
            gloss[1] = gloss[1].replace("\\", "")
            gloss[2] = ' '.join(gloss[2])
            gloss[2] = gloss[2].replace("\\", "")
            gloss[3] = ' '.join(gloss[3])
            gloss[3] = gloss[3].replace("\\", "")
            gloss[4] = ' '.join(gloss[4])

            aug_row = {
                'transcription': gloss[0],
                'segmentation': gloss[1],
                'pos_glosses': gloss[2],
                'glosses': gloss[3],
                'translation': gloss[4]
            }
            return aug_row
        
    elif not is_segmented:
        if gloss is not None:
            gloss[0] = ' '.join(gloss[0])
            gloss[1] = ' '.join(gloss[1])
            gloss[1] = gloss[1].replace("\\", "")
            gloss[2] = ' '.join(gloss[2])
            gloss[2] = gloss[2].replace("\\", "")
            gloss[3] = ' '.join(gloss[3])

            aug_row = {
                'transcription': gloss[0],
                'pos_glosses': gloss[1],
                'glosses': gloss[2],
                'translation': gloss[3]
            }
            return aug_row
