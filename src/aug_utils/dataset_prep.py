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
        gloss[1] = gloss[1].replace("\\", "")
        gloss[2] = ' '.join(gloss[2])
        gloss[2] = gloss[2].replace("\\", "")
        gloss[3] = ' '.join(gloss[3])
        
        aug_row = {
            'transcription': [gloss[0]],
            'pos_glosses': [gloss[1]],
            'glosses': [gloss[2]],          
            'translation': [gloss[3]]
         }
        return aug_row