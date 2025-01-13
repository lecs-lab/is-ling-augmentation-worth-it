import random

def random_delete(gloss, is_segmented):
    ''' Randomly deletes words from the gloss. 

    Args:
        gloss: An IGT gloss.
        is_segmented: A bool value to indicate whether the data is segmented

    Returns:
        processed_gloss: The original IGT gloss with a random token removed from each line.

    '''

    processed_gloss = []
    if is_segmented:
        if len(gloss[0]) == len(gloss[1]) == len(gloss[2]) == len(gloss[3]) == len(gloss[4]):
            x = random.randrange(len(gloss[0]))
            gloss[0].pop(x)
            gloss[1].pop(x)
            gloss[2].pop(x)
            gloss[3].pop(x)
            gloss[4].pop(x)
            if len(gloss[0]) > 0 and len(gloss[1]) > 0 and len(gloss[2]) > 0 and len(gloss[3]) > 0 and len(gloss[4]) > 0:
                processed_gloss.append(gloss[0])
                processed_gloss.append(gloss[1])
                processed_gloss.append(gloss[2])
                processed_gloss.append(gloss[3])
                processed_gloss.append(gloss[4])
                return processed_gloss
            
    elif not is_segmented:
        if len(gloss[0]) == len(gloss[1]) == len(gloss[2]) == len(gloss[3]):
            x = random.randrange(len(gloss[0]))
            gloss[0].pop(x)
            gloss[1].pop(x)
            gloss[2].pop(x)
            gloss[3].pop(x)
            if len(gloss[0]) > 0 and len(gloss[1]) > 0 and len(gloss[2]) > 0 and len(gloss[3]) > 0:
                processed_gloss.append(gloss[0])
                processed_gloss.append(gloss[1])
                processed_gloss.append(gloss[2])
                processed_gloss.append(gloss[3])
                return processed_gloss



