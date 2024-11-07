import random

def random_duplicate(gloss):
    ''' Randomly duplicates words in a gloss.

    Args:
        gloss: An IGT gloss.

    Returns:
        processed_gloss: The original IGT gloss with a random token duplicated within each line.

    '''

    processed_gloss = []

    if len(gloss[0]) == len(gloss[1]) == len(gloss[2]) == len(gloss[3]):
        x = random.randrange(len(gloss[0]))
        update_index = x+1
        gloss[0].insert(update_index, gloss[0][x])
        gloss[1].insert(update_index, gloss[1][x])
        gloss[2].insert(update_index, gloss[2][x])
        gloss[3].insert(update_index, gloss[3][x])
        processed_gloss.append(gloss[0])
        processed_gloss.append(gloss[1])
        processed_gloss.append(gloss[2])
        processed_gloss.append(gloss[3])
        return processed_gloss