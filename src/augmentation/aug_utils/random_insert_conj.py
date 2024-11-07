
def random_insert_beginning(gloss, conj):
    ''' Inserts a random conjunction or adverb at the start of each line of the gloss.

    Args:
        gloss: An IGT gloss.

    Returns:
        processed_gloss: The original IGT gloss with the random conjunction inserted at the beginnging. 

    '''

    processed_gloss = []

    if gloss[1][0] == 'ADV' or gloss[1][0] == 'CONJ':
        del gloss[0][0]
        del gloss[1][0]
        del gloss[2][0]
        del gloss[3][0]
        gloss[0].insert(0, conj[0])
        gloss[1].insert(0, conj[1])
        gloss[2].insert(0, conj[2])
        gloss[3].insert(0, conj[3])
        processed_gloss.append(gloss[0])
        processed_gloss.append(gloss[1])
        processed_gloss.append(gloss[2])
        processed_gloss.append(gloss[3])
        return processed_gloss
        
    else:
        gloss[0].insert(0, conj[0])
        gloss[1].insert(0, conj[1])
        gloss[2].insert(0, conj[2])
        gloss[3].insert(0, conj[3])
        processed_gloss.append(gloss[0])
        processed_gloss.append(gloss[1])
        processed_gloss.append(gloss[2])
        processed_gloss.append(gloss[3])
        return processed_gloss