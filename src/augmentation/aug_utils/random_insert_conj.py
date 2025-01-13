
def random_insert_beginning(gloss, conj, is_segmented):
    ''' Inserts a random conjunction or adverb at the start of each line of the gloss.

    Args:
        gloss: An IGT gloss.
        is_segmented: A bool value to indicate whether the data is segmented

    Returns:
        processed_gloss: The original IGT gloss with the random conjunction inserted at the beginnging. 

    '''

    processed_gloss = []
    
    if is_segmented:
        if gloss[2][0] == 'ADV' or gloss[2][0] == 'CONJ':
            del gloss[0][0]
            del gloss[1][0]
            del gloss[2][0]
            del gloss[3][0]
            del gloss[4][0]
            gloss[0].insert(0, conj[0])
            gloss[1].insert(0, conj[1])
            gloss[2].insert(0, conj[2])
            gloss[3].insert(0, conj[3])
            gloss[4].insert(0, conj[4])
            processed_gloss.append(gloss[0])
            processed_gloss.append(gloss[1])
            processed_gloss.append(gloss[2])
            processed_gloss.append(gloss[3])
            processed_gloss.append(gloss[4])
            return processed_gloss
            
        else:
            gloss[0].insert(0, conj[0])
            gloss[1].insert(0, conj[1])
            gloss[2].insert(0, conj[2])
            gloss[3].insert(0, conj[3])
            gloss[4].insert(0, conj[4])
            processed_gloss.append(gloss[0])
            processed_gloss.append(gloss[1])
            processed_gloss.append(gloss[2])
            processed_gloss.append(gloss[3])
            processed_gloss.append(gloss[4])
            return processed_gloss
        
    elif not is_segmented:
        if gloss[1][0] == 'ADV' or gloss[1][0] == 'CONJ':
            del gloss[0][0]
            del gloss[1][0]
            del gloss[2][0]
            del gloss[3][0]
            gloss[0].insert(0, conj[0])
            gloss[1].insert(0, conj[2])
            gloss[2].insert(0, conj[3])
            gloss[3].insert(0, conj[4])
            processed_gloss.append(gloss[0])
            processed_gloss.append(gloss[1])
            processed_gloss.append(gloss[2])
            processed_gloss.append(gloss[3])
            return processed_gloss
            
        else:
            gloss[0].insert(0, conj[0])
            gloss[1].insert(0, conj[2])
            gloss[2].insert(0, conj[3])
            gloss[3].insert(0, conj[4])
            processed_gloss.append(gloss[0])
            processed_gloss.append(gloss[1])
            processed_gloss.append(gloss[2])
            processed_gloss.append(gloss[3])
            return processed_gloss