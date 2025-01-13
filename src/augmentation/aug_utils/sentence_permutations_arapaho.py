from itertools import permutations

def permute(gloss, is_segmented):
    ''' Creates permutations of the first line of the gloss

    Args:
        gloss: An IGT gloss.
        is_segmented: A bool value to indicate whether the data is segmented.

    Returns:
        processed_gloss: A permutation of the first line of the gloss alone with the other two lines. 

    '''

    processed_gloss = []
    if is_segmented:
        gloss0 = [list(p) for p in permutations(gloss[0])]
        for g0 in gloss0:
            processed_gloss.append([g0, gloss[1], gloss[2], gloss[3]])
        return processed_gloss
    
    elif not is_segmented:
        gloss0 = [list(p) for p in permutations(gloss[0])]
        for g0 in gloss0:
            processed_gloss.append([g0, gloss[1], gloss[2]])
        return processed_gloss
        
