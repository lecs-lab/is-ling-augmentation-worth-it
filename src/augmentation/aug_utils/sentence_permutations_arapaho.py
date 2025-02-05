from itertools import permutations
import random

def permute(gloss, is_segmented, max_samples: int | None=10, seed=0):
    ''' Creates permutations of the first line of the gloss

    Args:
        gloss: An IGT gloss.
        is_segmented: A bool value to indicate whether the data is segmented.
        max_samples (int | None): If provided, limit the number of permutations

    Returns:
        processed_gloss: A permutation of the first line of the gloss alone with the other two lines.

    '''
    random.seed(seed)

    processed_gloss = []
    gloss0 = [list(p) for p in permutations(gloss[0])]
    if max_samples is not None and len(gloss0) > max_samples:
        gloss0 = random.sample(gloss0, max_samples)

    if is_segmented:
        for g0 in gloss0:
            processed_gloss.append([g0, gloss[1], gloss[2], gloss[3]])
        return processed_gloss

    elif not is_segmented:
        for g0 in gloss0:
            processed_gloss.append([g0, gloss[1], gloss[2]])
        return processed_gloss
