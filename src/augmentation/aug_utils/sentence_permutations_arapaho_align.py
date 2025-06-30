import random
import numpy as np
import math


def permute(gloss, is_segmented, max_samples: int = 10, seed=0):
    """Creates permutations of the first line of the gloss

    Args:
        gloss: An IGT gloss.
        is_segmented: A bool value to indicate whether the data is segmented.
        max_samples (int): Limit on the number of permutations.

    Returns:
        processed_gloss: A permutation of the first line of the gloss alone with the other two lines.

    """
    random.seed(seed)

    processed_gloss = []
    words = np.array(gloss[0])
    morphemes = np.array(gloss[1])
    word_map = dict(zip(words, morphemes))
    permutations = []

    # Limit the number of examples based on the number of possible unique combinations
    if len(gloss[0]) > 4:
        pass    # Leave as 10; 4! > 10
    else:
        max_samples = math.factorial(len(gloss[0])-1)

    for _ in range(max_samples):
        while True:
            perm = list(np.random.permutation(words))
            if perm in permutations:
                pass
            else:
                permutations.append(perm)

                if is_segmented:
                    p = list(perm)
                    g1 = []
                    for i in range(len(p)):
                        g1.append(word_map.get(p[i]))
                    processed_gloss.append([p, g1, gloss[2], gloss[3]])
                    break

                elif not is_segmented:
                    p = list(perm)
                    g1 = []
                    for i in range(len(p)):
                        g1.append(word_map.get(p[i]))
                    processed_gloss.append([p, g1, gloss[2]])
                    break
                break

    return processed_gloss

