
def random_insert_beginning(gloss, noise, is_segmented):
    ''' Inserts a random word at the start of each line of the gloss.

    Args:
        gloss: An IGT gloss.
        is_segmented: A bool value to indicate whether the data is segmented.

    Returns:
        processed_gloss: The original IGT gloss with the random word inserted at the beginning.

    '''
    noises = ['mule.deer', 'leaves', 'tomatoes', 'behold',  '1S-gun',  'who?',
              'in.the.brush', 'ten', 'old.men', 'beaver', 'snake',
              'snow', 'airplane', 'cheese', 'playing.ball', 'like',
              '1S-friend', 'Christmas',  'cellar,.hole.in.the.ground', 'Sun.Dance']

    processed_gloss = []
    if is_segmented:
        if gloss[2][0] in noises:
            del gloss[0][0]
            del gloss[1][0]
            del gloss[2][0]
            del gloss[3][0]
            gloss[0].insert(0, noise[0])
            gloss[1].insert(0, noise[1])
            gloss[2].insert(0, noise[2])
            gloss[3].insert(0, noise[3])
            processed_gloss.append(gloss[0])
            processed_gloss.append(gloss[1])
            processed_gloss.append(gloss[2])
            processed_gloss.append(gloss[3])
            return processed_gloss

        else:
            gloss[0].insert(0, noise[0])
            gloss[1].insert(0, noise[1])
            gloss[2].insert(0, noise[2])
            gloss[3].insert(0, noise[3])
            processed_gloss.append(gloss[0])
            processed_gloss.append(gloss[1])
            processed_gloss.append(gloss[2])
            processed_gloss.append(gloss[3])
            return processed_gloss

    elif not is_segmented:
        if gloss[1][0] in noises:
            del gloss[0][0]
            del gloss[1][0]
            del gloss[2][0]
            gloss[0].insert(0, noise[0])
            gloss[1].insert(0, noise[2])
            gloss[2].insert(0, noise[3])
            processed_gloss.append(gloss[0])
            processed_gloss.append(gloss[1])
            processed_gloss.append(gloss[2])
            return processed_gloss

        else:
            gloss[0].insert(0, noise[0])
            gloss[1].insert(0, noise[2])
            gloss[2].insert(0, noise[3])
            processed_gloss.append(gloss[0])
            processed_gloss.append(gloss[1])
            processed_gloss.append(gloss[2])
            return processed_gloss
