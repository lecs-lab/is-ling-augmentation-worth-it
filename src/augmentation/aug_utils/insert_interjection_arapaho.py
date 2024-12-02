
def random_insert_beginning(gloss, interjection):
    ''' Inserts an interjection, greeting, or conjunction at the start of each line of the gloss.

    Args:
        gloss: An IGT gloss.

    Returns:
        processed_gloss: The original IGT gloss with the random interjection, greeting, or conjunction inserted at the beginning. 

    '''
    interjections = ['allright!', 'yeah', 'okay', 'uhm', 'gee.whiz', 'laughter.at.joke',
                     'oh', 'uhm-hmm', 'no', 'ahm',  'hmm', 'maybe', 'yes', 'ouch', 'so',
                     'and', 'welcome', 'hello', 'but',  'IC.good.morning']

    processed_gloss = []

    if gloss[1][0] in interjections:
        del gloss[0][0]
        del gloss[1][0]
        del gloss[2][0]
        gloss[0].insert(0, interjection[0])
        gloss[1].insert(0, interjection[1])
        gloss[2].insert(0, interjection[2])
        processed_gloss.append(gloss[0])
        processed_gloss.append(gloss[1])
        processed_gloss.append(gloss[2])
        return processed_gloss

    else:
        gloss[0].insert(0, interjection[0])
        gloss[1].insert(0, interjection[1])
        gloss[2].insert(0, interjection[2])
        processed_gloss.append(gloss[0])
        processed_gloss.append(gloss[1])
        processed_gloss.append(gloss[2])
        return processed_gloss
