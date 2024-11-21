
def random_insert_beginning(gloss, n):
    ''' Inserts a random word at the start of each line of the gloss.

    Args:
        gloss: An IGT gloss.

    Returns:
        processed_gloss: The original IGT gloss with the random word inserted at the beginning. 

    '''
    noise = ['Rechi\'', 'Chiqe', 'Saneb\'','Keqiix', 'Baya','Inchk', 'Xte\'', 'Qája', 'Mismo', 'Tijk\'ey', 
            'Tib\'itaq', 'Tijut', 'Tilin','Aqaaj', 'Tiqatij', 'Mrel ánm', 'Jwi\'l tzaqoomch\'olaal ',
            'Kinye\' taq', 'Resureksyon','Tinloq\'e\'']
    processed_gloss = []

    if gloss[0][0] in noise:
        del gloss[0][0]
        del gloss[1][0]
        del gloss[2][0]
        del gloss[3][0]
        gloss[0].insert(0, n[0])
        gloss[1].insert(0, n[1])
        gloss[2].insert(0, n[2])
        gloss[3].insert(0, n[3])
        processed_gloss.append(gloss[0])
        processed_gloss.append(gloss[1])
        processed_gloss.append(gloss[2])
        processed_gloss.append(gloss[3])
        return processed_gloss
        
    else:
        gloss[0].insert(0, n[0])
        gloss[1].insert(0, n[1])
        gloss[2].insert(0, n[2])
        gloss[3].insert(0, n[3])
        processed_gloss.append(gloss[0])
        processed_gloss.append(gloss[1])
        processed_gloss.append(gloss[2])
        processed_gloss.append(gloss[3])
        return processed_gloss