import os 

def output_to_file(gloss, aug_file): 
    ''' Writes the processed glosses to the output file. 

    Args:
        gloss: A processed IGT gloss.
        aug_file: The name of the output file. 

    Returns:
        None

    '''
    with open(aug_file, 'a') as aug:
        gloss[0] = ' '.join(gloss[0])
        gloss[1] = ' '.join(gloss[1])
        gloss[1] = gloss[1].replace("\\", "")
        gloss[2] = ' '.join(gloss[2])
        gloss[2] = gloss[2].replace("\\", "")
        gloss[3] = ' '.join(gloss[3])
        if gloss is not None:
            aug.write(f"""
                        \\t  {gloss[0]}
                        \\p  {gloss[1]}
                        \\g  {gloss[2]}
                        \\l  {gloss[3]}
                """)