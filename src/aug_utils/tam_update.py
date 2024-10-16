import re
import re, os, string, spacy, mlconjug3, stanza, spacy_stanza
from mlconjug3 import Conjugator

stanza.download("es")
nlp = spacy_stanza.load_pipeline("es")

conjugator = Conjugator(language='es')  # Instantiate Spanish conjugator 

def tam_update(gloss):
    ''' Updates the TAM marker in verb constructions. 

    Args:
        gloss: An IGT gloss.

    Returns:
        processed_gloss: The original IGT gloss with the TAM marker updated in each line.

    '''
    processed_gloss = []
    person_markers = {'1Sing': 'yo', '2Sing': 'tú', '3Sing': 'él', '1Plur': 'nosotros', '2Plur': 'ellos', '3Plur': 'ellos'}
    
    for i in gloss[1]:
        matches = re.search('COM-V.*', i) or re.search('COM-.*-V.*', i) or re.search('INC-V.*', i) or re.search('INC-.*-V.*', i)
        if matches is not None:
            index = gloss[1].index(i)
            doc = nlp(' '.join(gloss[3]))
            for token in doc:
                # if token.dep_ == "root" and token.pos_ == "VERB" and token.head:
                if token.pos_ == "VERB":
                    number = token.morph.get("Number")
                    person = token.morph.get("Person")
                    if person and number:
                        marker =  ''.join(person+number)
                        orig_verb = token.text
                        lemma = token.lemma_
                        conjugations = conjugator.conjugate(lemma)
                        if conjugations is not None:
                            if len(gloss[2]) > index+1:
                                g0 = ' '.join(gloss[0])
                                g1  = ' '.join(gloss[1])
                                g2 = ' '.join(gloss[2])
                                if (matches.group()).startswith('COM'):
                                    new_verb = conjugations['Indicativo',
                                                                'Indicativo presente', person_markers[marker]]
                                    if new_verb != None:
                                        processed_gloss.append([''.join(re.sub(re.escape(gloss[0][index]), re.sub('^x', 't', re.escape(gloss[0][index]), count=1), g0))])
                                        processed_gloss.append([''.join(re.sub(re.escape(gloss[1][index]), re.sub('COM', 'INC', re.escape(gloss[1][index]), count=1), g1))])
                                        processed_gloss.append([''.join(re.sub(re.escape(gloss[2][index]), re.sub('COM', 'INC', re.escape(gloss[2][index]), count=1), g2))])
                                        processed_gloss.append(list(map(lambda x: x.replace(orig_verb, new_verb), gloss[3])))
                                        return processed_gloss
                                        
                                elif (matches.group()).startswith('INC'):
                                    new_verb = conjugations['Indicativo',
                                                                'Indicativo pretérito perfecto simple', person_markers[marker]]
                                    if new_verb != None:
                                        processed_gloss.append([''.join(re.sub(re.escape(gloss[0][index]), re.sub('^t', 'x', re.escape(gloss[0][index]), count=1), g0))])
                                        processed_gloss.append([''.join(re.sub(re.escape(gloss[1][index]), re.sub('INC', 'COM', re.escape(gloss[1][index]), count=1), g1))])
                                        processed_gloss.append([''.join(re.sub(re.escape(gloss[2][index]), re.sub('INC', 'COM', re.escape(gloss[2][index]), count=1), g2))])
                                        processed_gloss.append(list(map(lambda x: x.replace(orig_verb, new_verb), gloss[3])))
                                        return processed_gloss
                                    