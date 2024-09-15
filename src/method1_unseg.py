# %%
'''
This script uses existing Uspanteko IGT glosses to create augmented data. It identifies IGT glosses that contain either
the transitive or intransitive verb morpheme template, as well as the ergative or absolutive person marker used in it.
It then iterates through the list of relevant person markers and updates each line of the gloss, resulting in a total of
five new IGT glosses.
Example below:

Original                             Augmented
\t tanye' li b'ee           ->      \t  taye' li b'ee
\p INC-E1S-VT PREP S        ->      \p  INC-E2S-VT PREP S
\g INC-E1S-dar PREP camino  ->      \g  INC-E2S-dar PREP camino
\l Lo pongo en el camino.   ->      \l  Lo pones en el camino

'''

# %%
import pandas as pd
import re, string, spacy, mlconjug3, stanza, spacy_stanza, os.path
from mlconjug3 import Conjugator
from datasets import load_dataset, Dataset

# %%
stanza.download("es")
nlp = spacy_stanza.load_pipeline("es")

conjugator = Conjugator(language='es')

# %%
# The following section contains the dictionaries for both types of person markers - absolutive and ergative.
# Each dictionary consists of a list of three elements [Uspanteko morpheme, Glossing abbrev., Spanish pronoun]
# used to construct the IGT gloss.
# Note: The abs_check and erg_check dictionaries are used to verify which marker already exists in the data
# to prevent repeats from being created.

abs_check = {'A1S': ['in', 'A1S', 'yo'],
            'A2S': ['at', 'A2S', 'tú'],
            'A3S': ['', '', 'él'],
            'A1P': ['oj', 'A1P', 'nosotros'],
            'A2P': ['at', 'A2S', 'ellos'],
            'A3P': ['', '','ellos']}

absolutives = {'A1S': ['in', 'A1S', 'yo'],
            'A2S': ['at', 'A2S', 'tú'],
            'A3S': ['', '', 'él'],
            'A1P': ['oj', 'A1P', 'nosotros'],
            'A2P': ['at', 'A2S', 'ellos'],
            'A3P': ['', '','ellos']}

# Note: Each Uspanteko morpheme includes a common alternate spelling from the data and/or an allomorph.
erg_check = {'E1S': ['in|an', 'E1S', 'yo'],
            'E2S': ['a|aw', 'E2S', 'tú'],
            'E3S': ['j|r', 'E3S', 'él'],
            'E1P': ['qa|q', 'E1P', 'nosotros'],
            'E2P': ['a|aw','E2S', 'ellos'],
            'E3P': ['j|r', 'E3S', 'ellos']}

ergatives = {'E1S': ['in', 'E1S', 'yo'],
            'E2S': ['a', 'E2S', 'tú'],
            'E3S': ['j', 'E3S', 'él'],
            'E1P': ['qa', 'E1P', 'nosotros'],
            'E2P': ['a','E2S', 'ellos'],
            'E3P': ['j', 'E3S', 'ellos']}


def create_augmented_data(input: Dataset) -> Dataset:
    df = pd.DataFrame(input)
    df = df.drop(columns='segmentation')    # Drop the column with the segmented data

    # %%
    # This section converts the columns to strings and removes the punctuation marks that impact indexing
    # once the split is performed.
    df['transcription'] = df['transcription'].astype(str)
    df['transcription'] = df['transcription'].str.replace(',', '').str.replace('.', '').str.replace('?', '')

    df['pos_glosses'] = df['pos_glosses'].astype(str)

    df['glosses'] = df['glosses'].astype(str)
    df['glosses'] = df['glosses'].str.replace(',', '').str.replace('.', '').str.replace('?', '')

    df['translation'] = df['translation'].astype(str)
    df['translation'] = df['translation'].str.replace(',', '').str.replace('.', '').str.replace('?', '')

    # %%
    # This section converts each row into a list. Then, each line of the gloss is processed using split.
    # The resulting data is now formatted as a list of lists.
    glosses = df.values.tolist()

    for gloss in glosses:
        for idx, line in enumerate(gloss):
            gloss[idx] = line.split()

    # %%
    def gloss_processing(glosses):
        ''' Extracts key details from each gloss for use in verb conjugation.

        Args:
            glosses: Each list is an IGT gloss, and each row of the gloss is a list split on whitespace.

        Returns:
            processed_glosses: A list consisting of all of the information needed to update the gloss and create new data.
            Each list item is a list containing the person marker class, the specific person marker, the entire IGT gloss, the index of the verb
            to replace, the lemma of the verb (for use in conjugation), the original verb in Spanish to replace, the TAM marker, and
            the verb tense in Spanish.

        '''
        processed_glosses = []
        for idx, gloss in enumerate(glosses):
            for e in erg_check:
                verbs = ['INC-'+e+'-VT', 'INC-'+e+'-VI', 'COM-'+e+'-VT', 'COM-'+e+'-VI']    # Ergative-based verb templates
                for verb in verbs:
                    if verb in gloss[1]:    # Checks for a matching verb template in the \p line of the gloss
                        verb_index = gloss[1].index(verb)
                        doc = nlp(' '.join(gloss[3]))
                        for token in doc:
                            if token.dep_ == "root" and token.pos_ == "VERB" and token.head:
                                orig_verb = token.text
                                lemma = token.lemma_
                                if 'INC' in verb:
                                    processed_glosses.append(['Erg', e, gloss, verb_index, lemma, orig_verb,'INC', 'Indicativo presente'])
                                elif 'COM' in verb:
                                    processed_glosses.append([' Erg', e, gloss, verb_index, lemma, orig_verb, 'COM','Indicativo pretérito perfecto simple'])

            for a in abs_check:
                verbs = ['INC-'+a+'-VI', 'COM-'+a+'-VI']    # Absolutive-based verb templates
                for verb in verbs:
                    if verb in gloss[1]:   # Checks for a matching verb template in the \p line of the gloss
                        verb_index = gloss[1].index(verb)
                        doc = nlp(' '.join(gloss[3]))
                        for token in doc:
                            if token.dep_ == "root" and token.pos_ == "VERB" and token.head:
                                orig_verb = token.text
                                lemma = token.lemma_
                        if 'INC' in verb:
                            processed_glosses.append(['Abs', a, gloss, verb_index, lemma, orig_verb, 'INC', 'Indicativo presente'])
                        elif 'COM' in verb:
                            processed_glosses.append(['Abs', a, gloss, verb_index, lemma, orig_verb, 'COM', 'Indicativo pretérito perfecto simple'])

        return processed_glosses

    # %%
    processed_glosses = gloss_processing(glosses)

    augmented_data = []

    # This section writes the new data to the output file.
    for idx, gloss in enumerate(processed_glosses):
        if gloss[0] == 'Erg':
            verb_index = gloss[3]
            orig_erg = gloss[1]
            for erg in ergatives:
                if erg != orig_erg:
                    conjugations = conjugator.conjugate(gloss[4])
                    if conjugations:
                        l0 = gloss[2][0]
                        l1 = gloss[2][1]
                        l2 = gloss[2][2]
                        l3 = gloss[2][3]
                        line0 = ' '.join(gloss[2][0])
                        line1 = ' '.join(gloss[2][1])
                        line2 = ' '.join(gloss[2][2])
                        line3 = ' '.join(gloss[2][3])
                        verb_update = conjugations['Indicativo',
                                                gloss[7], ergatives[erg][2]]
                        if verb_update != None:
                            if len(l2) > verb_index+1:
                                # Output for plural ergative examples
                                if erg == "E2P" or erg == "E3P":
                                    # Prevents adding a duplicate plural marker
                                    if l0[verb_index+1] == 'taq':
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(erg_check[orig_erg][0], ergatives[erg][0], gloss[2][0][verb_index], count=1), line0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(erg_check[orig_erg][1], ergatives[erg][1], gloss[2][1][verb_index], count=1), line1)),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(erg_check[orig_erg][1], ergatives[erg][1], gloss[2][2][verb_index], count=1), line2)),
                                            'translation':      line3.replace(gloss[5], verb_update) ,
                                        }
                                        augmented_data.append(aug_row)
                                    elif l0[verb_index+1] != 'taq':     # Adds plural markers to the first three lines of the gloss if they don't already exist
                                        l0.insert(verb_index+1, 'taq')
                                        l1.insert(verb_index+1, 'PL')
                                        l2.insert(verb_index+1, 'PL')
                                        l0 = ' '.join(l0)
                                        l1 = ' '.join(l1)
                                        l2 = ' '.join(l2)
                                        l3 = ' '.join(l3)
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(erg_check[orig_erg][0], ergatives[erg][0], gloss[2][0][verb_index], count=1), l0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(erg_check[orig_erg][1], ergatives[erg][1], gloss[2][1][verb_index], count=1), l1)),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(erg_check[orig_erg][1], ergatives[erg][1], gloss[2][2][verb_index], count=1), l2)),
                                            'translation':      l3.replace(gloss[5], verb_update),
                                        }
                                        augmented_data.append(aug_row)
                                    else:
                                        pass
                                # Output for singular and E1P examples
                                elif erg != "E2P" or erg != "E3P":
                                    # Removes any lingering plural markers
                                    if l0[verb_index+1] == 'taq':
                                        l0.pop(verb_index+1)
                                        l1.pop(verb_index+1)
                                        l2.pop(verb_index+1)
                                        l0 = ' '.join(l0)
                                        l1 = ' '.join(l1)
                                        l2 = ' '.join(l2)
                                        l3 = ' '.join(l3)
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(erg_check[orig_erg][0], ergatives[erg][0], gloss[2][0][verb_index], count=1), l0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(erg_check[orig_erg][1], ergatives[erg][1], gloss[2][1][verb_index], count=1), l1)),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(erg_check[orig_erg][1], ergatives[erg][1], gloss[2][2][verb_index], count=1), l2)),
                                            'translation':      l3.replace(gloss[5], verb_update),
                                        }
                                        augmented_data.append(aug_row)

                                    elif l0[verb_index+1] != 'taq':
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(erg_check[orig_erg][0], ergatives[erg][0], gloss[2][0][verb_index], count=1), line0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(erg_check[orig_erg][1], ergatives[erg][1], gloss[2][1][verb_index], count=1), line1)),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(erg_check[orig_erg][1], ergatives[erg][1], gloss[2][2][verb_index], count=1), line2)),
                                            'translation':      line3.replace(gloss[5], verb_update),
                                        }
                                        augmented_data.append(aug_row)
                                    else:
                                        pass

        elif gloss[0] == 'Abs':
            verb_index = gloss[3]
            orig_abs = gloss[1]
            for abs in absolutives:
                if abs != orig_abs:
                    conjugations = conjugator.conjugate(gloss[4])
                    if conjugations:
                        l0 = gloss[2][0]
                        l1 = gloss[2][1]
                        l2 = gloss[2][2]
                        l3 = gloss[2][3]
                        line0 = ' '.join(gloss[2][0])
                        line1 = ' '.join(gloss[2][1])
                        line2 = ' '.join(gloss[2][2])
                        line3 = ' '.join(gloss[2][3])
                        verb_update = conjugations['Indicativo',
                                                gloss[7], absolutives[abs][2]]
                        if verb_update != None:
                            if len(l2) > verb_index+1:
                                # Output for A3S examples
                                # The Uspanteko morpheme for A3 is null, so the resulting "--" is removed from the \p and \g lines of the gloss
                                if abs == "A3S":
                                    # Removes any lingering plural markers
                                    if l0[verb_index+1] == 'taq':
                                        l0.pop(verb_index+1)
                                        l1.pop(verb_index+1)
                                        l2.pop(verb_index+1)
                                        l0 = ' '.join(l0)
                                        l1 = ' '.join(l1)
                                        l2 = ' '.join(l2)
                                        l3 = ' '.join(l3)
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(abs_check[orig_abs][0], absolutives[abs][0], gloss[2][0][verb_index], count=1), l0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][1][verb_index], count=1), l1)).replace('--', '-'),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][2][verb_index], count=1), l2)).replace('--', '-'),
                                            'translation':      l3.replace(gloss[5], verb_update),
                                        }
                                        augmented_data.append(aug_row)
                                    elif l0[verb_index+1] != 'taq':
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(abs_check[orig_abs][0], absolutives[abs][0], gloss[2][0][verb_index], count=1), line0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][1][verb_index], count=1), line1)).replace('--', '-'),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][2][verb_index], count=1), line2)).replace('--', '-'),
                                            'translation':      line3.replace(gloss[5], verb_update),
                                        }
                                        augmented_data.append(aug_row)
                                    else:
                                        pass
                                # Output for A2P examples
                                elif abs == "A2P":
                                    # Prevents adding a duplicate plural marker
                                    if l0[verb_index+1] == 'taq':
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(abs_check[orig_abs][0], absolutives[abs][0], gloss[2][0][verb_index], count=1), line0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][1][verb_index], count=1), line1)),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][2][verb_index], count=1), line2)),
                                            'translation':      line3.replace(gloss[5], verb_update),
                                        }
                                        augmented_data.append(aug_row)
                                    elif l0[verb_index+1] != 'taq':     # Adds plural markers to the first three lines of the gloss if they don't already exist
                                        l0.insert(verb_index+1, 'taq')
                                        l1.insert(verb_index+1, 'PL')
                                        l2.insert(verb_index+1, 'PL')
                                        l0 = ' '.join(l0)
                                        l1 = ' '.join(l1)
                                        l2 = ' '.join(l2)
                                        l3 = ' '.join(l3)
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(abs_check[orig_abs][0], absolutives[abs][0], gloss[2][0][verb_index], count=1), l0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][1][verb_index], count=1), l1)),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][2][verb_index], count=1), l2)),
                                            'translation':      l3.replace(gloss[5], verb_update),
                                        }
                                        augmented_data.append(aug_row)
                                    else:
                                        pass
                                # Output for A3P examples
                                # The Uspanteko morpheme for A3 is null, so the resulting "--" is removed from the \p and \g lines of the gloss
                                elif abs == "A3P":
                                    # Prevents adding a duplicate plural marker
                                    if l0[verb_index+1] == 'taq':
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(abs_check[orig_abs][0], absolutives[abs][0], gloss[2][0][verb_index], count=1), line0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][1][verb_index], count=1), line1)).replace('--', '-'),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][2][verb_index], count=1), line2)).replace('--', '-'),
                                            'translation':      line3.replace(gloss[5], verb_update),
                                        }
                                        augmented_data.append(aug_row)
                                    elif l0[verb_index+1] != 'taq':     # Adds plural markers to the first three lines of the gloss if they don't already exist
                                        l0.insert(verb_index+1, 'taq')
                                        l1.insert(verb_index+1, 'PL')
                                        l2.insert(verb_index+1, 'PL')
                                        l0 = ' '.join(l0)
                                        l1 = ' '.join(l1)
                                        l2 = ' '.join(l2)
                                        l3 = ' '.join(l3)
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(abs_check[orig_abs][0], absolutives[abs][0], gloss[2][0][verb_index], count=1), l0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][1][verb_index], count=1), l1)).replace('--', '-'),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][2][verb_index], count=1), l2)).replace('--', '-'),
                                            'translation':      l3.replace(gloss[5], verb_update),
                                        }
                                    else:
                                        pass
                                # Output for A1S, A2S, and A1P
                                else:
                                    # Removes any lingering plural markers
                                    if l0[verb_index+1] == 'taq':
                                        l0.pop(verb_index+1)
                                        l1.pop(verb_index+1)
                                        l2.pop(verb_index+1)
                                        l0 = ' '.join(l0)
                                        l1 = ' '.join(l1)
                                        l2 = ' '.join(l2)
                                        l3 = ' '.join(l3)
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(abs_check[orig_abs][0], absolutives[abs][0], gloss[2][0][verb_index], count=1), l0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][1][verb_index], count=1), l1)),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][2][verb_index], count=1), l2)),
                                            'translation':      l3.replace(gloss[5], verb_update),
                                        }
                                    elif l0[verb_index+1] != 'taq':
                                        aug_row = {
                                            'transcription':    ''.join(re.sub(gloss[2][0][verb_index], re.sub(abs_check[orig_abs][0], absolutives[abs][0], gloss[2][0][verb_index], count=1), line0)),
                                            'pos_glosses':      ''.join(re.sub(gloss[2][1][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][1][verb_index], count=1), line1)),
                                            'glosses':          ''.join(re.sub(gloss[2][2][verb_index], re.sub(abs_check[orig_abs][1], absolutives[abs][1], gloss[2][2][verb_index], count=1), line2)),
                                            'translation':      line3.replace(gloss[5], verb_update),
                                        }
                                    else:
                                        pass

    return Dataset.from_list(augmented_data)
