# %%
from aug_utils import output_to_file, random_insert_conj, glosses_to_list, create_dataframe, \
    tam_update, random_duplicate, random_delete, delete_w_exclusions
import os

# %%
# Data augmentation methods
# Set variable(s) to True to use the method(s).
run_random_insert_conj = False
run_tam_update = False
run_random_duplicate = False
run_random_delete = False
run_delete_w_exclusions = False

# %%
# Load in data
df = create_dataframe.create_dataframe()
df = df.dropna()

# %%
df.head()

# %%
# Move to correct folder for output
os.chdir('data/hallucinated')

# %%
# Declare aug file
aug_file = input("What should this file be called? Format: 'data/hallucinated/<output_filename.txt>'. Don't forget the extension. ")

# %%
# TAM update
if run_tam_update:
    tam_updates = []
    glosses = glosses_to_list.glosses_to_list(df)

    for gloss in glosses:
        tam_updates.append(tam_update.tam_update(gloss))
        
    for t in tam_updates:
        if t is not None:
            output_to_file.output_to_file(t, aug_file)

# %%
# Insert random conjunction/adverb at the start of sentence
# 20 conjunctions/adverbs grabbed from the Uspanteko gold standard data.
conjs = [['Toos', 'ADV', 'entonces', 'Entonces'], 
         ['Ójor', 'ADV', 'antiguamente', 'Antiguamente'],
         ['Kom', 'ADV', 'como', 'Como'],
         ['Kwando', 'ADV', 'cuando', 'cuando'],
         ['Pores', 'ADV', 'por@eso', 'Por eso'],
         ['Peer', 'ADV', 'pero', 'Pero'],
         ['Porke', 'ADV', 'porque', 'Porque'],
         ['Pent', 'ADV', 'de@repente', 'De repente'],
         ['Pwes', 'ADV', 'pues', 'Pues'],
         ['O', 'CONJ', 'o', 'O'],
         ['E', 'CONJ', 'CONJ', 'E'],
         ['I', 'CONJ', 'CONJ', 'Y'],
         ['Lamaas', 'ADV', 'donde', 'Donde'],
         ['Juntir', 'ADV', 'todo', 'Todos'],
         ['Jinon', 'ADV', 'asi', 'Así'],
         ['Dyunabes', 'ADV', 'de@una@vez', 'De una vez'],
         ['Usil', 'ADV', 'poco@a@poco', 'Poco a poco'],
         ['Loke', 'ADV', 'loque', 'Lo que'],
         ['Lojori', 'ADV', 'ahora', 'Ahora'],
         ['Si', 'ADV', 'si', 'Si']]

if run_random_insert_conj:
    glosses = glosses_to_list.glosses_to_list(df)

    for conj in conjs:
        for gloss in glosses:
            output_to_file.output_to_file(random_insert_conj.random_insert_beginning(gloss, conj), aug_file)

# %%
# Random Duplicates
if run_random_duplicate:
    duplicates = []
    glosses = glosses_to_list.glosses_to_list(df)

    for gloss in glosses:
        duplicates.append(random_duplicate.random_duplicate(gloss))
        
    for dup in duplicates:
        if dup is not None:
            output_to_file.output_to_file(dup, aug_file)

# %%
# Random delete
if run_random_delete:
    deleted = []
    glosses = glosses_to_list.glosses_to_list(df)

    for gloss in glosses:
        deleted.append(random_delete.random_delete(gloss))
        
    for d in deleted:
        output_to_file.output_to_file(d, aug_file)

# %%
# Delete with exclusions 
if run_delete_w_exclusions:
    exclusions = []
    glosses = glosses_to_list.glosses_to_list(df)

    for gloss in glosses:
        exclusions.append(delete_w_exclusions.exclusion_delete(gloss))
        
    for e in exclusions:
        if e is not None:
            output_to_file.output_to_file(e, aug_file)


