from aug_utils import create_dataframe, glosses_to_list, output_to_file, random_insert_conj, tam_update
import pandas as pd
import re, os, string, spacy, mlconjug3, stanza, spacy_stanza, sys
from mlconjug3 import Conjugator
from datasets import load_dataset

# %%
# Load in data
df = create_dataframe.create_dataframe()
df = df.dropna()

# %%
df.head()

# %%
# Move to correct folder for output
# You may need to change this depending on where in the file system you are running this from 
os.chdir('data/hallucinated')

# %%
# Declare aug file
aug_file = input("What should this file be called? Format: 'data/hallucinated/<output_filename.txt>'. Don't forget the extension. ")

# %%
# Data augmentation methods
# Uncomment blocks below to use the methods. The blocks are set up so that as many as you want can be run without issue. 

# # %%
# # TAM update
# tam_updates = []
# glosses = glosses_to_list.glosses_to_list(df)

# for gloss in glosses:
#     tam_updates.append(tam_update.tam_update(gloss))
# for t in tam_updates:
#     if t is not None:
#         output_to_file.output_to_file(t, aug_file)

# # %%
# # Insert random conjunction/adverb at the start of sentence
# # 20 conjunctions/adverbs grabbed from the Uspanteko gold standard data.
# conjs = [['Toos', 'ADV', 'entonces', 'Entonces'], 
#          ['Ójor', 'ADV', 'antiguamente', 'Antiguamente'],
#          ['Kom', 'ADV', 'como', 'Como'],
#          ['Kwando', 'ADV', 'cuando', 'cuando'],
#          ['Pores', 'ADV', 'por@eso', 'Por eso'],
#          ['Peer', 'ADV', 'pero', 'Pero'],
#          ['Porke', 'ADV', 'porque', 'Porque'],
#          ['Pent', 'ADV', 'de@repente', 'De repente'],
#          ['Pwes', 'ADV', 'pues', 'Pues'],
#          ['O', 'CONJ', 'o', 'O'],
#          ['E', 'CONJ', 'CONJ', 'E'],
#          ['I', 'CONJ', 'CONJ', 'Y'],
#          ['Lamaas', 'ADV', 'donde', 'Donde'],
#          ['Juntir', 'ADV', 'todo', 'Todos'],
#          ['Jinon', 'ADV', 'asi', 'Así'],
#          ['Dyunabes', 'ADV', 'de@una@vez', 'De una vez'],
#          ['Usil', 'ADV', 'poco@a@poco', 'Poco a poco'],
#          ['Loke', 'ADV', 'loque', 'Lo que'],
#          ['Lojori', 'ADV', 'ahora', 'Ahora'],
#          ['Si', 'ADV', 'si', 'Si']]
# glosses = glosses_to_list.glosses_to_list(df)

# for conj in conjs:
#     for gloss in glosses:
#         output_to_file.output_to_file(random_insert_conj.random_insert_beginning(gloss, conj), aug_file)


