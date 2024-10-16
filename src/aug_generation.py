# %%
from aug_utils import random_insert_conj, glosses_to_list, create_dataframe, dataset_prep, \
    tam_update, random_duplicate, random_delete, delete_w_exclusions, output_dataset


def aug_generation():
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

    final_list = []

    # %%
    # TAM update
    if run_tam_update:
        tam_updates = []
        tam_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for gloss in glosses:
            tam_updates.append(tam_update.tam_update(gloss))
            
        for t in tam_updates:
            if t is not None:
                tam_final.append(dataset_prep.dataset_prep(t))
        final_list.extend(tam_final)


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
        conj_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for conj in conjs:
            for gloss in glosses:
                conj_final.append(dataset_prep.dataset_prep(random_insert_conj.random_insert_beginning(gloss, conj)))
        final_list.extend(conj_final)

    # %%
    # Random Duplicates
    if run_random_duplicate:
        duplicates = []
        dup_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for gloss in glosses:
            duplicates.append(random_duplicate.random_duplicate(gloss))
            
        for dup in duplicates:
            if dup is not None:
                dup_final.append(dataset_prep.dataset_prep(dup))
        final_list.extend(dup_final)
                
    # %%
    # Random delete
    if run_random_delete:
        deleted = []
        del_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for gloss in glosses:
            deleted.append(random_delete.random_delete(gloss))
            
        for d in deleted:
            if d is not None:
                del_final.append(dataset_prep.dataset_prep(d))
        final_list.extend(del_final)
            

    # %%
    # Delete with exclusions 
    if run_delete_w_exclusions:
        exclusions = []
        excl_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for gloss in glosses:
            exclusions.append(delete_w_exclusions.exclusion_delete(gloss))
            
        for e in exclusions:
            if e is not None:
                excl_final.append(dataset_prep.dataset_prep(e))
        final_list.extend(excl_final)


    #   Create dataset from augmented data
    aug_dataset = output_dataset.output_dataset(final_list)
    return aug_dataset


