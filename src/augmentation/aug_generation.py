# %%
from datasets import Dataset

from augmentation.aug_parameters import AugmentationParameters

def aug_generation(
    initial_dataset: Dataset,
    fraction: float = 1,
    params: AugmentationParameters = AugmentationParameters()
):
    # Import locally so we don't do this every time we use the module
    from augmentation.aug_utils import random_insert_conj, glosses_to_list, create_dataframe, dataset_prep, \
        tam_update, random_duplicate, random_insert_noise, random_delete, delete_w_exclusions, output_dataset

    # %%
    # Load in data
    df = create_dataframe.create_dataframe(dataset=initial_dataset, fraction=fraction)
    df = df.dropna()

    # %%
    df.head()
    
    is_segmented = False
    if 'segmentation' in df.columns:
        is_segmented = True

    final_list = []

    # %%
    # TAM update
    if params.run_tam_update:
        tam_updates = []
        tam_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for gloss in glosses:
            tam_updates.append(tam_update.tam_update(gloss, is_segmented))

        for t in tam_updates:
            if t is not None:
                tam_final.append(dataset_prep.dataset_prep(t, is_segmented))
        final_list.extend(tam_final)


    # %%
    # Insert random conjunction/adverb at the start of sentence
    # 20 conjunctions/adverbs grabbed from the Uspanteko gold standard data.
    conjs = [['Toos', 'Toos', 'ADV', 'entonces', 'Entonces'],
        ['Ójor', 'Ójor', 'ADV', 'antiguamente', 'Antiguamente'],
        ['Kom', 'Kom', 'ADV', 'como', 'Como'],
        ['Kwando','Kwando', 'ADV', 'cuando', 'cuando'],
        ['Pores','Pores', 'ADV', 'por@eso', 'Por eso'],
        ['Peer', 'Peer', 'ADV', 'pero', 'Pero'],
        ['Porke', 'Porke', 'ADV', 'porque', 'Porque'],
        ['Pent', 'Pent', 'ADV', 'de@repente', 'De repente'],
        ['Pwes', 'Pwes', 'ADV', 'pues', 'Pues'],
        ['O', 'O', 'CONJ', 'o', 'O'],
        ['E', 'E', 'CONJ', 'CONJ', 'E'],
        ['I', 'I', 'CONJ', 'CONJ', 'Y'],
        ['Lamaas', 'Lamaas', 'ADV', 'donde', 'Donde'],
        ['Juntir', 'Juntir', 'ADV', 'todo', 'Todos'],
        ['Jinon', 'Jinon',  'ADV', 'asi', 'Así'],
        ['Dyunabes', 'Dyunabes', 'ADV', 'de@una@vez', 'De una vez'],
        ['Usil', 'Usil', 'ADV', 'poco@a@poco', 'Poco a poco'],
        ['Loke', 'Loke', 'ADV', 'loque', 'Lo que'],
        ['Lojori', 'Lojori', 'ADV', 'ahora', 'Ahora'],
        ['Si', 'Si', 'ADV', 'si', 'Si']]

    if params.run_random_insert_conj:
        conj_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for conj in conjs:
            for gloss in glosses:
                conj_final.append(dataset_prep.dataset_prep(random_insert_conj.random_insert_beginning(gloss, conj, is_segmented), is_segmented))
        final_list.extend(conj_final)

    # %%
    # Random Duplicates
    if params.run_random_duplicate:
        duplicates = []
        dup_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for gloss in glosses:
            duplicates.append(random_duplicate.random_duplicate(gloss, is_segmented))

        for dup in duplicates:
            if dup is not None:
                dup_final.append(dataset_prep.dataset_prep(dup, is_segmented))
        final_list.extend(dup_final)

    # %%
    # Random delete
    if params.run_random_delete:
        deleted = []
        del_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for gloss in glosses:
            deleted.append(random_delete.random_delete(gloss, is_segmented))

        for d in deleted:
            if d is not None:
                del_final.append(dataset_prep.dataset_prep(d, is_segmented))
        final_list.extend(del_final)


    # %%
    # Delete with exclusions
    if params.run_delete_w_exclusions:
        exclusions = []
        excl_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for gloss in glosses:
            exclusions.append(delete_w_exclusions.exclusion_delete(gloss, is_segmented))

        for e in exclusions:
            if e is not None:
                excl_final.append(dataset_prep.dataset_prep(e, is_segmented))
        final_list.extend(excl_final)


    # %
    # Insert random word at the start of sentence
    # 20 random words grabbed from the Uspanteko gold standard data.
    noise = [['Rechi\'', 'Rechi\'', 'E3-SREL-ENF', 'E3-SREL-ENF', 'Es de'],
            ['Chiqe', 'Chiqe', 'PREP-SREL', 'PREP-SREL', 'Entre'],
            ['Saneb\'','Saneb\'',  'S', 'arena@de@rio', 'Harenas del río'],
            ['Keqiix', 'Keqiix', 'S', 'culix', 'Kyeqiix'],
            ['Baya', 'Baya', 'VOC', 'baya', 'Baya'],
            ['Inchk','Inchk', 'A1S-S', 'A1S-trabajo', 'Mi trabajo'],
            ['Xte\'', 'Xte\'', 'COM-VT', 'COM-encontrar', 'Encontró'],
            ['Qája', 'Qája', 'E1-S', 'E1-agua', 'El agua'],
            ['Mismo', 'Mismo', 'ADV', 'mismo', 'Mismo'],
            ['Tijk\'ey', 'Tijk\'ey', 'INC-E3S-NEG', 'INC-E3S-NEG', 'No le gusta'],
            ['Tib\'itaq', 'Tib\'itaq', 'INC-VT-PL', 'INC-levantar-PL', 'Levantarse'],
            ['Tijut', 'Tijut', 'INC-VT', 'INC-meter', 'Metía'],
            ['Tilin', 'Tilin', 'NOM', 'catarina', 'Catarina'],
            ['Aqaaj', 'Aqaaj', 'E2S-S', 'E2S-papá', 'Tu papá'],
            ['Tiqatij', 'Tiqatij', 'INC-E1P-VT-ENF', 'INC-E1P-comer-ENF', 'Sufrimos'],
            ['Mrel ánm', 'Mrel ánm', 'ART-DIM S', 'ART-DIM mujer', 'La mujercita'],
            ['Jwi\'l tzaqoomch\'olaal ', 'Jwi\'l tzaqoomch\'olaal ','E3S-SREL VT-PAS-S-SAB', 'E3S-SREL botar-PAS-estómago-SAB', 'Por miedo'],
            ['Kinye\' taq', 'Kinye\' taq',  'VI-VT PL', 'quedar-dar PL', 'Dejarlo'],
            ['Resureksyon', 'Resureksyon', 'S', 'resurección', 'Resurección'],
            ['Tinloq\'e\'', 'Tinloq\'e\'', 'INC-A1S-VT-ENF', 'INC-A1S-comprar-ENF', 'Compro']]

    if params.run_random_insert_noise:
        noise_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for n in noise:
            for gloss in glosses:
                noise_final.append(dataset_prep.dataset_prep(random_insert_noise.random_insert_beginning(gloss, n, is_segmented), is_segmented))
        final_list.extend(noise_final)

    #   Create dataset from augmented data
    aug_dataset = output_dataset.output_dataset(final_list)
    return aug_dataset
