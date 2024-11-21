# %%
from dataclasses import dataclass
from dataclass_click import option
from typing import Annotated
from datasets import Dataset

@dataclass
class AugmentationParameters:
    run_random_insert_conj: Annotated[bool, option(is_flag=True)] = False
    run_tam_update: Annotated[bool, option(is_flag=True)] = False
    run_random_duplicate: Annotated[bool, option(is_flag=True)] = False
    run_random_delete: Annotated[bool, option(is_flag=True)] = False
    run_delete_w_exclusions: Annotated[bool, option(is_flag=True)] = False
    run_random_insert_noise: Annotated[bool, option(is_flag=True)] = False

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

    final_list = []

    # %%
    # TAM update
    if params.run_tam_update:
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

    if params.run_random_insert_conj:
        conj_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for conj in conjs:
            for gloss in glosses:
                conj_final.append(dataset_prep.dataset_prep(random_insert_conj.random_insert_beginning(gloss, conj)))
        final_list.extend(conj_final)

    # %%
    # Random Duplicates
    if params.run_random_duplicate:
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
    if params.run_random_delete:
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
    if params.run_delete_w_exclusions:
        exclusions = []
        excl_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for gloss in glosses:
            exclusions.append(delete_w_exclusions.exclusion_delete(gloss))

        for e in exclusions:
            if e is not None:
                excl_final.append(dataset_prep.dataset_prep(e))
        final_list.extend(excl_final)


    # %
    # Insert random word at the start of sentence
    # 20 random words grabbed from the Uspanteko gold standard data.
    noise = [['Rechi\'', 'E3-SREL-ENF', 'E3-SREL-ENF', 'Es de'],
            ['Chiqe', 'PREP-SREL', 'PREP-SREL', 'Entre'],
            ['Saneb\'', 'S', 'arena@de@rio', 'Harenas del río'],
            ['Keqiix', 'S', 'culix', 'Kyeqiix'],
            ['Baya', 'VOC', 'baya', 'Baya'],
            ['Inchk', 'A1S-S', 'A1S-trabajo', 'Mi trabajo'],
            ['Xte\'', 'COM-VT', 'COM-encontrar', 'Encontró'],
            ['Qája', 'E1-S', 'E1-agua', 'El agua'],
            ['Mismo', 'ADV', 'mismo', 'Mismo'],
            ['Tijk\'ey', 'INC-E3S-NEG', 'INC-E3S-NEG', 'No le gusta'],
            ['Tib\'itaq', 'INC-VT-PL', 'INC-levantar-PL', 'Levantarse'],
            ['Tijut', 'INC-VT', 'INC-meter', 'Metía'],
            ['Tilin', 'NOM', 'catarina', 'Catarina'],
            ['Aqaaj', 'E2S-S', 'E2S-papá', 'Tu papá'],
            ['Tiqatij', 'INC-E1P-VT-ENF', 'INC-E1P-comer-ENF', 'Sufrimos'],
            ['Mrel ánm', 'ART-DIM S', 'ART-DIM mujer', 'La mujercita'],
            ['Jwi\'l tzaqoomch\'olaal ', 'E3S-SREL VT-PAS-S-SAB', 'E3S-SREL botar-PAS-estómago-SAB', 'Por miedo'],
            ['Kinye\' taq', 'VI-VT PL', 'quedar-dar PL', 'Dejarlo'],
            ['Resureksyon', 'S', 'resurección', 'Resurección'],
            ['Tinloq\'e\'', 'INC-A1S-VT-ENF', 'INC-A1S-comprar-ENF', 'Compro']]

    if params.run_random_insert_noise:
        noise_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for n in noise:
            for gloss in glosses:
                noise_final.append(dataset_prep.dataset_prep(random_insert_noise.random_insert_beginning(gloss, n)))
        final_list.extend(noise_final)

    #   Create dataset from augmented data
    aug_dataset = output_dataset.output_dataset(final_list)
    return aug_dataset
