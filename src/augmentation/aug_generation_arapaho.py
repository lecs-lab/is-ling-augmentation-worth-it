# %%
from datasets import Dataset

from src.augmentation.aug_parameters import AugmentationParameters


def aug_generation(
    initial_dataset: Dataset,
    fraction: float = 1,
    params: AugmentationParameters = AugmentationParameters()
):
    # Import locally so we don't do this every time we use the module
    from augmentation.aug_utils import sentence_permutations_arapaho, glosses_to_list, create_dataframe_arapaho, dataset_prep_arapaho, \
        insert_noise_arapaho, insert_interjection_arapaho, output_dataset

    # %%
    # Load in data
    df = create_dataframe_arapaho.create_dataframe(dataset=initial_dataset, fraction=fraction)
    df = df.dropna()

    # %%
    df.head()

    final_list = []

    # %%
    # Word order permutations
    if params.run_sentence_permutations:
        permutations_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for gloss in glosses:
            permutations_final.append(dataset_prep_arapaho.dataset_prep(sentence_permutations_arapaho.permute(gloss)))

        final_list.extend(permutations_final)


    # %%
    # Insert random interjection, greeting, or conjunction at the start of sentence
    # 20 interjections/greetings/conjunctions grabbed from the Arapaho gold standard data.
    interjections = [['Hih\'oo', 'allright!', 'All right !'],
                ['Yeah', 'yeah', 'Yeah'],
                ['Wohei', 'okay', 'Well'],
                ['Uhm', 'uhm', 'Uhm'],
                ['Yeheihoo', 'gee.whiz', 'Gee whiz'],
                ['Yiiih', 'laughter.at.joke', 'Haah !'],
                ['Oh', 'oh', 'Oh'],
                ['Uhm-hmm', 'uhm-hmm', 'Uhm-hmm'],
                ['Hiiko', 'no', 'No'],
                ['Ahm', 'ahm', 'Ahm'],
                ['Hmm', 'hmm', 'Hmm'],
                ['Nooxeihi', 'maybe', 'Maybe'],
                ['Hee', 'yes', 'Yes'],
                ['\'O\'xu\'', 'ouch', 'Ouch'],
                ['So', 'so', 'So'],
                ['Noh', 'and', 'And'],
                ['Wouukohei', 'welcome', 'Welcome'],
                ['Tous', 'hello', 'Hello'],
                ['\'Oh ', 'but', 'But'],
                ['Nii\'ooke\'', 'IC.good.morning', 'Good morning']]

    if params.run_insert_interjection:
        interjections_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for interjection in interjections:
            for gloss in glosses:
                interjections_final.append(dataset_prep_arapaho.dataset_prep(insert_interjection_arapaho.random_insert_beginning(gloss, interjection)))
        final_list.extend(interjections_final)

    # %
    # Insert random word at the start of sentence
    # 20 random words grabbed from the Arapaho gold standard data.
    noises = [['Bih\'ih', 'mule.deer', 'Mule deer'],
                ['Biiciixo', 'leaves', 'Leaves'],
                ['Yeinou\'u', 'tomatoes', 'Tomatoes'],
                ['Nonii', 'behold', 'Behold'],
                ['Nokokiy', '1S-gun', 'My gun'],
                ['Henee\'', 'who?', 'Who'],
                ['Hiikoo\'', 'in.the.brush', 'In the brush'],
                ['Beteetox ', 'ten', 'Ten'],
                ['Beh\'eihoho\'', 'old.men', 'Old men'],
                ['Hebes', 'beaver', 'Beaver'],
                ['Siisiiyei', 'snake', 'Snake'],
                ['Hiii', 'snow', 'Snow'],
                ['Niinih\'ohuunoo\'', 'airplane', 'Airplane'],
                ['Ciiis', 'cheese', 'Cheese'],
                ['Kokoh\'owoot', 'playing.ball', 'Basketball'],
                ['Wootii', 'like', 'Like'],
                ['Neiteh\'ei', '1S-friend', 'My friend'],
                ['Hoonou3oot', 'Christmas', 'Christmas'],
                ['Tono\'wuuhee', 'cellar,.hole.in.the.ground', 'Cellar'],
                ['Hoseihoowu\'', 'Sun.Dance', 'Sun Dance']]

    if params.run_random_insert_noise:
        noise_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for noise in noises:
            for gloss in glosses:
                noise_final.append(dataset_prep_arapaho.dataset_prep(insert_noise_arapaho.random_insert_beginning(gloss, noise)))
        final_list.extend(noise_final)


    #   Create dataset from augmented data
    aug_dataset = output_dataset.output_dataset(final_list)
    return aug_dataset
