# %%
from datasets import Dataset

from augmentation.aug_parameters import AugmentationParameters


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

    is_segmented = False
    if 'segmentation' in df.columns:
        is_segmented = True

    # %%
    # Word order permutations
    if params.run_sentence_permutations:
        permutations_updates = []
        permutations_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for gloss in glosses:
             permutations_updates.extend(sentence_permutations_arapaho.permute(gloss, is_segmented))
        for p in permutations_updates:
            if p is not None:
                permutations_final.append(dataset_prep_arapaho.dataset_prep(p, is_segmented))

        final_list.extend(permutations_final)


    # %%
    # Insert random interjection, greeting, or conjunction at the start of sentence
    # 20 interjections/greetings/conjunctions grabbed from the Arapaho gold standard data.
    interjections = [['Hih\'oo', 'Hih\'oo', 'allright!', 'All right !'],
                ['Yeah', 'Yeah', 'yeah', 'Yeah'],
                ['Wohei', 'Wohei', 'okay', 'Well'],
                ['Uhm', 'Uhm', 'uhm', 'Uhm'],
                ['Yeheihoo', 'Yeheihoo', 'gee.whiz', 'Gee whiz'],
                ['Yiiih', 'Yiiih', 'laughter.at.joke', 'Haah !'],
                ['Oh', 'Oh', 'oh', 'Oh'],
                ['Uhm-hmm', 'Uhm-hmm',  'uhm-hmm', 'Uhm-hmm'],
                ['Hiiko', 'Hiiko', 'no', 'No'],
                ['Ahm', 'Ahm', 'ahm', 'Ahm'],
                ['Hmm', 'Hmm', 'hmm', 'Hmm'],
                ['Nooxeihi', 'Nooxeihi', 'maybe', 'Maybe'],
                ['Hee', 'Hee', 'yes', 'Yes'],
                ['\'O\'xu\'', '\'O\'xu\'', 'ouch', 'Ouch'],
                ['So', 'So', 'so', 'So'],
                ['Noh', 'Noh', 'and', 'And'],
                ['Wouukohei', 'Wouukohei', 'welcome', 'Welcome'],
                ['Tous', 'Tous', 'hello', 'Hello'],
                ['\'Oh ', '\'Oh ','but', 'But'],
                ['Nii\'ooke\'', 'Nii\'ooke\'', 'IC.good.morning', 'Good morning']]

    if params.run_insert_interjection:
        interjections_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for interjection in interjections:
            for gloss in glosses:
                interjections_final.append(dataset_prep_arapaho.dataset_prep(insert_interjection_arapaho.random_insert_beginning(gloss, interjection, is_segmented), is_segmented))
        final_list.extend(interjections_final)

    # %
    # Insert random word at the start of sentence
    # 20 random words grabbed from the Arapaho gold standard data.
    noises = [['Bih\'ih', 'Bih\'ih', 'mule.deer', 'Mule deer'],
                ['Biiciixo', 'Biiciixo', 'leaves', 'Leaves'],
                ['Yeinou\'u', 'Yeinou\'u', 'tomatoes', 'Tomatoes'],
                ['Nonii', 'Nonii', 'behold', 'Behold'],
                ['Nokokiy', 'Nokokiy', '1S-gun', 'My gun'],
                ['Henee\'', 'Henee\'',  'who?', 'Who'],
                ['Hiikoo\'', 'Hiikoo\'', 'in.the.brush', 'In the brush'],
                ['Beteetox ', 'Beteetox ', 'ten', 'Ten'],
                ['Beh\'eihoho\'', 'Beh\'eihoho\'','old.men', 'Old men'],
                ['Hebes', 'Hebes', 'beaver', 'Beaver'],
                ['Siisiiyei', 'Siisiiyei', 'snake', 'Snake'],
                ['Hiii', 'Hiii', 'snow', 'Snow'],
                ['Niinih\'ohuunoo\'', 'Niinih\'ohuunoo\'', 'airplane', 'Airplane'],
                ['Ciiis', 'Ciiis', 'cheese', 'Cheese'],
                ['Kokoh\'owoot', 'Kokoh\'owoot',  'playing.ball', 'Basketball'],
                ['Wootii', 'Wootii', 'like', 'Like'],
                ['Neiteh\'ei', 'Neiteh\'ei', '1S-friend', 'My friend'],
                ['Hoonou3oot', 'Hoonou3oot', 'Christmas', 'Christmas'],
                ['Tono\'wuuhee', 'Tono\'wuuhee', 'cellar,.hole.in.the.ground', 'Cellar'],
                ['Hoseihoowu\'', 'Hoseihoowu\'', 'Sun.Dance', 'Sun Dance']]

    if params.run_random_insert_noise:
        noise_final = []
        glosses = glosses_to_list.glosses_to_list(df)

        for noise in noises:
            for gloss in glosses:
                noise_final.append(dataset_prep_arapaho.dataset_prep(insert_noise_arapaho.random_insert_beginning(gloss, noise, is_segmented), is_segmented))
        final_list.extend(noise_final)


    #   Create dataset from augmented data
    aug_dataset = output_dataset.output_dataset(final_list)
    return aug_dataset
