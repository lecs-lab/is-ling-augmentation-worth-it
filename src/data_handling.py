"""Defines models and functions for loading, manipulating, and writing task data"""

from typing import Literal, cast

import datasets
import glossing

from augmentation.aug_generation import AugmentationParameters, aug_generation as usp_gen
from augmentation.aug_generation_arapaho import aug_generation as arp_gen

# from method1_unseg import create_augmented_data as create_m1_data
from utils import AUGMENTATION_TYPE


def create_dataset(
    language: Literal['usp', 'arp'],
    augmentation_type: AUGMENTATION_TYPE,
    params: AugmentationParameters | None,
    sample_train_size: int | None = None,
    seed: int = 0,
):
    dataset_key = {'usp': "usp-igt-resplit", 'arp': "arp-igt"}[language]
    dataset = cast(
        datasets.DatasetDict, datasets.load_dataset(f"lecslab/{dataset_key}")
    ).with_format("torch")

    # Make a small validation split, different each tiem
    train_eval_split = dataset["train"].train_test_split(test_size=0.05, seed=seed)
    dataset["train"] = train_eval_split["train"]
    dataset["eval"] = train_eval_split["test"]

    # Sample some of the training data
    if sample_train_size is not None:
        dataset["train"] = (
            dataset["train"].shuffle(seed=seed).select(range(sample_train_size))
        )  # Need to be deterministic for reproducibility
    initial_train_size = len(dataset["train"])

    # Add in hallucinated data as needed
    def load_aug_data(path: str) -> datasets.Dataset:
        aug_data = glossing.load_igt_file(path)
        return datasets.Dataset.from_list([ex.__dict__() for ex in aug_data])

    if augmentation_type != "baseline":
        print(f"Creating augmented data with method {augmentation_type}...")
        if augmentation_type == "aug_m1":
            raise NotImplementedError()
            # dataset["aug_train"] = create_m1_data(dataset["train"])
        elif augmentation_type == "aug_m2":
            dataset["aug_train"] = load_aug_data("../data/hallucinated/method2.txt")
        elif augmentation_type == "combo":
            if params is None:
                raise ValueError()

            if language == "usp":
                aug_data = usp_gen(
                    initial_dataset=dataset["train"], fraction=1, params=params
                )
            elif language == "arp":
                aug_data = arp_gen(
                    initial_dataset=dataset["train"], fraction=1, params=params
                )
            if len(aug_data) > 0:
                dataset["aug_train"] = aug_data
            else:
                dataset["aug_train"] = dataset["train"]
        else:
            raise Exception("Invalid choice!")

        print(
            f"Created {len(dataset['aug_train'])} augmented rows "
            f"from {initial_train_size} initial_train_size a total of {len(dataset['aug_train']) + initial_train_size}"
        )

    return dataset
