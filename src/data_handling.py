"""Defines models and functions for loading, manipulating, and writing task data"""
from functools import reduce
from typing import Optional, List, Union, cast, Literal
import datasets
import glossing
from method1_unseg import create_augmented_data as create_m1_data

from utils import AUGMENTATION_TYPE

def create_dataset(
    model_type: AUGMENTATION_TYPE,
    sample_train_size: int | None = None,
    seed: int = 0
):
    dataset = cast(
        datasets.DatasetDict, datasets.load_dataset("lecslab/usp-igt-resplit")
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

    if model_type != "baseline":
        print(f"Creating augmented data with method {model_type}...")
        if model_type == "aug_m1":
            dataset["aug_train"] = create_m1_data(dataset["train"])
        elif model_type == "aug_m2":
            dataset["aug_train"] = load_aug_data("../data/hallucinated/method2.txt")
        else:
            raise Exception("Invalid choice!")

        print(
            f"Created {len(dataset['aug_train'])} augmented rows "
            f"from {initial_train_size} initial_train_size a total of {len(dataset['aug_train']) + initial_train_size}"
        )

    return dataset
