from dataclasses import dataclass
from dataclass_click import option
from typing import Annotated

@dataclass
class AugmentationParameters:
    # Shared
    run_random_insert_noise: Annotated[bool, option(is_flag=True)] = False

    # Uspanteko
    run_random_insert_conj: Annotated[bool, option(is_flag=True)] = False
    run_tam_update: Annotated[bool, option(is_flag=True)] = False
    run_random_duplicate: Annotated[bool, option(is_flag=True)] = False
    run_random_delete: Annotated[bool, option(is_flag=True)] = False
    run_delete_w_exclusions: Annotated[bool, option(is_flag=True)] = False

    # Arapaho
    run_insert_interjection: Annotated[bool, option(is_flag=True)] = False
    run_sentence_permutations: Annotated[bool, option(is_flag=True)] = False
