"""Maps improperly named aug keys to use underscores"""

import wandb
from tqdm import tqdm

project = "augmorph/augmorph-usp-transl-transc"
runs = wandb.Api().runs(path=project)

mapped = 0
for run in tqdm(runs):
    mapped_row = False

    config = run.config
    if "aug" in run.config:
        mapped_row = True
        for key in run.config["aug"]:
            run.config["aug_" + key] = run.config["aug"][key]
        del run.config["aug"]

    for key in run.config:
        if key.startswith("aug."):
            mapped_row = True
            run.config[key.replace("aug.", "aug_")] = run.config[key]
            del run.config[key]

    if "aug_random_insert_noise" in run.config:
        mapped_row = True
        del run.config["aug_random_insert_noise"]

    for missing_key in ["aug_run_random_insert_noise", "aug_run_insert_interjection", "aug_run_sentence_permutations"]:
        if missing_key not in run.config:
            mapped_row = True
            run.config[missing_key] = False

    if mapped_row:
        mapped += 1
    run.update()

print(f"Successfully mapped {mapped} runs")
